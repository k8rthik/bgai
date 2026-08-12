"""Shard-block sampling: shuffle within a bounded working set of shards.

A globally shuffled index over the population corpus is unusable. Shards
decompress to ~193 MB each and there are 323 of them (62 GB), so they
cannot all stay resident; but with a global shuffle every batch of 512
draws from ~512 distinct shards, so *any* bounded cache misses on nearly
every access and re-decompresses a 193 MB shard per sample.

This sampler walks shards in a random order, in blocks of
``shards_per_block``, and shuffles all records *within* the current block.
The resident set is therefore bounded by the block size, while records
still arrive well mixed -- the standard trade for sharded corpora.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ShardBlockSampler(Sampler[int]):
    """Yields dataset positions grouped into blocks of whole shards.

    ``index`` is the dataset's ``(shard_index, row)`` list; the sampler
    emits positions into it, so it composes with a normal ``DataLoader``
    via ``sampler=``.
    """

    def __init__(
        self,
        index: Sequence[tuple[int, int]],
        shards_per_block: int,
        seed: int = 0,
    ) -> None:
        if shards_per_block < 1:
            raise ValueError(f"shards_per_block must be >= 1, got {shards_per_block}")
        by_shard: dict[int, list[int]] = defaultdict(list)
        for position, (shard, _row) in enumerate(index):
            by_shard[shard].append(position)
        self._by_shard = {
            shard: np.asarray(positions, dtype=np.int64)
            for shard, positions in by_shard.items()
        }
        self._shards_per_block = shards_per_block
        self._seed = seed
        self._epoch = 0
        self._length = len(index)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the next pass. Without this every epoch would
        present the same block composition and the same order."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self._seed, self._epoch])
        shards = rng.permutation(np.fromiter(self._by_shard, dtype=np.int64))
        for start in range(0, len(shards), self._shards_per_block):
            block = shards[start : start + self._shards_per_block]
            positions = np.concatenate([self._by_shard[int(s)] for s in block])
            rng.shuffle(positions)
            yield from (int(p) for p in positions)
