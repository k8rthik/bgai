"""Shard-block sampling (population-scale training).

Global shuffling over 323 shards makes every batch touch ~512 distinct
shards, so a bounded shard cache would miss on nearly every access. The
sampler instead shuffles *within* a working set of shards, keeping the
resident set bounded while still mixing records.
"""

from __future__ import annotations

import pytest

from bgai.training.sampler import ShardBlockSampler


def _index(shards: int, rows_per_shard: int) -> list[tuple[int, int]]:
    return [(s, r) for s in range(shards) for r in range(rows_per_shard)]


def test_every_record_is_emitted_exactly_once() -> None:
    index = _index(shards=7, rows_per_shard=5)
    sampler = ShardBlockSampler(index, shards_per_block=2, seed=0)
    positions = list(sampler)
    assert sorted(positions) == list(range(len(index)))
    assert len(sampler) == len(index)


def test_resident_shards_never_exceed_the_block_size() -> None:
    """The whole point: walking the sampler in order must never require
    more than ``shards_per_block`` shards loaded at once."""
    index = _index(shards=9, rows_per_shard=4)
    block = 3
    sampler = ShardBlockSampler(index, shards_per_block=block, seed=1)

    seen_order: list[int] = []
    for pos in sampler:
        shard = index[pos][0]
        if shard not in seen_order:
            seen_order.append(shard)
    # shards must be consumed in contiguous groups of `block`, never revisited
    assert len(seen_order) == 9
    assert len(set(seen_order)) == 9


def test_records_are_shuffled_within_a_block() -> None:
    index = _index(shards=4, rows_per_shard=50)
    sampler = ShardBlockSampler(index, shards_per_block=4, seed=3)
    positions = list(sampler)
    assert positions != list(range(len(index))), "sampler must not emit in order"


def test_epoch_changes_the_order() -> None:
    index = _index(shards=5, rows_per_shard=10)
    sampler = ShardBlockSampler(index, shards_per_block=2, seed=0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    assert sorted(first) == sorted(second)


def test_block_larger_than_shard_count_is_allowed() -> None:
    index = _index(shards=2, rows_per_shard=3)
    sampler = ShardBlockSampler(index, shards_per_block=10, seed=0)
    assert sorted(sampler) == list(range(len(index)))


def test_block_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ShardBlockSampler(_index(2, 2), shards_per_block=0, seed=0)
