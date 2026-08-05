"""Torch dataset + collate over imitation shards (plan Task 6).

Shards are loaded lazily per worker and cached; candidate sets are ragged
so ``collate`` pads to the batch maximum and returns a bool mask the
model uses to blank out padding before the softmax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.vocab import ENCODING_VERSION

Split = Literal["train", "val"]


class ImitationDataset(Dataset):
    """Decision records from a shard directory, filtered to one split.

    Split is by season (``manifest['val_season_min']``): a time-based
    boundary, so validation contains only games played after every
    training game -- no future leakage (master plan Phase 5).
    """

    def __init__(self, shard_dir: Path, split: Split) -> None:
        self.shard_dir = Path(shard_dir)
        manifest = json.loads((self.shard_dir / "manifest.json").read_text())
        if manifest["encoding_version"] != ENCODING_VERSION:
            raise ValueError(
                f"shards were built with encoding v{manifest['encoding_version']}, "
                f"code is v{ENCODING_VERSION} -- rebuild the dataset"
            )
        self.val_season_min = int(manifest["val_season_min"])
        self.division_weights = {
            int(k): float(v) for k, v in manifest["division_weights"].items()
        }
        self._files = [self.shard_dir / s["file"] for s in manifest["shards"]]
        self._cache: dict[int, dict[str, np.ndarray]] = {}

        # (shard_index, row_index) for every record on this side of the split
        self.index: list[tuple[int, int]] = []
        for shard_index, path in enumerate(self._files):
            with np.load(path) as data:
                seasons = data["season"]
            is_val = seasons >= self.val_season_min
            keep = np.flatnonzero(is_val if split == "val" else ~is_val)
            self.index.extend((shard_index, int(row)) for row in keep)

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.get(shard_index)
        if cached is None:
            with np.load(self._files[shard_index]) as data:
                cached = {k: data[k] for k in data.files}
            # offsets for the flat candidate array
            cached["cand_offsets"] = np.concatenate(
                [[0], np.cumsum(cached["cand_counts"])]
            ).astype(np.int64)
            self._cache[shard_index] = cached
        return cached

    def __getitem__(self, i: int) -> dict[str, Any]:
        shard_index, row = self.index[i]
        shard = self._shard(shard_index)
        start = int(shard["cand_offsets"][row])
        end = int(shard["cand_offsets"][row + 1])
        division = int(shard["division"][row])
        final = shard["final_vps"][row].astype(np.float32)
        # value target: each seat's share of the table's final VP, so the
        # head is scale-free and sums to 1 across the 4 mover-relative seats
        share = final / max(float(final.sum()), 1.0)
        return {
            "hex_planes": torch.from_numpy(shard["hex_planes"][row].astype(np.float32)),
            "globals": torch.from_numpy(shard["globals"][row].astype(np.float32)),
            "candidates": torch.from_numpy(shard["cand_flat"][start:end].astype(np.int64)),
            "chosen": int(shard["chosen"][row]),
            "value": torch.from_numpy(share),
            "weight": float(self.division_weights.get(division, 1.0)),
            "faction": int(shard["mover_faction"][row]),
        }


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_cands = max(item["candidates"].shape[0] for item in batch)
    n = len(batch)
    cand = torch.zeros((n, max_cands, MOVE_FIELDS), dtype=torch.long)
    mask = torch.zeros((n, max_cands), dtype=torch.bool)
    for i, item in enumerate(batch):
        k = item["candidates"].shape[0]
        cand[i, :k] = item["candidates"]
        mask[i, :k] = True
    return {
        "hex_planes": torch.stack([b["hex_planes"] for b in batch]),
        "globals": torch.stack([b["globals"] for b in batch]),
        "candidates": cand,
        "cand_mask": mask,
        "chosen": torch.tensor([b["chosen"] for b in batch], dtype=torch.long),
        "value": torch.stack([b["value"] for b in batch]),
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
        "faction": torch.tensor([b["faction"] for b in batch], dtype=torch.long),
    }
