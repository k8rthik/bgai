"""Shard builder (plan 2026-08-04-imitation-phase5, Task 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bgai.training.dataset_build import build
from bgai.training.vocab import ENCODING_VERSION


def test_build_mini_dataset(tmp_path: Path) -> None:
    stats = build(tmp_path, limit=6)
    assert stats.record_count > 300
    assert stats.unmatched == 0
    assert stats.failed_games == ()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["encoding_version"] == ENCODING_VERSION
    assert manifest["record_count"] == stats.record_count
    assert sum(s["records"] for s in manifest["shards"]) == stats.record_count

    shard = np.load(tmp_path / manifest["shards"][0]["file"])
    n = shard["chosen"].shape[0]
    assert shard["hex_planes"].shape == (n, 113, 17)
    assert shard["cand_counts"].sum() == shard["cand_flat"].shape[0]
    assert (shard["chosen"] < shard["cand_counts"]).all()
    assert (shard["cand_counts"] >= 2).all()
    assert shard["final_vps"].shape == (n, 4)
