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


def test_weight_power_sharpens_strength_weighting(tmp_path) -> None:
    """The baked-in weights span only 0.6-1.0 (1.67x), so training treats
    an elite table almost like a weak one. weight_power re-shapes them at
    load time, without rebuilding a 2.5 GB shard set."""
    import numpy as np

    from bgai.training.dataset import ImitationDataset

    build(tmp_path, limit=4)
    flat = ImitationDataset(tmp_path, "train", weight_power=1.0)
    sharp = ImitationDataset(tmp_path, "train", weight_power=8.0)
    fw = np.array([flat[i]["weight"] for i in range(min(200, len(flat)))])
    sw = np.array([sharp[i]["weight"] for i in range(min(200, len(sharp)))])
    assert np.allclose(sw, fw**8)
    if fw.max() > fw.min():
        assert (sw.max() / max(sw.min(), 1e-9)) > (fw.max() / max(fw.min(), 1e-9))


def test_mover_mu_is_stored_and_min_mover_mu_keeps_only_stronger_movers(tmp_path) -> None:
    import pytest

    from bgai.training.dataset import ImitationDataset
    from bgai.training.vocab import SHARD_FORMAT

    build(tmp_path, limit=4)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["shard_format"] == SHARD_FORMAT
    mus = np.concatenate(
        [np.load(tmp_path / s["file"])["mover_mu"] for s in manifest["shards"]]
    )
    assert mus.dtype == np.float32 and np.isfinite(mus).any()
    assert ((mus > 0) | np.isnan(mus)).all()

    everyone = ImitationDataset(tmp_path, "train")
    cut = float(np.nanmedian(mus))
    strong = ImitationDataset(tmp_path, "train", min_mover_mu=cut)
    assert 0 < len(strong) < len(everyone)
    shards = [np.load(tmp_path / s["file"])["mover_mu"] for s in manifest["shards"]]
    assert all(shards[s][r] >= cut for s, r in strong.index)
    assert set(strong.index) <= set(everyone.index)

    manifest["shard_format"] = SHARD_FORMAT - 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format"):
        ImitationDataset(tmp_path, "train")


def test_parts_merge_into_the_full_build(tmp_path: Path, monkeypatch) -> None:
    import bgai.training.dataset_build as db

    monkeypatch.setattr(db, "GAMES_PER_SHARD", 2)
    full, parted = tmp_path / "full", tmp_path / "parted"
    db.build(full, limit=5)
    for i in range(2):
        db.build(parted, limit=5, part=(i, 2))
    db.merge_parts(parted)

    want = json.loads((full / "manifest.json").read_text())
    got = json.loads((parted / "manifest.json").read_text())
    assert got == want
    assert not list(parted.glob("manifest.part*.json"))
    for shard in want["shards"]:
        a, b = np.load(full / shard["file"]), np.load(parted / shard["file"])
        assert all(np.array_equal(a[k], b[k], equal_nan=k == "mover_mu") for k in a.files)
