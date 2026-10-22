"""Training loop smoke test (plan 2026-08-04-imitation-phase5, Task 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bgai.training.dataset_build import build
from bgai.training.train import TrainConfig, pick_device, train
from bgai.training.vocab import ENCODING_VERSION


def test_train_runs_and_checkpoints(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    # span the season split (val_season_min=67) so both sides are non-empty
    build(
        shards,
        game_ids=[
            "4pLeague_S1_D1L1_G1",
            "4pLeague_S10_D1L1_G1",
            "4pLeague_S70_D3L1_G4",
            "4pLeague_S69_D3L3_G1",
        ],
    )
    out = tmp_path / "run"
    last = train(
        TrainConfig(
            shards=shards,
            out=out,
            epochs=1,
            batch_size=32,
            device="cpu",
            max_steps=8,
            log_every=1000,
        )
    )
    assert (out / "last.pt").exists()
    assert (out / "metrics.jsonl").exists()
    ckpt = torch.load(out / "last.pt", weights_only=False)
    assert ckpt["encoding_version"] == ENCODING_VERSION
    assert ckpt["step"] == 8
    # the checkpoint pins its full architecture (see test_model_config_checkpoint)
    assert ckpt["config"]["model_config"]["hidden"] == 1024
    assert 0.0 <= last["top1"] <= 1.0 and last["top3"] >= last["top1"]


def test_pick_device_respects_explicit_request() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}


def test_parse_arch_overrides_types_and_rejects_unknown_fields() -> None:
    from bgai.training.train import parse_arch

    assert parse_arch("") == {}
    assert parse_arch("torso=hexgnn,hex_dim=64,depth=4,dropout=0.05") == {
        "torso": "hexgnn", "hex_dim": 64, "depth": 4, "dropout": 0.05,
    }
    import pytest

    with pytest.raises(ValueError, match="nope"):
        parse_arch("nope=1")
    with pytest.raises(ValueError, match="value_simplex"):
        parse_arch("value_simplex=1")  # has its own flag; one source of truth


def test_train_builds_the_requested_architecture(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    build(shards, game_ids=["4pLeague_S1_D1L1_G1", "4pLeague_S70_D3L1_G4"])
    out = tmp_path / "run"
    train(TrainConfig(shards=shards, out=out, epochs=1, batch_size=16, device="cpu",
                      max_steps=2, log_every=1000,
                      arch={"torso": "resmlp", "hidden": 64, "depth": 2}))
    cfg = torch.load(out / "last.pt", weights_only=False)["config"]["model_config"]
    assert (cfg["torso"], cfg["hidden"], cfg["depth"]) == ("resmlp", 64, 2)


def test_lr_schedule_constant_and_cosine() -> None:
    from bgai.training.train import lr_at

    assert lr_at(0, 1e-3, "constant", 100, 10) == lr_at(99, 1e-3, "constant", 100, 10) == 1e-3
    # cosine: linear warmup, peak at warmup end, decays to the floor at max_steps
    assert lr_at(0, 1e-3, "cosine", 100, 10) < lr_at(5, 1e-3, "cosine", 100, 10) < 1e-3
    assert lr_at(10, 1e-3, "cosine", 100, 10) == 1e-3
    mid, end = lr_at(55, 1e-3, "cosine", 100, 10), lr_at(100, 1e-3, "cosine", 100, 10)
    assert 0.4e-3 < mid < 0.6e-3 and end == pytest.approx(1e-5)
    assert lr_at(500, 1e-3, "cosine", 100, 10) == pytest.approx(1e-5)  # past the end: floor


def test_cosine_schedule_needs_a_step_budget(tmp_path: Path) -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="max_steps"):
        train(TrainConfig(shards=tmp_path, out=tmp_path / "o", lr_schedule="cosine"))
