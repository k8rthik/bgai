"""Training loop smoke test (plan 2026-08-04-imitation-phase5, Task 7)."""

from __future__ import annotations

from pathlib import Path

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
    assert (out / "checkpoint.pt").exists()
    assert (out / "metrics.jsonl").exists()
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
    assert ckpt["encoding_version"] == ENCODING_VERSION
    assert ckpt["step"] == 8
    assert 0.0 <= last["top1"] <= 1.0 and last["top3"] >= last["top1"]


def test_pick_device_respects_explicit_request() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "mps", "cuda"}
