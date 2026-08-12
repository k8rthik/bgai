"""Checkpoint retention rules.

Three failures found the hard way on the first population-scale run:
the best weights were overwritten by whatever validated last, the
trailing partial interval was trained and then discarded, and the run
wrote into a directory that already held another run's checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bgai.training.dataset_build import build
from bgai.training.train import TrainConfig, train

GAMES = [
    "4pLeague_S1_D1L1_G1",
    "4pLeague_S10_D1L1_G1",
    "4pLeague_S70_D3L1_G4",
    "4pLeague_S69_D3L3_G1",
]


@pytest.fixture(scope="module")
def shards(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("shards")
    build(out, game_ids=GAMES)
    return out


def test_refuses_a_non_empty_output_directory(shards: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoint.pt").write_bytes(b"someone else's run")
    with pytest.raises(ValueError, match="not empty"):
        train(TrainConfig(shards=shards, out=out, epochs=1, max_steps=1, device="cpu"))
    assert (out / "checkpoint.pt").read_bytes() == b"someone else's run"


def test_best_and_last_are_both_written(shards: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = TrainConfig(
        shards=shards,
        out=out,
        epochs=1,
        batch_size=32,
        device="cpu",
        max_steps=4,
        val_every_steps=2,
        log_every=1000,
    )
    train(cfg)
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()

    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert rows, "expected validation rows"
    best_row = min(rows, key=lambda r: r["loss"])

    import torch

    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert best["val"]["loss"] == pytest.approx(best_row["loss"])


def test_final_step_is_validated_not_discarded(shards: Path, tmp_path: Path) -> None:
    """With a step schedule, the trailing steps that do not land on the
    interval must still be validated and saved."""
    out = tmp_path / "run"
    cfg = TrainConfig(
        shards=shards,
        out=out,
        epochs=1,
        batch_size=32,
        device="cpu",
        max_steps=5,  # not a multiple of val_every_steps
        val_every_steps=2,
        log_every=1000,
    )
    result = train(cfg)
    assert result["step"] == 5, "final partial interval must be validated"
