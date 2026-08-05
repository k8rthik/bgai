"""Decision extraction (plan 2026-08-04-imitation-phase5, Tasks 2/5)."""

from __future__ import annotations

import polars as pl
import pytest

from bgai.data.ledger_parser import Kind
from bgai.engine.tm.replay import replay_game


@pytest.fixture(scope="module")
def dfs() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
    )


def test_capture_hook_transparent_and_fires(dfs) -> None:
    moves_df, deltas_df = dfs
    base = replay_game("4pLeague_S10_D1L1_G1", moves_df, deltas_df)
    captured = []

    def hook(state, faction, cmd):
        captured.append((state.round, faction, cmd))

    hooked = replay_game("4pLeague_S10_D1L1_G1", moves_df, deltas_df, on_decision=hook)
    assert (hooked.error, hooked.mismatches) == (base.error, base.mismatches)
    assert base.error is None and base.mismatches == ()
    assert hooked.rows_checked == base.rows_checked
    assert len(captured) > 100
    assert all(cmd.kind is Kind.DECISION for _, _, cmd in captured)
    # first decisions are setup dwelling placements (round 0)
    assert captured[0][0] == 0 and captured[0][2].verb == "build"
