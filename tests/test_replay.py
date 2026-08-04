"""tests/test_replay.py

Validates :mod:`bgai.engine.tm.replay`'s deltas-oracle comparison in
isolation on a synthetic state (task brief: "Mismatch comparison logic
unit-tested on a synthetic case before running real games"), then replays
the reference game (``4pLeague_S10_D1L1_G1``) and the first 10 *loadable*
corpus games end to end.

Note on "first 10 games": Task 2 found 10 corpus games with ``nofaction``
placeholder seats that raise ``ValueError`` from ``load_setup``. None of
those fall within the first 10 games of ``games_meta.parquet`` sorted by
``game_id`` (verified directly -- the first 10 sorted game_ids are all
loadable), so ``_first_n_loadable_game_ids`` below still skips unloadable
games defensively for whichever corpus slice a future change to
``games_meta`` might produce, but no skip is currently exercised.
"""

from __future__ import annotations

import polars as pl
import pytest

from bgai.engine.tm.replay import Mismatch, _row_mismatches, replay_game
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, cult_string

GAME_ID = "4pLeague_S10_D1L1_G1"


@pytest.fixture(scope="session")
def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
    )


# --------------------------------------------------------------------------
# Step 1: deltas-oracle comparison, synthetic case (no real game involved)
# --------------------------------------------------------------------------


def test_row_mismatches_is_empty_when_every_field_matches() -> None:
    state = GameState.initial(load_setup(GAME_ID))
    faction = "engineers"
    fs = state.factions[faction]
    delta = {
        "vp_value": fs.vp,
        "c_value": fs.coins,
        "w_value": fs.workers,
        "p_value": fs.priests,
        "pw": fs.power.as_str(),
        "cult": cult_string(state, faction),
    }
    assert _row_mismatches(GAME_ID, 1, faction, state, delta, "raw text") == []


def test_row_mismatches_reports_only_the_differing_fields() -> None:
    state = GameState.initial(load_setup(GAME_ID))
    faction = "engineers"
    fs = state.factions[faction]
    delta = {
        "vp_value": fs.vp + 5,  # wrong
        "c_value": fs.coins,  # matches
        "w_value": fs.workers,  # matches
        "p_value": fs.priests,  # matches
        "pw": "9/9/9",  # wrong
        "cult": cult_string(state, faction),  # matches
    }
    mismatches = _row_mismatches(GAME_ID, 7, faction, state, delta, "raw text")
    assert {m.field for m in mismatches} == {"vp", "pw"}
    assert mismatches[0] == Mismatch(
        game_id=GAME_ID,
        row=7,
        faction=faction,
        field="vp",
        expected=str(fs.vp + 5),
        actual=str(fs.vp),
        raw="raw text",
    )


# --------------------------------------------------------------------------
# Step 2/3: real games
# --------------------------------------------------------------------------


def test_reference_game_replays_clean(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    r = replay_game(GAME_ID, *frames)
    assert r.error is None
    assert r.mismatches == ()
    assert r.rows_checked > 200


def _first_n_loadable_game_ids(n: int) -> list[str]:
    games_meta = pl.read_parquet("data/datasets/games_meta.parquet").sort("game_id")
    loadable: list[str] = []
    for game_id in games_meta["game_id"].to_list():
        try:
            load_setup(game_id)
        except ValueError:
            continue
        loadable.append(game_id)
        if len(loadable) == n:
            break
    return loadable


def test_first_ten_games_replay_clean(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    moves_df, deltas_df = frames
    for game_id in _first_n_loadable_game_ids(10):
        result = replay_game(game_id, moves_df, deltas_df)
        assert result.error is None, f"{game_id}: {result.error}"
        assert result.mismatches == (), f"{game_id}: {result.mismatches}"
        assert result.rows_checked > 0


# --------------------------------------------------------------------------
# Task 14: ACTC compound-turn bundling (round_flow.is_turn_boundary)
# --------------------------------------------------------------------------


def test_actc_bundled_row_replays_past_its_own_row_without_a_turn_order_error(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S10_D2L1_G4`` row 309 (``convert; pass BON1``, chaos
    magicians) is the first row after the game's chaosmagicians ACTC row
    (302: ``action ACTC; action FAV6; gain_cult n1=1; pass BON3``) that a
    different faction (engineers) acts in. Before this fix, the harness
    called ``advance_turn`` once per row regardless of how many Perl-level
    full actions it bundled -- 1 call for a 3-action ACTC row -- so
    ``active_index`` never made it past chaosmagicians and this exact row
    raised ``EngineError: faction acted out of turn (active is
    'chaosmagicians')`` before any oracle row past 302 could even be
    checked. Pinning full-game replay (not just the row) since the bug was
    in how many rows the harness could reach at all, not a value mismatch.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S10_D2L1_G4", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()
    assert result.rows_checked > 309
