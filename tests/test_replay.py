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

from dataclasses import replace

import polars as pl
import pytest

from bgai.engine.tm.replay import Mismatch, _release_finished_drops, _row_mismatches, replay_game
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, cult_string, with_faction

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
# Task 14: a round missing one faction's cult_income_for_faction row
# entirely must not strand the harness in Phase.CLEANUP forever
# --------------------------------------------------------------------------


def test_missing_cult_income_row_does_not_strand_the_harness_in_cleanup(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S11_D3L1_G5``: the raw ledger's round 1->2 transition
    (verified directly against the crawled game JSON) has
    ``cult_income_for_faction`` rows for cultists (x2)/witches/
    chaosmagicians but *none* for darklings -- a genuine ledger gap, not a
    parsing bug. ``_advance_after_row``'s old "wait for every faction's
    cult_income_for_faction row" gate never fires when a faction's row is
    simply absent, stranding the harness in ``Phase.CLEANUP`` for the rest
    of the game -- eventually a hard error much later (row 128,
    ``EngineError: power action space ACT4 is blocked this round``, since
    ``command_start``'s per-round unblock never ran either). Seeing a
    round's first ``other_income_for_faction`` row while still in CLEANUP
    is itself proof the cult-income phase has ended, missing rows or not.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S11_D3L1_G5", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()
    assert result.rows_checked > 300


def test_seat_order_rotation_without_variable_turn_order(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S1_D1L1_G1`` (no ``variable-turn-order``): round 2's
    turn order starts with whoever passed first in round 1 (rotating
    *seat* order, not simply reusing it unrotated) -- and round 3's start
    diverges from round 2's plain pass order, since a faction that passes
    early keeps its original seat *position* for the next round's
    rotation rather than jumping to the front by pass *timestamp*
    (``round_flow.end_of_round``'s own docstring has the full citation
    trail). Before this fix, an earlier revision either always reused raw
    seat order (never rotated) or always used passed_order directly
    (right only under variable-turn-order) -- both wrong here, producing
    "faction acted out of turn" hard errors by round 3.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S1_D1L1_G1", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()
    assert result.rows_checked > 300


def test_release_finished_drops_leaves_a_still_playing_faction_alone() -> None:
    """A dropped faction hasn't necessarily dropped *yet* by any given
    row -- ``_release_finished_drops`` only releases it once the ledger
    has moved strictly past its last-ever appearance, never before."""
    state = GameState.initial(load_setup(GAME_ID))
    fs = replace(state.factions["darklings"], bonus="BON1")
    state = with_faction(state, "darklings", fs)
    state2 = _release_finished_drops(state, row=50, dropped_last_row={"darklings": 60})
    assert state2.factions["darklings"].bonus == "BON1"
    assert not state2.factions["darklings"].passed


def test_release_finished_drops_releases_bonus_and_marks_passed_once_past_last_row() -> None:
    """Task-14 fix: proactively releases a dropped faction's held bonus
    tile (and marks it passed) as soon as the ledger crosses its last-
    ever appearance -- earlier than ``_skip_dropped_factions``'s reactive
    trigger, which only fires once *another* faction's turn-order
    mismatch reveals the drop. Matters for ``round_flow._bumped_bonus_
    coins`` timing (that function's own citation trail): a tile still
    marked "held" too long accrues no coins for the rounds in between.
    """
    state = GameState.initial(load_setup(GAME_ID))
    fs = replace(state.factions["darklings"], bonus="BON1")
    state = with_faction(state, "darklings", fs)
    state2 = _release_finished_drops(state, row=61, dropped_last_row={"darklings": 60})
    assert state2.factions["darklings"].bonus is None
    assert state2.factions["darklings"].passed


def test_dropped_faction_is_skipped_in_the_round_robin(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S12_D2L1_G6``: darklings drops (raw JSON ``"dropped":
    1``) partway through round 1, after playing real early moves --
    ``GameSetup.dropped_factions``/``replay._skip_dropped_factions``
    reactively skip its turn once the ledger's own silence about it makes
    that clear, since no ``drop-faction`` ledger verb exists to pinpoint
    exactly when. Pins the replay reaching well past where the dropped
    faction's turn would otherwise strand the harness (row 51's turn-order
    gate, then row 111's ``pass is not legal during INCOME``, both fixed
    together) -- not necessarily a fully clean replay for this specific
    game, since dropped-faction games can carry other, unrelated
    mismatches this task's loop didn't chase further.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S12_D2L1_G6", moves_df, deltas_df)
    assert result.rows_checked > 130


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


def test_cult_blocked_does_not_survive_into_a_later_turn(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S10_D3L1_G4``: chaosmagicians' FIRE cult step is capped
    at 9 for lack of a key at row 329 (``gain_favor FAV5``); a key arrives
    7 rows/one full turn-cycle later at row 336 (``gain_town TW3``), in a
    different turn. Before the task-14 fix, ``round_flow.py``'s turn
    machinery never reset ``FactionState.cult_blocked`` between turns, so
    ``_retry_blocked_cults`` wrongly retroactively bumped FIRE to 10 at
    row 336 -- Perl's own ``start_full_move`` deletes that memory every
    fresh turn, so a key from an unrelated later turn has nothing queued
    to retry."""
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S10_D3L1_G4", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()
    assert result.rows_checked > 300
