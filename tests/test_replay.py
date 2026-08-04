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

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.replay import (
    Mismatch,
    _advance_after_row,
    _apply_pending_drops,
    _dedupe_income_commands,
    _row_mismatches,
    replay_game,
)
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, Phase, cult_string, with_faction


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.BOOKKEEPING, raw=verb, **fields)  # type: ignore[arg-type]

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


# --------------------------------------------------------------------------
# Task 14: Cultists' bundled gain_cult + cult_income_for_faction row is a
# raw-ledger duplicate; a batch missing a different faction's own row must
# still grant that faction its income before end_of_round fires.
# --------------------------------------------------------------------------


def test_dedupe_income_commands_drops_cult_income_bundled_with_gain_cult() -> None:
    """Corpus ``4pLeague_S32_D3L3_G7`` row 187 shape (``_dedupe_income_
    commands``'s own docstring, full citation trail): a ``cult_income_
    for_faction`` command sharing its row with a ``gain_cult`` is a
    raw-ledger duplicate (always Cultists, 29 corpus occurrences, this
    exact bundling) and must be dropped -- every other command in the row
    is untouched.
    """
    cmds = (_cmd("gain_cult", cult="EARTH", n1=1), _cmd("cult_income_for_faction"))
    result = _dedupe_income_commands(cmds)
    assert result == (_cmd("gain_cult", cult="EARTH", n1=1),)


def test_dedupe_income_commands_leaves_a_standalone_cult_income_row_alone() -> None:
    cmds = (_cmd("cult_income_for_faction"),)
    assert _dedupe_income_commands(cmds) == cmds


def test_dedupe_income_commands_leaves_other_income_alone_even_with_gain_cult() -> None:
    """The bundling anomaly is specific to ``cult_income_for_faction`` --
    ``other_income_for_faction`` is never affected, even sharing a row
    with ``gain_cult`` (not an observed corpus shape, but the rule must
    not overreach to it regardless)."""
    cmds = (_cmd("gain_cult", cult="EARTH", n1=1), _cmd("other_income_for_faction"))
    assert _dedupe_income_commands(cmds) == cmds


def _round_flow_state() -> GameState:
    s = GameState.initial(load_setup(GAME_ID))
    # score_tiles[1] (round 2's own tile): EARTH, req 1, income {"C": 1}.
    return replace(s, round=2, phase=Phase.CLEANUP)


def test_advance_after_row_grants_missing_cult_income_before_early_end_of_round() -> None:
    """Corpus ``4pLeague_S32_D3L3_G7`` shape: Witches' own
    ``cult_income_for_faction`` row is genuinely absent from its round's
    batch. The *other* three factions' rows arrive, then a different
    faction's ``other_income_for_faction`` row (proof the cult-income
    phase ended, module docstring) forces ``end_of_round`` early --
    Witches must still receive that round's cult income
    (round 2's tile here: EARTH, req 1, income ``{"C": 1}``) at that
    point, not lose it outright for lack of a ledger row.
    """
    s = _round_flow_state()
    factions = dict(s.factions)
    factions["darklings"] = replace(factions["darklings"], coins=0)
    s = replace(s, factions=factions, cults={**s.cults, "darklings": {**s.cults["darklings"], "EARTH": 4}})

    other_income_done: set[str] = set()
    cult_income_done: set[str] = {"engineers", "nomads", "mermaids"}  # darklings' row never arrives
    s2, other_income_done, cult_income_done = _advance_after_row(
        s, "engineers", (_cmd("other_income_for_faction"),), other_income_done, cult_income_done
    )
    assert s2.factions["darklings"].coins == 4  # floor(4/1) * 1 C, granted despite no ledger row
    assert s2.phase == Phase.INCOME  # end_of_round did fire
    assert cult_income_done == set()
    assert other_income_done == {"engineers"}


def test_advance_after_row_skips_missing_cult_income_for_a_dropped_faction() -> None:
    """A dropped faction is never owed further income at all (every other
    post-drop exclusion in this module agrees) -- the catch-up must not
    grant it one just because its row is (unsurprisingly) also absent.
    """
    s = _round_flow_state()
    factions = dict(s.factions)
    factions["darklings"] = replace(factions["darklings"], coins=0, dropped=True)
    s = replace(s, factions=factions, cults={**s.cults, "darklings": {**s.cults["darklings"], "EARTH": 4}})

    cult_income_done: set[str] = {"engineers", "nomads", "mermaids"}
    s2, _, _ = _advance_after_row(
        s, "engineers", (_cmd("other_income_for_faction"),), set(), cult_income_done
    )
    assert s2.factions["darklings"].coins == 0  # not granted -- dropped


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


def test_apply_pending_drops_leaves_a_not_yet_dropped_faction_alone() -> None:
    """A dropped faction hasn't necessarily dropped *yet* by any given
    row -- ``_apply_pending_drops`` only applies the drop once the ledger
    has moved strictly past its exact ``dropped_at_row`` entry, never
    before."""
    state = GameState.initial(load_setup(GAME_ID))
    fs = replace(state.factions["darklings"], bonus="BON1")
    state = with_faction(state, "darklings", fs)
    state2 = _apply_pending_drops(state, row=50, dropped_at_row={"darklings": 60})
    assert state2.factions["darklings"].bonus == "BON1"
    assert not state2.factions["darklings"].dropped


def test_apply_pending_drops_releases_bonus_and_marks_dropped_once_past_the_drop_row() -> None:
    """Task-14 fix (dropped-faction ledger model): the raw ledger's own
    ``"<faction> dropped from the game"`` comment (``GameSetup.
    dropped_at_row`` docstring) pins the exact row the drop happened --
    ``_apply_pending_drops`` applies it, proactively, the moment the
    ledger crosses that row: releases the held bonus tile and permanently
    marks the faction ``dropped`` (excluded from turn order for the rest
    of the game, unlike the old ``passed``-based marker this replaced,
    which round_flow's own ``end_of_round`` would have reset every
    round). Matters for ``round_flow._bumped_bonus_coins`` timing (that
    function's own citation trail): a tile still marked "held" too long
    accrues no coins for the rounds in between.
    """
    state = GameState.initial(load_setup(GAME_ID))
    fs = replace(state.factions["darklings"], bonus="BON1")
    state = with_faction(state, "darklings", fs)
    state2 = _apply_pending_drops(state, row=61, dropped_at_row={"darklings": 60})
    assert state2.factions["darklings"].bonus is None
    assert state2.factions["darklings"].dropped


def test_apply_pending_drops_advances_away_even_when_the_dropped_factions_own_row_is_next() -> None:
    """The drop *comment* is not always the chronologically-last thing a
    faction does -- corpus ``4pLeague_S22_D3L1_G1`` row 32 ("mermaids
    dropped from the game") is immediately followed by row 33, mermaids'
    own ``build F4``. Task-14 fix (superseding an earlier revision that
    special-cased this): that row's own ``deltas.parquet`` entry proved
    byte-identical to mermaids' pre-drop state -- a genuine no-op in real
    Perl, not a legitimately-attempted action -- so there is no "rightful
    next turn" left to protect. ``_apply_pending_drops`` now advances
    ``active_index`` away from a just-dropped faction unconditionally,
    even when the very next row still names that same (now-dropped)
    faction (``replay_game`` is what actually skips applying that row's
    commands, per its own docstring -- not this function).
    """
    state = GameState.initial(load_setup(GAME_ID))
    state = replace(
        state, phase=Phase.SETUP_DWELLINGS, turn_order=("darklings", "engineers"), active_index=0
    )
    state2 = _apply_pending_drops(state, row=61, dropped_at_row={"darklings": 60})
    assert state2.factions["darklings"].dropped
    assert state2.active_index == 1  # advanced past darklings to engineers regardless


def test_dropped_faction_is_excluded_from_turn_order(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S12_D2L1_G6``: darklings drops (raw JSON ``"dropped":
    1``, ledger comment "darklings dropped from the game" at row 62)
    partway through round 1, after playing real early moves. Pins the
    replay reaching well past where the dropped faction's turn would
    otherwise strand the harness (row 51's turn-order gate, then row
    111's ``pass is not legal during INCOME``, both fixed together) --
    not necessarily a fully clean replay for this specific game, since
    dropped-faction games can carry other, unrelated mismatches this
    task's loop didn't chase further.
    """
    assert load_setup("4pLeague_S12_D2L1_G6").dropped_at_row == {"darklings": 62}
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S12_D2L1_G6", moves_df, deltas_df)
    assert result.rows_checked > 130


def test_dropped_factions_still_standing_building_avoids_the_isolated_tp_surcharge(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S64_D1L1_G6``: alchemists drops at row 276; row 348's
    nomads ``upgrade E3 to TP`` is directly adjacent to two of
    alchemists' still-standing dwellings. ``leech.has_leechable_neighbor``
    fix (that module's docstring, full citation trail): this must not be
    charged the "isolated" doubled TP cost (6 C) just because alchemists
    fell out of ``offers_for_build``'s live-faction seat-order walk under
    ``variable-turn-order`` -- the real ledger charges the un-isolated
    3 C. A full clean replay (not just past row 348) since this game has
    no other unrelated mismatches once this fix lands.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S64_D1L1_G6", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


def test_dropped_faction_mid_setup_dwellings_loses_both_snake_slots(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S45_D3L4_G1``: darklings drops at row 30, exactly when
    its own forward-order dwelling pick would start (round 0,
    ``Phase.SETUP_DWELLINGS``) -- ``round_flow._first_live_setup_index``
    fix (that function's own citation trail): darklings loses *both* its
    dwelling-snake slots (forward and reverse) outright, not just the one
    it was about to take. A full clean replay, not just past row 30/31.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S45_D3L4_G1", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


def test_dropped_faction_mid_setup_bonus_loses_its_bonus_tile_pick(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S12_D2L1_G4``: engineers drops at row 38, mid-
    ``Phase.SETUP_BONUS`` (after its own dwelling picks, before its own
    reverse-order bonus-tile pick) -- same
    ``round_flow._first_live_setup_index`` fix, exercised on the
    ``SETUP_BONUS`` phase instead of ``SETUP_DWELLINGS``. A full clean
    replay.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S12_D2L1_G4", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


def test_dropped_faction_never_strands_round_1_active_index_at_seat_zero(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S21_D3L1_G4``: darklings (seat index 0) drops mid-
    ``SETUP_BONUS`` (row 41); ``round_flow.begin_actions``'s
    ``_first_eligible_index`` fix (that function's own citation trail)
    is what lets round 1 correctly start with engineers instead of
    wrongly re-seeding ``active_index=0`` straight back onto the dropped
    darklings. A full clean replay.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S21_D3L1_G4", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


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


# --------------------------------------------------------------------------
# Task 14 phase 4: round-0 (SETUP_DWELLINGS/SETUP_BONUS) has no active-
# faction enforcement in real Perl, and a dropped faction's own stale
# in-flight setup row is applied under whichever faction `setup_order`'s
# front now names, not the identity baked into the request.
# --------------------------------------------------------------------------


def test_dropped_factions_stale_setup_row_applies_under_the_new_setup_order_front(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S22_D3L1_G1``: mermaids drops right as its own first
    ``SETUP_DWELLINGS`` turn would start. Row 33's ledger `faction` is
    still "mermaids" (`build F4`, a stale already-in-flight client
    submission), but real Perl's raw JSON `map` has F4 colored green --
    witches' home color, not mermaids' blue -- because `commands.pm`'s
    `$assert_active_faction` skips its active-player check entirely for
    `$game{round} == 0`, and `acting.pm`'s `setup_action`/
    `shift_setup_order` blindly pops whichever faction `setup_order`'s
    front now names (witches, once mermaids' own remaining entries are
    filtered out by the drop) regardless of who actually submitted the
    command. Applying this row under mermaids' own (dropped) identity
    either hard-errors (`EngineError`: wrong home color, since F4 isn't
    blue) or, if skipped as a no-op, strands this engine's own
    `_setup_dwellings_order` pointer one slot behind real Perl's --
    surfacing several rows later as a spurious "faction acted out of
    turn" on cultists' very next legitimate build. A full clean replay.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S22_D3L1_G1", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


def test_dropped_factions_stale_setup_row_can_apply_under_a_later_faction_too(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S22_D3L1_G6``: witches drops right as its own first
    ``SETUP_DWELLINGS`` turn would start (a 4-player game with Nomads, so
    `setup_order`'s post-drop front lands on nomads' *forward* dwelling
    slot, not the very next faction in seat order). Row 31's ledger
    `faction` is "witches" (`build F3`), but the raw JSON's `map` has F3
    colored yellow -- nomads' home color -- and nomads' own row 137
    (`upgrade F3 to TP`) has no intervening "build F3" anywhere in the
    ledger, so row 31 must be where nomads' dwelling actually landed.
    Same fix as the sibling `S22_D3L1_G1` test, pinned separately because
    the post-drop `setup_order` front here is a *different* faction than
    the very next seat, exercising `_first_live_setup_index`'s general
    skip-ahead rather than the simple case.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S22_D3L1_G6", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()


def test_dropped_factions_stale_setup_row_resolves_without_further_fixes(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """``4pLeague_S22_D3L1_G4``: alchemists drops mid-``SETUP_BONUS``; the
    same round-0 setup-order-front substitution (no separate fix needed)
    resolves this game's own "faction acted out of turn" too, confirming
    the fix generalizes across both `SETUP_DWELLINGS` and `SETUP_BONUS`.
    """
    moves_df, deltas_df = frames
    result = replay_game("4pLeague_S22_D3L1_G4", moves_df, deltas_df)
    assert result.error is None, result.error
    assert result.mismatches == ()
