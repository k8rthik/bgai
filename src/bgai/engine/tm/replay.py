"""Replay harness: drive a crawled game's ledger through ``apply()`` and
cross-check the result against the ``deltas.parquet`` oracle.

This is the module where every earlier task's handler meets a real game for
the first time -- ``round_flow.py``'s "Task 12 contract" docstring already
specifies the exact call sequence a row-grouping caller owes the engine;
this module *is* that caller.

**Row grouping.** ``moves.parquet`` has no per-row "this is the
turn-owning verb" flag, but the contract only needs one boolean per
command: does it use a main-track verb (``_MAIN_TRACK_VERBS`` -- exactly
round_flow's own list: build/upgrade/action/advance/pass/send/dig/
transform/bridge/connect), or is it a sub-decision answer (leech/decline/
gain_favor/gain_town/gain_cult) or a bookkeeping row (setup/income/
score_*)? Almost every ledger row is exactly one full action (one
``advance_turn`` call, e.g. "dig 1. build A3. connect R1. gain_town TW1",
one Mermaids turn, corpus row 208 -- round_flow's own docstring cites
this), but Chaos Magicians' ACTC can grant 2 *extra* full actions that get
submitted as part of the *same* row as the ACTC action itself -- e.g. row
308, "action ACTC. action BON2. +1CULT. pass BON3" is 3 Perl-level full
actions bundled into one row. ``_apply_row_commands`` calls
``round_flow.is_turn_boundary`` per command (that module's own docstring
has the full citation trail) to decide, per main-track command, whether it
completes a fresh full action (warranting its own ``advance_turn`` call)
or is a continuation of the row's already-in-progress action (a transform/
build spending spades or a marker an earlier command in the same row just
granted) -- with a floor of at least one call for any row containing a
main-track verb, matching the previous (pre-ACTC-fix) behavior for the
common single-action-per-row case. This decides ``advance_turn``
eligibility across all three phases that use it (``SETUP_DWELLINGS``/
``SETUP_BONUS``/``ACTIONS`` -- ``advance_turn`` is a documented no-op for
``INCOME``/``CLEANUP``/``FINISHED``).

``other_income_for_faction``/``cult_income_for_faction``/
``all_income_for_faction`` rows are gate-exempt and phase-agnostic
(round_flow's Step-1 finding: row order is provably irrelevant), so this
harness tracks which factions have received each income component *this
round* and fires ``begin_actions``/``end_of_round`` once every faction is
accounted for, mirroring rather than re-deriving the corpus's own
grouping. The ``end_of_round`` check runs before the ``begin_actions``
check each row so a same-row ``all_income_for_faction`` batch
(``merge-income-phases``, not exercised by the reference game or the
first 10 corpus games -- see this module's tests) can't spuriously fire
``begin_actions`` off carry-over bookkeeping from the ``end_of_round``
call that just ran. A faction's ``cult_income_for_faction`` row can be
missing from the ledger entirely (a genuine corpus gap, not a parsing
bug -- verified directly against the raw crawled JSON, task-14 fix:
``4pLeague_S11_D3L1_G5``'s round 1->2 transition simply never emits one
for darklings), so "every faction accounted for" can never become true on
its own; seeing the round's first ``other_income_for_faction`` row while
still in ``Phase.CLEANUP`` is itself proof the cult-income phase ended
regardless, and forces ``end_of_round`` early in that case.

**Deltas oracle.** After every command of a ledger row has been applied
(and any phase transition above has run), if ``deltas.parquet`` has a row
for ``(game_id, row, faction)`` its ``vp``/``c``/``w``/``p`` *absolute*
values, ``pw`` bowl string, and ``cult`` string are compared against the
just-applied state for that faction (task brief). A mismatch never aborts
the replay outright -- it is collected as a :class:`Mismatch` and replay
continues, up to ``stop_after`` mismatches, so one bad row doesn't hide
the next distinct bug class. An exception raised by ``apply()`` (or by the
phase-transition calls above) *does* abort -- caught, wrapped with row
context, and returned as :class:`ReplayResult.error`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import apply
from bgai.engine.tm.round_flow import (
    advance_turn,
    begin_actions,
    end_of_round,
    grant_missing_cult_income,
    is_turn_boundary,
    never_starts_action,
    start_setup,
)
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, Phase, active_faction, cult_string, with_faction

_MAIN_TRACK_VERBS = frozenset(
    {"build", "upgrade", "action", "advance", "pass", "send", "dig", "transform", "bridge", "connect"}
)
_OTHER_INCOME_VERBS = frozenset({"other_income_for_faction", "all_income_for_faction"})
_CULT_INCOME_VERBS = frozenset({"cult_income_for_faction", "all_income_for_faction"})
# Split ``other_income_for_faction`` back out from ``_OTHER_INCOME_VERBS``
# for the "cult-income phase implicitly ended" check below -- that check
# must not also fire on ``all_income_for_faction`` (the merge-income-phases
# option, still deferred per the task brief) the moment its first faction's
# row lands, since that verb legitimately opens *both* phases' bookkeeping
# on its own, phase-agnostically, in one row.
_PURE_OTHER_INCOME_VERBS = frozenset({"other_income_for_faction"})

_MOVES_PATH = Path("data/datasets/moves.parquet")
_DELTAS_PATH = Path("data/datasets/deltas.parquet")
_GAMES_META_PATH = Path("data/datasets/games_meta.parquet")


@dataclass(frozen=True)
class Mismatch:
    game_id: str
    row: int
    faction: str
    field: str
    expected: str
    actual: str
    raw: str


@dataclass(frozen=True)
class ReplayResult:
    game_id: str
    rows_checked: int
    mismatches: tuple[Mismatch, ...]
    error: str | None


# --------------------------------------------------------------------------
# Row reconstruction: moves.parquet record -> ParsedCommand
# --------------------------------------------------------------------------

_COMMAND_FIELDS = (
    "loc", "loc2", "building", "tile", "cult", "color", "target", "reason", "res1", "res2", "n1", "n2",
)


def _reconstruct_raw(rec: dict[str, Any]) -> str:
    """``moves.parquet`` doesn't carry the original ledger text -- this
    rebuilds a readable stand-in from the parsed fields, used only for
    ``EngineError`` context and ``Mismatch.raw`` diagnostics (never
    re-parsed).
    """
    parts = [str(rec["verb"])]
    for key in _COMMAND_FIELDS:
        value = rec[key]
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _parsed_command_from_row(rec: dict[str, Any]) -> ParsedCommand:
    return ParsedCommand(
        verb=rec["verb"],
        kind=Kind(rec["kind"]),
        raw=_reconstruct_raw(rec),
        **{field: rec[field] for field in _COMMAND_FIELDS},
    )


def _iter_rows(game_moves: pl.DataFrame) -> list[tuple[int, str, tuple[ParsedCommand, ...]]]:
    """``(row, faction, commands)`` triples in ledger order (``row`` then
    ``seq`` ascending -- caller sorts). One faction per row (verified
    against the corpus: no row ever mixes two factions' commands).
    """
    rows: list[tuple[int, str, tuple[ParsedCommand, ...]]] = []
    current_row: int | None = None
    current_faction = ""
    current_cmds: list[ParsedCommand] = []
    for rec in game_moves.iter_rows(named=True):
        row = int(rec["row"])
        if row != current_row:
            if current_row is not None:
                rows.append((current_row, current_faction, tuple(current_cmds)))
            current_row = row
            current_faction = rec["faction"]
            current_cmds = []
        current_cmds.append(_parsed_command_from_row(rec))
    if current_row is not None:
        rows.append((current_row, current_faction, tuple(current_cmds)))
    return rows


def _delta_lookup(deltas_df: pl.DataFrame, game_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    game_deltas = deltas_df.filter(pl.col("game_id") == game_id)
    return {(int(rec["row"]), rec["faction"]): rec for rec in game_deltas.iter_rows(named=True)}


# --------------------------------------------------------------------------
# Deltas-oracle comparison (unit-tested standalone on a synthetic case --
# see tests/test_replay.py -- before it is ever run against real games).
# --------------------------------------------------------------------------


def _row_mismatches(
    game_id: str, row: int, faction: str, state: GameState, delta: dict[str, Any], raw: str
) -> list[Mismatch]:
    fs = state.factions[faction]
    checks = (
        ("vp", delta["vp_value"], fs.vp),
        ("c", delta["c_value"], fs.coins),
        ("w", delta["w_value"], fs.workers),
        ("p", delta["p_value"], fs.priests),
        ("pw", delta["pw"], fs.power.as_str()),
        ("cult", delta["cult"], cult_string(state, faction)),
    )
    return [
        Mismatch(
            game_id=game_id, row=row, faction=faction, field=field,
            expected=str(expected), actual=str(actual), raw=raw,
        )
        for field, expected, actual in checks
        if str(expected) != str(actual)
    ]


# --------------------------------------------------------------------------
# Phase-transition driving (round_flow.py's "Task 12 contract")
# --------------------------------------------------------------------------


def _apply_pending_drops(state: GameState, row: int, dropped_at_row: Mapping[str, int]) -> GameState:
    """Apply every faction's drop event whose exact ledger row
    (``GameSetup.dropped_at_row``, sourced straight from the raw ledger's
    own ``"<faction> dropped from the game"`` comment) the replay has now
    moved past -- proactively and precisely, mirroring ``commands.pm``'s
    ``drop-faction`` handler (~1578-1602) as the single atomic event it
    is in real Perl: ``allowed_actions`` zeroed (this engine's
    ``extra_actions``), the held bonus tile released back to the pool
    (``find_bonus_to_discard``/``adjust_resource(-1)``), and the faction
    permanently marked ``FactionState.dropped`` -- excluded from turn
    order for the rest of the game by ``round_flow._advance_actions``'s
    own ``not fs.dropped`` check, matching ``acting.pm``'s ``!dropped``
    filter everywhere (``next_faction_in_turn``, ``in_play``'s
    ``all_passed``).

    Called once per iterated row, before that row's commands are applied.
    A drop's own comment row is comment-only (no ``commands`` key), so it
    never survives into ``moves.parquet`` (``build_moves.py`` skips any
    ledger row without one) and is therefore never itself an iterated row
    here -- this function is how its effect still lands, exactly once,
    the first time ``row`` moves strictly past it.

    Every applicable drop is committed to ``state`` before any turn-order
    fixup runs, so a fixup that lands on a second faction whose own drop
    row is also ``< row`` (both dropped within the same processed-row gap)
    sees it as already ``dropped`` too, rather than depending on dict
    iteration order.

    The turn-order fixup now fires **unconditionally** whenever the
    just-dropped faction was ``active_faction`` -- no longer gated to
    "only when the upcoming row belongs to someone else" (task-14 fix,
    superseding an earlier revision of this function). That gate was
    solving the wrong problem: ``commands.pm``'s drop handler sets
    ``allowed_actions = 0`` and calls ``dismiss_action`` as part of the
    single atomic drop event, so a dropped faction can never act again,
    full stop -- there is no "rightful next turn" for even an
    already-in-flight same-faction row to receive. Corpus
    ``4pLeague_S22_D3L1_G1`` row 32 ("mermaids dropped from the game")
    is immediately followed by row 33, mermaids' own ``build F4``: the
    row's own ``deltas.parquet`` entry is byte-identical to mermaids'
    *pre-drop* row 28 -- zero state change, proving real Perl treated it
    as a no-op, not a legitimately-attempted-then-failed action. This
    function no longer takes an ``upcoming_faction`` parameter at all
    (the caller, ``replay_game``, now skips applying a dropped faction's
    own row's commands entirely -- see its own docstring -- so there is
    nothing left for a same-faction exception to protect).
    """
    newly_dropped = [
        faction
        for faction, drop_row in dropped_at_row.items()
        if row > drop_row and not state.factions[faction].dropped
    ]
    for faction in newly_dropped:
        fs = state.factions[faction]
        state = with_faction(state, faction, replace(fs, dropped=True, bonus=None, extra_actions=0))
    # If the faction that just dropped happened to still be
    # ``active_faction`` (e.g. it dropped between its own last action and
    # the next row this harness sees), ``round_flow.advance_turn``'s
    # phase-appropriate skip-forward loop (``_advance_setup_dwellings``/
    # ``_advance_setup_bonus``/``_advance_actions``, all now dropped-aware)
    # resolves it in one call, however many consecutive dropped entries
    # that takes -- not gated to ``Phase.ACTIONS`` alone (task-14 fix,
    # corpus: several games drop a faction mid-``SETUP_DWELLINGS``/
    # ``SETUP_BONUS``, before round 1 even starts, e.g.
    # ``4pLeague_S45_D3L4_G1`` row 30). ``advance_turn`` is already a
    # documented no-op outside these three phases, so calling it
    # unconditionally here is safe.
    if newly_dropped and active_faction(state) in newly_dropped:
        state = advance_turn(state)
    return state


def _apply_row_commands(
    state: GameState, faction: str, cmds: tuple[ParsedCommand, ...], *, oracle_cult: str | None = None
) -> GameState:
    """Apply every command of one ledger row via ``apply()``, calling
    ``advance_turn`` once per genuinely independent full action within the
    row (``round_flow.is_turn_boundary`` -- ordinarily exactly one, but a
    Chaos Magicians ACTC-funded row can bundle several, module docstring
    "Row grouping").

    A full action's own continuations (a build claiming the spades a
    preceding transform/dig/ACT5/ACT6/BON1 just granted) must finish
    applying *before* ``advance_turn`` closes it out -- closing the moment
    ``action ACT6`` itself lands would hand control to the next faction
    before this faction's own ``transform``/``build`` sub-commands in the
    *same* row get to run at all. So the boundary call fires lazily: right
    *before* applying the next command ``is_turn_boundary`` classifies as
    fresh (never before), and once more at the end of the row to close
    whatever full action is still open. A row-*leading* ``transform``/
    ``connect`` (``never_starts_action``) never opens a segment by itself
    either -- it waits to be folded into the row's real fresh command,
    whichever side of it that lands on (corpus: ``connect R10. gain_town
    TW8. pass`` -- ``pass``, not the leading ``connect``, is the actual
    action; closing right after ``connect`` would hand control away before
    ``pass`` could even apply). Either way, at least one call is guaranteed
    for any row containing a main-track verb -- the floor that covers a
    row where *every* command is a never-starts-action verb (e.g. a lone
    ``transform`` row).

    ``oracle_cult`` is this row's deltas-oracle ``cult`` string for
    ``faction`` (``replay_game``'s own delta lookup, looked up once per
    row and threaded straight through) -- passed to every ``apply()`` call
    in the row, but only ``gain_town``'s handler ever reads it (module
    docstring "Deltas oracle"; ``apply.py``/``actions_build.py``'s
    ``_cult_gain_order`` docstrings have the full citation trail). Task 14,
    user adjudication 2026-08-04.
    """
    prev_verb: str | None = None
    open_action = False
    any_main_track = False
    for cmd in cmds:
        if cmd.verb not in _MAIN_TRACK_VERBS:
            state = apply(state, faction, cmd, oracle_cult=oracle_cult)
            continue
        any_main_track = True
        if open_action and is_turn_boundary(cmd, state, faction, prev_verb):
            state = advance_turn(state)
            open_action = False
        state = apply(state, faction, cmd, oracle_cult=oracle_cult)
        prev_verb = cmd.verb
        if open_action or not never_starts_action(cmd.verb):
            open_action = True
    if any_main_track:
        state = advance_turn(state)
    return state


def _dedupe_income_commands(cmds: tuple[ParsedCommand, ...]) -> tuple[ParsedCommand, ...]:
    """Drop a ``cult_income_for_faction`` command that shares its ledger
    row with a ``gain_cult`` command.

    A raw-ledger anomaly, always on Cultists (29 occurrences corpus-wide,
    every one this exact shape): task-14 fix, corpus
    ``4pLeague_S32_D3L3_G7`` rows 187/190 -- row 187 reads ``gain_cult
    cult=EARTH n1=1; cult_income_for_faction`` (Cultists' reactive
    leech-bonus resolution, ``leech.py``'s ``[opponent accepted power]``
    mechanic, bundled with a ``cult_income_for_faction`` in the *same*
    row) and grants **0** C in the real ledger despite Cultists' EARTH
    position already being 9 by then; row 190, a lone standalone
    ``cult_income_for_faction`` a few rows later with no ``gain_cult``
    alongside it, is where the real +9 C lands. Every one of the 22
    *other* corpus occurrences of this same bundling shape (this replay
    harness already passes those clean) also has a bundled row that
    grants 0 -- consistent with the same real Perl behavior (this bundled
    entry is not a real income event; likely a crawler/parser artifact of
    the ledger-writer using a similar entry format for both events),
    just not previously visible as a bug because those particular games'
    cult positions happened to make 0 the "expected" value anyway even
    without this fix. The always-following standalone
    ``cult_income_for_faction`` row (this engine's own sequencing
    already computes an identical result whichever of the two rows
    actually grants, since the position doesn't change between them) is
    always left untouched -- this only ever removes the bundled one.
    """
    has_gain_cult = any(cmd.verb == "gain_cult" for cmd in cmds)
    if not has_gain_cult:
        return cmds
    return tuple(cmd for cmd in cmds if cmd.verb != "cult_income_for_faction")


def _ensure_actions_phase_started(state: GameState, cmds: tuple[ParsedCommand, ...]) -> GameState:
    """Symmetric counterpart to the CLEANUP-phase check in
    ``_advance_after_row``: a round's main-track ACTIONS-phase rows only
    ever start once every faction's ``other_income_for_faction`` row has
    landed and ``begin_actions`` has fired -- so a main-track command
    arriving while still ``Phase.INCOME`` is itself proof that phase has
    genuinely ended, missing rows or not (a dropped faction, task-14 fix,
    never gets an ``other_income_for_faction`` row for any round after it
    drops either -- the mirror image of the missing-``cult_income_for_
    faction`` case). Must run *before* the row's commands are applied
    (unlike the CLEANUP-side check, which can react afterward): a stale
    ``Phase.INCOME`` makes ``handle_pass`` raise immediately
    (``actions_pass.py``: "pass is not legal during INCOME") before any
    later bookkeeping would get a chance to fix the phase.

    ``transform`` is deliberately excluded from the trigger set: task-13
    fix #6 established that a cult-income SPADE payout legitimately forces
    an immediate out-of-turn ``transform`` *while still* ``Phase.INCOME``
    (reference-game row 98) -- treating that as "actions phase started"
    would be wrong and reintroduces that already-fixed bug.
    """
    trigger_verbs = _MAIN_TRACK_VERBS - {"transform"}
    if state.phase == Phase.INCOME and any(cmd.verb in trigger_verbs for cmd in cmds):
        state = begin_actions(state)
    return state


def _cult_income_pending_later(income_rows: list[tuple[int, str, str]], row: int, faction: str) -> bool:
    """Scans ``income_rows`` (precomputed once per game: every
    ``(row, faction, verb)`` for the three income verbs, sorted by row)
    forward from just past ``row`` for ``faction``'s own
    ``cult_income_for_faction``/``all_income_for_faction`` entry, stopping
    at the first *any-faction* ``other_income_for_faction``/
    ``all_income_for_faction`` row (proof, per ``_ensure_actions_phase_
    started``'s own reasoning, that the cult-income batch has ended).
    Returns ``True`` the moment ``faction``'s own entry is found -- a real,
    just-not-yet-reached row, not a genuinely missing one.
    """
    for r, f, verb in income_rows:
        if r <= row:
            continue
        if verb in _CULT_INCOME_VERBS and f == faction:
            return True
        if verb in _OTHER_INCOME_VERBS:
            return False
    return False


def _ensure_cult_income_landed(
    state: GameState,
    apply_faction: str,
    cmds: tuple[ParsedCommand, ...],
    cult_income_done: set[str],
    row: int,
    income_rows: list[tuple[int, str, str]],
) -> tuple[GameState, set[str]]:
    """Mirror image of ``_ensure_actions_phase_started``'s INCOME-side
    check, scoped to a *single* faction rather than the whole-batch
    ``Phase.CLEANUP`` -> ``Phase.INCOME`` transition: a faction's own
    genuinely-missing ``cult_income_for_faction`` row (module docstring,
    ``4pLeague_S11_D3L1_G5``'s darklings precedent) can grant a SPADE that
    forces an immediate out-of-turn ``transform`` *within the same
    cult-income batch*, before any other faction's row would prove the
    whole batch is over -- ``_grant_missing_cult_income`` alone only
    catches this once ``end_of_round`` fires, too late for a `transform`
    stranded mid-batch waiting on a spade nothing has granted yet. Corpus
    ``4pLeague_S7_D3L3_G4`` row 147: darklings never gets a round 2 -> 3
    ``cult_income_for_faction`` row at all (confirmed absent, not just
    reordered, by scanning every darklings row between the prior round's
    income and the next), yet row 147 is darklings' own ``transform H7``
    forced by exactly that missing grant's SPADE.

    ``_cult_income_pending_later`` (look-ahead, required -- task-14 fix)
    guards against double-granting: a main-track verb landing on
    ``apply_faction`` during ``Phase.CLEANUP`` isn't *always* proof this
    faction's own income is missing -- corpus ``4pLeague_S49_D3L1_G7`` row
    323, darklings' own ``transform H8`` lands 6 rows *before* their very
    real row 329 ``cult_income_for_faction`` (some other, unrelated spade
    source, not investigated further), which an earlier, unguarded
    revision of this function double-granted by assuming any not-yet-seen
    faction was missing outright.

    Row order is provably irrelevant to a cult-income amount (module
    docstring, "Step 1: ... row order is provably irrelevant") -- granting
    a faction's income a few rows early changes nothing about the amount,
    since only that faction's own cult-track position (unaffected by any
    other faction's actions) determines it.
    """
    if state.phase != Phase.CLEANUP or apply_faction in cult_income_done:
        return state, cult_income_done
    if state.factions[apply_faction].dropped:
        return state, cult_income_done
    # This same row's own cult_income_for_faction command (if any) must be
    # left to apply normally -- proactively granting here too would
    # double-grant it (real row, not missing; just bundled with the
    # main-track command it unblocks in the same ledger row).
    if any(cmd.verb in _CULT_INCOME_VERBS for cmd in cmds):
        return state, cult_income_done
    if not any(cmd.verb in _MAIN_TRACK_VERBS for cmd in cmds):
        return state, cult_income_done
    if _cult_income_pending_later(income_rows, row, apply_faction):
        return state, cult_income_done
    state = grant_missing_cult_income(state, apply_faction)
    cult_income_done = cult_income_done | {apply_faction}
    return state, cult_income_done


def _grant_missing_cult_income(state: GameState, all_factions: set[str], cult_income_done: set[str]) -> GameState:
    """Retroactively grant ``cult_income_for_faction`` (``round_flow.
    grant_missing_cult_income``) to any *live* faction not yet in
    ``cult_income_done`` -- called right before either place in
    :func:`_advance_after_row` that fires ``end_of_round``, so a
    genuinely-missing ledger row (this function's docstring/module
    docstring both already establish this is a real, accepted gap, e.g.
    ``4pLeague_S11_D3L1_G5``'s darklings) doesn't also cost that faction
    the income itself, just the ledger row that would have shown it.
    Dropped factions are excluded (a dropped faction is never owed
    further income at all, matching every other post-drop exclusion in
    this module).

    Corpus: ``4pLeague_S32_D3L3_G7`` round 3's cult-income batch has
    Cultists' own real row correctly identified now (`_dedupe_income_
    commands` above), but Witches' row is separately, genuinely absent
    from the ledger for this same batch -- without this catch-up,
    Witches would simply never receive round 3's EARTH-based C income at
    all, a persistent 1 C shortfall for the rest of the game.
    """
    missing = sorted(all_factions - cult_income_done)
    for faction in missing:
        if state.factions[faction].dropped:
            continue
        state = grant_missing_cult_income(state, faction)
    return state


def _advance_after_row(
    state: GameState,
    faction: str,
    cmds: tuple[ParsedCommand, ...],
    other_income_done: set[str],
    cult_income_done: set[str],
) -> tuple[GameState, set[str], set[str]]:
    # A dropped faction never gets another income row (every other
    # post-drop exclusion in this module agrees), so it must not count
    # toward "every faction accounted for" -- task-14 fix, corpus
    # `4pLeague_S3_D1L1_G1`: dwarves drops mid-round-5, and this game's
    # `merge-income-phases` option means every income row is
    # `all_income_for_faction` (never a bare `other_income_for_faction`),
    # so the `_PURE_OTHER_INCOME_VERBS` reactive-proof fallback below never
    # fires either -- the raw completeness check below was the *only*
    # path that could ever trigger this round's `end_of_round`, and an
    # unfiltered `all_factions` (still counting dropped dwarves) made it
    # permanently unsatisfiable, stranding `state.round` one round behind
    # real Perl for the rest of the game (`power_actions_taken` never
    # reset, surfacing several rows later as a spurious "power action
    # space ACT2 is blocked this round").
    all_factions = {f for f in state.setup.factions if not state.factions[f].dropped}
    for cmd in cmds:
        if cmd.verb in _PURE_OTHER_INCOME_VERBS and state.phase == Phase.CLEANUP:
            # A round's other_income_for_faction rows only ever start once
            # the cult-income phase has genuinely ended -- so seeing one
            # while still in CLEANUP is itself proof the cult-income phase
            # is over, even if not every faction actually got a
            # cult_income_for_faction row (the ledger can just be missing
            # one entirely: task-14 fix, corpus ``4pLeague_S11_D3L1_G5``,
            # darklings never gets a round 1->2 cult_income_for_faction row
            # at all, so the "wait for every faction" gate below would
            # otherwise never fire and strand the harness in Phase.CLEANUP
            # for the rest of the game). The grant itself, not just the
            # phase transition, is still retroactively applied for any
            # such faction (`_grant_missing_cult_income` above) before
            # `end_of_round` runs.
            state = _grant_missing_cult_income(state, all_factions, cult_income_done)
            state = end_of_round(state)
            cult_income_done = set()
            other_income_done = set()
        if cmd.verb in _OTHER_INCOME_VERBS:
            other_income_done.add(faction)
        if cmd.verb in _CULT_INCOME_VERBS:
            cult_income_done.add(faction)

    if state.phase == Phase.CLEANUP and cult_income_done >= all_factions:
        state = end_of_round(state)
        cult_income_done = set()
        other_income_done = set()
    if state.phase == Phase.INCOME and other_income_done >= all_factions:
        state = begin_actions(state)
        other_income_done = set()

    return state, other_income_done, cult_income_done


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def replay_game(
    game_id: str, moves_df: pl.DataFrame, deltas_df: pl.DataFrame, *, stop_after: int = 5
) -> ReplayResult:
    """Replay one game's ledger through ``apply()``, cross-checking every
    row against the ``deltas.parquet`` oracle (module docstring).

    A row whose own faction is *already* ``dropped`` (checked right
    after ``_apply_pending_drops`` runs for this row, so this also
    catches the row immediately following that same faction's own drop)
    skips command application entirely -- ``commands.pm``'s drop handler
    zeroes ``allowed_actions``/calls ``dismiss_action`` as part of the
    single atomic drop event, so a dropped faction can never act again in
    real Perl, even for an already-in-flight client submission (task-14
    fix, ``_apply_pending_drops``'s own docstring has the full corpus
    citation and delta-oracle proof: such a row's recorded delta is
    byte-identical to that faction's pre-drop state -- a genuine no-op,
    not a legitimately-attempted-then-failed action). Only bookkeeping
    (row count, oracle comparison) still runs for it; the comparison
    trivially passes since the state is genuinely unchanged.
    """
    try:
        setup = load_setup(game_id)
    except Exception as exc:  # noqa: BLE001 -- loud, contextualized failure
        return ReplayResult(game_id=game_id, rows_checked=0, mismatches=(), error=str(exc))

    state = start_setup(GameState.initial(setup))

    game_moves = moves_df.filter(pl.col("game_id") == game_id).sort(["row", "seq"])
    delta_lookup = _delta_lookup(deltas_df, game_id)
    income_rows: list[tuple[int, str, str]] = [
        (r["row"], r["faction"], r["verb"])
        for r in game_moves.filter(
            pl.col("verb").is_in(_CULT_INCOME_VERBS | _OTHER_INCOME_VERBS)
        ).iter_rows(named=True)
    ]

    mismatches: list[Mismatch] = []
    rows_checked = 0
    other_income_done: set[str] = set()
    cult_income_done: set[str] = set()

    for row, faction, cmds in _iter_rows(game_moves):
        raw = "; ".join(cmd.raw for cmd in cmds)
        delta = delta_lookup.get((row, faction))
        # Looked up before the row applies (not just for the after-the-fact
        # mismatch check below) so `_apply_row_commands` can forward this
        # row's own recorded outcome to `gain_town`'s cult-tiebreak
        # ambiguity check (`apply.py`/`actions_build.py`'s `_cult_gain_order`
        # docstrings) -- REPLAY mode, task 14, user adjudication 2026-08-04.
        oracle_cult = delta["cult"] if delta is not None else None
        try:
            state = _apply_pending_drops(state, row, setup.dropped_at_row)
            # Round 0 (SETUP_DWELLINGS/SETUP_BONUS) is the one place a
            # dropped faction's own ledger row is not applied under its own
            # name. `commands.pm`'s `$assert_active_faction` (~1217-1221)
            # skips its "is active player" check entirely when
            # `$game{round} == 0`, and `acting.pm`'s `setup_action` (called
            # from both `command_build` and `command_pass`) blindly
            # `shift_setup_order()`s the *front* of the queue regardless of
            # which faction's command triggered it -- so a stale,
            # already-in-flight client submission from a faction that just
            # dropped still gets processed by the *server*, but against
            # whichever faction `setup_order`'s front now names post-drop
            # (this engine's own `active_faction(state)`, already
            # recomputed by `_apply_pending_drops` above), not the identity
            # baked into the stale request. Verified against both corpus
            # `4pLeague_S22_D3L1_G1` (mermaids drops; row 33's ledger
            # `faction` is still "mermaids", raw command `build F4`, but the
            # raw game JSON's final `map` has F4 colored green -- witches'
            # home color, the faction `active_faction(state)` computes as
            # next in `setup_order` once mermaids' entries are filtered
            # out) and `4pLeague_S22_D3L1_G6` (witches drops; row 31 "build
            # F3" ends up yellow -- nomads' color, and nomads' row 137
            # later upgrades F3 with no intervening "build F3" from nomads
            # anywhere in the ledger, so row 31 must be where that dwelling
            # actually landed). A dropped faction's *own* row remains a
            # true no-op for every other phase (round > 0 genuinely
            # enforces active-only, per `_apply_pending_drops`'s docstring)
            # -- this substitution is narrowly scoped to round 0.
            apply_faction = faction
            if state.factions[faction].dropped and state.phase in (
                Phase.SETUP_DWELLINGS,
                Phase.SETUP_BONUS,
            ):
                apply_faction = active_faction(state)
            if not state.factions[apply_faction].dropped:
                was_income = state.phase == Phase.INCOME
                state = _ensure_actions_phase_started(state, cmds)
                if was_income and state.phase != Phase.INCOME:
                    other_income_done = set()
                state, cult_income_done = _ensure_cult_income_landed(
                    state, apply_faction, cmds, cult_income_done, row, income_rows
                )
                apply_cmds = _dedupe_income_commands(cmds)
                state = _apply_row_commands(state, apply_faction, apply_cmds, oracle_cult=oracle_cult)
                state, other_income_done, cult_income_done = _advance_after_row(
                    state, apply_faction, cmds, other_income_done, cult_income_done
                )
        except Exception as exc:  # noqa: BLE001 -- loud, contextualized failure
            return ReplayResult(
                game_id=game_id,
                rows_checked=rows_checked,
                mismatches=tuple(mismatches),
                error=f"row {row} ({faction}, {raw!r}): {exc}",
            )

        rows_checked += 1
        if delta is not None:
            mismatches.extend(_row_mismatches(game_id, row, faction, state, delta, raw))
            if len(mismatches) >= stop_after:
                break

    return ReplayResult(game_id=game_id, rows_checked=rows_checked, mismatches=tuple(mismatches), error=None)


# --------------------------------------------------------------------------
# CLI: `uv run python -m bgai.engine.tm.replay --limit N [--game-id X] [--report out.json]`
# --------------------------------------------------------------------------


def _select_game_ids(limit: int, game_id: str | None) -> list[str]:
    if game_id is not None:
        return [game_id]
    games_meta = pl.read_parquet(_GAMES_META_PATH).sort("game_id")
    return games_meta["game_id"].to_list()[:limit]


def _mismatch_verb(mismatch: Mismatch) -> str:
    """First token of the reconstructed row text -- the responsible verb,
    for the CLI's (verb, field) frequency triage.
    """
    return mismatch.raw.split(" ", 1)[0]


def _max_row_reached(moves_df: pl.DataFrame, game_id: str, rows_checked: int) -> int | None:
    """The ledger ``row`` id of the last row this replay actually reached
    (``rows_checked`` distinct rows in, or ``None`` if none were).  Used to
    bound the per-faction pass-rate denominator to rows the replay actually
    saw, rather than every oracle row in the game (relevant when a replay
    stopped early on an error or ``stop_after``).
    """
    if rows_checked == 0:
        return None
    distinct_rows = (
        moves_df.filter(pl.col("game_id") == game_id)
        .select("row")
        .unique()
        .sort("row")["row"]
        .to_list()
    )
    return distinct_rows[rows_checked - 1] if rows_checked <= len(distinct_rows) else distinct_rows[-1]


def _faction_pass_rates(
    moves_df: pl.DataFrame, deltas_df: pl.DataFrame, result: ReplayResult
) -> dict[str, tuple[int, int]]:
    """``{faction: (clean_rows, total_oracle_rows_reached)}`` for one game,
    bounded to the rows this replay actually reached.
    """
    max_row = _max_row_reached(moves_df, result.game_id, result.rows_checked)
    if max_row is None:
        return {}
    game_deltas = deltas_df.filter((pl.col("game_id") == result.game_id) & (pl.col("row") <= max_row))
    totals = Counter(game_deltas["faction"].to_list())
    mismatched_rows: dict[str, set[int]] = {}
    for m in result.mismatches:
        mismatched_rows.setdefault(m.faction, set()).add(m.row)
    return {
        faction: (total - len(mismatched_rows.get(faction, ())), total) for faction, total in totals.items()
    }


def _print_report(results: list[ReplayResult], moves_df: pl.DataFrame, deltas_df: pl.DataFrame) -> None:
    all_mismatches: list[Mismatch] = []
    faction_clean: Counter[str] = Counter()
    faction_total: Counter[str] = Counter()

    for result in results:
        status = "FAIL" if result.error or result.mismatches else "PASS"
        print(
            f"{status} {result.game_id}: rows_checked={result.rows_checked} "
            f"mismatches={len(result.mismatches)} error={result.error}"
        )
        all_mismatches.extend(result.mismatches)
        for faction, (clean, total) in _faction_pass_rates(moves_df, deltas_df, result).items():
            faction_clean[faction] += clean
            faction_total[faction] += total

    if faction_total:
        print("\nPer-faction row pass rates:")
        for faction, total in sorted(faction_total.items()):
            clean = faction_clean[faction]
            print(f"  {faction}: {clean}/{total} ({100 * clean / total:.1f}%)")

    if all_mismatches:
        print("\nMismatch summary (verb, field) by frequency:")
        by_verb_field: Counter[tuple[str, str]] = Counter(
            (_mismatch_verb(m), m.field) for m in all_mismatches
        )
        for (verb, field), count in by_verb_field.most_common():
            print(f"  {count:4d}  {verb} / {field}")


def _write_report(path: str, results: list[ReplayResult]) -> None:
    payload = [
        {
            "game_id": r.game_id,
            "rows_checked": r.rows_checked,
            "error": r.error,
            "mismatches": [
                {
                    "row": m.row, "faction": m.faction, "field": m.field,
                    "expected": m.expected, "actual": m.actual, "raw": m.raw,
                }
                for m in r.mismatches
            ],
        }
        for r in results
    ]
    Path(path).write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay crawled games against apply() + the deltas oracle.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args(argv)

    moves_df = pl.read_parquet(_MOVES_PATH)
    deltas_df = pl.read_parquet(_DELTAS_PATH)
    game_ids = _select_game_ids(args.limit, args.game_id)

    results = [replay_game(gid, moves_df, deltas_df) for gid in game_ids]
    _print_report(results, moves_df, deltas_df)
    if args.report:
        _write_report(args.report, results)


if __name__ == "__main__":
    main()
