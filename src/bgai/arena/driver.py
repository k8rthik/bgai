"""Live-game driver: the Task-12 contract for games with no ledger.

`replay.py` mirrors corpus rows; this module *generates* the bookkeeping
those rows record (income, cult income, end-of-round, final scoring) and
routes genuine decisions to agents. Engine hooks handle reactive effects
(e.g. Cultists' leech bonus fires inside handle_leech/handle_decline via
_resolve_cultist_watch), so the driver never emits those verbs.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import Kind
from bgai.engine.tm.apply import apply
from bgai.engine.tm.legal import BLOCKING_PENDING_KINDS
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import begin_actions, end_of_round, start_setup
from bgai.engine.tm.scoring import final_scoring
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState, PendingDecision, Phase, active_faction

MAIN_TRACK_VERBS = frozenset(
    {"build", "upgrade", "action", "advance", "pass", "send", "dig", "transform", "bridge",
     "connect"}
)
ALWAYS_END_VERBS = frozenset({"pass", "send", "advance"})
CONTINUATION_MARKER_KINDS = frozenset({"free_d", "free_tp", "free_tf", "bridge"})

_CLEANUP_INCOME_DONE = "driver_cleanup_income_done"
"""Driver-owned pending marker: this round's cult income has been granted.

Cult income can grant a SPADE, whose forced transform is a real decision
the caller must route to an agent before ``end_of_round`` -- so the
cleanup batch is re-entrant, and the driver needs a marker inside the
(immutable) state to avoid granting income twice. The kind is unknown to
every engine scan (legal.py/leech.py/round_flow.py all match specific
kinds), and it is popped before ``end_of_round`` runs.
"""


class DriverError(Exception):
    """Driver-level failure (budget exhausted, agent returned illegal move)."""


def new_game(setup: GameSetup) -> GameState:
    return start_setup(GameState.initial(setup))


def live_factions(state: GameState) -> tuple[str, ...]:
    return tuple(f for f in state.setup.factions if not state.factions[f].dropped)


def _has_blocking_pending(state: GameState) -> bool:
    return any(p.kind in BLOCKING_PENDING_KINDS for p in state.pending)


def _has_unspent_spades(state: GameState) -> bool:
    return any(state.factions[f].spades_available > 0 for f in live_factions(state))


def spade_holder(state: GameState) -> str | None:
    """During INCOME/CLEANUP: the first live faction sitting on an unspent
    spade balance (a cult-income SPADE grant forces an out-of-turn
    transform before the batch finishes -- legal.py, Design decision 4).
    """
    if state.phase not in (Phase.INCOME, Phase.CLEANUP):
        return None
    for faction in live_factions(state):
        if state.factions[faction].spades_available > 0:
            return faction
    return None


def next_actor(state: GameState) -> str | None:
    """The faction owing a decision right now: the oldest blocking pending's
    faction wins; else a forced income/cleanup spade transform; else the
    active faction during decision phases; else None (bookkeeping owes an
    advance).
    """
    for p in state.pending:
        if p.kind in BLOCKING_PENDING_KINDS:
            return p.faction
    holder = spade_holder(state)
    if holder is not None:
        return holder
    if state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS, Phase.ACTIONS):
        actor = active_faction(state)
        fs = state.factions[actor]
        if not fs.passed and not fs.dropped:
            return actor
    return None


def _cleanup_income_done(state: GameState) -> bool:
    return any(p.kind == _CLEANUP_INCOME_DONE for p in state.pending)


def _push_cleanup_marker(state: GameState) -> GameState:
    marker = PendingDecision(faction="", kind=_CLEANUP_INCOME_DONE)
    return replace(state, pending=state.pending + (marker,))


def _pop_cleanup_marker(state: GameState) -> GameState:
    pending = tuple(p for p in state.pending if p.kind != _CLEANUP_INCOME_DONE)
    return replace(state, pending=pending)


def advance_bookkeeping(state: GameState) -> GameState:
    """Run INCOME/CLEANUP grants + phase transitions; stop at any decision
    point, at FINISHED (after one-shot final_scoring), or when nothing is
    owed. Returns `state` unchanged (identity) when there is nothing to do.

    Cleanup is re-entrant: cult income is granted exactly once per round
    (guarded by the ``_CLEANUP_INCOME_DONE`` marker), and ``end_of_round``
    is deferred while any blocking pending or forced spade transform is
    outstanding -- the caller routes those to agents and calls back in.
    """
    while True:
        if state.phase is Phase.INCOME:
            if _has_blocking_pending(state) or spade_holder(state) is not None:
                return state
            for faction in live_factions(state):
                state = apply(
                    state, faction, cmd("other_income_for_faction", kind=Kind.BOOKKEEPING)
                )
            state = begin_actions(state)
            continue
        if state.phase is Phase.CLEANUP:
            if _has_blocking_pending(state) or spade_holder(state) is not None:
                return state
            if not _cleanup_income_done(state):
                for faction in live_factions(state):
                    state = apply(
                        state, faction, cmd("cult_income_for_faction", kind=Kind.BOOKKEEPING)
                    )
                state = _push_cleanup_marker(state)
                continue  # re-check pendings/spades the income just created
            state = end_of_round(_pop_cleanup_marker(state))
            if state.phase is Phase.FINISHED:
                return final_scoring(state)
            continue
        return state
