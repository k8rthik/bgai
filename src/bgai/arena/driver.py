"""Live-game driver: the Task-12 contract for games with no ledger.

`replay.py` mirrors corpus rows; this module *generates* the bookkeeping
those rows record (income, cult income, end-of-round, final scoring) and
routes genuine decisions to agents. Engine hooks handle reactive effects
(e.g. Cultists' leech bonus fires inside handle_leech/handle_decline via
_resolve_cultist_watch), so the driver never emits those verbs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from bgai.agents.base import Agent
from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import apply
from bgai.engine.tm.legal import BLOCKING_PENDING_KINDS, legal_moves_for
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import advance_turn, begin_actions, end_of_round, start_setup
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


def _can_spend_spades(state: GameState, faction: str) -> bool:
    """Whether `faction`'s spade balance has any legal transform to spend it
    on. Giants are the canonical False case: their transforms always cost
    2 spades, so a 1-spade cult-income grant is unusable (Perl handles the
    same situation with `lose_spade`).
    """
    return any(m.verb == "transform" for m in legal_moves_for(state, faction))


def _forfeit_unusable_spades(state: GameState) -> GameState:
    """Zero any live faction's unspendable spade balance (Perl's turn-end
    ``lose_spade`` -- pure bookkeeping, no VP effect in this engine).
    """
    for faction in live_factions(state):
        fs = state.factions[faction]
        if fs.spades_available > 0 and not _can_spend_spades(state, faction):
            new_factions = dict(state.factions)
            new_factions[faction] = replace(fs, spades_available=0)
            state = replace(state, factions=new_factions)
    return state


def spade_holder(state: GameState) -> str | None:
    """During INCOME/CLEANUP: the first live faction sitting on an unspent,
    *spendable* spade balance (a cult-income SPADE grant forces an
    out-of-turn transform before the batch finishes -- legal.py, Design
    decision 4).
    """
    if state.phase not in (Phase.INCOME, Phase.CLEANUP):
        return None
    for faction in live_factions(state):
        if state.factions[faction].spades_available > 0 and _can_spend_spades(state, faction):
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
        state = _forfeit_unusable_spades(state)
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


# --------------------------------------------------------------------------
# Turn protocol + game loop (plan task 4)
# --------------------------------------------------------------------------

DONE = cmd("done")
"""Driver-level end-of-turn sentinel: interpreted by the driver (advance_turn),
never passed to apply() -- the engine's own `done` handler is a no-op."""

_PENDING_ANSWER_VERBS: dict[str, tuple[str, ...]] = {
    "leech": ("leech", "decline"),
    "cult_choice": ("gain_cult",),
    "gain_favor": ("gain_favor",),
    "gain_town": ("gain_town",),
    "convert_w_to_p": ("convert",),
}


def canonical_order(moves: tuple[ParsedCommand, ...]) -> tuple[ParsedCommand, ...]:
    """Sort moves by their full field tuple. Engine enumeration iterates
    frozensets, whose order depends on the per-process hash seed -- without
    this, seeded agents pick different moves run-to-run and games are not
    reproducible across processes.
    """

    def key(m: ParsedCommand) -> tuple:
        return (
            m.verb, m.loc or "", m.loc2 or "", m.building or "", m.tile or "", m.cult or "",
            m.color or "", m.target or "", m.res1 or "", m.res2 or "", m.n1 or 0, m.n2 or 0,
        )

    return tuple(sorted(moves, key=key))


def must_continue(state: GameState, faction: str) -> bool:
    """The open action cannot end yet: an unspent spade balance or a queued
    one-shot marker (free_d/free_tp/free_tf/bridge) must be spent or
    declined first (markers' decline moves are in the legal set).
    """
    fs = state.factions[faction]
    if fs.spades_available > 0 and _can_spend_spades(state, faction):
        return True
    return any(
        p.faction == faction and p.kind in CONTINUATION_MARKER_KINDS for p in state.pending
    )


def _end_action(state: GameState, faction: str) -> GameState:
    """Close `faction`'s full action: forfeit any leftover (unspendable or
    voluntarily unspent) spade balance -- Perl's turn-end ``lose_spade`` --
    then advance the turn.
    """
    fs = state.factions[faction]
    if fs.spades_available > 0:
        new_factions = dict(state.factions)
        new_factions[faction] = replace(fs, spades_available=0)
        state = replace(state, factions=new_factions)
    return advance_turn(state)


def offered_moves(state: GameState, faction: str, turn_open: bool) -> tuple[ParsedCommand, ...]:
    """What `faction` may do right now. Adds the driver's DONE sentinel once
    a main-track verb has landed and nothing forces a continuation. During
    INCOME/CLEANUP a forced spade balance restricts the offer to the
    transform moves that spend it (mechanical forcing, per legal.py's
    Design decision 4 -- not a judgment call).
    """
    moves = canonical_order(legal_moves_for(state, faction))
    # The engine's own bare `done` (a no-op verb) is not a turn: Perl's
    # require_action demands a real action per turn, and offering it lets
    # an agent silently skip its setup dwelling or whole turns. The
    # driver's DONE sentinel below is the only sanctioned turn-ender.
    moves = tuple(m for m in moves if m.verb != "done")
    if state.phase in (Phase.INCOME, Phase.CLEANUP) and spade_holder(state) == faction:
        forced = tuple(m for m in moves if m.verb == "transform")
        if forced:
            return forced
    if turn_open and not must_continue(state, faction):
        moves = moves + (DONE,)
    return moves


def _oldest_blocking_pending(state: GameState) -> PendingDecision | None:
    for p in state.pending:
        if p.kind in BLOCKING_PENDING_KINDS:
            return p
    return None


def _check_offered(
    move: ParsedCommand, moves: tuple[ParsedCommand, ...], agent: Agent, faction: str
) -> None:
    if move not in moves:
        raise DriverError(f"agent {agent.name!r} for {faction} returned unoffered move {move!r}")


def _pop_pending(state: GameState, pending: PendingDecision) -> GameState:
    idx = state.pending.index(pending)
    return replace(state, pending=state.pending[:idx] + state.pending[idx + 1 :])


def _answer_step(
    state: GameState,
    agent: Agent,
    faction: str,
    pending: PendingDecision,
    events: list[str],
) -> GameState:
    """One blocking-pending answer (no advance_turn -- sub-decisions carry
    no turn-order weight). An unanswerable pending (e.g. gain_favor with
    every eligible tile gone) is popped, mirroring Perl's skip.
    """
    answer_verbs = _PENDING_ANSWER_VERBS[pending.kind]
    moves = canonical_order(
        tuple(m for m in legal_moves_for(state, faction) if m.verb in answer_verbs)
    )
    if pending.kind == "leech":
        moves = tuple(m for m in moves if m.target == pending.source)
    if not moves:
        events.append(f"-- {faction}: pending {pending.kind} had no legal answer, skipped")
        return _pop_pending(state, pending)
    move = agent.choose(state, faction, moves)
    _check_offered(move, moves, agent, faction)
    events.append(f"{faction}: {move.raw or move.verb}")
    return apply(state, faction, move)


def _spade_step(state: GameState, agent: Agent, faction: str, events: list[str]) -> GameState:
    """One forced income/cleanup spade transform (phase-exempt; advance_turn
    is a documented no-op in these phases).
    """
    moves = offered_moves(state, faction, turn_open=False)
    move = agent.choose(state, faction, moves)
    _check_offered(move, moves, agent, faction)
    events.append(f"{faction}: {move.raw or move.verb}")
    return apply(state, faction, move)


@dataclass
class _Budget:
    remaining: int

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise DriverError("command budget exhausted (runaway game or agent loop)")


def _bot_full_action(
    state: GameState, agent: Agent, faction: str, events: list[str], budget: _Budget
) -> GameState:
    """One full main-track action for the structurally active faction, per
    the plan's turn protocol: DONE or an ALWAYS_END verb ends it; so does
    build/upgrade/connect/bridge/action with no forced continuation;
    transform/dig leave it open (optional build continuation).
    """
    prev_main: str | None = None
    while True:
        moves = offered_moves(state, faction, turn_open=prev_main is not None)
        move = agent.choose(state, faction, moves)
        _check_offered(move, moves, agent, faction)
        if move.verb == "done":
            return _end_action(state, faction)
        state = apply(state, faction, move)
        budget.spend()
        events.append(f"{faction}: {move.raw or move.verb}")
        if move.verb not in MAIN_TRACK_VERBS:
            continue  # aux (convert/burn) or an interleaved own-pending answer
        prev_main = move.verb
        if move.verb in ALWAYS_END_VERBS:
            return _end_action(state, faction)
        if move.verb in ("transform", "dig"):
            continue  # dig leaves spades; transform leaves an optional build
        if not must_continue(state, faction):
            return _end_action(state, faction)


def play_until_decision(
    state: GameState,
    bots: Mapping[str, Agent],
    external: frozenset[str],
    events: list[str],
    max_commands: int = 10000,
) -> GameState:
    """Advance the game -- bot decisions and bookkeeping -- until an
    external seat owes a decision or the game is FINISHED (final scoring
    applied). Appends one event line per applied command ("faction: cmd")
    and "-- note" lines for driver notes.
    """
    budget = _Budget(remaining=max_commands)
    while True:
        if state.phase is Phase.FINISHED:
            return state
        actor = next_actor(state)
        if actor is None:
            new_state = advance_bookkeeping(state)
            if new_state is state:
                raise DriverError(f"stuck: no actor and no bookkeeping owed (phase={state.phase})")
            state = new_state
            continue
        if actor in external:
            return state
        agent = bots[actor]
        pending = _oldest_blocking_pending(state)
        if pending is not None and pending.faction == actor:
            state = _answer_step(state, agent, actor, pending, events)
            budget.spend()
        elif spade_holder(state) == actor:
            state = _spade_step(state, agent, actor, events)
            budget.spend()
        else:
            state = _bot_full_action(state, agent, actor, events, budget)


@dataclass(frozen=True)
class GameResult:
    game_id: str
    vp: Mapping[str, int]
    winners: tuple[str, ...]
    commands_applied: int


def run_game(
    setup: GameSetup, agents: Mapping[str, Agent], max_commands: int = 10000
) -> GameResult:
    """Play a full game with an agent on every seat; returns final scores.
    Winners = all factions tied at max VP (snellman shares wins).
    """
    events: list[str] = []
    state = play_until_decision(new_game(setup), agents, frozenset(), events, max_commands)
    vp = {f: state.factions[f].vp for f in setup.factions}
    top = max(vp.values())
    winners = tuple(f for f in setup.factions if vp[f] == top)
    commands = sum(1 for e in events if not e.startswith("--"))
    return GameResult(game_id=setup.game_id, vp=vp, winners=winners, commands_applied=commands)
