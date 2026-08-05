"""Live-game driver: the Task-12 contract for games with no ledger.

`replay.py` mirrors corpus rows; this module *generates* the bookkeeping
those rows record (income, cult income, end-of-round, final scoring) and
routes genuine decisions to agents. Engine hooks handle reactive effects
(e.g. Cultists' leech bonus fires inside handle_leech/handle_decline via
_resolve_cultist_watch), so the driver never emits those verbs.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace

from bgai.agents.base import Agent
from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import apply
from bgai.engine.tm.legal import (
    BLOCKING_PENDING_KINDS,
    legal_moves_for,
    pending_answer_moves,
)
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import advance_turn, begin_actions, end_of_round, start_setup
from bgai.engine.tm.scoring import final_scoring
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState, PendingDecision, Phase, active_faction
from bgai.mcp.render import render_command

MAIN_TRACK_VERBS = frozenset(
    {"build", "upgrade", "action", "advance", "pass", "send", "dig", "transform", "bridge",
     "connect"}
)
AUX_VERBS: frozenset[str] = frozenset({"convert", "burn", "wait"})
"""Order-exempt resource/no-op moves -- a first-legal-move picker (test
helpers) must skip these or it would convert forever."""
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


def progress_moves(moves: tuple[ParsedCommand, ...]) -> tuple[ParsedCommand, ...]:
    """Moves that advance the game (non-AUX); falls back to `moves` if empty."""
    filtered = tuple(m for m in moves if m.verb not in AUX_VERBS)
    return filtered or moves


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


def end_action(state: GameState, faction: str) -> GameState:
    """Close `faction`'s full action: forfeit any leftover (unspendable or
    voluntarily unspent) spade balance -- Perl's turn-end ``lose_spade`` --
    and, if the faction just passed, its unused extra actions (an ACTC
    ticket dies with the pass; advance_turn would otherwise keep the
    passed faction structurally active and strand the round). Then
    advance the turn.
    """
    fs = state.factions[faction]
    if fs.spades_available > 0 or (fs.passed and fs.extra_actions > 0):
        new_factions = dict(state.factions)
        new_factions[faction] = replace(
            fs,
            spades_available=0,
            extra_actions=0 if fs.passed else fs.extra_actions,
        )
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
    # The engine's bare `done`/`wait` no-ops are not turns: Perl's
    # require_action demands a real action per turn, and a uniform-random
    # agent offered `wait` would no-op forever. The driver's DONE sentinel
    # below is the only sanctioned turn-ender.
    moves = tuple(m for m in moves if m.verb not in ("done", "wait"))
    if must_continue(state, faction):
        restricted = tuple(m for m in moves if m.verb in _continuation_verbs(state, faction))
        if restricted:
            return restricted
        return moves  # defensive: should not happen (decline moves always exist)
    if turn_open:
        moves = moves + (DONE,)
    return moves


def _continuation_verbs(state: GameState, faction: str) -> frozenset[str]:
    """Verbs that resolve `faction`'s forced continuation (spend or decline
    the outstanding spade balance / one-shot markers). While one is
    outstanding, offering anything else would let an agent open a second
    main action inside the same turn -- the engine does not gate
    one-action-per-turn (that is this driver's job), so the turn
    accounting would silently desync (observed: heuristic dwarves taking
    two power actions in one turn on an unspent ACT1 bridge marker).
    """
    fs = state.factions[faction]
    allowed: set[str] = {"lose_marker"}
    if fs.spades_available > 0 and _can_spend_spades(state, faction):
        allowed.add("transform")
    kinds = {
        p.kind
        for p in state.pending
        if p.faction == faction and p.kind in CONTINUATION_MARKER_KINDS
    }
    if "free_d" in kinds:
        allowed.add("build")
    if "free_tp" in kinds:
        allowed.add("upgrade")
    if "free_tf" in kinds:
        allowed |= {"transform", "build"}
    if "bridge" in kinds:
        allowed.add("bridge")
    return frozenset(allowed)


def oldest_blocking_pending(state: GameState) -> PendingDecision | None:
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


def answer_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """Every answer `faction` can give to its own outstanding blocking
    pendings (legal.py's public pending API), canonically ordered."""
    return canonical_order(pending_answer_moves(state, faction))


def _answer_step(
    state: GameState,
    agent: Agent,
    faction: str,
    pending: PendingDecision,
    events: list[str],
    rng: random.Random,
) -> GameState:
    """One blocking-pending answer (no advance_turn -- sub-decisions carry
    no turn-order weight). A faction whose outstanding pendings have no
    legal answer at all (e.g. gain_favor with every eligible tile gone)
    gets its oldest one popped, mirroring Perl's skip.
    """
    moves = answer_moves(state, faction)
    if not moves:
        events.append(f"-- {faction}: pending {pending.kind} had no legal answer, skipped")
        return _pop_pending(state, pending)
    move = agent.choose(state, faction, moves, rng)
    _check_offered(move, moves, agent, faction)
    events.append(f"{faction}: {render_command(move)}")
    return apply(state, faction, move)


def _spade_step(
    state: GameState, agent: Agent, faction: str, events: list[str], rng: random.Random
) -> GameState:
    """One forced income/cleanup spade transform (phase-exempt; advance_turn
    is a documented no-op in these phases).
    """
    moves = offered_moves(state, faction, turn_open=False)
    move = agent.choose(state, faction, moves, rng)
    _check_offered(move, moves, agent, faction)
    events.append(f"{faction}: {render_command(move)}")
    return apply(state, faction, move)


@dataclass
class _Budget:
    remaining: int

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise DriverError("command budget exhausted (runaway game or agent loop)")


def _bot_full_action(
    state: GameState,
    agent: Agent,
    faction: str,
    events: list[str],
    budget: _Budget,
    rng: random.Random,
) -> GameState:
    """One full main-track action for the structurally active faction, per
    the plan's turn protocol: DONE or an ALWAYS_END verb ends it; so does
    build/upgrade/connect/bridge/action with no forced continuation;
    transform/dig leave it open (optional build continuation).
    """
    prev_main: str | None = None
    while True:
        moves = offered_moves(state, faction, turn_open=prev_main is not None)
        move = agent.choose(state, faction, moves, rng)
        _check_offered(move, moves, agent, faction)
        if move.verb == "done":
            return end_action(state, faction)
        state = apply(state, faction, move)
        budget.spend()
        events.append(f"{faction}: {render_command(move)}")
        if move.verb not in MAIN_TRACK_VERBS:
            continue  # aux (convert/burn) or an interleaved own-pending answer
        prev_main = move.verb
        if move.verb in ALWAYS_END_VERBS:
            return end_action(state, faction)
        if move.verb in ("transform", "dig"):
            continue  # dig leaves spades; transform leaves an optional build
        if not must_continue(state, faction):
            return end_action(state, faction)


def play_until_decision(
    state: GameState,
    bots: Mapping[str, Agent],
    external: frozenset[str],
    events: list[str],
    max_commands: int = 10000,
    rng: random.Random | None = None,
) -> GameState:
    """Advance the game -- bot decisions and bookkeeping -- until an
    external seat owes a decision or the game is FINISHED (final scoring
    applied). Appends one event line per applied command ("faction: cmd")
    and "-- note" lines for driver notes.
    """
    rng = rng if rng is not None else random.Random(0)
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
        pending = oldest_blocking_pending(state)
        if pending is not None and pending.faction == actor:
            state = _answer_step(state, agent, actor, pending, events, rng)
            budget.spend()
        elif spade_holder(state) == actor:
            state = _spade_step(state, agent, actor, events, rng)
            budget.spend()
        else:
            state = _bot_full_action(state, agent, actor, events, budget, rng)


@dataclass(frozen=True)
class GameResult:
    game_id: str
    vp: Mapping[str, int]
    winners: tuple[str, ...]
    commands_applied: int


def run_game(
    setup: GameSetup,
    agents: Mapping[str, Agent],
    max_commands: int = 10000,
    seed: int = 0,
) -> GameResult:
    """Play a full game with an agent on every seat; returns final scores.
    Winners = all factions tied at max VP (snellman shares wins). Agents
    draw randomness only from the seeded rng (Agent protocol), so the
    result is a pure function of (setup, agents, seed).
    """
    events: list[str] = []
    state = play_until_decision(
        new_game(setup), agents, frozenset(), events, max_commands, rng=random.Random(seed)
    )
    vp = {f: state.factions[f].vp for f in setup.factions}
    top = max(vp.values())
    winners = tuple(f for f in setup.factions if vp[f] == top)
    commands = sum(1 for e in events if not e.startswith("--"))
    return GameResult(game_id=setup.game_id, vp=vp, winners=winners, commands_applied=commands)
