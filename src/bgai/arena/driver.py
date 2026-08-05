"""Steppable, immutable game driver (Phase 6, decision D6.1).

``sim.run_game`` originally kept the turn protocol's bookkeeping --
whether this turn's fresh action has been taken, and the previous
main-track verb -- in local variables. That is fine for playing one game
forward, but MCTS must *branch*: clone a position, try a move, and come
back. Anything living in a local variable cannot be cloned.

So the driver's control state is explicit and frozen here:

    SimState = (GameState, fresh_taken, prev_verb, decisions)

``advance(sim, choice)`` returns a NEW SimState, and every intermediate
forced transition (income grants, phase flips, engine-forced no-choice
spade forfeits) is applied inside ``_settle`` until the game is either
finished or waiting on a genuine agent decision. Cloning is therefore
free: hold the reference.

``decision(sim)`` returns ``(faction, offer)`` for the pending decision,
or ``None`` when the game is over. The offer is canonically ordered, so
every consumer -- arena, imitation labels, MCTS children -- indexes
candidates identically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.legal import legal_moves, legal_moves_for, pending_answer_moves
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import (
    advance_turn,
    begin_actions,
    end_of_round,
    is_turn_boundary,
    start_setup,
)
from bgai.engine.tm.scoring import final_scoring
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState, Phase, active_faction

MAIN_TRACK_VERBS = frozenset(
    {
        "build", "upgrade", "dig", "bridge", "transform", "connect",
        "action", "pass", "advance", "send",
    }
)
_NOOP_VERBS = frozenset({"wait", "done"})
_FREE_VERBS = frozenset({"convert", "burn"})


def canonical_moves(
    moves: tuple[ParsedCommand, ...] | list[ParsedCommand],
) -> tuple[ParsedCommand, ...]:
    """Deterministic offer ordering. Engine legal-move generation iterates
    sets in places, so raw ordering varies with interpreter hash
    randomization; sorting by every command field makes a game
    reproducible from (setup, seats, seed) across processes.
    """
    return tuple(
        sorted(
            moves,
            key=lambda m: (
                m.verb,
                m.loc or "",
                m.loc2 or "",
                m.building or "",
                m.tile or "",
                m.cult or "",
                m.color or "",
                m.target or "",
                m.res1 or "",
                m.res2 or "",
                m.n1 if m.n1 is not None else -1,
                m.n2 if m.n2 is not None else -1,
            ),
        )
    )


@dataclass(frozen=True)
class SimState:
    """A game paused at a decision point (or finished)."""

    game: GameState
    fresh_taken: bool = False
    prev_verb: str | None = None
    decisions: int = 0
    income_marker: tuple[int, str] | None = None
    """``(round, phase name)`` whose income/cleanup grants have already
    been applied. The engine's grant handlers are idempotent per *call*,
    not per round, so the driver must remember. It lives here rather than
    on ``GameState`` because MCTS clones ``SimState`` -- a marker stashed
    on the shared game object would leak across branches.
    """

    @property
    def finished(self) -> bool:
        return self.game.phase == Phase.FINISHED


def new_game(setup: GameSetup) -> SimState:
    """A game settled to its first decision point."""
    return _settle(SimState(game=start_setup(GameState.initial(setup))))


def decision(sim: SimState) -> tuple[str, tuple[ParsedCommand, ...]] | None:
    """``(faction, canonical offer)`` for the pending decision, or None
    if the game is finished. Never returns an empty offer.
    """
    if sim.finished:
        return None
    forced = _forced_answer(sim.game)
    if forced is not None:
        faction, answers = forced
        return faction, canonical_moves(answers)
    if sim.game.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        faction = active_faction(sim.game)
        want = "build" if sim.game.phase == Phase.SETUP_DWELLINGS else "pass"
        offer = tuple(m for m in legal_moves(sim.game) if m.verb == want)
        if not offer:
            raise EngineError(
                f"no {want!r} moves during {sim.game.phase}",
                state=sim.game,
                faction=faction,
                cmd=cmd("wait"),
            )
        return faction, canonical_moves(offer)
    spade = _spade_decision(sim.game)
    if spade is not None:
        return spade
    return _actions_offer(sim)


def advance(sim: SimState, choice: ParsedCommand) -> SimState:
    """Apply ``choice`` for the current decision's faction, then settle
    forward to the next decision point (or the finished state).
    """
    current = decision(sim)
    if current is None:
        raise ValueError("advance called on a finished game")
    faction, offer = current
    if choice not in offer:
        raise EngineError(
            "choice is not in the current offer",
            state=sim.game,
            faction=faction,
            cmd=choice,
        )

    pre = sim.game
    game = apply(sim.game, faction, choice)
    fresh_taken, prev_verb = sim.fresh_taken, sim.prev_verb
    end_turn = False

    if _is_actions_turn_choice(sim, faction):
        if choice.verb == "done":
            end_turn = True
        else:
            if choice.verb in MAIN_TRACK_VERBS:
                if is_turn_boundary(choice, pre, faction, prev_verb):
                    fresh_taken = True
                prev_verb = choice.verb
            if choice.verb == "pass":
                end_turn = True

    nxt = replace(
        sim,
        game=game,
        fresh_taken=fresh_taken,
        prev_verb=prev_verb,
        decisions=sim.decisions + 1,
    )
    if end_turn:
        nxt = replace(
            _end_actions_turn(nxt), fresh_taken=False, prev_verb=None
        )
    if sim.game.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS) and _forced_answer(
        nxt.game
    ) is None:
        # A setup placement ends that faction's setup turn immediately --
        # unless it queued a pending answer (a leech offer from the very
        # first dwellings), which must be answered before the turn passes.
        nxt = replace(nxt, game=advance_turn(nxt.game))
    return _settle(nxt)


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _forced_answer(game: GameState) -> tuple[str, tuple[ParsedCommand, ...]] | None:
    """The head blocking pending decision, if any (leech offers enqueue
    clockwise from the builder, so queue order is snellman's own answer
    order).
    """
    for pending in game.pending:
        answers = pending_answer_moves(game, pending.faction)
        if answers:
            return pending.faction, answers
    return None


def _spade_decision(game: GameState) -> tuple[str, tuple[ParsedCommand, ...]] | None:
    """During INCOME/CLEANUP a live spade balance forces an immediate
    transform-or-forfeit before the batch proceeds. Only a real choice
    (at least one legal transform) becomes a decision; a balance with no
    legal transform is forfeited by ``_settle`` without consulting anyone.
    """
    if game.phase not in (Phase.INCOME, Phase.CLEANUP):
        return None
    for faction in game.turn_order:
        if game.factions[faction].spades_available <= 0:
            continue
        transforms = tuple(
            m for m in legal_moves_for(game, faction) if m.verb == "transform"
        )
        if not transforms:
            return None  # settled automatically
        balance = game.factions[faction].spades_available
        forfeit = cmd("lose_spade", kind=Kind.BOOKKEEPING, n1=balance)
        return faction, transforms + (forfeit,)
    return None


def _is_actions_turn_choice(sim: SimState, faction: str) -> bool:
    """Whether the decision just answered belonged to the ACTIONS turn
    protocol (as opposed to a forced pending answer or a spade window),
    which is what turn bookkeeping tracks.
    """
    if sim.game.phase != Phase.ACTIONS:
        return False
    if _forced_answer(sim.game) is not None:
        return False
    return faction == active_faction(sim.game)


def _actions_offer(sim: SimState) -> tuple[str, tuple[ParsedCommand, ...]]:
    """One ACTIONS turn: exactly one fresh main-track action, then its
    continuations (``is_turn_boundary`` False) plus free convert/burn,
    until ``done``.
    """
    game = sim.game
    faction = active_faction(game)
    if not sim.fresh_taken:
        offer = tuple(m for m in legal_moves(game) if m.verb not in _NOOP_VERBS)
    else:
        moves = legal_moves(game)
        continuations = tuple(
            m
            for m in moves
            if m.verb in MAIN_TRACK_VERBS
            and m.verb != "pass"
            and not is_turn_boundary(m, game, faction, sim.prev_verb)
        )
        frees = tuple(m for m in moves if m.verb in _FREE_VERBS)
        offer = continuations + frees + (cmd("done"),)
    return faction, canonical_moves(offer)


def _end_actions_turn(sim: SimState) -> SimState:
    return replace(sim, game=advance_turn(sim.game))


def _settle(sim: SimState) -> SimState:
    """Apply every forced, choice-free transition until the game is
    finished or a genuine decision is pending: income/cleanup grants,
    phase flips, and spade balances with no legal transform.
    """
    game = sim.game
    marker = sim.income_marker
    while True:
        if game.phase == Phase.FINISHED:
            return replace(sim, game=final_scoring(game), income_marker=marker)
        if _forced_answer(game) is not None or game.phase in (
            Phase.SETUP_DWELLINGS,
            Phase.SETUP_BONUS,
        ):
            return replace(sim, game=game, income_marker=marker)

        if game.phase in (Phase.INCOME, Phase.CLEANUP):
            verb = (
                "other_income_for_faction"
                if game.phase == Phase.INCOME
                else "cult_income_for_faction"
            )
            batch = (game.round, game.phase.name)
            if marker != batch:
                for faction in game.turn_order:
                    game = apply(game, faction, cmd(verb, kind=Kind.INCOME))
                marker = batch
                continue
            settled = _forfeit_choiceless_spades(game)
            if settled is not None:
                game = settled
                continue
            if _spade_decision(game) is not None or _forced_answer(game) is not None:
                return replace(sim, game=game, income_marker=marker)
            game = begin_actions(game) if game.phase == Phase.INCOME else end_of_round(game)
            continue

        return replace(sim, game=game, income_marker=marker)  # ACTIONS: decision pending


def _forfeit_choiceless_spades(game: GameState) -> GameState | None:
    """Forfeit any spade balance whose holder has no legal transform --
    leaving one live would let ``is_turn_boundary`` treat the next build
    as a free continuation. Returns None if there was nothing to do.
    """
    for faction in game.turn_order:
        balance = game.factions[faction].spades_available
        if balance <= 0:
            continue
        if any(m.verb == "transform" for m in legal_moves_for(game, faction)):
            continue
        return apply(
            game,
            faction,
            cmd("lose_spade", kind=Kind.BOOKKEEPING, n1=balance),
        )
    return None
