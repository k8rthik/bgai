"""Self-driving Terra Mystica simulation loop (plan Task 4).

Composes the replay-validated engine primitives into a complete headless
game between :class:`bgai.agents.base.Agent` implementations:

- setup phases: one placement/pick per snake-order turn
  (``round_flow.start_setup`` / ``advance_turn``);
- INCOME: synthetic ``other_income_for_faction`` per faction (the
  documented equivalent of ``round_flow.handle_income_row``'s ledger
  dispatch), then the spade window, then ``begin_actions``;
- ACTIONS: drain blocking pending decisions (leech answers, favor/town
  picks, cult choices) in queue order, then one full-turn protocol for
  ``active_faction`` per iteration -- one fresh main-track action plus
  its continuations, classified by ``round_flow.is_turn_boundary``;
- CLEANUP: synthetic ``cult_income_for_faction`` per faction (the round
  tile's cult reward, ``score_tiles[round - 1]`` while ``round`` is
  still r), the spade window, then ``end_of_round``;
- FINISHED: ``scoring.final_scoring`` exactly once (simulation mode:
  fixed FIRE>WATER>EARTH>AIR tiebreaks, no oracle).

Leftover income-window spades are forfeited via ``lose_spade``
(``actions_terraform.handle_lose_spade``) before play resumes --
``is_turn_boundary`` treats any live spade balance as licence for a free
build continuation, so a leak here would hand out free actions.

Every ``GameState`` remains immutable; the mutable ``_Driver`` instance
is per-game bookkeeping (current state reference, decision counter),
discarded after ``run_game`` returns.

Timing (Apple-silicon MacBook, random vs random, 200 corpus-sampled
setups, 2026-08-04): 48 ms per headless 4p game -- ~20x under the master
plan's <1s target, no optimization warranted this phase.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping

from bgai.agents.base import Agent
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

# The verbs round_flow's turn-boundary classification is defined over
# (its _ALWAYS_FRESH / _NEVER_FRESH / marker universe). convert/burn and
# pending answers are never turn-relevant (is_turn_boundary docstring:
# "ignoring non-main-track commands like gain_cult/burn in between").
MAIN_TRACK_VERBS = frozenset(
    {"build", "upgrade", "dig", "bridge", "transform", "connect", "action", "pass", "advance", "send"}
)

_NOOP_VERBS = frozenset({"wait", "done"})
_FREE_VERBS = frozenset({"convert", "burn"})


def _canonical(moves: tuple[ParsedCommand, ...] | list[ParsedCommand]) -> tuple[ParsedCommand, ...]:
    """Deterministic offer ordering. Engine legal-move generation iterates
    sets in places (hex candidate sets and friends), so raw ordering
    varies with interpreter hash randomization -- sorting by every command
    field makes a game reproducible from (setup, seats, seed) across
    processes, which the Agent protocol promises.
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
class GameResult:
    """Outcome of one arena game.

    ``seats``/``vps``/``ranks`` are built in seat order
    (``setup.factions``) -- consumers (``ratings.placement_table``) rely
    on dict insertion order for seat indices. ``ranks`` is 0-based
    competition ranking (ties share a rank). ``error`` non-None means the
    game aborted (engine rejection or decision cap) -- rating updates
    skip it, reports list it verbatim.
    """

    setup_game_id: str
    seats: Mapping[str, str]
    vps: Mapping[str, int]
    ranks: Mapping[str, int]
    decisions: int
    error: str | None = None
    anomalies: tuple[str, ...] = ()


class _CapExceeded(Exception):
    pass


@dataclass
class _Driver:
    """Per-game bookkeeping around an immutable GameState sequence."""

    state: GameState
    seats: Mapping[str, Agent]
    rng: random.Random
    max_decisions: int
    decisions: int = 0
    anomalies: list[str] = field(default_factory=list)

    # -- agent interaction ------------------------------------------------

    def _agent_apply(self, faction: str, offer: tuple[ParsedCommand, ...]) -> ParsedCommand:
        if self.decisions >= self.max_decisions:
            raise _CapExceeded(f"decision cap exceeded ({self.max_decisions})")
        offer = _canonical(offer)
        choice = self.seats[faction].choose(self.state, faction, offer, self.rng)
        if choice not in offer:
            raise EngineError(
                f"agent {self.seats[faction].name!r} returned a move outside its offer",
                state=self.state,
                faction=faction,
                cmd=choice,
            )
        self.state = apply(self.state, faction, choice)
        self.decisions += 1
        return choice

    # -- pending decisions ------------------------------------------------

    def _drain(self, skip: str | None = None) -> None:
        """Answer blocking pending decisions in queue order (leech offers
        enqueue clockwise from the builder; head-first matches snellman's
        default answer order). ``skip`` defers the named faction's own
        answers to its turn protocol's answers-first branch.
        """
        while True:
            for pending in self.state.pending:
                faction = pending.faction
                if faction == skip:
                    continue
                answers = pending_answer_moves(self.state, faction)
                if answers:
                    self._agent_apply(faction, answers)
                    break
            else:
                return

    # -- setup phases -----------------------------------------------------

    def _setup_step(self) -> None:
        faction = active_faction(self.state)
        want = "build" if self.state.phase == Phase.SETUP_DWELLINGS else "pass"
        offer = tuple(m for m in legal_moves(self.state) if m.verb == want)
        if not offer:
            raise EngineError(
                f"no {want!r} moves during {self.state.phase}",
                state=self.state,
                faction=faction,
                cmd=cmd("wait"),
            )
        self._agent_apply(faction, offer)
        self._drain()
        self.state = advance_turn(self.state)

    # -- income / cleanup windows -----------------------------------------

    def _spade_window(self) -> None:
        """Let each faction spend (or forfeit) income-window spades.
        ``legal_moves_for``'s INCOME/CLEANUP branch serves the transform
        moves; forfeiting is a plain ``lose_spade`` decrement.
        """
        while True:
            faction = next(
                (f for f in self.state.turn_order if self.state.factions[f].spades_available > 0),
                None,
            )
            if faction is None:
                return
            balance = self.state.factions[faction].spades_available
            forfeit = cmd("lose_spade", kind=Kind.BOOKKEEPING, n1=balance)
            transforms = tuple(
                m for m in legal_moves_for(self.state, faction) if m.verb == "transform"
            )
            if not transforms:
                self.state = apply(self.state, faction, forfeit)
                continue
            self._agent_apply(faction, transforms + (forfeit,))
            self._drain()

    def _income_window(self) -> None:
        for faction in self.state.turn_order:
            self.state = apply(
                self.state, faction, cmd("other_income_for_faction", kind=Kind.INCOME)
            )
        self._drain()
        self._spade_window()
        self.state = begin_actions(self.state)

    def _cleanup_window(self) -> None:
        for faction in self.state.turn_order:
            self.state = apply(
                self.state, faction, cmd("cult_income_for_faction", kind=Kind.INCOME)
            )
        self._drain()
        self._spade_window()
        self.state = end_of_round(self.state)

    # -- the turn protocol -------------------------------------------------

    def _actions_turn(self) -> None:
        """One full turn for ``active_faction``: forced pending answers
        first, then exactly one fresh main-track action, then its
        continuations (``is_turn_boundary`` False) plus free
        convert/burn until the agent picks ``done``. ``advance_turn``
        afterwards re-serves the same faction while ACTC
        ``extra_actions`` remain.
        """
        faction = active_faction(self.state)
        fresh_taken = False
        prev_verb: str | None = None
        while True:
            answers = pending_answer_moves(self.state, faction)
            if answers:
                offer = answers
            elif not fresh_taken:
                offer = tuple(
                    m for m in legal_moves(self.state) if m.verb not in _NOOP_VERBS
                )
            else:
                moves = legal_moves(self.state)
                continuations = tuple(
                    m
                    for m in moves
                    if m.verb in MAIN_TRACK_VERBS
                    and m.verb != "pass"
                    and not is_turn_boundary(m, self.state, faction, prev_verb)
                )
                frees = tuple(m for m in moves if m.verb in _FREE_VERBS)
                offer = continuations + frees + (cmd("done"),)
            pre_state = self.state
            choice = self._agent_apply(faction, offer)
            if choice.verb == "done":
                break
            self._drain(skip=faction)
            if choice.verb in MAIN_TRACK_VERBS:
                if is_turn_boundary(choice, pre_state, faction, prev_verb):
                    fresh_taken = True
                prev_verb = choice.verb
            if choice.verb == "pass":
                break
        self.state = advance_turn(self.state)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        self.state = start_setup(self.state)
        while self.state.phase != Phase.FINISHED:
            phase = self.state.phase
            if phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
                self._setup_step()
            elif phase == Phase.INCOME:
                self._income_window()
            elif phase == Phase.ACTIONS:
                self._drain()
                if self.state.phase == Phase.ACTIONS:
                    self._actions_turn()
            elif phase == Phase.CLEANUP:
                self._cleanup_window()
            else:  # pragma: no cover - Phase enum is exhaustive above
                raise EngineError(
                    f"unexpected phase {phase}",
                    state=self.state,
                    faction=self.state.turn_order[0],
                    cmd=cmd("wait"),
                )
        self.state = final_scoring(self.state)


def _competition_ranks(vps: Mapping[str, int]) -> dict[str, int]:
    ordered = sorted(vps, key=lambda f: -vps[f])
    ranks: dict[str, int] = {}
    for index, faction in enumerate(ordered):
        ranks[faction] = index if vps[faction] != vps[ordered[index - 1]] or index == 0 else ranks[ordered[index - 1]]
    return ranks


def run_game(
    setup: GameSetup,
    seats: Mapping[str, Agent],
    rng: random.Random,
    max_decisions: int = 5000,
) -> GameResult:
    """Play one complete game; never raises for engine rejections -- an
    ``EngineError``/cap breach is recorded on the result (the arena
    doubles as a ``legal_moves``-soundness fuzzer; findings must be
    visible, not fatal). Driver bugs (any other exception) propagate.
    """
    if set(seats) != set(setup.factions):
        raise ValueError(f"seats {sorted(seats)} != setup factions {sorted(setup.factions)}")
    driver = _Driver(
        state=GameState.initial(setup), seats=seats, rng=rng, max_decisions=max_decisions
    )
    error: str | None = None
    try:
        driver.run()
    except EngineError as exc:
        error = f"decision {driver.decisions}: {exc} (cmd={exc.cmd.verb if hasattr(exc, 'cmd') else '?'})"
    except _CapExceeded as exc:
        error = str(exc)
    state = driver.state
    vps = {f: state.factions[f].vp for f in setup.factions}
    seat_names = {f: seats[f].name for f in setup.factions}
    ranks_unordered = _competition_ranks(vps)
    ranks = {f: ranks_unordered[f] for f in setup.factions}
    return GameResult(
        setup_game_id=setup.game_id,
        seats=seat_names,
        vps=vps,
        ranks=ranks,
        decisions=driver.decisions,
        error=error,
        anomalies=tuple(driver.anomalies),
    )
