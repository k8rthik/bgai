"""Play complete headless games between agents (Phase 4).

The turn machinery lives in ``bgai.arena.driver`` as an explicit,
immutable ``SimState`` (Phase 6, decision D6.1 -- MCTS must clone and
branch positions, which local variables cannot express). This module is
the thin loop over it: ask the driver for the pending decision, ask that
seat's agent to choose, advance, repeat.

Timing (Apple-silicon MacBook, random vs random, 200 corpus-sampled
setups): ~48 ms per headless 4p game -- ~20x under the master plan's
<1s target, no optimization warranted this phase.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from bgai.agents.base import Agent
from bgai.arena.driver import (
    MAIN_TRACK_VERBS,
    SimState,
    advance,
    canonical_moves,
    decision,
    new_game,
)
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.setup import GameSetup

__all__ = [
    "GameResult",
    "MAIN_TRACK_VERBS",
    "canonical_moves",
    "run_game",
]

# Backwards-compatible alias: this was module-private before the driver
# was extracted, and the training pipeline indexes candidates in exactly
# this order.
_canonical = canonical_moves


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


def _competition_ranks(vps: Mapping[str, int]) -> dict[str, int]:
    ordered = sorted(vps, key=lambda f: -vps[f])
    ranks: dict[str, int] = {}
    for index, faction in enumerate(ordered):
        if index == 0 or vps[faction] != vps[ordered[index - 1]]:
            ranks[faction] = index
        else:
            ranks[faction] = ranks[ordered[index - 1]]
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

    error: str | None = None
    sim: SimState = new_game(setup)
    try:
        while True:
            pending = decision(sim)
            if pending is None:
                break
            faction, offer = pending
            if sim.decisions >= max_decisions:
                error = f"decision cap exceeded ({max_decisions})"
                break
            agent = seats[faction]
            # Optional protocol extension (Phase 6): a search agent needs
            # the whole SimState to branch from, not just the GameState.
            # Ordinary agents implement `choose` and never see the driver.
            chooser = getattr(agent, "choose_sim", None)
            choice = (
                chooser(sim, faction, offer, rng)
                if chooser is not None
                else agent.choose(sim.game, faction, offer, rng)
            )
            if choice not in offer:
                raise EngineError(
                    f"agent {agent.name!r} returned a move outside its offer",
                    state=sim.game,
                    faction=faction,
                    cmd=choice,
                )
            sim = advance(sim, choice)
    except EngineError as exc:
        error = f"decision {sim.decisions}: {exc}"

    game = sim.game
    vps = {f: game.factions[f].vp for f in setup.factions}
    ranks_unordered = _competition_ranks(vps)
    return GameResult(
        setup_game_id=setup.game_id,
        seats={f: seats[f].name for f in setup.factions},
        vps=vps,
        ranks={f: ranks_unordered[f] for f in setup.factions},
        decisions=sim.decisions,
        error=error,
    )
