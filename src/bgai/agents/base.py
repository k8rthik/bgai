"""The Agent protocol: the single interface every Terra Mystica player
implements (master plan Phase 4; plan 2026-08-04-arena-baselines Task 1).
"""

from __future__ import annotations

import random
from typing import Protocol

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState


class Agent(Protocol):
    """Anything that can play Terra Mystica in the arena.

    The arena calls ``choose`` once per decision -- main actions, setup
    dwelling placements, leech answers, favor picks, and income-window
    spade transforms all arrive through this single method (the engine's
    pending-decision queue makes them all ordinary moves). ``offer`` is
    always non-empty and the return value must be one of its elements.
    ``rng`` is the arena's seeded generator: agents must draw randomness
    only from it so games are reproducible from (setup, seats, seed).
    """

    name: str

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand: ...
