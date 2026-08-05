"""Uniform-random baseline agent (plan Task 1)."""

from __future__ import annotations

import random

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState


class RandomAgent:
    """Uniform-random baseline: the floor of the rating scale."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        return offer[rng.randrange(len(offer))]
