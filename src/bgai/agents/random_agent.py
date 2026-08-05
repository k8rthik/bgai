"""Uniform-random baseline over progress moves. Floor of the arena ladder."""

from __future__ import annotations

import random

from bgai.agents.base import progress_moves
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState


class RandomAgent:
    def __init__(self, seed: int, name: str = "random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose(
        self, state: GameState | None, faction: str, moves: tuple[ParsedCommand, ...]
    ) -> ParsedCommand:
        if not moves:
            raise ValueError(f"{faction}: no legal moves offered")
        return self._rng.choice(progress_moves(moves))
