"""Agent protocol shared by baseline bots and the arena driver."""

from __future__ import annotations

from typing import Protocol

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState

AUX_VERBS: frozenset[str] = frozenset({"convert", "burn", "wait"})
"""Order-exempt resource moves, always legal -- an unbiased random pick over
the full legal set would convert forever and never end its turn."""


class Agent(Protocol):
    name: str

    def choose(
        self, state: GameState, faction: str, moves: tuple[ParsedCommand, ...]
    ) -> ParsedCommand: ...


def progress_moves(moves: tuple[ParsedCommand, ...]) -> tuple[ParsedCommand, ...]:
    """Moves that advance the game (non-AUX); falls back to `moves` if empty."""
    filtered = tuple(m for m in moves if m.verb not in AUX_VERBS)
    return filtered or moves
