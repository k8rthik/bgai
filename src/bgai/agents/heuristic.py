"""Simple scripted opponent -- a yardstick above random, NOT a strategy
oracle. Never exposed through MCP tools (spec: the LLM gets facts only).

Priority policy (plan task 5): answer leech by amount; favor/town tile
preferences; setup builds prefer central rows; then a fixed action
priority (big upgrades > small upgrades > build > transform > send >
power/special action > dig > pass-with-best-coin-tile); DONE when open
and nothing matched; seeded-random progress move as the last resort.
"""

from __future__ import annotations

import random

from bgai.agents.base import progress_moves
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState
from bgai.engine.tm.tiles import BONUS_TILES

_FAVOR_PREFERENCE = ("FAV11", "FAV10")
_TOWN_PREFERENCE = ("TW3", "TW1")
_LEECH_ACCEPT_MAX = 2  # rough VP-cost rule of thumb
_CENTRAL_ROWS = "DEF"


def _first(moves: tuple[ParsedCommand, ...], verb: str, **fields: object) -> ParsedCommand | None:
    for m in moves:
        if m.verb == verb and all(getattr(m, k) == v for k, v in fields.items()):
            return m
    return None


def _best_pass(moves: tuple[ParsedCommand, ...]) -> ParsedCommand | None:
    passes = [m for m in moves if m.verb == "pass"]
    if not passes:
        return None
    setup_or_round_pass = [m for m in passes if m.tile is not None]
    if not setup_or_round_pass:
        return passes[0]  # round-6 pass carries no tile

    def coins(m: ParsedCommand) -> int:
        tile = BONUS_TILES.get(m.tile or "")
        return tile.income.get("C", 0) if tile is not None else 0

    return max(setup_or_round_pass, key=coins)


def _tile_preference(
    moves: tuple[ParsedCommand, ...], verb: str, preference: tuple[str, ...]
) -> ParsedCommand | None:
    offered = [m for m in moves if m.verb == verb]
    if not offered:
        return None
    for tile in preference:
        for m in offered:
            if m.tile == tile:
                return m
    return max(offered, key=lambda m: m.tile or "")


def _pick(
    state: GameState | None,
    faction: str,
    moves: tuple[ParsedCommand, ...],
    rng: random.Random,
) -> ParsedCommand:
    if not moves:
        raise ValueError(f"{faction}: no legal moves offered")

    leech = _first(moves, "leech")
    if leech is not None:
        if (leech.n1 or 0) <= _LEECH_ACCEPT_MAX:
            return leech
        decline = _first(moves, "decline")
        if decline is not None:
            return decline

    favor = _tile_preference(moves, "gain_favor", _FAVOR_PREFERENCE)
    if favor is not None:
        return favor
    town = _tile_preference(moves, "gain_town", _TOWN_PREFERENCE)
    if town is not None:
        return town

    builds = [m for m in moves if m.verb == "build"]
    for building in ("SH", "SA", "TP", "TE"):
        upgrade = _first(moves, "upgrade", building=building)
        if upgrade is not None:
            return upgrade
    if builds:
        central = [m for m in builds if (m.loc or " ")[0] in _CENTRAL_ROWS]
        return central[0] if central else builds[0]

    for verb in ("transform", "send", "action", "dig"):
        move = _first(moves, verb)
        if move is not None:
            return move

    best_pass = _best_pass(moves)
    if best_pass is not None:
        return best_pass

    done = _first(moves, "done")
    if done is not None:
        return done

    return rng.choice(progress_moves(moves))


class HeuristicAgent:
    def __init__(self, seed: int, name: str = "heuristic") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose(
        self, state: GameState | None, faction: str, moves: tuple[ParsedCommand, ...]
    ) -> ParsedCommand:
        return _pick(state, faction, moves, self._rng)
