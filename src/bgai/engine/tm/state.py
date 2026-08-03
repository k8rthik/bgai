"""Immutable game-state model: `GameState`/`FactionState` + tiny read helpers.

Pure construction only -- `GameState.initial`/`FactionState.initial` build
the pre-round-1 snapshot from a :class:`~bgai.engine.tm.setup.GameSetup`.
No move legality, no `apply`: that belongs to a later task. Field names and
shapes are frozen API per the task-3 brief; later tasks add fields via
migration rather than repurposing these.

Starting VP is 20 for every faction (deltas-oracle setup rows). Priest pool
starts at ``MAX_PRIESTS`` for every faction regardless of starting priests
in hand, per the brief's explicit ``FactionState.initial`` contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from bgai.engine.tm.board import base_board
from bgai.engine.tm.cults import PRIEST_SLOT_STEPS
from bgai.engine.tm.factions_data import CULTS, FACTIONS, MAX_PRIESTS, FactionData
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.tiles import FAVOR_POOL_COUNTS, TILE_OPTIONS, TOWN_POOL_COUNTS


class Phase(Enum):
    SETUP_DWELLINGS = "SETUP_DWELLINGS"
    SETUP_BONUS = "SETUP_BONUS"
    INCOME = "INCOME"
    ACTIONS = "ACTIONS"
    CLEANUP = "CLEANUP"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class HexState:
    color: str
    building: str | None
    owner: str | None


@dataclass(frozen=True)
class PendingDecision:
    """A forced sub-decision queued ahead of the turn-order successor.

    ``kind`` values used by later tasks: "leech", "gain_favor", "gain_town",
    "cult_choice", "convert_w_to_p", "halflings_spades", "bonus_choice",
    "spade_use".
    """

    faction: str
    kind: str
    amount: int = 0
    source: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactionState:
    name: str
    coins: int
    workers: int
    priests: int
    priest_pool: int
    power: Power
    vp: int
    shipping: int
    dig_level: int
    teleport_level: int
    buildings: Mapping[str, frozenset[str]]
    favors: tuple[str, ...]
    bonus: str | None
    towns: tuple[str, ...]
    keys: int
    passed: bool
    actions_used: frozenset[str]
    cult_blocked: frozenset[str]

    @classmethod
    def initial(cls, data: FactionData) -> FactionState:
        return cls(
            name=data.name,
            coins=data.coins,
            workers=data.workers,
            priests=data.priests,
            priest_pool=MAX_PRIESTS,
            power=Power(*data.power),
            vp=20,
            shipping=data.shipping.level,
            dig_level=data.dig.level,
            teleport_level=0,
            buildings={building: frozenset() for building in data.buildings},
            favors=(),
            bonus=None,
            towns=(),
            keys=0,
            passed=False,
            actions_used=frozenset(),
            cult_blocked=frozenset(),
        )


def _initial_hexes() -> dict[str, HexState]:
    board = base_board()
    return {
        key: HexState(color=hex_.color, building=None, owner=None)
        for key, hex_ in board.hexes.items()
    }


def _initial_towns_pool(setup: GameSetup) -> dict[str, int]:
    return {
        town: count
        for town, count in TOWN_POOL_COUNTS.items()
        if TILE_OPTIONS.get(town) != "mini-expansion-1" or setup.options.mini_expansion_1
    }


@dataclass(frozen=True)
class GameState:
    setup: GameSetup
    round: int
    phase: Phase
    turn_order: tuple[str, ...]
    passed_order: tuple[str, ...]
    active_index: int
    pending: tuple[PendingDecision, ...]
    hexes: Mapping[str, HexState]
    bridges: frozenset[frozenset[str]]
    factions: Mapping[str, FactionState]
    cults: Mapping[str, Mapping[str, int]]
    cult_10: Mapping[str, str | None]
    priest_slots: Mapping[str, tuple[str | None, ...]]
    favors_pool: Mapping[str, int]
    towns_pool: Mapping[str, int]
    bonus_coins: Mapping[str, int]
    power_actions_taken: frozenset[str]

    @classmethod
    def initial(cls, setup: GameSetup) -> GameState:
        factions = {name: FactionState.initial(FACTIONS[name]) for name in setup.factions}
        cults = {name: dict(FACTIONS[name].cults) for name in setup.factions}
        empty_priest_slots = tuple(None for _ in PRIEST_SLOT_STEPS)
        return cls(
            setup=setup,
            round=0,
            phase=Phase.SETUP_DWELLINGS,
            turn_order=setup.factions,
            passed_order=setup.factions,
            active_index=0,
            pending=(),
            hexes=_initial_hexes(),
            bridges=frozenset(),
            factions=factions,
            cults=cults,
            cult_10={cult: None for cult in CULTS},
            priest_slots={cult: empty_priest_slots for cult in CULTS},
            favors_pool=dict(FAVOR_POOL_COUNTS),
            towns_pool=_initial_towns_pool(setup),
            bonus_coins={bon: 0 for bon in setup.bonus_tiles},
            power_actions_taken=frozenset(),
        )


def active_faction(state: GameState) -> str:
    """Head of `pending` if any, else the turn-order successor."""
    if state.pending:
        return state.pending[0].faction
    return state.turn_order[state.active_index]


def with_faction(state: GameState, name: str, fs: FactionState) -> GameState:
    """Return a new GameState with `name`'s FactionState replaced by `fs`."""
    new_factions = dict(state.factions)
    new_factions[name] = fs
    return replace(state, factions=new_factions)


def cult_string(state: GameState, faction: str) -> str:
    """Cult-track positions for `faction` as "F/W/E/A", e.g. "0/1/1/0"."""
    positions = state.cults[faction]
    return "/".join(str(positions[cult]) for cult in CULTS)
