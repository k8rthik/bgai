"""State -> numpy encoding, v1 (plan Task 3; bump ``ENCODING_VERSION`` on
any layout change).

Everything is mover-relative: seat blocks are ordered [mover, next
clockwise, ...] by rotating ``state.setup.factions`` to the mover, and
hex ownership is a mover-relative seat one-hot. Values are stored as
small ints (counts, not normalized); the torch data loader converts to
float and scales.

Layout (verified by tests/test_encode_state.py):
- ``hex_planes`` (113, HEX_FEAT_DIM) int8, row = vocab.HEXES order:
  terrain color one-hot (7) | river flag (1) | building one-hot (5) |
  owner mover-relative seat one-hot (4)
- ``globals`` (GLOBAL_DIM,) int16:
  4 seat blocks of SEAT_BLOCK each, then game context:
  round one-hot (7, index=round 0..6) | phase one-hot (6) |
  6 score tiles x (cult one-hot incl CULT_P (5) | req (1)) |
  bonus-pool multi-hot (10) | power-actions-taken multi-hot (6) |
  pending-decision count (1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgai.engine.tm.state import GameState, Phase
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES
from bgai.training.vocab import (
    BUILDING_INDEX,
    BUILDINGS,
    COLOR_INDEX,
    COLORS,
    CULT_INDEX,
    CULTS4,
    FACTION_INDEX,
    FACTION_NAMES,
    HEXES,
)

_BON_IDS: tuple[str, ...] = tuple(sorted(BONUS_TILES))
_BON_INDEX = {t: i for i, t in enumerate(_BON_IDS)}
_FAV_IDS: tuple[str, ...] = tuple(sorted(FAVOR_TILES))
_FAV_INDEX = {t: i for i, t in enumerate(_FAV_IDS)}
_PHASES: tuple[Phase, ...] = tuple(Phase)
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}

HEX_FEAT_DIM = len(COLORS) + 1 + len(BUILDINGS) + 4  # 7 + 1 + 5 + 4 = 17

# Per-seat block: faction one-hot | C,W,P,priest_pool,VP | bowls (3) |
# shipping,dig | spades,extra_actions | passed | is_active | cults (4) |
# keys | bridges_built | bonus one-hot (10) | favors multi-hot (12) |
# towns count
_FACTION_OH = len(FACTION_NAMES)  # 14
_RES_OFFSET = _FACTION_OH
_N_RES = 5
_BOWLS_OFFSET = _RES_OFFSET + _N_RES
_TRACKS_OFFSET = _BOWLS_OFFSET + 3
_SPADES_OFFSET = _TRACKS_OFFSET + 2
_FLAGS_OFFSET = _SPADES_OFFSET + 2
_CULTS_OFFSET = _FLAGS_OFFSET + 2
_KEYS_OFFSET = _CULTS_OFFSET + len(CULTS4)
_BON_OFFSET = _KEYS_OFFSET + 2
_FAV_OFFSET = _BON_OFFSET + len(_BON_IDS)
_TOWNS_OFFSET = _FAV_OFFSET + len(_FAV_IDS)
SEAT_BLOCK = _TOWNS_OFFSET + 1  # 57

_CTX_ROUND = 4 * SEAT_BLOCK
_CTX_PHASE = _CTX_ROUND + 7
_CTX_SCORE = _CTX_PHASE + len(_PHASES)
_SCORE_TILE_DIM = len(CULTS4) + 1 + 1  # cult one-hot + CULT_P flag folded + req
_CTX_BON_POOL = _CTX_SCORE + 6 * _SCORE_TILE_DIM
_CTX_ACTS = _CTX_BON_POOL + len(_BON_IDS)
_CTX_PENDING = _CTX_ACTS + 6
GLOBAL_DIM = _CTX_PENDING + 1


@dataclass(frozen=True)
class StateEncoding:
    hex_planes: np.ndarray
    globals: np.ndarray


def _seat_order(state: GameState, faction: str) -> tuple[str, ...]:
    seats = state.setup.factions
    pivot = seats.index(faction)
    return seats[pivot:] + seats[:pivot]


def encode_state(state: GameState, faction: str) -> StateEncoding:
    order = _seat_order(state, faction)
    seat_of = {name: i for i, name in enumerate(order)}

    planes = np.zeros((len(HEXES), HEX_FEAT_DIM), dtype=np.int8)
    for row, hex_key in enumerate(HEXES):
        hx = state.hexes[hex_key]
        if hx.color in COLOR_INDEX:
            planes[row, COLOR_INDEX[hx.color]] = 1
        else:  # river
            planes[row, len(COLORS)] = 1
        if hx.building is not None:
            planes[row, len(COLORS) + 1 + BUILDING_INDEX[hx.building]] = 1
        if hx.owner is not None and hx.owner in seat_of:
            planes[row, len(COLORS) + 1 + len(BUILDINGS) + seat_of[hx.owner]] = 1

    g = np.zeros(GLOBAL_DIM, dtype=np.int16)
    active = state.turn_order[state.active_index] if state.turn_order else None
    for seat, name in enumerate(order):
        fs = state.factions[name]
        base = seat * SEAT_BLOCK
        g[base + FACTION_INDEX[name]] = 1
        g[base + _RES_OFFSET : base + _RES_OFFSET + _N_RES] = (
            fs.coins, fs.workers, fs.priests, fs.priest_pool, fs.vp
        )
        g[base + _BOWLS_OFFSET : base + _BOWLS_OFFSET + 3] = (
            fs.power.bowl1, fs.power.bowl2, fs.power.bowl3
        )
        g[base + _TRACKS_OFFSET : base + _TRACKS_OFFSET + 2] = (fs.shipping, fs.dig_level)
        g[base + _SPADES_OFFSET : base + _SPADES_OFFSET + 2] = (
            fs.spades_available, fs.extra_actions
        )
        g[base + _FLAGS_OFFSET] = int(fs.passed)
        g[base + _FLAGS_OFFSET + 1] = int(name == active)
        for cult in CULTS4:
            g[base + _CULTS_OFFSET + CULT_INDEX[cult]] = state.cults[name][cult]
        g[base + _KEYS_OFFSET] = fs.keys
        g[base + _KEYS_OFFSET + 1] = fs.bridges_built
        if fs.bonus is not None:
            g[base + _BON_OFFSET + _BON_INDEX[fs.bonus]] = 1
        for fav in fs.favors:
            g[base + _FAV_OFFSET + _FAV_INDEX[fav]] += 1
        g[base + _TOWNS_OFFSET] = len(fs.towns)

    g[_CTX_ROUND + state.round] = 1
    g[_CTX_PHASE + _PHASE_INDEX[state.phase]] = 1
    for r, tile in enumerate(state.setup.score_tiles):
        base = _CTX_SCORE + r * _SCORE_TILE_DIM
        if tile.cult in CULT_INDEX:
            g[base + CULT_INDEX[tile.cult]] = 1
        else:  # CULT_P (temple-scoring-tile)
            g[base + len(CULTS4)] = 1
        g[base + len(CULTS4) + 1] = tile.req
    for bon in state.setup.bonus_tiles:
        g[_CTX_BON_POOL + _BON_INDEX[bon]] = 1
    for act in state.power_actions_taken:
        # ACT1..ACT6 only; faction specials tracked per-faction via actions_used
        if act.startswith("ACT") and act[3:].isdigit():
            g[_CTX_ACTS + int(act[3:]) - 1] = 1
    g[_CTX_PENDING] = len(state.pending)

    return StateEncoding(hex_planes=planes, globals=g)
