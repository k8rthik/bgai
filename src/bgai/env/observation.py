"""Mover-relative observation encoding for 2-5 player tables.

This is a *sibling* of ``training.encode_state``, not a replacement:
that encoder is pinned at four seats (``_CTX_ROUND = 4 * SEAT_BLOCK``,
4-wide hex-owner one-hot) because every one of the 3,563 corpus games is
4-player, and every existing checkpoint depends on its exact layout. The
env has to serve 2-5 players from one fixed space, so it carries
``MAX_SEATS`` seat blocks plus a ``seat_present`` mask and a
``seat_count`` one-hot, and absent seats stay zero.

Everything is mover-relative: seat blocks are ordered [mover, next in
seat order, ...] and hex ownership is a mover-relative seat one-hot, so a
policy is seat-equivariant for free.

Layout (pinned by tests/test_env_observation.py):

``hex_planes`` (113, HEX_FEAT_DIM) int8, rows in ``vocab.HEXES`` order
    terrain colour one-hot (7) | river flag (1) | building one-hot (5) |
    owner mover-relative seat one-hot (MAX_SEATS)

``globals`` (GLOBAL_DIM,) int16
    MAX_SEATS seat blocks of SEAT_BLOCK, then game context:
    seat-count one-hot (4, for 2..5) | seat-present mask (MAX_SEATS) |
    round one-hot (7) | phase one-hot (6) |
    6 score tiles x (cult one-hot (4) | temple flag (1) | req (1)) |
    bonus-pool multi-hot (10) | power-actions-taken multi-hot (6) |
    pending-decision count (1) | offer size (1)

``candidates`` (max_candidates, MOVE_FIELDS) int16
    ``training.encode_move`` features per offered move in canonical order,
    zero-padded. Row i is meaningful iff ``action_mask[i]`` is 1.

``action_mask`` (max_candidates,) int8
    1 for every index the engine currently offers. Derived straight from
    ``driver.decision``'s offer -- there is no second copy of the rules
    here, and there cannot be one by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState, Phase
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES
from bgai.env.config import (
    MAX_SEATS,
    MIN_PLAYERS,
    SCORE_TILES_PER_GAME,
    OfferOverflowError,
)
from bgai.training.encode_move import MOVE_FIELDS, encode_move
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

OBS_VERSION = 1
"""Bump on ANY change to the layouts above; a checkpoint trained on one
version cannot read another."""

_BON_IDS: tuple[str, ...] = tuple(sorted(BONUS_TILES))
_BON_INDEX = {t: i for i, t in enumerate(_BON_IDS)}
_FAV_IDS: tuple[str, ...] = tuple(sorted(FAVOR_TILES))
_FAV_INDEX = {t: i for i, t in enumerate(_FAV_IDS)}
_PHASES: tuple[Phase, ...] = tuple(Phase)
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASES)}

N_ROUNDS_SLOTS = 7  # round 0 (setup) .. 6
N_POWER_ACTIONS = 6  # ACT1..ACT6
N_SEAT_COUNTS = MAX_SEATS - MIN_PLAYERS + 1  # 2,3,4,5

HEX_FEAT_DIM = len(COLORS) + 1 + len(BUILDINGS) + MAX_SEATS  # 7+1+5+5 = 18

# -- seat block offsets ----------------------------------------------------
_SB_FACTION = 0
_SB_RES = _SB_FACTION + len(FACTION_NAMES)  # coins, workers, priests, pool, vp
_SB_N_RES = 5
_SB_BOWLS = _SB_RES + _SB_N_RES
_SB_TRACKS = _SB_BOWLS + 3  # shipping, dig_level, teleport_level
_SB_SPADES = _SB_TRACKS + 3  # spades_available, extra_actions
_SB_FLAGS = _SB_SPADES + 2  # passed, is_active, dropped
_SB_CULTS = _SB_FLAGS + 3
_SB_KEYS = _SB_CULTS + len(CULTS4)  # keys, bridges_built
_SB_BONUS = _SB_KEYS + 2
_SB_FAVORS = _SB_BONUS + len(_BON_IDS)
_SB_TOWNS = _SB_FAVORS + len(_FAV_IDS)
SEAT_BLOCK = _SB_TOWNS + 1

# -- context offsets -------------------------------------------------------
_CTX = MAX_SEATS * SEAT_BLOCK
_CTX_SEAT_COUNT = _CTX
_CTX_SEAT_PRESENT = _CTX_SEAT_COUNT + N_SEAT_COUNTS
_CTX_ROUND = _CTX_SEAT_PRESENT + MAX_SEATS
_CTX_PHASE = _CTX_ROUND + N_ROUNDS_SLOTS
_SCORE_TILE_DIM = len(CULTS4) + 1 + 1
_CTX_SCORE = _CTX_PHASE + len(_PHASES)
_CTX_BON_POOL = _CTX_SCORE + SCORE_TILES_PER_GAME * _SCORE_TILE_DIM
_CTX_ACTS = _CTX_BON_POOL + len(_BON_IDS)
_CTX_PENDING = _CTX_ACTS + N_POWER_ACTIONS
_CTX_OFFER = _CTX_PENDING + 1
GLOBAL_DIM = _CTX_OFFER + 1


@dataclass(frozen=True, eq=False)
class Observation:
    """One agent's view of one decision point. Immutable by contract --
    the arrays are never written to after construction.

    ``eq=False`` because a generated ``__eq__`` would compare numpy arrays
    with ``==`` and raise on the ambiguous truth value; :meth:`__eq__`
    below does the elementwise comparison a round-trip test wants.
    """

    hex_planes: np.ndarray  # (113, HEX_FEAT_DIM) int8
    globals: np.ndarray  # (GLOBAL_DIM,) int16
    candidates: np.ndarray  # (max_candidates, MOVE_FIELDS) int16
    action_mask: np.ndarray  # (max_candidates,) int8

    @property
    def n_legal(self) -> int:
        return int(self.action_mask.sum())

    _ARRAYS = ("hex_planes", "globals", "candidates", "action_mask")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Observation):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            and getattr(self, name).dtype == getattr(other, name).dtype
            for name in self._ARRAYS
        )

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, name).tobytes() for name in self._ARRAYS))


def seat_order(state: GameState, faction: str) -> tuple[str, ...]:
    """``state.setup.factions`` rotated so ``faction`` is seat 0."""
    seats = state.setup.factions
    pivot = seats.index(faction)
    return seats[pivot:] + seats[:pivot]


def encode_hex_planes(state: GameState, seat_of: dict[str, int]) -> np.ndarray:
    planes = np.zeros((len(HEXES), HEX_FEAT_DIM), dtype=np.int8)
    building_base = len(COLORS) + 1
    owner_base = building_base + len(BUILDINGS)
    for row, hex_key in enumerate(HEXES):
        hx = state.hexes[hex_key]
        if hx.color in COLOR_INDEX:
            planes[row, COLOR_INDEX[hx.color]] = 1
        else:  # river
            planes[row, len(COLORS)] = 1
        if hx.building is not None:
            planes[row, building_base + BUILDING_INDEX[hx.building]] = 1
        if hx.owner is not None and hx.owner in seat_of:
            planes[row, owner_base + seat_of[hx.owner]] = 1
    return planes


def _encode_seat_block(
    g: np.ndarray, base: int, state: GameState, name: str, active: str | None
) -> None:
    fs = state.factions[name]
    g[base + _SB_FACTION + FACTION_INDEX[name]] = 1
    g[base + _SB_RES : base + _SB_RES + _SB_N_RES] = (
        fs.coins,
        fs.workers,
        fs.priests,
        fs.priest_pool,
        fs.vp,
    )
    g[base + _SB_BOWLS : base + _SB_BOWLS + 3] = (
        fs.power.bowl1,
        fs.power.bowl2,
        fs.power.bowl3,
    )
    g[base + _SB_TRACKS : base + _SB_TRACKS + 3] = (
        fs.shipping,
        fs.dig_level,
        fs.teleport_level,
    )
    g[base + _SB_SPADES : base + _SB_SPADES + 2] = (
        fs.spades_available,
        fs.extra_actions,
    )
    g[base + _SB_FLAGS] = int(fs.passed)
    g[base + _SB_FLAGS + 1] = int(name == active)
    g[base + _SB_FLAGS + 2] = int(fs.dropped)
    for cult in CULTS4:
        g[base + _SB_CULTS + CULT_INDEX[cult]] = state.cults[name][cult]
    g[base + _SB_KEYS] = fs.keys
    g[base + _SB_KEYS + 1] = fs.bridges_built
    if fs.bonus is not None:
        g[base + _SB_BONUS + _BON_INDEX[fs.bonus]] = 1
    for fav in fs.favors:
        g[base + _SB_FAVORS + _FAV_INDEX[fav]] += 1
    g[base + _SB_TOWNS] = len(fs.towns)


def encode_globals(
    state: GameState, order: tuple[str, ...], offer_size: int
) -> np.ndarray:
    g = np.zeros(GLOBAL_DIM, dtype=np.int16)
    active = state.turn_order[state.active_index] if state.turn_order else None
    for seat, name in enumerate(order):
        _encode_seat_block(g, seat * SEAT_BLOCK, state, name, active)

    n_seats = len(order)
    g[_CTX_SEAT_COUNT + (n_seats - MIN_PLAYERS)] = 1
    g[_CTX_SEAT_PRESENT : _CTX_SEAT_PRESENT + n_seats] = 1
    g[_CTX_ROUND + state.round] = 1
    g[_CTX_PHASE + _PHASE_INDEX[state.phase]] = 1
    for r, tile in enumerate(state.setup.score_tiles[:SCORE_TILES_PER_GAME]):
        base = _CTX_SCORE + r * _SCORE_TILE_DIM
        if tile.cult in CULT_INDEX:
            g[base + CULT_INDEX[tile.cult]] = 1
        else:  # CULT_P (temple-scoring-tile)
            g[base + len(CULTS4)] = 1
        g[base + len(CULTS4) + 1] = tile.req
    for bon in state.setup.bonus_tiles:
        g[_CTX_BON_POOL + _BON_INDEX[bon]] = 1
    for act in state.power_actions_taken:
        if act.startswith("ACT") and act[3:].isdigit():
            index = int(act[3:]) - 1
            if 0 <= index < N_POWER_ACTIONS:
                g[_CTX_ACTS + index] = 1
    g[_CTX_PENDING] = len(state.pending)
    g[_CTX_OFFER] = offer_size
    return g


def encode_candidates(
    state: GameState,
    faction: str,
    offer: tuple[ParsedCommand, ...],
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(candidates, action_mask)`` for one offer, zero-padded.

    The mask is the engine's own legal set, nothing else: one bit per
    index of ``offer``, which came from ``driver.decision`` and therefore
    from ``legal_moves``. Raises :class:`OfferOverflowError` rather than
    truncating, because a truncated mask would silently delete legal moves.
    """
    if len(offer) > max_candidates:
        raise OfferOverflowError(
            f"engine offered {len(offer)} candidates to {faction!r} but the "
            f"action space holds {max_candidates}; raise "
            f"EnvConfig.max_candidates (see config.MAX_CANDIDATES for the "
            f"measured distribution)"
        )
    candidates = np.zeros((max_candidates, MOVE_FIELDS), dtype=np.int16)
    mask = np.zeros(max_candidates, dtype=np.int8)
    for index, move in enumerate(offer):
        candidates[index] = encode_move(move, state, faction)
        mask[index] = 1
    return candidates, mask


def encode_observation(
    state: GameState,
    faction: str,
    offer: tuple[ParsedCommand, ...],
    max_candidates: int,
) -> Observation:
    """The full observation for ``faction`` at ``state`` with ``offer``.

    ``offer`` is empty for a seat that is not the one to decide (and for
    every seat once the game is over); the mask is then all zeros, the
    PettingZoo convention for "not your move".
    """
    order = seat_order(state, faction)
    seat_of = {name: i for i, name in enumerate(order)}
    candidates, mask = encode_candidates(state, faction, offer, max_candidates)
    return Observation(
        hex_planes=encode_hex_planes(state, seat_of),
        globals=encode_globals(state, order, len(offer)),
        candidates=candidates,
        action_mask=mask,
    )
