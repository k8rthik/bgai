"""State -> numpy encoding, v3 (plan Task 3; bump ``ENCODING_VERSION`` on
any layout change, and only ever append to ``globals``: a net trained on an
older encoding reads its prefix, ``ModelConfig.global_dim``).

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
  -- that is the whole v1 vector, ``GLOBAL_DIM_V1`` = 294 entries. v2
  appends the economy block, engine-derived facts an MLP over raw cells
  cannot compute (connectivity, income):
  4 economy seat blocks of ECON_SEAT_BLOCK (17) each, same seat order:
  next-round income C, W, P, PW (4) | best open cluster power,
  structures (SA counts twice), town power threshold, open clusters at
  or above the threshold (4) | structures left in supply D, TP, TE, SH,
  SA (5) | unbuilt reachable land hexes at 0, 1, 2, 3 spades from home
  terrain (4)
  then town tiles left per tile id (8, sorted) | favor tiles left per
  tile id (12, sorted)
  -- that is the whole v2 vector, ``GLOBAL_DIM_V2`` = 382 entries. v3
  appends the scoring block:
  4 scoring seat blocks of SCORE_SEAT_BLOCK (15) each, same seat order:
  network size, end-game network VP, cult-majority VP over the 4 tracks,
  leftover-resource VP (4) | rank on each cult track (4) | steps to the
  next power-granting cult space on each track (4) | own dwellings next
  to an opponent structure, leech power offered to this seat, leech
  power this seat offers the others (3).

Economy semantics:
- income is ``income.faction_income`` summed over its base, building,
  bonus-tile and favor buckets. Round-scoring cult income is not in it
  (the engine pays that separately at round end).
- an open cluster is a plain (direct adjacency + bridge) cluster that
  overlaps no founded town, as ``town_shaping.town_progress`` reads it;
  Mermaids' river-skip clusters are not included. "Best" is the cluster
  closest to a town by town_progress's measure, ties broken on power then
  structures. These stay populated when the town pool is empty; the pool
  counts say whether a town can still be founded.
- spade distance is the colour-wheel distance from the hex's terrain to
  the faction's home terrain for every faction, including Giants (who
  pay 2 spades for any transform) and Fakirs/Dwarves (whose reach counts
  carpet/tunnel range via ``connectivity.reachable``).

Scoring semantics (v3):
- the first four entries are ``scoring.projected_vp``'s parts if the
  game ended now: ``network_size`` (shipping, tunnelling, carpet and the
  Mermaids river skip count), its 18/12/6 award against every seat's
  current size, the 8/4/2 cult awards with the engine's tie split, and
  ``resource_vp``. ``projected_vp`` = VP (v1 seat block) + these three
  awards.
- cult rank counts the other seats at the same or a higher position, so
  0 is alone in first and a tie ranks every tied seat below the top
  (the engine splits tied awards and breaks no ties).
- cult steps is the smallest number of steps ``cults.advance`` would
  turn into power from here: 3/5/7 always, 10 only while the 10 space is
  unclaimed and the seat holds an unspent key. 0 when no step can pay.
- contact follows leech's geometry (``connectivity.directly_adjacent``,
  bridges included; structure power D 1, TP/TE 2, SH/SA 3; an opponent
  structure is any building of another colour). A dwelling next to one
  upgrades to a trading post at the undoubled price. Offered-to-seat sums,
  over every opponent structure, this seat's adjacent structure power --
  what it would be offered if each of them were upgraded once.
  Offered-by-seat is the mirror over this seat's structures. Neither is
  capped by bowls or gated on the round, unlike a live leech offer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import is_

import numpy as np

from bgai.engine.tm.connectivity import directly_adjacent, effective_shipping, reachable
from bgai.engine.tm.cults import advance
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.memo import state_memo
from bgai.engine.tm.scoring import compute_cult_scoring, network_awards, network_size, resource_vp
from bgai.engine.tm.state import GameState, HexState, Phase
from bgai.engine.tm.terraform import spade_distance
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES, TOWN_TILES
from bgai.engine.tm.towns import _town_threshold, building_power_value, open_clusters
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
_TOWN_IDS: tuple[str, ...] = tuple(sorted(TOWN_TILES))
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
GLOBAL_DIM_V1 = _CTX_PENDING + 1
"""Width of the v1 globals vector. Later encodings only append, so a net
trained at this width reads the leading ``GLOBAL_DIM_V1`` entries of any
later encoding (``ModelConfig.global_dim``)."""

_INCOME_RESOURCES = ("C", "W", "P", "PW")
_TOWN_STRUCTURES = 4
_MAX_SPADES = 3
_ECON_INCOME = 0
_ECON_TOWN = _ECON_INCOME + len(_INCOME_RESOURCES)
_ECON_SUPPLY = _ECON_TOWN + 4
_ECON_TARGETS = _ECON_SUPPLY + len(BUILDINGS)
ECON_SEAT_BLOCK = _ECON_TARGETS + _MAX_SPADES + 1  # 17

_ECON_BASE = GLOBAL_DIM_V1
_CTX_TOWN_POOL = _ECON_BASE + 4 * ECON_SEAT_BLOCK
_CTX_FAVOR_POOL = _CTX_TOWN_POOL + len(_TOWN_IDS)
GLOBAL_DIM_V2 = _CTX_FAVOR_POOL + len(_FAV_IDS)
"""Width of the v2 globals vector (v1 plus the economy block)."""

_SCORE_PROJECTION = 0  # network size, network VP, cult VP, resource VP
_SCORE_CULT_RANK = _SCORE_PROJECTION + 4
_SCORE_CULT_STEPS = _SCORE_CULT_RANK + len(CULTS4)
_SCORE_CONTACT = _SCORE_CULT_STEPS + len(CULTS4)
SCORE_SEAT_BLOCK = _SCORE_CONTACT + 3  # 15

_SCORE_BASE = GLOBAL_DIM_V2
GLOBAL_DIM = _SCORE_BASE + 4 * SCORE_SEAT_BLOCK

_INT16 = np.iinfo(np.int16)


@dataclass(frozen=True)
class StateEncoding:
    hex_planes: np.ndarray
    globals: np.ndarray


def _seat_order(state: GameState, faction: str) -> tuple[str, ...]:
    seats = state.setup.factions
    pivot = seats.index(faction)
    return seats[pivot:] + seats[:pivot]


@state_memo(maxsize=2048)
def _hex_planes(hexes: Mapping[str, HexState], faction: str, seats: tuple[str, ...]) -> np.ndarray:
    """Hex rows, memoised on the ``state.hexes`` mapping object.

    The engine rebuilds that mapping only when a hex changes (a build,
    upgrade or transform); every other move -- converts, passes, power
    actions, leech answers, cult sends -- hands the child state the very
    same mapping object, so a search tree shares it widely. The rows
    depend on nothing else but the mover-relative seat order, which is a
    function of ``(seats, faction)``.
    """
    pivot = seats.index(faction)
    seat_of = {name: i for i, name in enumerate(seats[pivot:] + seats[:pivot])}
    planes = np.zeros((len(HEXES), HEX_FEAT_DIM), dtype=np.int8)
    for row, hex_key in enumerate(HEXES):
        hx = hexes[hex_key]
        if hx.color in COLOR_INDEX:
            planes[row, COLOR_INDEX[hx.color]] = 1
        else:  # river
            planes[row, len(COLORS)] = 1
        if hx.building is not None:
            planes[row, len(COLORS) + 1 + BUILDING_INDEX[hx.building]] = 1
        if hx.owner is not None and hx.owner in seat_of:
            planes[row, len(COLORS) + 1 + len(BUILDINGS) + seat_of[hx.owner]] = 1
    planes.setflags(write=False)
    return planes


def _income(state: GameState, name: str) -> list[int]:
    buckets = faction_income(state.factions[name], FACTIONS[name]).values()
    return [sum(b.get(r, 0) for b in buckets) for r in _INCOME_RESOURCES]


def _town_progress(state: GameState, name: str) -> list[int]:
    threshold = _town_threshold(state.factions[name])
    found = open_clusters(state, name)

    def closeness(cluster: tuple[int, int]) -> tuple[float, int, int]:
        power, count = cluster
        return (min(1.0, power / threshold) * min(1.0, count / _TOWN_STRUCTURES), power, count)

    power, count = max(found, key=closeness, default=(0, 0))
    ready = sum(1 for p, _ in found if p >= threshold)
    return [power, count, threshold, ready]


def _supply(state: GameState, name: str) -> list[int]:
    built = state.factions[name].buildings
    tracks = FACTIONS[name].buildings
    return [tracks[b].max_count - len(built.get(b, ())) for b in BUILDINGS]


def _expansion_targets(state: GameState, name: str) -> list[int]:
    home = FACTIONS[name].color
    buckets = [0] * (_MAX_SPADES + 1)
    for hex_key in reachable(state, name):
        hx = state.hexes[hex_key]
        if hx.building is None and hx.color in COLOR_INDEX:
            buckets[spade_distance(hx.color, home)] += 1
    return buckets


def _contact(state: GameState) -> dict[str, list[int]]:
    """Per faction: [own dwellings next to an opponent structure, leech
    power offered to it, leech power it offers] (module docstring)."""
    out: dict[str, list[int]] = {}
    for name, fs in state.factions.items():
        color = FACTIONS[name].color
        touching = offered_to = offered_by = 0
        for building, hexes in fs.buildings.items():
            power = building_power_value(building)
            for hex_key in hexes:
                rivals = [
                    building_power_value(hx.building)
                    for hx in (state.hexes[n] for n in directly_adjacent(state, hex_key))
                    if hx.building is not None and hx.color != color
                ]
                touching += building == "D" and bool(rivals)
                offered_to += power * len(rivals)
                offered_by += sum(rivals)
        out[name] = [touching, offered_to, offered_by]
    return out


def _next_power_step(position: int, keys: int, track_open: bool) -> int:
    for steps in range(1, 11 - position):
        if advance(position, steps, keys, track_open).power_gained:
            return steps
    return 0


def _cult_standing(state: GameState) -> dict[str, list[int]]:
    """Per faction: [cult-majority VP now, rank per track (4), steps to the
    next power-granting space per track (4)] (module docstring)."""
    awards = compute_cult_scoring(state)
    out: dict[str, list[int]] = {}
    for name, fs in state.factions.items():
        mine = state.cults[name]
        vp = sum(awards[cult].get(name, 0) for cult in CULTS4)
        ranks = [
            sum(state.cults[other][cult] >= mine[cult] for other in state.factions if other != name)
            for cult in CULTS4
        ]
        steps = [
            _next_power_step(mine[cult], fs.keys, state.cult_10[cult] is None) for cult in CULTS4
        ]
        out[name] = [vp, *ranks, *steps]
    return out


class _IdentityCache[V]:
    """Values keyed on the identity of immutable engine objects plus
    hashable scalars. Engine objects are never mutated after construction
    (see ``engine.tm.memo``) and each entry pins the objects it keys on, so
    an id cannot be recycled while its entry lives. FIFO eviction."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._table: dict[tuple, tuple[tuple[object, ...], V]] = {}

    def get(self, objs: tuple[object, ...], scalars: tuple, compute: Callable[[], V]) -> V:
        key = (*map(id, objs), *scalars)
        hit = self._table.get(key)
        if hit is not None and all(map(is_, hit[0], objs)):
            return hit[1]
        value = compute()
        if len(self._table) >= self.maxsize:
            del self._table[next(iter(self._table))]
        self._table[key] = (objs, value)
        return value

    def clear(self) -> None:
        self._table.clear()


# ``reachable``/``clusters`` memoise per GameState, which is new at every
# search leaf. The parts below are keyed on exactly what they read, which a
# leaf shares with its parent for every seat (and every part) its move did
# not touch: a convert or a pass leaves the mover's buildings, and so its
# holdings, untouched; only a build changes ``hexes`` and the targets.
_holdings_cache: _IdentityCache[list[int]] = _IdentityCache(8192)
_targets_cache: _IdentityCache[list[int]] = _IdentityCache(8192)
_network_cache: _IdentityCache[int] = _IdentityCache(8192)
_resource_cache: _IdentityCache[int] = _IdentityCache(8192)
_cult_cache: _IdentityCache[dict[str, list[int]]] = _IdentityCache(2048)
_contact_cache: _IdentityCache[dict[str, list[int]]] = _IdentityCache(2048)
_CACHES = (
    _holdings_cache, _targets_cache, _network_cache, _resource_cache, _cult_cache, _contact_cache
)


def _seat_economy(state: GameState, name: str) -> list[int]:
    fs = state.factions[name]
    founded = state.founded_towns.get(name, ())
    holdings = _holdings_cache.get(
        (fs.buildings, fs.favors, state.bridges, founded),
        (name, fs.bonus),
        lambda: _income(state, name) + _town_progress(state, name) + _supply(state, name),
    )
    targets = _targets_cache.get(
        (fs.buildings, state.bridges, state.hexes),
        (name, fs.bonus, fs.shipping, fs.passed, fs.teleport_level),
        lambda: _expansion_targets(state, name),
    )
    return holdings + targets


def _seat_network_size(state: GameState, name: str) -> int:
    fs = state.factions[name]
    return _network_cache.get(
        (fs.buildings, state.bridges),
        (name, effective_shipping(state, name), fs.teleport_level),
        lambda: network_size(state, name),
    )


@functools.lru_cache(maxsize=4096)
def _network_awards(sizes: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return network_awards(dict(sizes))


def _seat_resource_vp(state: GameState, name: str) -> int:
    # keyed on what the conversion reads, not on ``fs``, which every move
    # of this seat replaces
    fs = state.factions[name]
    return _resource_cache.get(
        (fs.power,), (name, fs.coins, fs.workers, fs.priests), lambda: resource_vp(fs, name)
    )


def clear_caches() -> None:
    """Drop the economy and scoring memo tables (benchmarks and tests)."""
    for cache in _CACHES:
        cache.clear()
    _network_awards.cache_clear()


def _economy(state: GameState, order: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for name in order:
        values += _seat_economy(state, name)
    values += [state.towns_pool.get(t, 0) for t in _TOWN_IDS]
    values += [state.favors_pool.get(t, 0) for t in _FAV_IDS]
    return values


def _scoring(state: GameState, order: tuple[str, ...]) -> list[int]:
    sizes = {name: _seat_network_size(state, name) for name in state.factions}
    network_vp = _network_awards(tuple(sizes.items()))
    keys = tuple(fs.keys for fs in state.factions.values())
    cults = _cult_cache.get((state.cults, state.cult_10), keys, lambda: _cult_standing(state))
    contact = _contact_cache.get((state.hexes, state.bridges), (), lambda: _contact(state))
    values: list[int] = []
    for name in order:
        cult_vp, *standing = cults[name]
        values += [sizes[name], network_vp.get(name, 0), cult_vp, _seat_resource_vp(state, name)]
        values += standing
        values += contact[name]
    return values


def encode_state(state: GameState, faction: str) -> StateEncoding:
    order = _seat_order(state, faction)
    # Read-only shared array: consumers copy (astype/np.stack) before use.
    planes = _hex_planes(state.hexes, faction, state.setup.factions)

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
    derived = _economy(state, order) + _scoring(state, order)
    g[_ECON_BASE:] = np.clip(derived, _INT16.min, _INT16.max)

    return StateEncoding(hex_planes=planes, globals=g)
