"""State -> numpy encoding, v4 (plan Task 3; bump ``ENCODING_VERSION`` on
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
  power this seat offers the others (3)
  then each round's score tile id one-hot (6 x 9, SCORE1..SCORE9): the
  v1 cult/req pair cannot tell SCORE3 (D >> 2, 4 WATER -> P) from
  SCORE6 (TP >> 3, 4 WATER -> spade).
  -- that is the whole v3 vector, ``GLOBAL_DIM_V3`` = 496 entries. v4
  appends the plan block, what this round and the next pay for and how
  far each seat is from paying for it:
  4 plan seat blocks of PLAN_SEAT_BLOCK (18) each, same seat order:
  workers and priests per spade, spades affordable from workers/priests,
  spades from one power action (4) | best open cluster's power and
  structures still missing for a town (2) | empty land hexes adjacent to
  that cluster at 0, 1, 2, 3 spades (4) | its D->TP, TP->TE, TP->SH,
  TE->SA upgrades affordable now (4) | empty land hexes adjacent to any
  own structure at 0, 1, 2, 3 spades (4)
  then the current and the next round's score tile, PLAN_TILE_DIM (17)
  each: VP per D, TP, TE, SH/SA, spade, town (6) | its end-of-round cult
  one-hot FIRE, WATER, EARTH, AIR, CULT_P (5) | cult steps per reward
  (1) | reward per step-multiple in C, W, P, PW, SPADE (5)
  then rounds left after this one (1).

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

Plan semantics (v4):
- per-spade cost is the faction's dig cost at its dig level
  (``FactionData.dig``; Darklings pay a priest). Spades from resources
  is how many digs that cost covers now. Spades from power is the most
  one open spade power action (ACT5, ACT6) yields with bowl 3 plus
  everything bowl 2 can burn; it is not gated on the seat having passed.
- the cluster is the one v2's town progress picks. Missing power and
  structures are 0 once it is big enough. Adjacent hexes use
  ``connectivity.directly_adjacent`` (bridges included), which is how a
  new building joins a cluster. An upgrade counts when the seat has the
  building in supply and holds its cost now; a trading post costs double
  coins with no other colour adjacent (``leech.has_leechable_neighbor``).
  Each upgrade is priced alone, not jointly with the others.
- spade buckets here are what the seat pays to make the hex home terrain
  (``spade_transform_cost``: Giants pay 2 for any change), unlike v2's
  colour-wheel buckets.
- score tile VP is ``tiles.scored_vp`` for the event's mode (build for
  structures, gain for spades and towns); a tile pays for one of them.
  Round 0 (setup) has no current tile; round 6 has no next tile. Both
  read as zeros.
- ACT1-ACT6 availability is not encoded again: it is the complement of
  v1's taken flags, and their power costs never change.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import is_

import numpy as np

from bgai.engine.tm import actions_terraform as _registers_giants_hooks  # noqa: F401
from bgai.engine.tm.connectivity import directly_adjacent, effective_shipping, reachable
from bgai.engine.tm.cults import advance
from bgai.engine.tm.factions.hooks import hooks_for
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.leech import has_leechable_neighbor
from bgai.engine.tm.legal_shared import affordable, resource_amount
from bgai.engine.tm.memo import state_memo
from bgai.engine.tm.scoring import compute_cult_scoring, network_awards, network_size, resource_vp
from bgai.engine.tm.state import GameState, HexState, Phase
from bgai.engine.tm.terraform import spade_distance
from bgai.engine.tm.tiles import (
    BONUS_TILES,
    FAVOR_TILES,
    POWER_ACTIONS,
    SCORE_TILES,
    TOWN_TILES,
    ScoringTile,
    scored_vp,
)
from bgai.engine.tm.towns import (
    _town_threshold,
    building_power_value,
    open_clusters,
    open_town_clusters,
)
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
_CTX_SCORE_IDS = _SCORE_BASE + 4 * SCORE_SEAT_BLOCK
GLOBAL_DIM_V3 = _CTX_SCORE_IDS + 6 * len(SCORE_TILES)
"""Width of the v3 globals vector (v2 plus the scoring block)."""

_UPGRADES: tuple[tuple[str, str], ...] = (("D", "TP"), ("TP", "TE"), ("TP", "SH"), ("TE", "SA"))
_PLAN_SPADES = 0  # W per spade, P per spade, spades from resources, spades from power
_PLAN_TOWN_NEEDS = _PLAN_SPADES + 4
_PLAN_TOWN_EDGE = _PLAN_TOWN_NEEDS + 2
_PLAN_UPGRADES = _PLAN_TOWN_EDGE + _MAX_SPADES + 1
_PLAN_TARGETS = _PLAN_UPGRADES + len(_UPGRADES)
PLAN_SEAT_BLOCK = _PLAN_TARGETS + _MAX_SPADES + 1  # 18

# (event types, scoring mode): the tile pays the best of the types
_TILE_EVENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("D",), "build"),
    (("TP",), "build"),
    (("TE",), "build"),
    (("SH", "SA"), "build"),
    (("SPADE",), "gain"),
    (_TOWN_IDS, "gain"),
)
_TILE_CULTS = (*CULTS4, "CULT_P")
_TILE_REWARDS = ("C", "W", "P", "PW", "SPADE")
PLAN_TILE_DIM = len(_TILE_EVENTS) + len(_TILE_CULTS) + 1 + len(_TILE_REWARDS)  # 17

_PLAN_BASE = GLOBAL_DIM_V3
_CTX_PLAN_TILES = _PLAN_BASE + 4 * PLAN_SEAT_BLOCK
_CTX_ROUNDS_LEFT = _CTX_PLAN_TILES + 2 * PLAN_TILE_DIM
GLOBAL_DIM = _CTX_ROUNDS_LEFT + 1

_SPADE_ACTIONS: tuple[tuple[str, int, int], ...] = tuple(
    (name, act.cost_power, act.gain["SPADE"])
    for name, act in POWER_ACTIONS.items()
    if "SPADE" in act.gain
)


def _score_tile_key(tile: ScoringTile) -> tuple:
    # the rules a tile carries; display strings vary with where it was parsed
    return (tile.cult, tile.req, tile.vp_mode, tile.vp, tile.cult_income)


_SCORE_TILE_INDEX = {
    _score_tile_key(SCORE_TILES[name]): i for i, name in enumerate(sorted(SCORE_TILES))
}

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


def _closeness(cluster: tuple[int, int], threshold: int) -> tuple[float, int, int]:
    power, count = cluster
    return (min(1.0, power / threshold) * min(1.0, count / _TOWN_STRUCTURES), power, count)


def _town_progress(state: GameState, name: str) -> list[int]:
    threshold = _town_threshold(state.factions[name])
    found = open_clusters(state, name)
    power, count = max(found, key=lambda c: _closeness(c, threshold), default=(0, 0))
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
                if building == "D" and rivals:
                    touching += 1
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


def _spade_economy(state: GameState, name: str) -> list[int]:
    fs = state.factions[name]
    dig = FACTIONS[name].dig.cost[fs.dig_level]
    paid = [(res, amount) for res, amount in dig.items() if amount > 0]
    from_resources = min((resource_amount(fs, res) // amount for res, amount in paid), default=0)
    power = fs.power.bowl3 + fs.power.bowl2 // 2
    from_power = max(
        (
            spades
            for act, cost, spades in _SPADE_ACTIONS
            if act not in state.power_actions_taken and power >= cost
        ),
        default=0,
    )
    return [dig.get("W", 0), dig.get("P", 0), from_resources, from_power]


def _land_by_spades(state: GameState, name: str, hexes: frozenset[str]) -> list[int]:
    home = FACTIONS[name].color
    cost = hooks_for(name).spade_transform_cost
    buckets = [0] * (_MAX_SPADES + 1)
    for hex_key in hexes:
        hx = state.hexes[hex_key]
        if hx.building is None and hx.color in COLOR_INDEX:
            buckets[cost(state, name, hx.color, home)] += 1
    return buckets


def _neighbours(state: GameState, hexes: frozenset[str]) -> frozenset[str]:
    return frozenset().union(*(directly_adjacent(state, h) for h in hexes)) - hexes


def _upgrade_cost(state: GameState, name: str, new: str, hex_key: str) -> dict[str, int]:
    cost = dict(FACTIONS[name].buildings[new].cost)
    if new == "TP" and not has_leechable_neighbor(state, name, hex_key):
        cost["C"] = cost.get("C", 0) * 2
    return cost


@dataclass(frozen=True)
class _PlanShape:
    """What a seat's own buildings, favors and the bridges fix: the town
    shortfall of the best open cluster, the hexes next to it and next to
    any own structure, and the cluster's upgrades in supply as
    (``_UPGRADES`` index, new building, hex)."""

    needs: list[int]
    cluster_edge: frozenset[str]
    own_edge: frozenset[str]
    upgrades: tuple[tuple[int, str, str], ...]


def _plan_shape(state: GameState, name: str) -> _PlanShape:
    fs = state.factions[name]
    threshold = _town_threshold(fs)
    found = list(zip(open_town_clusters(state, name), open_clusters(state, name), strict=True))
    cluster, (power, count) = max(
        found, key=lambda c: _closeness(c[1], threshold), default=(frozenset(), (0, 0))
    )
    tracks = FACTIONS[name].buildings
    return _PlanShape(
        needs=[max(0, threshold - power), max(0, _TOWN_STRUCTURES - count)],
        cluster_edge=_neighbours(state, cluster),
        own_edge=_neighbours(state, frozenset().union(*fs.buildings.values())),
        upgrades=tuple(
            (i, new, hex_key)
            for i, (old, new) in enumerate(_UPGRADES)
            if len(fs.buildings.get(new, ())) < tracks[new].max_count
            for hex_key in cluster
            if state.hexes[hex_key].building == old
        ),
    )


@dataclass(frozen=True)
class _PlanGeometry:
    """``_PlanShape`` read against the board: the town shortfall and the
    cluster-edge land buckets, upgrade costs, own-edge land buckets."""

    town: list[int]
    upgrades: tuple[tuple[int, dict[str, int]], ...]
    targets: list[int]


def _plan_geometry(state: GameState, name: str, shape: _PlanShape) -> _PlanGeometry:
    return _PlanGeometry(
        town=shape.needs + _land_by_spades(state, name, shape.cluster_edge),
        upgrades=tuple(
            (i, _upgrade_cost(state, name, new, hex_key)) for i, new, hex_key in shape.upgrades
        ),
        targets=_land_by_spades(state, name, shape.own_edge),
    )


def _affordable_upgrades(state: GameState, name: str, geometry: _PlanGeometry) -> list[int]:
    fs = state.factions[name]
    counts = [0] * len(_UPGRADES)
    for i, cost in geometry.upgrades:
        counts[i] += affordable(fs, cost)
    return counts


def _plan_tile(tile: ScoringTile | None) -> list[int]:
    if tile is None:
        return [0] * PLAN_TILE_DIM
    vp = [max(scored_vp(tile, t, mode) for t in types) for types, mode in _TILE_EVENTS]
    reward = dict(tile.cult_income)
    return (
        vp
        + [int(tile.cult == cult) for cult in _TILE_CULTS]
        + [tile.req]
        + [reward.get(r, 0) for r in _TILE_REWARDS]
    )


def _plan_context(tiles: tuple[ScoringTile, ...], round_: int) -> list[int]:
    current = tiles[round_ - 1] if round_ >= 1 else None
    upcoming = tiles[round_] if round_ < len(tiles) else None
    return _plan_tile(current) + _plan_tile(upcoming) + [len(tiles) - round_]


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
_seat_plan_cache: _IdentityCache[tuple[int, ...]] = _IdentityCache(8192)
_spade_cache: _IdentityCache[list[int]] = _IdentityCache(8192)
_shape_cache: _IdentityCache[_PlanShape] = _IdentityCache(8192)
_geometry_cache: _IdentityCache[_PlanGeometry] = _IdentityCache(8192)
_upgrade_cache: _IdentityCache[list[int]] = _IdentityCache(8192)
_plan_context_cache: _IdentityCache[list[int]] = _IdentityCache(64)
_CACHES = (
    _holdings_cache,
    _targets_cache,
    _network_cache,
    _resource_cache,
    _cult_cache,
    _contact_cache,
    _seat_plan_cache,
    _spade_cache,
    _shape_cache,
    _geometry_cache,
    _upgrade_cache,
    _plan_context_cache,
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


def _seat_plan_parts(state: GameState, name: str) -> tuple[int, ...]:
    fs = state.factions[name]
    spades = _spade_cache.get(
        (fs.power, state.power_actions_taken),
        (name, fs.dig_level, fs.workers, fs.priests),
        lambda: _spade_economy(state, name),
    )
    # an opponent's build changes ``hexes`` but not this seat's shape
    shape = _shape_cache.get(
        (fs.buildings, fs.favors, state.bridges, state.founded_towns.get(name, ())),
        (name,),
        lambda: _plan_shape(state, name),
    )
    geometry = _geometry_cache.get(
        (shape, state.hexes), (name,), lambda: _plan_geometry(state, name, shape)
    )
    upgrades = _upgrade_cache.get(
        (geometry,),
        (fs.coins, fs.workers, fs.priests),
        lambda: _affordable_upgrades(state, name, geometry),
    )
    return (*spades, *geometry.town, *upgrades, *geometry.targets)


def _seat_plan(state: GameState, name: str) -> tuple[int, ...]:
    # a leaf shares every seat but the mover's FactionState with its parent
    fs = state.factions[name]
    return _seat_plan_cache.get(
        (fs, state.hexes, state.bridges, state.power_actions_taken, state.founded_towns),
        (name,),
        lambda: _seat_plan_parts(state, name),
    )


def _plan(state: GameState, order: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for name in order:
        values += _seat_plan(state, name)
    tiles = state.setup.score_tiles
    values += _plan_context_cache.get(
        (tiles,), (state.round,), lambda: _plan_context(tiles, state.round)
    )
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
    g[_ECON_BASE:_CTX_SCORE_IDS] = np.clip(derived, _INT16.min, _INT16.max)
    for r, tile in enumerate(state.setup.score_tiles):
        g[_CTX_SCORE_IDS + r * len(SCORE_TILES) + _SCORE_TILE_INDEX[_score_tile_key(tile)]] = 1
    g[_PLAN_BASE:] = _plan(state, order)

    return StateEncoding(hex_planes=planes, globals=g)
