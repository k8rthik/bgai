"""Town formation detection and town-tile rewards.

Ported from the reference implementation (jsnell/terra-mystica, MIT),
``src/towns.pm`` (whole file, 154 lines) plus the building-strength table
from ``src/Game/Constants.pm`` (lines 25-31):

- ``%building_strength`` (Constants.pm lines 25-31): D=1, TP=2, TE=2, SH=3,
  SA=3. (No base faction overrides this per-faction -- that hook,
  ``$faction->{building_strength}{$type} // $building_strength{$type}``, is
  only ever populated by the Fire & Ice expansion's Yetis, out of scope
  here -- so ``building_power_value`` is faction-independent.)
- ``detect_towns_from`` (towns.pm lines 24-87): given a hex, walks the
  faction's connected building component from it (``adjacent_own_buildings``,
  direct adjacency + bridges), summing power and a building *count* where a
  Sanctuary (SA) counts twice (``$count++ if $map{$loc}{building} eq 'SA'``,
  line 67) while still only contributing its single 3-power building
  strength. A component founds a town when ``power >= TOWN_SIZE`` (default
  7, reduced to 6 while FAV5 is held -- FAV5's ``TOWN_SIZE => -1`` passive,
  see ``tiles.py`` ``FAVOR_TILES["FAV5"]``) and ``count >= 4``, *and* the
  town-tile pool has at least one tile left (line 77-79: ``$town_tile_count
  = grep {/^TW/ and $game{pool}{$_} > 0} keys %{$game{pool}}``). A hex
  already marked ``$map{$where}{town}`` is skipped entirely (line 28) --
  this module ports that as "any current cluster that shares a hex with an
  already-founded town's hex-set is not a new town" (buildings only ever get
  added in Terra Mystica, never removed, so a founded town's hex-set can
  only ever be a subset of its component going forward; that growing
  component is the *same* town, not a new one).
- ``check_mermaid_river_connection_town``/``update_mermaid_town_connections``
  (towns.pm lines 89-152) is the Mermaids-only "connect across one river
  hex" mechanic: for a single river hex, the faction's building hexes
  directly touching it are unioned into one component (via
  ``adjacent_own_buildings``, which for Mermaids also consults a ``{skip}``
  table populated by a *previous* successful connect -- not relevant when
  scanning for hexes that could newly qualify), and the same power/count/
  pool-availability test applies. **Decision on Task 5's ``clusters(...,
  river_skip=True)``**: that flag joins *any two* building hexes sharing
  *any* common river neighbor, unioned transitively across *every* river
  hex on the board simultaneously -- Task 5's own docstring calls this an
  intentional over-approximation of the real one-river-hex-at-a-time
  mechanic. Reusing it here would let this module report a component as a
  qualifying town when reaching that size actually requires two *separate*
  river-connect actions (hex A joins hex B via river R1, hex B joins hex C
  via river R2, but no single river hex touches all of A, B, and C) --
  unsound for "does founding a town via one river hex qualify right now".
  So this module does **not** consume ``river_skip``; it re-implements the
  one-river-hex BFS directly (``_mermaid_river_cluster``), matching
  towns.pm lines 99-125 exactly: seed with the faction's own building hexes
  directly adjacent to a single river hex, then expand only through plain
  building-to-building adjacency (``connectivity.directly_adjacent``, which
  already folds in bridges) -- never through a second river hop.
- ``resources.pm`` ``adjust_resource`` (lines 250-395), the ``$type =~
  /^TW/`` branch (lines 355-362) and the generic end-of-function
  ``maybe_gain_faction_special $faction, $type, 'gain'`` call (line 391):
  taking a town tile grants ``%tiles{$type}{gain}`` (here split into
  ``TownTile.vp`` + ``TownTile.gain`` by ``tiles.py``) and, if the faction
  has a ``special`` entry for that specific tile id, that gain too --
  Witches ``{VP => 5}`` and Swarmlings ``{W => 3}`` for every TW1-8
  (``factions_data.py`` collapses the per-tile-id Perl hash to one
  ``special_gain["TOWN"]`` key since the value is identical for all eight).

``apply_town_tile`` applies the mechanically simple gain keys directly:
VP, KEY, C, W, P, PW (the last via ``Power.gain``, ports ``gain_power``
in resources.pm). ``TownTile.gain`` also carries keys this module
deliberately does **not** apply, per the task brief ("do NOT import cult
logic here; just expose the gain"):

- FIRE/WATER/EARTH/AIR (TW5/TW6): cult-track steps. Applying these needs
  ``cults.advance`` (crossing 3/5/7/10 grants power/keys and can block a
  track), which is orchestration this module has no business doing --
  the caller must read ``TOWN_TILES[tile].gain`` for these keys and drive
  ``cults.advance`` itself.
- GAIN_SHIP/carpet_range (TW7): shipping/carpet-flight track advances
  (``resources.pm`` ``GAIN_(TELEPORT|SHIP)`` branch, lines 277-286) --
  track-level state this module does not touch either.

These keys are left in the faction's resources unapplied by design (not
silently dropped: they are still visible via ``TOWN_TILES[tile].gain`` for
a later task to finish).
"""

from __future__ import annotations

from dataclasses import replace

from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import clusters, directly_adjacent
from bgai.engine.tm.factions_data import FACTIONS, TOWN_SIZE
from bgai.engine.tm.state import FactionState, GameState, with_faction
from bgai.engine.tm.tiles import FAVOR_TILES, TOWN_TILES, scored_vp

# Constants.pm %building_strength, lines 25-31.
BUILDING_POWER_VALUES: dict[str, int] = {"D": 1, "TP": 2, "TE": 2, "SH": 3, "SA": 3}

# Deferred gain keys (see module docstring): applied by a later task, not here.
_DEFERRED_GAIN_KEYS: frozenset[str] = frozenset(
    {"FIRE", "WATER", "EARTH", "AIR", "GAIN_SHIP", "carpet_range"}
)

# FAV5's passive (tiles.py FAVOR_TILES["FAV5"].passive), read rather than
# hardcoded per the task brief.
_TOWN_SIZE_DELTA: int = FAVOR_TILES["FAV5"].passive.get("TOWN_SIZE", 0)


def building_power_value(building: str) -> int:
    """towns.pm/Constants.pm ``%building_strength``: D=1, TP=2, TE=2, SH=3, SA=3."""
    return BUILDING_POWER_VALUES[building]


def _faction_buildings(fs: FactionState) -> frozenset[str]:
    return frozenset().union(*fs.buildings.values())


def _cluster_power_and_count(state: GameState, cluster: frozenset[str]) -> tuple[int, int]:
    """towns.pm lines 61-68: sum power, count buildings (SA counts twice)."""
    power = 0
    count = 0
    for hex_key in cluster:
        building = state.hexes[hex_key].building
        if building is None:
            raise ValueError(f"cluster hex {hex_key!r} has no building")
        power += building_power_value(building)
        count += 1
        if building == "SA":
            count += 1
    return power, count


def _town_threshold(fs: FactionState) -> int:
    """towns.pm ``$faction->{TOWN_SIZE}``: base 7, -1 per held FAV5 (stacks)."""
    return TOWN_SIZE + _TOWN_SIZE_DELTA * fs.favors.count("FAV5")


def _town_tiles_available(state: GameState) -> bool:
    """towns.pm lines 77-79/127-128: at least one TW tile left in the pool."""
    return any(count > 0 for count in state.towns_pool.values())


def _qualifies(state: GameState, fs: FactionState, cluster: frozenset[str]) -> bool:
    power, count = _cluster_power_and_count(state, cluster)
    return power >= _town_threshold(fs) and count >= 4


def _overlaps_founded(cluster: frozenset[str], founded: tuple[frozenset[str], ...]) -> bool:
    """A cluster sharing any hex with an already-founded town is not new."""
    return any(cluster & prior for prior in founded)


def _plain_candidates(state: GameState, faction: str) -> tuple[frozenset[str], ...]:
    """towns.pm ``detect_towns_from``: plain direct-adjacency (+bridge) clusters."""
    if not _town_tiles_available(state):
        return ()
    fs = state.factions[faction]
    founded = state.founded_towns.get(faction, ())
    return tuple(
        c
        for c in clusters(state, faction)
        if _qualifies(state, fs, c) and not _overlaps_founded(c, founded)
    )


def _mermaid_river_cluster(
    state: GameState, faction: str, building_hexes: frozenset[str], river: str
) -> frozenset[str] | None:
    """towns.pm ``check_mermaid_river_connection_town`` lines 99-125.

    Seeds with the faction's building hexes directly touching ``river``
    (plain board adjacency -- rivers don't carry bridges), then expands
    through ordinary building-to-building adjacency only (no second river
    hop). Returns ``None`` if no building touches ``river`` at all.
    """
    board = base_board()
    seed = {h for h in board.adjacent.get(river, frozenset()) if h in building_hexes}
    if not seed:
        return None
    seen = set(seed)
    stack = list(seed)
    while stack:
        node = stack.pop()
        for neighbor in directly_adjacent(state, node) & building_hexes:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return frozenset(seen)


def _mermaid_candidates(state: GameState, faction: str) -> tuple[frozenset[str], ...]:
    """towns.pm ``update_mermaid_town_connections`` lines 135-152, restricted
    to river hexes that would newly qualify (mirrors ``detect_towns_from``'s
    role for the plain case)."""
    if faction != "mermaids" or not _town_tiles_available(state):
        return ()
    fs = state.factions[faction]
    building_hexes = _faction_buildings(fs)
    if not building_hexes:
        return ()
    founded = state.founded_towns.get(faction, ())
    board = base_board()

    results: list[frozenset[str]] = []
    seen_clusters: set[frozenset[str]] = set()
    for river, hex_state in board.hexes.items():
        if hex_state.color != RIVER:
            continue
        # towns.pm lines 92-97: bail if a building already adjacent to this
        # river hex is already part of a founded town.
        seed = {h for h in board.adjacent.get(river, frozenset()) if h in building_hexes}
        if any(_overlaps_founded(frozenset({h}), founded) for h in seed):
            continue
        cluster = _mermaid_river_cluster(state, faction, building_hexes, river)
        if cluster is None or cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        if _qualifies(state, fs, cluster) and not _overlaps_founded(cluster, founded):
            results.append(cluster)
    return tuple(results)


def new_towns(state: GameState, faction: str) -> tuple[frozenset[str], ...]:
    """Clusters of ``faction``'s buildings that now qualify as a town.

    Combines plain direct-adjacency clusters (``detect_towns_from``,
    towns.pm lines 24-87) with, for Mermaids only, single-river-hex
    connections (``check_mermaid_river_connection_town``, see module
    docstring for why this is re-derived rather than consuming
    ``connectivity.clusters(..., river_skip=True)``). Excludes any cluster
    overlapping a hex already recorded in ``state.founded_towns[faction]``.
    """
    plain = _plain_candidates(state, faction)
    combined = list(plain)
    for cluster in _mermaid_candidates(state, faction):
        if cluster not in combined:
            combined.append(cluster)
    return tuple(combined)


def record_founded_town(state: GameState, faction: str, cluster: frozenset[str]) -> GameState:
    """Record ``cluster`` as a founded town's hex-set for ``faction``.

    Not itself present in towns.pm (which tags hexes with a town id
    in-place, ``$map{$_}{town} = ...``, towns.pm line 81/145); this is the
    immutable equivalent needed to make ``state.founded_towns`` -- and
    therefore ``new_towns``'s already-founded exclusion -- actually work.
    A later task's action-application code is expected to call this once a
    ``new_towns`` candidate is chosen (e.g. resolving a ``"gain_town"``
    pending decision), immediately followed by ``apply_town_tile`` for the
    chosen tile.
    """
    founded = state.founded_towns.get(faction, ()) + (cluster,)
    new_founded = dict(state.founded_towns)
    new_founded[faction] = founded
    return replace(state, founded_towns=new_founded)


def apply_town_tile(state: GameState, faction: str, tile: str) -> GameState:
    """Grant ``tile``'s VP + resource gains (and faction TOWN passives).

    Ports the ``$type =~ /^TW/`` branch of ``resources.pm``
    ``adjust_resource`` (lines 355-362: apply ``%tiles{$type}{gain}``) plus
    the generic ``maybe_gain_faction_special $faction, $type, 'gain'`` call
    (line 391) that grants Witches +5 VP / Swarmlings +3 W per town founded
    (``factions_data.py`` ``special_gain["TOWN"]``), plus that same generic
    loop's sibling call, ``maybe_score_current_score_tile $faction, $type,
    'gain'`` (``tiles.scored_vp`` -- e.g. round 4's ``TOWN >> 5`` tile
    grants +5 VP per town founded that round, keyed by the specific TWx id
    granted). Decrements
    ``state.towns_pool[tile]`` and appends ``tile`` to ``fs.towns``
    (``resources.pm``'s generic pool-decrement, lines 320-326). See the
    module docstring for the gain keys (cult steps, GAIN_SHIP, carpet_range)
    this function deliberately does not apply.
    """
    if tile not in TOWN_TILES:
        raise ValueError(f"unknown town tile {tile!r}")
    if state.towns_pool.get(tile, 0) < 1:
        raise ValueError(f"town tile {tile!r} not available in pool")

    town_tile = TOWN_TILES[tile]
    fs = state.factions[faction]

    vp = town_tile.vp
    keys = 0
    coins = 0
    workers = 0
    priests = 0
    power_gain = 0
    for resource, amount in town_tile.gain.items():
        if resource in _DEFERRED_GAIN_KEYS:
            continue
        if resource == "KEY":
            keys += amount
        elif resource == "C":
            coins += amount
        elif resource == "W":
            workers += amount
        elif resource == "P":
            priests += amount
        elif resource == "PW":
            power_gain += amount
        elif resource == "VP":
            vp += amount
        else:
            raise ValueError(f"unhandled town-tile gain key {resource!r} in {tile}")

    if state.round >= 1:
        current_tile = state.setup.score_tiles[state.round - 1]
        vp += scored_vp(current_tile, tile, "gain")

    passive = FACTIONS[faction].special_gain.get("TOWN", {})
    vp += passive.get("VP", 0)
    coins += passive.get("C", 0)
    workers += passive.get("W", 0)
    priests += passive.get("P", 0)
    power_gain += passive.get("PW", 0)
    keys += passive.get("KEY", 0)

    fs = replace(
        fs,
        vp=fs.vp + vp,
        keys=fs.keys + keys,
        coins=fs.coins + coins,
        workers=fs.workers + workers,
        priests=fs.priests + priests,
        power=fs.power.gain(power_gain) if power_gain else fs.power,
        towns=fs.towns + (tile,),
    )

    new_pool = dict(state.towns_pool)
    new_pool[tile] = new_pool[tile] - 1
    return replace(with_faction(state, faction, fs), towns_pool=new_pool)
