"""Board connectivity: adjacency, build/transform reachability, and clusters.

Ported from the reference implementation (jsnell/terra-mystica, MIT),
``src/map.pm``:

- ``setup_hex_ranges``/``setup_ranges`` (lines 131-162): precomputes, for
  every hex, a distance table to every other hex under two travel modes.
  ``river_only=1`` (used for shipping) only lets a path continue through
  river hexes after the first hop -- the recorded distance equals the
  count of river hexes crossed. ``river_only=0`` (used for tunnel/carpet)
  is an unrestricted BFS over the whole adjacency graph where the
  recorded distance is ``true hex_distance - 1`` (the BFS accumulator
  starts at -1 for the source hex, so immediate neighbors record 0).
- ``check_reachable`` (lines 209-273): a hex is reachable if (a) it is
  directly adjacent to one of the faction's own building hexes
  (``$map{$where}{adjacent}{$loc}``, mutated in place by
  ``commands.pm`` ``command_bridge`` lines 734-735 to fold bridge
  endpoints into plain adjacency -- this is why ``directly_adjacent``
  below folds in ``state.bridges``), or (b) within shipping range via
  the river-only distance table (``range{1}{$loc} <= ship level``,
  lines 238-245), or (c) within tunnel/carpet range via the unrestricted
  distance table (``range{0}{$loc} <= teleport range``, lines 255-270;
  since that table's recorded distance is ``hex_distance - 1``, the
  condition is ``hex_distance <= teleport_range + 1``).
- ``adjacent_own_buildings``/``find_building_cliques`` (lines 277-344):
  clusters ("cliques") of a faction's buildings are the connected
  components of the direct-adjacency graph (bridges included, since they
  were merged into ``{adjacent}``). ``find_building_cliques`` also has
  an ``$allow_indirect`` mode used for network-size scoring (out of
  scope here) and a Mermaids-only ``{skip}`` edge set (see below).
- ``commands.pm`` ``command_connect`` (lines 930-962): when Mermaids
  found a river town, every pair of land hexes adjacent to the
  qualifying river hex is marked mutually reachable via a ``{skip}``
  table, consulted only for the Mermaids faction (``map.pm`` lines
  281-282, 326-328). This module does not track which river hex was
  used to found a town (that is Task 6's state), so ``clusters(...,
  river_skip=True)`` ports the underlying geometric primitive: any two
  of the faction's building hexes that share a common river-hex
  neighbor are treated as joined, which is the maximal set of pairs the
  real ``{skip}`` table could ever produce for a single river hex.

``reachable`` is intentionally pure geometry: it does not filter out
hexes that already carry a building, are the wrong terrain color, or are
otherwise illegal to build/transform on -- that legality check is a
later task's (build-command validation), per ``check_reachable`` itself,
which is only ever called from build-location enumeration
(``update_reachable_build_locations``, lines 666-694) alongside a
separate ``build_color_ok``/occupancy check.

- ``check_reachable`` (lines 214-222): BON4 (``special => { ship => 1 }``,
  ``Constants.pm`` line 130) adds 1 to the acting faction's shipping range
  while ``$faction->{BON4}`` is truthy and ``$faction->{max_level}`` (the
  track exists at all) -- ``and !$faction->{passed}`` ("Bon4 doesn't apply
  in phase III" per the Perl's own comment: a faction that has already
  passed for the round no longer benefits). Grepping ``resources.pm``
  (``adjust_resource``, the ``$type =~ /^BON/`` branch, lines 313-397)
  shows ``$faction->{BON4}`` is not an independent counter with its own
  set/unset lifecycle -- it is the exact same generic
  ``$faction->{$type} += $delta`` mechanism every resource gain uses,
  incremented to 1 when the tile is taken and decremented back to 0 when
  it is given up at a later pass. That is precisely
  ``FactionState.bonus == "BON4"`` in this port -- a live field read fresh
  at query time, not a snapshot -- so the effective-shipping helper below
  (computed on demand from ``fs.bonus``) is exactly faithful to the Perl,
  not merely an approximation of some other "grant on take, remove on
  return" mechanism.
"""

from __future__ import annotations

from collections import deque

from bgai.engine.tm.board import RIVER, Board, base_board, hex_distance
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.state import GameState
from bgai.engine.tm.tiles import BONUS_TILES


def _faction_buildings(state: GameState, faction: str) -> frozenset[str]:
    fs = state.factions[faction]
    return frozenset().union(*fs.buildings.values())


def directly_adjacent(state: GameState, hex_key: str) -> frozenset[str]:
    """Board adjacency plus any bridge endpoints touching ``hex_key``.

    Ports ``map.pm`` ``$map{$where}{adjacent}{$loc}`` after
    ``commands.pm`` ``command_bridge`` (lines 734-735) has mutated the
    map's adjacency table to include built bridges.
    """
    board = base_board()
    neighbors = set(board.adjacent.get(hex_key, frozenset()))
    for bridge in state.bridges:
        if hex_key in bridge:
            neighbors |= bridge - {hex_key}
    return frozenset(neighbors)


def _shipping_reach(board: Board, source: str, level: int) -> frozenset[str]:
    """Land hexes reachable from ``source`` via <= ``level`` river hexes.

    Ports ``map.pm`` ``setup_hex_ranges`` with ``river_only=1``: the path
    must leave ``source`` onto a river hex, may only continue through
    further river hexes, and terminates the moment it reaches land.
    """
    if level <= 0:
        return frozenset()

    visited_river: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for river in board.adjacent.get(source, frozenset()):
        if board.hexes[river].color == RIVER:
            visited_river[river] = 1
            queue.append((river, 1))

    result: set[str] = set()
    while queue:
        node, count = queue.popleft()
        for neighbor in board.adjacent.get(node, frozenset()):
            if board.hexes[neighbor].color == RIVER:
                next_count = count + 1
                if next_count <= level and visited_river.get(neighbor, next_count + 1) > next_count:
                    visited_river[neighbor] = next_count
                    queue.append((neighbor, next_count))
            elif neighbor != source:
                result.add(neighbor)
    return frozenset(result)


def _teleport_reach(board: Board, source: str, effective_range: int) -> frozenset[str]:
    """Hexes within tunnel/carpet range of ``source``.

    Ports ``map.pm`` ``range{0}`` (unrestricted BFS whose recorded value
    is ``hex_distance - 1``), so ``recorded <= effective_range`` becomes
    ``hex_distance <= effective_range + 1``.
    """
    if effective_range <= 0:
        return frozenset()
    max_distance = effective_range + 1
    return frozenset(
        h for h in board.hexes if h != source and hex_distance(source, h, board) <= max_distance
    )


def effective_shipping(state: GameState, faction: str) -> int:
    """``faction``'s shipping range for reachability purposes: track level
    plus BON4's +1 passive while held (module docstring). BON4 only ever
    applies to a faction with a shipping track at all (``max_level > 0``
    -- e.g. never Dwarves) and only before ``faction`` has passed this
    round.
    """
    fs = state.factions[faction]
    level = fs.shipping
    track = FACTIONS[faction].shipping
    if track.max_level > 0 and not fs.passed and fs.bonus is not None:
        level += BONUS_TILES[fs.bonus].passive.get("ship", 0)
    return level


def reachable(state: GameState, faction: str) -> frozenset[str]:
    """Hexes ``faction`` may build/transform on: direct, shipping, or teleport range.

    Purely geometric (see module docstring): does not exclude occupied
    hexes or check terrain-color legality.
    """
    board = base_board()
    fs = state.factions[faction]
    building_hexes = _faction_buildings(state, faction)
    teleport = FACTIONS[faction].teleport
    ship_level = effective_shipping(state, faction)

    result: set[str] = set()
    for loc in building_hexes:
        result |= directly_adjacent(state, loc)
        if ship_level > 0:
            result |= _shipping_reach(board, loc, ship_level)
        if teleport is not None:
            effective_range = min(teleport.range + fs.teleport_level, teleport.max_range)
            result |= _teleport_reach(board, loc, effective_range)
    return frozenset(result)


def clusters(
    state: GameState, faction: str, *, river_skip: bool = False
) -> tuple[frozenset[str], ...]:
    """Connected components of ``faction``'s building hexes.

    Edges are direct adjacency plus bridges (``directly_adjacent``); with
    ``river_skip=True``, two building hexes sharing a common river-hex
    neighbor are also joined (see module docstring on the Mermaids
    ``{skip}`` port).
    """
    building_hexes = _faction_buildings(state, faction)
    if not building_hexes:
        return ()

    board = base_board()
    river_neighbors: dict[str, frozenset[str]] = {}
    if river_skip:
        river_neighbors = {
            h: frozenset(
                n for n in board.adjacent.get(h, frozenset()) if board.hexes[n].color == RIVER
            )
            for h in building_hexes
        }

    remaining = set(building_hexes)
    components: list[frozenset[str]] = []
    while remaining:
        start = next(iter(remaining))
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            neighbors = set(directly_adjacent(state, node) & building_hexes)
            if river_skip:
                for other in building_hexes:
                    if other not in seen and river_neighbors[node] & river_neighbors[other]:
                        neighbors.add(other)
            for neighbor in neighbors - seen:
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(frozenset(seen))
        remaining -= seen
    return tuple(components)
