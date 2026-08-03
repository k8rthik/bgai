"""tests/test_connectivity.py"""

from dataclasses import replace

from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import clusters, directly_adjacent, reachable
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, with_faction

BOARD = base_board()


def _river_gap() -> tuple[str, str, str]:
    """(land_a, river, land_b): two land hexes joined only via one river hex."""
    for r in (k for k, h in BOARD.hexes.items() if h.color == RIVER):
        lands = [n for n in BOARD.adjacent[r] if BOARD.hexes[n].color != RIVER]
        for a in lands:
            for b in lands:
                if a != b and b not in BOARD.adjacent[a]:
                    return a, r, b
    raise AssertionError("no river gap on base map")


def _place(state: GameState, faction: str, hex_key: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building="D", owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, "D": fs.buildings["D"] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _fresh() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))


def _with_dwarves(state: GameState) -> GameState:
    return with_faction(state, "dwarves", FactionState.initial(FACTIONS["dwarves"]))


def test_shipping_zero_cannot_cross_river() -> None:
    a, _r, b = _river_gap()
    s = _place(_fresh(), "engineers", a)  # engineers ship level 0 initially
    assert b not in reachable(s, "engineers")


def test_shipping_one_reaches_across_single_river_hex() -> None:
    a, _r, b = _river_gap()
    s = _place(_fresh(), "engineers", a)
    s = replace(
        s, factions={**s.factions, "engineers": replace(s.factions["engineers"], shipping=1)}
    )
    assert b in reachable(s, "engineers")


def test_direct_neighbors_always_reachable() -> None:
    a, _r, _b = _river_gap()
    s = _place(_fresh(), "engineers", a)
    land_neighbors = {n for n in BOARD.adjacent[a] if BOARD.hexes[n].color != RIVER}
    assert land_neighbors <= reachable(s, "engineers")


def test_clusters_merge_via_bridge() -> None:
    # Find two land hexes at hex_distance 2 with a river between (bridge shape
    # rules ported in Task 8; here just assert clusters() honors state.bridges).
    a, _r, b = _river_gap()
    s = _place(_place(_fresh(), "engineers", a), "engineers", b)
    assert len(clusters(s, "engineers")) == 2
    s = replace(s, bridges=frozenset({frozenset({a, b})}))
    assert len(clusters(s, "engineers")) == 1


def test_dwarves_tunnel_skips_one_land_hex() -> None:
    # Locate land->land->land chain where the ends are not adjacent.
    for mid in (k for k, h in BOARD.hexes.items() if h.color != RIVER):
        lands = [n for n in BOARD.adjacent[mid] if BOARD.hexes[n].color != RIVER]
        pair = next(
            ((a, b) for a in lands for b in lands if a != b and b not in BOARD.adjacent[a]), None
        )
        if pair:
            a, b = pair
            s = _place(_fresh(), "darklings", a)  # non-dwarves: not reachable
            assert b not in reachable(s, "darklings")
            s = _place(_with_dwarves(_fresh()), "dwarves", a)
            assert b in reachable(s, "dwarves")
            return
    raise AssertionError("no skip-chain found")


def test_directly_adjacent_includes_bridge_endpoints() -> None:
    a, _r, b = _river_gap()
    s = replace(_fresh(), bridges=frozenset({frozenset({a, b})}))
    assert b in directly_adjacent(s, a)
    assert a in directly_adjacent(s, b)


def test_reachable_excludes_occupied_hexes_is_not_enforced() -> None:
    # reachable() is purely geometric: it does not filter out hexes that
    # already carry a building (occupancy/color legality belongs to a
    # later task's build-command validation).
    a, _r, b = _river_gap()
    s = _place(_place(_fresh(), "engineers", a), "engineers", b)
    s = replace(
        s, factions={**s.factions, "engineers": replace(s.factions["engineers"], shipping=1)}
    )
    assert b in reachable(s, "engineers")
