"""tests/test_towns.py"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, with_faction
from bgai.engine.tm.towns import (
    apply_town_tile,
    building_power_value,
    new_towns,
    record_founded_town,
)

BOARD = base_board()


def _fresh() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))


def _with_faction(state: GameState, name: str) -> GameState:
    return with_faction(state, name, FactionState.initial(FACTIONS[name]))


def _place(state: GameState, faction: str, hex_key: str, building: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _place_all(state: GameState, faction: str, layout: dict[str, str]) -> GameState:
    for hex_key, building in layout.items():
        state = _place(state, faction, hex_key, building)
    return state


def _chain(n: int) -> tuple[str, ...]:
    """``n`` land hexes forming a single connected component (plain adjacency)."""
    for start in BOARD.land_hexes():
        seen = [start]
        seen_set = {start}
        stack = [start]
        while stack and len(seen) < n:
            node = stack.pop()
            for neighbor in sorted(BOARD.adjacent[node]):
                if BOARD.hexes[neighbor].color != RIVER and neighbor not in seen_set:
                    seen_set.add(neighbor)
                    seen.append(neighbor)
                    stack.append(neighbor)
                    if len(seen) >= n:
                        break
        if len(seen) >= n:
            return tuple(seen[:n])
    raise AssertionError(f"no {n}-hex land chain found")


def _mermaid_river_layout() -> tuple[str, str, str, str, str]:
    """(a, a2, river, b, b2): two 2-hex land groups joined only via one river hex."""

    def land_neighbors(x: str, exclude: frozenset[str] = frozenset()) -> list[str]:
        return [n for n in BOARD.adjacent[x] if BOARD.hexes[n].color != RIVER and n not in exclude]

    for river, hex_ in BOARD.hexes.items():
        if hex_.color != RIVER:
            continue
        lands = land_neighbors(river)
        for a in lands:
            for b in lands:
                if a == b or b in BOARD.adjacent[a]:
                    continue
                for a2 in land_neighbors(a, frozenset({b, river})):
                    if a2 in BOARD.adjacent[b]:
                        continue
                    for b2 in land_neighbors(b, frozenset({a, river, a2})):
                        if b2 in BOARD.adjacent[a] or b2 in BOARD.adjacent[a2]:
                            continue
                        return a, a2, river, b, b2
    raise AssertionError("no mermaid river layout found")


def test_building_power_values() -> None:
    assert building_power_value("D") == 1
    assert building_power_value("TP") == 2
    assert building_power_value("TE") == 2
    assert building_power_value("SH") == 3
    assert building_power_value("SA") == 3


def test_four_buildings_power_seven_qualifies() -> None:
    a, b, c, d = _chain(4)
    s = _place_all(_fresh(), "engineers", {a: "D", b: "TP", c: "TP", d: "TP"})
    assert new_towns(s, "engineers") == (frozenset({a, b, c, d}),)


def test_three_buildings_power_eight_does_not_qualify() -> None:
    a, b, c = _chain(3)
    s = _place_all(_fresh(), "engineers", {a: "SH", b: "SH", c: "TP"})
    assert new_towns(s, "engineers") == ()


def test_sanctuary_counts_double_and_town_size_six_favor() -> None:
    a, b, c, d = _chain(4)
    s = _place_all(_fresh(), "engineers", {a: "SA", b: "D", c: "D", d: "D"})
    # power = 3 + 1 + 1 + 1 = 6: below the default threshold of 7.
    assert new_towns(s, "engineers") == ()
    # FAV5 passive lowers TOWN_SIZE to 6: now qualifies. Count = 4 physical
    # buildings + 1 (SA counts twice) = 5 >= 4 either way.
    fs = replace(s.factions["engineers"], favors=("FAV5",))
    s2 = with_faction(s, "engineers", fs)
    assert new_towns(s2, "engineers") == (frozenset({a, b, c, d}),)


def test_mermaid_river_join_qualifies_only_for_mermaids() -> None:
    a, a2, river, b, b2 = _mermaid_river_layout()
    layout = {a: "SH", a2: "D", b: "TE", b2: "D"}
    # power = (3+1) + (2+1) = 7, count = 4, split across two 2-building
    # components only connected through `river`.
    s = _place_all(_fresh(), "mermaids", layout)
    assert new_towns(s, "mermaids") == (frozenset({a, a2, b, b2}),)

    s_witches = _place_all(_with_faction(_fresh(), "witches"), "witches", layout)
    assert new_towns(s_witches, "witches") == ()


def test_founding_twice_returns_nothing_new() -> None:
    a, b, c, d = _chain(4)
    s = _place_all(_fresh(), "engineers", {a: "D", b: "TP", c: "TP", d: "TP"})
    found = new_towns(s, "engineers")
    assert found == (frozenset({a, b, c, d}),)
    s2 = record_founded_town(s, "engineers", found[0])
    assert new_towns(s2, "engineers") == ()


def test_apply_town_tile_grants_vp_and_gain() -> None:
    s = _fresh()
    before = s.factions["engineers"]
    before_pool = s.towns_pool["TW1"]
    s2 = apply_town_tile(s, "engineers", "TW1")
    after = s2.factions["engineers"]
    assert after.vp == before.vp + 5
    assert after.keys == before.keys + 1
    assert after.coins == before.coins + 6
    assert after.towns == ("TW1",)
    assert s2.towns_pool["TW1"] == before_pool - 1


def test_apply_town_tile_power_gain_uses_power_bowl() -> None:
    s = _fresh()
    before = s.factions["engineers"]
    s2 = apply_town_tile(s, "engineers", "TW4")  # gain={"KEY": 1, "PW": 8}
    after = s2.factions["engineers"]
    assert after.power == before.power.gain(8)
    assert after.keys == before.keys + 1


def test_apply_town_tile_witches_passive_vp() -> None:
    s = _with_faction(_fresh(), "witches")
    before = s.factions["witches"]
    s2 = apply_town_tile(s, "witches", "TW1")
    after = s2.factions["witches"]
    assert after.vp == before.vp + 5 + 5  # tile VP + Witches special_gain


def test_apply_town_tile_swarmlings_passive_workers() -> None:
    s = _with_faction(_fresh(), "swarmlings")
    before = s.factions["swarmlings"]
    s2 = apply_town_tile(s, "swarmlings", "TW1")  # tile gain has no W
    after = s2.factions["swarmlings"]
    assert after.workers == before.workers + 3  # Swarmlings special_gain only


def test_apply_town_tile_defers_cult_step_keys() -> None:
    s = _fresh()
    before_cults = s.cults["engineers"]
    before = s.factions["engineers"]
    s2 = apply_town_tile(s, "engineers", "TW5")  # gain incl. FIRE/WATER/EARTH/AIR
    after = s2.factions["engineers"]
    assert after.vp == before.vp + 8
    assert after.keys == before.keys + 1
    assert s2.cults["engineers"] == before_cults  # cult steps NOT applied here


def test_apply_town_tile_rejects_unknown_tile() -> None:
    with pytest.raises(ValueError):
        apply_town_tile(_fresh(), "engineers", "TW99")


def test_apply_town_tile_rejects_exhausted_pool() -> None:
    s = _fresh()
    s = replace(s, towns_pool={**s.towns_pool, "TW8": 0})
    with pytest.raises(ValueError):
        apply_town_tile(s, "engineers", "TW8")
