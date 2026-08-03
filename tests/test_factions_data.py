"""Sanity tests for the base-game faction data, hand-verified against the
reference Perl source (jsnell/terra-mystica, src/Game/Factions/*.pm)."""

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from bgai.engine.tm.factions_data import (
    COLOR_WHEEL,
    CULTS,
    FACTIONS,
    FactionData,
)

BASE_FACTION_NAMES = {
    "alchemists",
    "auren",
    "chaosmagicians",
    "cultists",
    "darklings",
    "dwarves",
    "engineers",
    "fakirs",
    "giants",
    "halflings",
    "mermaids",
    "nomads",
    "swarmlings",
    "witches",
}


def test_exactly_14_base_factions() -> None:
    assert set(FACTIONS) == BASE_FACTION_NAMES
    assert len(FACTIONS) == 14


def test_keys_match_names_and_types() -> None:
    for name, faction in FACTIONS.items():
        assert isinstance(faction, FactionData)
        assert faction.name == name


def test_every_faction_has_four_cult_values() -> None:
    for faction in FACTIONS.values():
        assert tuple(faction.cults) == CULTS
        assert all(0 <= v <= 2 for v in faction.cults.values())


def test_home_colors_each_appear_exactly_twice() -> None:
    counts = Counter(f.color for f in FACTIONS.values())
    assert set(counts) == set(COLOR_WHEEL)
    assert all(count == 2 for count in counts.values())


def test_every_faction_has_all_five_buildings() -> None:
    expected_max = {"D": 8, "TP": 4, "TE": 3, "SH": 1, "SA": 1}
    for faction in FACTIONS.values():
        assert set(faction.buildings) == set(expected_max)
        for key, track in faction.buildings.items():
            assert track.max_count == expected_max[key]
            for resource_track in track.income.values():
                assert len(resource_track) == track.max_count + 1


def test_track_lengths_are_consistent() -> None:
    for faction in FACTIONS.values():
        dig = faction.dig
        assert len(dig.cost) == dig.max_level + 1
        assert len(dig.advance_vp) == dig.max_level
        ship = faction.shipping
        if ship.max_level:
            assert len(ship.advance_vp) == ship.max_level
        else:
            assert ship.advance_cost is None and ship.advance_vp == ()


def test_power_bowls_total_twelve() -> None:
    # Every base faction starts with 12 power tokens, none in bowl 3.
    for faction in FACTIONS.values():
        assert sum(faction.power) == 12
        assert faction.power[2] == 0


# Spot checks, hand-verified from the Perl faction files.


def test_witches_spot_check() -> None:
    witches = FACTIONS["witches"]
    assert witches.color == "green"
    assert witches.cults == {"FIRE": 0, "WATER": 0, "EARTH": 0, "AIR": 2}
    assert witches.special_gain == {"TOWN": {"VP": 5}}
    assert witches.buildings["SH"].build_gain == ({"ACTW": 1},)


def test_alchemists_spot_check() -> None:
    alchemists = FACTIONS["alchemists"]
    assert alchemists.color == "black"
    assert alchemists.exchange_rate_overrides == {"C": {"VP": 2}, "VP": {"C": 1}}
    assert alchemists.special_gain == {"SPADE": {"PW": 2}}
    assert alchemists.special_requires_sh is True
    assert alchemists.buildings["SH"].build_gain == ({"PW": 12},)
    assert alchemists.buildings["TP"].income["C"] == (0, 2, 4, 7, 11)


def test_darklings_spot_check() -> None:
    darklings = FACTIONS["darklings"]
    assert darklings.color == "black"
    assert (darklings.workers, darklings.priests) == (1, 1)
    assert darklings.dig.max_level == 0
    assert darklings.dig.cost == ({"P": 1},)
    assert darklings.dig.dig_gain == ({"SPADE": 1, "VP": 2},)
    assert darklings.buildings["SA"].cost == {"W": 4, "C": 10}
    assert darklings.buildings["SA"].income["P"] == (0, 2)


def test_swarmlings_spot_check() -> None:
    swarmlings = FACTIONS["swarmlings"]
    assert (swarmlings.coins, swarmlings.workers) == (20, 8)
    assert swarmlings.cults == {"FIRE": 1, "WATER": 1, "EARTH": 1, "AIR": 1}
    assert swarmlings.buildings["SA"].cost == {"W": 5, "C": 8}
    assert swarmlings.buildings["SH"].cost == {"W": 5, "C": 8}
    assert swarmlings.buildings["D"].cost == {"W": 2, "C": 3}


def test_engineers_spot_check() -> None:
    engineers = FACTIONS["engineers"]
    assert engineers.color == "gray"
    assert (engineers.coins, engineers.workers) == (10, 2)
    assert engineers.power == (3, 9, 0)
    assert engineers.buildings["D"].cost == {"W": 1, "C": 1}
    assert engineers.buildings["TP"].cost == {"W": 1, "C": 2}
    assert engineers.buildings["TE"].cost == {"W": 1, "C": 4}
    assert engineers.buildings["SH"].cost == {"W": 3, "C": 6}
    assert engineers.buildings["SA"].cost == {"W": 3, "C": 6}
    assert engineers.special_actions == ("ACTE",)


def test_mermaids_shipping_and_dwarves_fakirs_teleport() -> None:
    mermaids = FACTIONS["mermaids"]
    assert (mermaids.shipping.level, mermaids.shipping.max_level) == (1, 5)
    assert mermaids.shipping.advance_vp == (0, 2, 3, 4, 5)
    for name, kind in (("dwarves", "tunnel"), ("fakirs", "carpet")):
        faction = FACTIONS[name]
        assert faction.shipping.max_level == 0
        assert faction.teleport is not None
        assert faction.teleport.kind == kind


def test_setup_dwelling_counts() -> None:
    assert FACTIONS["chaosmagicians"].start_dwellings == 1
    assert FACTIONS["nomads"].start_dwellings == 3
    others = BASE_FACTION_NAMES - {"chaosmagicians", "nomads"}
    assert all(FACTIONS[name].start_dwellings == 2 for name in others)


def test_faction_data_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        FACTIONS["witches"].coins = 99  # type: ignore[misc]
