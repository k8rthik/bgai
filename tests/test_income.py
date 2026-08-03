"""Income computation tests.

Step 6 is the load-bearing test: it validates ``faction_income`` against the
actual round-1 income deltas recorded for game ``4pLeague_S10_D1L1_G1`` in
``data/datasets/deltas.parquet`` (ledger rows 42-45, ``other_income_for_faction``
commands -- no ``cult_income_for_faction`` rows exist in round 1 since nobody
holds a favor tile yet). Steps 1-5's unit tests are scaffolding around the
individual income sources (buildings, bonus, favors); this data-driven check
is the oracle.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.state import FactionState
from bgai.engine.tm.tiles import FAVOR_TILES

_EMPTY_BUILDINGS = {k: frozenset() for k in ("D", "TP", "TE", "SH", "SA")}


def _fs(name: str, **kw: object) -> FactionState:
    fs = FactionState.initial(FACTIONS[name])
    return replace(fs, **kw)


def test_engineers_two_dwellings() -> None:
    fs = _fs(
        "engineers",
        buildings={**_EMPTY_BUILDINGS, "D": frozenset({"E7", "F6"})},
    )
    inc = faction_income(fs, FACTIONS["engineers"])
    assert inc["buildings"].get("W", 0) == 2  # track (0,1,2,...) index 2


def test_bonus_tile_income() -> None:
    fs = _fs("engineers", bonus="BON3", buildings=_EMPTY_BUILDINGS)
    inc = faction_income(fs, FACTIONS["engineers"])
    assert inc["bonus"] == {"C": 6}


def test_favor_income() -> None:
    # Pick a favor tile whose income dict is non-empty (there is at least one
    # coin/power/worker-income favor tile) and assert inc["favors"] equals
    # that tile's income dict verbatim.
    fav_id = next(f for f, t in FAVOR_TILES.items() if t.income)
    fs = _fs("darklings", favors=(fav_id,), buildings=_EMPTY_BUILDINGS)
    inc = faction_income(fs, FACTIONS["darklings"])
    assert inc["favors"] == dict(FAVOR_TILES[fav_id].income)


def test_no_buildings_no_bonus_no_favors_is_all_empty() -> None:
    fs = _fs("darklings", buildings=_EMPTY_BUILDINGS)
    inc = faction_income(fs, FACTIONS["darklings"])
    assert inc["bonus"] == {}
    assert inc["favors"] == {}
    assert inc["buildings"] == {}
    # Darklings' D-track floor (index 0) is 1 W -- a real per-faction base
    # income granted even with zero dwellings built (Darklings.pm line 26).
    assert inc["base"] == {"W": 1}


def test_engineers_base_income_is_zero() -> None:
    # Engineers.pm line 27: D income track starts at 0, unlike every other
    # base faction's D track (which starts at 1 W; Swarmlings starts at 2 W).
    fs = _fs("engineers", buildings=_EMPTY_BUILDINGS)
    inc = faction_income(fs, FACTIONS["engineers"])
    assert inc["base"] == {}


def _sum_categories(inc: dict[str, dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for bucket in inc.values():
        for resource, amount in bucket.items():
            total[resource] = total.get(resource, 0) + amount
    return total


def test_matches_reference_game_round1_income() -> None:
    """Pin income arithmetic to the deltas-parquet oracle before apply() exists.

    Post-setup state (ledger rows 28-36 for dwellings, rows 37-40 for bonus
    tile picks): engineers has dwellings E7+F6 and BON4; darklings has G5+E10
    and BON3; nomads has D3+F3+G4 (3 dwellings) and BON5; mermaids has D2+D5
    and BON1. Nobody holds a favor tile yet, so cult income is not present in
    round 1 -- the recorded rows are ``other_income_for_faction`` only
    (ledger rows 42-45, i.e. ``deltas.parquet`` rows 42-45 for this game).
    """
    setups: dict[str, tuple[str, ...]] = {
        "engineers": ("E7", "F6"),
        "darklings": ("G5", "E10"),
        "nomads": ("D3", "F3", "G4"),
        "mermaids": ("D2", "D5"),
    }
    bonuses: dict[str, str] = {
        "engineers": "BON4",
        "darklings": "BON3",
        "nomads": "BON5",
        "mermaids": "BON1",
    }
    row_by_faction = {"engineers": 42, "darklings": 43, "nomads": 44, "mermaids": 45}

    deltas = pl.read_parquet("data/datasets/deltas.parquet").filter(
        (pl.col("game_id") == "4pLeague_S10_D1L1_G1")
        & (pl.col("row").is_in(list(row_by_faction.values())))
    )

    for faction, dwellings in setups.items():
        fs = _fs(
            faction,
            buildings={**_EMPTY_BUILDINGS, "D": frozenset(dwellings)},
            bonus=bonuses[faction],
        )
        inc = faction_income(fs, FACTIONS[faction])
        total = _sum_categories(inc)

        row = deltas.filter(
            (pl.col("faction") == faction) & (pl.col("row") == row_by_faction[faction])
        )
        assert row.height == 1, f"expected exactly one deltas row for {faction}"
        record = row.row(0, named=True)

        assert total.get("C", 0) == record["c_delta"], faction
        assert total.get("W", 0) == record["w_delta"], faction
        assert total.get("P", 0) == record["p_delta"], faction

        starting_power = FactionState.initial(FACTIONS[faction]).power
        expected_power = starting_power.gain(total.get("PW", 0))
        assert expected_power.as_str() == record["pw"], faction
