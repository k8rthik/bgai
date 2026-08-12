"""Per-game provenance: time key and sample weight (population rebuild).

The league corpus carried season/division in the game id. The population
corpus does not, so provenance is resolved from the catalogue instead --
and the league rules have to keep producing exactly their old answers, or
the C4/C5/C6 numbers stop being comparable.
"""

from __future__ import annotations

import polars as pl
import pytest

from bgai.training.provenance import (
    DIVISION_WEIGHTS,
    VAL_SEASON_MIN,
    GameProvenance,
    build_provenance,
    league_season_division,
    month_to_period,
    strength_weight,
)


def test_league_id_parses_season_and_division() -> None:
    assert league_season_division("4pLeague_S30_D3L4_G4") == (30, 3)
    assert league_season_division("4pLeague_S7_D1L12_G7") == (7, 1)


@pytest.mark.parametrize(
    "game_id", ["TMStreet240002", "mkreur", "SkyIsFalling13", "TMTourS12ChatRoom", ""]
)
def test_non_league_ids_have_no_season(game_id: str) -> None:
    assert league_season_division(game_id) is None


def test_month_to_period_is_monotone_and_dense() -> None:
    assert month_to_period("2014-06") - month_to_period("2014-05") == 1
    assert month_to_period("2015-01") - month_to_period("2014-12") == 1
    assert month_to_period("2026-07") > month_to_period("2014-05")


def test_strength_weight_spans_the_division_range() -> None:
    """Weakest decile matches Div 3's weight, strongest matches Div 1's."""
    assert strength_weight(0) == pytest.approx(DIVISION_WEIGHTS[3])
    assert strength_weight(9) == pytest.approx(DIVISION_WEIGHTS[1])
    assert strength_weight(4) < strength_weight(5)


def test_league_game_keeps_its_id_derived_provenance() -> None:
    population = pl.DataFrame(
        {"game_id": ["4pLeague_S70_D1L1_G1"], "month": ["2024-03"], "player": ["a"]}
    )
    ratings = pl.DataFrame({"player": ["a"], "conservative": [30.0]})
    prov = build_provenance(["4pLeague_S70_D1L1_G1"], population, ratings)

    entry = prov["4pLeague_S70_D1L1_G1"]
    assert (entry.season, entry.division) == (70, 1)
    assert entry.weight == pytest.approx(DIVISION_WEIGHTS[1])
    assert entry.period == month_to_period("2024-03")


def test_population_game_gets_date_and_strength_provenance() -> None:
    population = pl.DataFrame(
        {
            "game_id": ["casual1", "casual1", "casual2", "casual2"],
            "month": ["2019-08", "2019-08", "2019-08", "2019-08"],
            "player": ["a", "b", "c", "d"],
        }
    )
    ratings = pl.DataFrame(
        {"player": ["a", "b", "c", "d"], "conservative": [10.0, 12.0, 40.0, 42.0]}
    )
    prov = build_provenance(["casual1", "casual2"], population, ratings)

    weak, strong = prov["casual1"], prov["casual2"]
    assert (weak.season, weak.division) == (-1, -1)
    assert weak.period == month_to_period("2019-08")
    # the stronger table is worth more, and both stay inside the division range
    assert strong.weight > weak.weight
    for entry in (weak, strong):
        assert DIVISION_WEIGHTS[3] <= entry.weight <= DIVISION_WEIGHTS[1]


def test_missing_catalogue_row_is_not_silently_dropped() -> None:
    """A game with no catalogue entry must be absent, so the caller can
    count it rather than train on a fabricated weight."""
    population = pl.DataFrame({"game_id": ["known"], "month": ["2020-01"], "player": ["a"]})
    ratings = pl.DataFrame({"player": ["a"], "conservative": [25.0]})
    prov = build_provenance(["known", "orphan"], population, ratings)
    assert "orphan" not in prov
    assert "known" in prov


def test_val_rule_preserves_the_league_boundary() -> None:
    """League games split on season exactly as before; population games
    split on the wall-clock date that boundary corresponds to."""
    old_val = GameProvenance(season=VAL_SEASON_MIN, division=1, period=100, weight=1.0)
    old_train = GameProvenance(season=VAL_SEASON_MIN - 1, division=1, period=999, weight=1.0)
    assert old_val.is_val(val_period_min=0) is True
    # a late period must NOT drag an old league game into val
    assert old_train.is_val(val_period_min=0) is False

    late = GameProvenance(season=-1, division=-1, period=500, weight=0.8)
    early = GameProvenance(season=-1, division=-1, period=100, weight=0.8)
    assert late.is_val(val_period_min=400) is True
    assert early.is_val(val_period_min=400) is False
