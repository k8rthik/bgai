"""Per-game provenance: the time key and sample weight every training
record carries.

The league corpus encoded both in the game id (``4pLeague_S30_D3L4_G4``),
so extraction parsed them with a regex. The population corpus does not --
its ids look like ``mkreur`` or ``TMStreet240002`` -- and that regex was
silently dropping 43% of the crawled games.

Provenance is therefore resolved from the catalogue, using signals that
exist for *every* game:

* **period** -- months since year 0, from ``population.month``. A
  date-based split is what the season split was approximating anyway.
* **weight** -- table strength from the self-computed player ratings,
  bucketed into deciles and mapped onto the division-weight range. This
  is the same strength signal ``crawl.population_order`` already uses to
  stratify the crawl, so weighting and sampling now agree.

League games keep their id-derived season and division so their split and
weights come out bit-identical to the pre-population runs -- otherwise the
C4/C5/C6 measurements would stop being comparable to anything new.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

LEAGUE_ID_RE = re.compile(r"4pLeague_S(\d+)_D(\d)L\d+_G\d+$")

VAL_SEASON_MIN = 67
"""Seasons >= this go to validation: a time-based split, so no future
game informs a prediction about an earlier one (master plan Phase 5)."""

DIVISION_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6}
"""Table-quality weighting (master plan: Div 1 > Div 2 > Div 3)."""

STRENGTH_DECILES = 10
"""Population games are bucketed into this many strength bands, matching
the crawl's own stratification."""

NO_SEASON = -1
"""Sentinel for a game whose id carries no league season/division."""


@dataclass(frozen=True)
class GameProvenance:
    """Where a game sits in time and how much its decisions are worth."""

    season: int
    division: int
    period: int
    weight: float

    @property
    def is_league(self) -> bool:
        return self.season != NO_SEASON

    def is_val(self, val_period_min: int) -> bool:
        """League games split on season exactly as they always did; every
        other game splits on the wall-clock boundary that season implies."""
        if self.is_league:
            return self.season >= VAL_SEASON_MIN
        return self.period >= val_period_min


def league_season_division(game_id: str) -> tuple[int, int] | None:
    """``(season, division)`` for a league id, ``None`` for anything else."""
    match = LEAGUE_ID_RE.match(game_id)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def month_to_period(month: str) -> int:
    """``"2019-08"`` -> a dense month counter, monotone across year ends."""
    year, month_of_year = month.split("-")
    return int(year) * 12 + int(month_of_year) - 1


def strength_weight(decile: int) -> float:
    """Map a strength decile onto the division-weight range, so the
    weakest population tables count for what Div 3 counts for and the
    strongest for what Div 1 counts for."""
    low, high = DIVISION_WEIGHTS[3], DIVISION_WEIGHTS[1]
    span = STRENGTH_DECILES - 1
    return low + (high - low) * (decile / span)


def _table_strength(population: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Mean conservative rating of each game's seats. Unrated players
    score 0.0, matching ``crawl.population_order``."""
    return (
        population.join(ratings.select("player", "conservative"), on="player", how="left")
        .with_columns(pl.col("conservative").fill_null(0.0))
        .group_by("game_id")
        .agg(pl.col("conservative").mean().alias("strength"))
    )


def _strength_deciles(strength: pl.DataFrame) -> pl.DataFrame:
    """Rank games by table strength and bucket them into deciles."""
    ranked = strength.sort(["strength", "game_id"]).with_row_index("rank")
    decile = (pl.col("rank") * STRENGTH_DECILES // max(ranked.height, 1)).clip(
        0, STRENGTH_DECILES - 1
    )
    return ranked.with_columns(decile.cast(pl.Int32).alias("decile")).select("game_id", "decile")


def build_provenance(
    game_ids: list[str],
    population: pl.DataFrame,
    ratings: pl.DataFrame,
) -> dict[str, GameProvenance]:
    """Resolve provenance for ``game_ids``.

    Games with no catalogue row are *omitted* rather than defaulted, so a
    caller can count them instead of training on a fabricated weight.
    """
    wanted = set(game_ids)
    months = dict(
        population.filter(pl.col("game_id").is_in(wanted))
        .select("game_id", "month")
        .unique(subset=["game_id"])
        .iter_rows()
    )
    # deciles are ranked over the *whole* population, not just the games
    # being built, so a game's weight does not depend on what else is in
    # the batch -- a subset build and a full build agree
    deciles = dict(_strength_deciles(_table_strength(population, ratings)).iter_rows())

    resolved: dict[str, GameProvenance] = {}
    for game_id in game_ids:
        month = months.get(game_id)
        if month is None:
            continue
        league = league_season_division(game_id)
        season, division = league if league is not None else (NO_SEASON, NO_SEASON)
        weight = (
            DIVISION_WEIGHTS[division]
            if league is not None and division in DIVISION_WEIGHTS
            else strength_weight(deciles.get(game_id, 0))
        )
        resolved[game_id] = GameProvenance(
            season=season,
            division=division,
            period=month_to_period(month),
            weight=weight,
        )
    return resolved


def val_period_min(provenance: dict[str, GameProvenance]) -> int:
    """The wall-clock boundary the season split implies: the earliest month
    any validation-side league game was played.

    Falling back to the max period when there are no league games keeps a
    league-free corpus entirely in train rather than silently splitting it
    on an arbitrary date.
    """
    val_periods = [
        entry.period
        for entry in provenance.values()
        if entry.is_league and entry.season >= VAL_SEASON_MIN
    ]
    if not val_periods:
        periods = [entry.period for entry in provenance.values()]
        return max(periods) + 1 if periods else 0
    return min(val_periods)
