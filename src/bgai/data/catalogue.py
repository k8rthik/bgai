"""Build the qualifying-game catalogue from tmtour.

Output (under ``data/catalogue/``):
- ``games.parquet``  — one row per qualifying game (Div <= 3, finished)
- ``seats.parquet``  — one row per seat (4 per game): username, faction, score, rank
- ``report.md``      — counts per division/season, exclusions, unparseable names
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

from bgai.data.tmtour import (
    QualifyingResult,
    fetch_season_games,
    fetch_seasons,
    open_client,
    qualifying_games,
)

MAX_DIVISION = 3
INTER_REQUEST_SLEEP_SECONDS = 0.5


def _games_frame(result: QualifyingResult) -> pl.DataFrame:
    rows = [
        {
            "game_id": g.game_id,
            "season": g.name_parts.season,
            "division": g.name_parts.division,
            "league": g.name_parts.league,
            "game_number": g.name_parts.game_number,
            "n_players": len(g.players),
        }
        for g in result.games
    ]
    return pl.DataFrame(rows)


def _seats_frame(result: QualifyingResult) -> pl.DataFrame:
    rows = [
        {
            "game_id": g.game_id,
            "season": g.name_parts.season,
            "division": g.name_parts.division,
            "username": p.username,
            "faction": p.faction,
            "score": p.score,
            "rank": p.rank,
            "seat": p.seat,
        }
        for g in result.games
        for p in g.players
    ]
    return pl.DataFrame(rows)


def _report(games: pl.DataFrame, result: QualifyingResult, n_seasons: int) -> str:
    by_division = games.group_by("division").len().sort("division")
    by_season = games.group_by("season", "division").len().sort("season", "division")
    lines = [
        "# Qualifying-game catalogue report",
        "",
        f"Seasons fetched: {n_seasons}",
        f"Qualifying games (Div <= {MAX_DIVISION}, finished): {games.height}",
        f"Excluded: division > {MAX_DIVISION}: {result.excluded_division}, "
        f"unfinished: {result.excluded_unfinished}, unparseable names: {len(result.unparseable)}",
        "",
        "## Games per division",
        str(by_division),
        "",
        "## Games per season/division (csv)",
        by_season.write_csv() if games.height else "(none)",
    ]
    if result.unparseable:
        lines += ["## Unparseable names", *result.unparseable, ""]
    return "\n".join(lines)


def build_catalogue(out_dir: Path) -> pl.DataFrame:
    seasons = fetch_seasons()
    season_ids = sorted(s["id"] for s in seasons)
    print(f"seasons: {len(season_ids)} (ids {season_ids[0]}..{season_ids[-1]})", flush=True)

    all_raw: list[dict] = []
    with open_client() as client:
        for season_id in season_ids:
            raw = fetch_season_games(season_id, client)
            all_raw.extend(raw)
            print(f"  season {season_id}: {len(raw)} games", flush=True)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    result = qualifying_games(all_raw, max_division=MAX_DIVISION)
    games = _games_frame(result)
    seats = _seats_frame(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    games.write_parquet(out_dir / "games.parquet")
    seats.write_parquet(out_dir / "seats.parquet")
    report = _report(games, result, n_seasons=len(season_ids))
    (out_dir / "report.md").write_text(report)
    print(report)
    return games


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/catalogue")
    build_catalogue(target)
