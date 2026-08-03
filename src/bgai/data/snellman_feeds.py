"""Download and index snellman's crawler-permitted monthly events feeds.

``https://terra.snellman.net/data/events/<YYYY-MM>.json`` lists every game
finished that month: id, base_map hash, player_count, last_update, factions,
per-faction event counters, and global options. We use these to verify the
scope filter (map, options, player count) for qualifying games without
touching the robots-disallowed ``/app/`` endpoints.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import httpx
import orjson
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from bgai.data.tmtour import USER_AGENT

EVENTS_BASE_URL = "https://terra.snellman.net/data/events"
FIRST_MONTH = (2014, 5)  # tmtour season 1 started 2014-05
INTER_REQUEST_SLEEP_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 60.0


def month_range(first: tuple[int, int], last: tuple[int, int]) -> list[str]:
    """Inclusive list of YYYY-MM strings from first to last."""
    months: list[str] = []
    year, month = first
    while (year, month) <= last:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=60))
def _fetch_month(client: httpx.Client, month: str) -> bytes | None:
    response = client.get(f"{EVENTS_BASE_URL}/{month}.json")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def download_events(raw_dir: Path) -> list[Path]:
    """Download all monthly feeds not already cached. Resumable by file presence."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    months = month_range(FIRST_MONTH, (today.year, today.month))
    paths: list[Path] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
    ) as client:
        for month in months:
            path = raw_dir / f"{month}.json"
            if path.exists():
                paths.append(path)
                continue
            content = _fetch_month(client, month)
            if content is None:
                print(f"  {month}: 404 (skipped)", flush=True)
                continue
            path.write_bytes(content)
            paths.append(path)
            print(f"  {month}: {len(content) / 1e6:.1f} MB", flush=True)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)
    return paths


def build_events_index(raw_dir: Path, game_ids: set[str]) -> pl.DataFrame:
    """One row per matching game: id, base_map, player_count, last_update, options."""
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        for game in orjson.loads(path.read_bytes()):
            game_id = game.get("game")
            if game_id not in game_ids:
                continue
            options = game.get("events", {}).get("global", {})
            rows.append(
                {
                    "game_id": game_id,
                    "month": path.stem,
                    "base_map": game.get("base_map"),
                    "player_count": game.get("player_count"),
                    "last_update": game.get("last_update"),
                    "options": sorted(
                        k.removeprefix("option-")
                        for k in options
                        if k.startswith("option-")
                    ),
                }
            )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/events")
    download_events(raw_dir)
    catalogue = pl.read_parquet("data/catalogue/games.parquet")
    index = build_events_index(raw_dir, set(catalogue["game_id"]))
    index.write_parquet("data/catalogue/events_index.parquet")
    print(f"indexed {index.height} / {catalogue.height} qualifying games")
    print("player_count values:", index["player_count"].unique().to_list())
    print("base_map values:", index["base_map"].unique().to_list())
    print(
        "option sets:",
        index.group_by("options").len().sort("len", descending=True),
    )
