"""Polite, resumable crawler for full snellman game logs.

Fetches ``/app/view-game/?game=<id>`` for every catalogued qualifying game.
That path is robots-disallowed, so this crawler is deliberately conservative:
one request at a time, >= 1 s apart, an identifying User-Agent with contact
info, and a resumable on-disk cache so nothing is ever fetched twice.

Run: ``python -m bgai.data.crawl [catalogue_dir] [raw_games_dir]``
Progress/status: ``python -m bgai.data.crawl --status``
"""

from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import httpx
import orjson
import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential

from bgai.data.tmtour import USER_AGENT

VIEW_GAME_URL = "https://terra.snellman.net/app/view-game/"
INTER_REQUEST_SLEEP_SECONDS = 1.2
REQUEST_TIMEOUT_SECONDS = 60.0
PROGRESS_EVERY = 25

DEFAULT_CATALOGUE_DIR = Path("data/catalogue")
DEFAULT_RAW_GAMES_DIR = Path("data/raw/games")


def crawl_order(games: pl.DataFrame) -> pl.DataFrame:
    """Best tables first: Division 1 -> 2 -> 3, newest season first within each."""
    return games.sort(
        ["division", "season", "league", "game_number"],
        descending=[False, True, False, False],
    )


def cache_path(raw_dir: Path, game_id: str) -> Path:
    return raw_dir / f"{game_id}.json.gz"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=5, max=120))
def _fetch_game(client: httpx.Client, game_id: str) -> bytes:
    response = client.get(VIEW_GAME_URL, params={"game": game_id})
    response.raise_for_status()
    return response.content


def _validate(content: bytes, game_id: str) -> str | None:
    """Return an error string if the payload is not a usable finished-game log."""
    try:
        payload = orjson.loads(content)
    except orjson.JSONDecodeError:
        return "response is not JSON"
    errors = payload.get("error") or []
    if errors:
        return f"api error: {errors}"
    if not payload.get("ledger"):
        return "missing ledger"
    if not payload.get("finished"):
        return "game not finished"
    return None


def _record_error(raw_dir: Path, game_id: str, message: str) -> None:
    line = orjson.dumps({"game_id": game_id, "error": message}) + b"\n"
    with (raw_dir / "_errors.jsonl").open("ab") as f:
        f.write(line)
    print(f"  ERROR {game_id}: {message}", flush=True)


def crawl(catalogue_dir: Path, raw_dir: Path) -> None:
    games = crawl_order(pl.read_parquet(catalogue_dir / "games.parquet"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in games["game_id"].to_list() if not cache_path(raw_dir, g).exists()]
    print(f"catalogue: {games.height} games, already cached: {games.height - len(todo)}, "
          f"to fetch: {len(todo)}", flush=True)

    fetched = 0
    started = time.monotonic()
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
    ) as client:
        for game_id in todo:
            try:
                content = _fetch_game(client, game_id)
            except httpx.HTTPError as exc:
                _record_error(raw_dir, game_id, f"http failure after retries: {exc}")
                time.sleep(INTER_REQUEST_SLEEP_SECONDS)
                continue
            problem = _validate(content, game_id)
            if problem is None:
                with gzip.open(cache_path(raw_dir, game_id), "wb") as f:
                    f.write(content)
            else:
                _record_error(raw_dir, game_id, problem)
            fetched += 1
            if fetched % PROGRESS_EVERY == 0:
                rate = fetched / (time.monotonic() - started)
                remaining = (len(todo) - fetched) / rate if rate else float("inf")
                print(f"  {fetched}/{len(todo)} ({rate:.2f} games/s, "
                      f"~{remaining / 60:.0f} min left)", flush=True)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)
    print(f"done: fetched {fetched}, cache now "
          f"{len(list(raw_dir.glob('*.json.gz')))} games", flush=True)


def status(catalogue_dir: Path, raw_dir: Path) -> None:
    total = pl.read_parquet(catalogue_dir / "games.parquet").height
    cached = len(list(raw_dir.glob("*.json.gz")))
    errors_file = raw_dir / "_errors.jsonl"
    n_errors = sum(1 for _ in errors_file.open()) if errors_file.exists() else 0
    print(f"cached {cached}/{total} games ({cached / total:.1%}), errors logged: {n_errors}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--status"]
    catalogue_dir = Path(args[0]) if args else DEFAULT_CATALOGUE_DIR
    raw_dir = Path(args[1]) if len(args) > 1 else DEFAULT_RAW_GAMES_DIR
    if "--status" in sys.argv:
        status(catalogue_dir, raw_dir)
    else:
        crawl(catalogue_dir, raw_dir)
