"""Polite, resumable crawler for full snellman game logs.

Fetches ``/app/view-game/?game=<id>`` for every catalogued qualifying game.
That path is robots-disallowed, so this crawler is deliberately conservative:
one request at a time, never less than 1 s between request starts (paced
start-to-start, so latency cannot make it drift either way), an identifying
User-Agent with contact info, and a resumable on-disk cache so nothing is
ever fetched twice.

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

from bgai.data.player_ratings import playable
from bgai.data.identity import user_agent

VIEW_GAME_URL = "https://terra.snellman.net/app/view-game/"
MIN_REQUEST_INTERVAL_SECONDS = 1.0
"""Minimum seconds between request *starts* -- i.e. a hard ceiling of 1 req/s."""
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


def population_order(population: pl.DataFrame, ratings: pl.DataFrame) -> list[str]:
    """Stratified interleave of the whole playable population, best-balanced first.

    A 24-hour crawl will be interrupted, so the ordering has to make every
    *prefix* useful rather than back-loading one end of the skill range. Games
    are scored by their table's mean player rating, split into deciles, then
    emitted round-robin across deciles -- so stopping at any point leaves a
    sample that still spans weak-to-strong play, which is what the value head's
    coverage deficit (C5) actually needs.
    """
    seats = playable(population)
    table = (
        seats.join(ratings.select("player", "conservative"), on="player", how="left")
        .with_columns(pl.col("conservative").fill_null(0.0))
        .group_by("game_id")
        .agg(pl.col("conservative").mean().alias("strength"))
    )
    ranked = table.sort(["strength", "game_id"]).with_row_index("rank")
    decile = (pl.col("rank") * 10 // max(ranked.height, 1)).clip(0, 9)
    buckets: dict[int, list[str]] = {d: [] for d in range(10)}
    for game_id, d in ranked.with_columns(decile.alias("d")).select("game_id", "d").iter_rows():
        buckets[int(d)].append(game_id)

    ordered: list[str] = []
    for index in range(max((len(b) for b in buckets.values()), default=0)):
        for d in range(10):
            if index < len(buckets[d]):
                ordered.append(buckets[d][index])
    return ordered


def crawl_ids(game_ids: list[str], raw_dir: Path, sleep_seconds: float) -> None:
    """Fetch an explicit id list at a fixed rate, skipping anything cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in game_ids if not cache_path(raw_dir, g).exists()]
    print(f"target: {len(game_ids)} games, already cached: {len(game_ids) - len(todo)}, "
          f"to fetch: {len(todo)} (~{len(todo) * sleep_seconds / 3600:.1f} h at "
          f"{sleep_seconds}s/req)", flush=True)
    _crawl_loop(todo, raw_dir, sleep_seconds)


def crawl(catalogue_dir: Path, raw_dir: Path) -> None:
    games = crawl_order(pl.read_parquet(catalogue_dir / "games.parquet"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for g in games["game_id"].to_list() if not cache_path(raw_dir, g).exists()]
    print(f"catalogue: {games.height} games, already cached: {games.height - len(todo)}, "
          f"to fetch: {len(todo)}", flush=True)
    _crawl_loop(todo, raw_dir, MIN_REQUEST_INTERVAL_SECONDS)


def _crawl_loop(todo: list[str], raw_dir: Path, min_interval_seconds: float) -> None:
    """Fetch sequentially, pacing on the interval between request *starts*.

    Sleeping a fixed amount *after* each response makes the real rate depend on
    server latency -- it drifts slower when the site is slow, and would silently
    speed past the limit if the site got fast. Pacing start-to-start pins the
    rate at exactly ``1 / min_interval_seconds`` requests per second whatever
    the latency does, which is the promise this crawler actually makes.
    """
    fetched = 0
    started = time.monotonic()
    next_allowed = time.monotonic()
    with httpx.Client(
        headers={"User-Agent": user_agent()}, timeout=REQUEST_TIMEOUT_SECONDS
    ) as client:
        for game_id in todo:
            delay = next_allowed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_allowed = time.monotonic() + min_interval_seconds
            try:
                content = _fetch_game(client, game_id)
            except httpx.HTTPError as exc:
                _record_error(raw_dir, game_id, f"http failure after retries: {exc}")
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
    print(f"done: fetched {fetched}, cache now "
          f"{len(list(raw_dir.glob('*.json.gz')))} games", flush=True)


def status(catalogue_dir: Path, raw_dir: Path) -> None:
    total = pl.read_parquet(catalogue_dir / "games.parquet").height
    cached = len(list(raw_dir.glob("*.json.gz")))
    errors_file = raw_dir / "_errors.jsonl"
    n_errors = sum(1 for _ in errors_file.open()) if errors_file.exists() else 0
    print(f"cached {cached}/{total} games ({cached / total:.1%}), errors logged: {n_errors}")


def crawl_population(catalogue_dir: Path, raw_dir: Path, sleep_seconds: float) -> None:
    """Crawl the whole playable population, stratified by table strength."""
    population = pl.read_parquet(catalogue_dir / "population.parquet")
    ratings = pl.read_parquet(catalogue_dir / "player_ratings.parquet")
    crawl_ids(population_order(population, ratings), raw_dir, sleep_seconds)


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    catalogue_dir = Path(positional[0]) if positional else DEFAULT_CATALOGUE_DIR
    raw_dir = Path(positional[1]) if len(positional) > 1 else DEFAULT_RAW_GAMES_DIR

    sleep_seconds = MIN_REQUEST_INTERVAL_SECONDS
    for flag in flags:
        if flag.startswith("--sleep="):
            sleep_seconds = float(flag.removeprefix("--sleep="))

    if "--status" in flags:
        status(catalogue_dir, raw_dir)
    elif "--population" in flags:
        crawl_population(catalogue_dir, raw_dir, sleep_seconds)
    else:
        crawl(catalogue_dir, raw_dir)
