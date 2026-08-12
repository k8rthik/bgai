"""Incremental ingest: parse and validate crawled games as they land.

The population crawl runs for ~17 hours. Batching all parsing until it
finishes would mean discovering problems at the end, on a corpus 20x larger
than anything the parser and engine have faced -- and ``build_moves.build``
fails the *whole* run on one unparseable command, which is right for a
curated corpus and wrong for 60k general-population games where a single
odd 2015 log would destroy 59,999 good ones.

So this module processes whatever is on disk, repeatedly, alongside the
crawl. Each pass is idempotent: it looks at what is already parsed or
quarantined and does only the remainder, so it can be re-run, interrupted,
or looped (``--follow``) with identical results. Two failure policies make
that work:

- **Unparseable game -> quarantine, never fatal.** The pass continues and
  reports the distinct unknown vocabulary aggregated across all failures.
  Fix a grammar rule, re-run, and quarantined games reprocess automatically
  because the pass is idempotent -- the quarantine drains itself.
- **Undecompressable game -> skip silently, retry next pass.** The crawler
  is writing these files concurrently; a half-written gzip is a race, not a
  defect, and must not be mistaken for a parse failure.

Output is per-batch parquet *parts* (parquet cannot append, and polars scans
a glob as one dataset). ``compact`` merges parts into the canonical
``moves.parquet``/``deltas.parquet``/``games_meta.parquet`` that every
existing consumer reads, so nothing downstream changes until we choose to
compact.

NOT done here: extraction into training records. ``training.extract`` derives
each record's ``season`` (the train/val boundary) and ``division`` (the
sample weight) by regex from tmtour game ids, and population ids like
``danvert01b`` carry neither. Choosing their replacement -- dates for the
split, table TrueSkill for the weight -- is a modelling decision tied to the
value/policy training split, deliberately left to that work rather than
invented here.

Run: ``python -m bgai.data.ingest [--follow] [--interval 300] [--compact]``
"""

from __future__ import annotations

import argparse
import gzip
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import polars as pl

from bgai.data.build_moves import MOVES_SCHEMA, parse_game
from bgai.data.ledger_parser import parse_commands

DEFAULT_RAW_DIR = Path("data/raw/games")
DEFAULT_PARTS_DIR = Path("data/datasets/parts")
DEFAULT_OUT_DIR = Path("data/datasets")
GAMES_PER_PART = 500


@dataclass(frozen=True)
class PassStats:
    """What one pass did. ``skipped`` are races with the crawler, not errors."""

    parsed: int = 0
    quarantined: int = 0
    skipped: int = 0
    parts_written: int = 0
    unknown_commands: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        head = (
            f"parsed {self.parsed}, quarantined {self.quarantined}, "
            f"skipped(in-flight) {self.skipped}, parts {self.parts_written}"
        )
        if not self.unknown_commands:
            return head
        top = ", ".join(f"{cmd!r}x{n}" for cmd, n in self.unknown_commands.most_common(5))
        return f"{head}\n  unknown vocabulary: {top}"


def _quarantine_path(parts_dir: Path) -> Path:
    return parts_dir / "_quarantine.jsonl"


def quarantined_ids(parts_dir: Path) -> set[str]:
    path = _quarantine_path(parts_dir)
    if not path.exists():
        return set()
    return {orjson.loads(line)["game_id"] for line in path.read_text().splitlines() if line}


def parsed_ids(parts_dir: Path) -> set[str]:
    """Game ids already present in written parts. Derived from the parts
    themselves rather than a side-car state file, so there is nothing to
    corrupt or resynchronise after an interrupted pass."""
    ids: set[str] = set()
    for part in parts_dir.glob("moves-*.parquet"):
        ids.update(pl.read_parquet(part, columns=["game_id"])["game_id"].unique().to_list())
    return ids


def _load_game(path: Path) -> dict | None:
    """None if the file cannot be read yet -- the crawler may be mid-write."""
    try:
        return orjson.loads(gzip.decompress(path.read_bytes()))
    except (OSError, EOFError, gzip.BadGzipFile, orjson.JSONDecodeError):
        return None


def _unknown_vocabulary(game: dict) -> Counter[str]:
    """The commands the parser could not read, for the aggregate report."""
    unknown: Counter[str] = Counter()
    for row in game.get("ledger", []):
        commands = row.get("commands")
        if commands:
            _, bad = parse_commands(commands)
            unknown.update(bad)
    return unknown


def _next_part_index(parts_dir: Path) -> int:
    existing = sorted(parts_dir.glob("moves-*.parquet"))
    if not existing:
        return 0
    return int(existing[-1].stem.removeprefix("moves-")) + 1


def _write_part(
    parts_dir: Path, index: int, moves: list[dict], deltas: list[dict], meta: list[dict]
) -> None:
    pl.DataFrame(moves, schema=MOVES_SCHEMA).write_parquet(parts_dir / f"moves-{index:04d}.parquet")
    pl.DataFrame(deltas).write_parquet(parts_dir / f"deltas-{index:04d}.parquet")
    pl.DataFrame(meta).write_parquet(parts_dir / f"meta-{index:04d}.parquet")


def run_pass(
    raw_dir: Path = DEFAULT_RAW_DIR,
    parts_dir: Path = DEFAULT_PARTS_DIR,
    games_per_part: int = GAMES_PER_PART,
) -> PassStats:
    """Parse every cached game not yet parsed or quarantined."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    # Quarantine is DERIVED, not a tombstone: previously-failed games are
    # retried every pass and the file is rewritten from this pass's failures.
    # That is what makes "fix a grammar rule, re-run, quarantine drains" true
    # by construction rather than by remembering to clear something. The cost
    # is re-parsing a handful of games per pass, which is why quarantine must
    # stay small -- if it ever grows large, that is the signal, not the cost.
    already_parsed = parsed_ids(parts_dir)
    todo = sorted(
        p
        for p in raw_dir.glob("*.json.gz")
        if p.name.removesuffix(".json.gz") not in already_parsed
    )
    if not todo:
        return PassStats()
    failures: list[tuple[str, str]] = []

    part_index = _next_part_index(parts_dir)
    moves: list[dict] = []
    deltas: list[dict] = []
    meta: list[dict] = []
    parsed = quarantined = skipped = parts_written = 0
    unknown: Counter[str] = Counter()

    for path in todo:
        game_id = path.name.removesuffix(".json.gz")
        game = _load_game(path)
        if game is None:  # in-flight write; next pass will pick it up
            skipped += 1
            continue
        try:
            game_moves, game_deltas, game_meta = parse_game(game_id, game)
        except ValueError as exc:
            unknown.update(_unknown_vocabulary(game))
            failures.append((game_id, str(exc)))
            quarantined += 1
            continue
        moves.extend(game_moves)
        deltas.extend(game_deltas)
        meta.append(
            {
                **game_meta,
                "options": ",".join(game_meta["options"]),
                "factions": ",".join(game_meta["factions"]),
                "final_vp": orjson.dumps(game_meta["final_vp"]).decode(),
            }
        )
        parsed += 1
        if len(meta) >= games_per_part:
            _write_part(parts_dir, part_index, moves, deltas, meta)
            part_index += 1
            parts_written += 1
            moves, deltas, meta = [], [], []

    if meta:
        _write_part(parts_dir, part_index, moves, deltas, meta)
        parts_written += 1

    _rewrite_quarantine(parts_dir, failures)
    return PassStats(parsed, quarantined, skipped, parts_written, unknown)


def _rewrite_quarantine(parts_dir: Path, failures: list[tuple[str, str]]) -> None:
    """Replace the quarantine with this pass's failures. Rewritten rather than
    appended so a game fixed by a grammar change simply stops appearing."""
    path = _quarantine_path(parts_dir)
    if not failures:
        path.unlink(missing_ok=True)
        return
    payload = b"".join(
        orjson.dumps({"game_id": game_id, "reason": reason}) + b"\n"
        for game_id, reason in sorted(failures)
    )
    path.write_bytes(payload)


def _merge_by_game(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate ``frames``, keeping every row of a game from the first
    frame that contains it.

    Deduping must happen at the *game* level, not the row level:
    ``moves``/``deltas`` carry thousands of rows per game, so a row-level
    ``unique(subset=["game_id"])`` keeps one arbitrary row per game and
    silently discards the rest of the corpus.
    """
    seen: set[str] = set()
    kept: list[pl.DataFrame] = []
    for frame in frames:
        fresh = frame.filter(~pl.col("game_id").is_in(seen))
        if fresh.height == 0:
            continue
        seen = seen | set(fresh["game_id"].unique().to_list())
        kept.append(fresh)
    if not kept:
        return frames[0].clear() if frames else pl.DataFrame()
    return pl.concat(kept, how="vertical_relaxed")


def compact(parts_dir: Path = DEFAULT_PARTS_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> None:
    """Merge parts *and* the existing canonical files into fresh canonical
    files. Existing consumers read ``moves.parquet``; until this runs, they
    see exactly the corpus they saw before, which is what keeps seeded arena
    runs and pinned gates reproducible mid-crawl."""
    for name, glob in (
        ("moves", "moves-*.parquet"),
        ("deltas", "deltas-*.parquet"),
        ("games_meta", "meta-*.parquet"),
    ):
        parts = sorted(parts_dir.glob(glob))
        if not parts:
            continue
        frames = [pl.read_parquet(p) for p in parts]
        canonical = out_dir / f"{name}.parquet"
        if canonical.exists():
            frames.insert(0, pl.read_parquet(canonical))
        merged = _merge_by_game(frames)
        merged.write_parquet(canonical)
        print(
            f"  {canonical}: {merged['game_id'].n_unique()} games, "
            f"{merged.height} rows",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true", help="loop instead of one pass")
    parser.add_argument("--interval", type=float, default=300.0, help="seconds between passes")
    parser.add_argument("--compact", action="store_true", help="merge parts into canonical files")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    args = parser.parse_args(argv)

    if args.compact:
        compact(args.parts_dir)
        return

    while True:
        stats = run_pass(args.raw_dir, args.parts_dir)
        print(stats.summary(), flush=True)
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
