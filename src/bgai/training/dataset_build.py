"""Shard builder CLI: corpus -> imitation training shards (plan Task 5).

    uv run python -m bgai.training.dataset_build --out data/datasets/imitation

Each shard is a compressed npz holding a fixed number of games' decision
records, with ragged candidate sets stored flat plus an offsets array.
``manifest.json`` records the encoding version, per-shard counts, the
train/val season boundary, and the division sample weights -- the loader
refuses shards whose version doesn't match the code.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from bgai.arena.setups import clean_game_ids
from bgai.training.extract import DecisionRecord, extract_game
from bgai.training.provenance import (
    DIVISION_WEIGHTS,
    VAL_SEASON_MIN,
    build_provenance,
    val_period_min,
)
from bgai.training.vocab import ENCODING_VERSION, SHARD_FORMAT

GAMES_PER_SHARD = 200

POPULATION_PATH = "data/catalogue/population.parquet"
RATINGS_PATH = "data/catalogue/player_ratings.parquet"


@dataclass(frozen=True)
class ShardStats:
    shard_count: int
    record_count: int
    train_records: int
    val_records: int
    unmatched: int
    failed_games: tuple[str, ...]


def _pack(records: list[DecisionRecord]) -> dict[str, np.ndarray]:
    cand_counts = np.array([r.candidates.shape[0] for r in records], dtype=np.int32)
    return {
        "hex_planes": np.stack([r.hex_planes for r in records]),
        "globals": np.stack([r.globals for r in records]),
        "cand_flat": np.concatenate([r.candidates for r in records]),
        "cand_counts": cand_counts,
        "chosen": np.array([r.chosen for r in records], dtype=np.int32),
        "final_vps": np.stack([r.final_vps for r in records]),
        "season": np.array([r.season for r in records], dtype=np.int16),
        "division": np.array([r.division for r in records], dtype=np.int8),
        "period": np.array([r.period for r in records], dtype=np.int32),
        "weight": np.array([r.weight for r in records], dtype=np.float32),
        "mover_faction": np.array([r.mover_faction_id for r in records], dtype=np.int8),
        "mover_mu": np.array([r.mover_mu for r in records], dtype=np.float32),
    }


def build(
    out_dir: Path,
    limit: int | None = None,
    raw_dir: Path | None = None,
    game_ids: Sequence[str] | None = None,
    part: tuple[int, int] | None = None,
) -> ShardStats:
    """Extract ``game_ids`` (default: every clean corpus game, optionally
    the first ``limit`` of them) into shards under ``out_dir``.

    ``part=(i, n)`` builds only shards whose index is ``i`` mod ``n`` and
    writes ``manifest.part{i}.json``; ``merge_parts`` joins the parts into
    the manifest a full build would have written.
    """
    moves_df = pl.read_parquet("data/datasets/moves.parquet")
    deltas_df = pl.read_parquet("data/datasets/deltas.parquet")
    meta_df = pl.read_parquet("data/datasets/games_meta.parquet")
    if game_ids is None:
        game_ids = list(clean_game_ids() if raw_dir is None else clean_game_ids(raw_dir))
    else:
        game_ids = list(game_ids)
    if limit is not None:
        game_ids = game_ids[:limit]

    provenance = build_provenance(
        game_ids,
        pl.read_parquet(POPULATION_PATH),
        pl.read_parquet(RATINGS_PATH),
    )
    uncatalogued = [g for g in game_ids if g not in provenance]
    game_ids = [g for g in game_ids if g in provenance]
    boundary = val_period_min(provenance)

    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    unmatched_total = 0
    failed: list[str] = []
    train_total = val_total = 0

    for shard_index, start in enumerate(range(0, len(game_ids), GAMES_PER_SHARD)):
        if part is not None and shard_index % part[1] != part[0]:
            continue
        batch = game_ids[start : start + GAMES_PER_SHARD]
        records: list[DecisionRecord] = []
        n_val = 0
        for game_id in batch:
            try:
                game_records, unmatched = extract_game(
                    game_id, moves_df, deltas_df, meta_df, provenance=provenance[game_id]
                )
            except Exception as exc:  # a game that cannot replay is data noise, not fatal
                failed.append(f"{game_id}: {type(exc).__name__}: {exc}")
                continue
            unmatched_total += unmatched
            records.extend(game_records)
            if provenance[game_id].is_val(boundary):
                n_val += len(game_records)
        if not records:
            continue
        path = out_dir / f"shard_{shard_index:04d}.npz"
        np.savez_compressed(path, **_pack(records))
        train_total += len(records) - n_val
        val_total += n_val
        shards.append({"file": path.name, "records": len(records), "games": len(batch)})
        print(f"{path.name}: {len(records)} records from {len(batch)} games", flush=True)

    manifest = {
        "encoding_version": ENCODING_VERSION,
        "shard_format": SHARD_FORMAT,
        "val_season_min": VAL_SEASON_MIN,
        "val_period_min": boundary,
        "division_weights": {str(k): v for k, v in DIVISION_WEIGHTS.items()},
        "games_per_shard": GAMES_PER_SHARD,
        "shards": shards,
        "record_count": train_total + val_total,
        "train_records": train_total,
        "val_records": val_total,
        "unmatched_commands": unmatched_total,
        "uncatalogued_games": uncatalogued,
        "failed_games": failed,
    }
    name = "manifest.json" if part is None else f"manifest.part{part[0]}.json"
    (out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ShardStats(
        shard_count=len(shards),
        record_count=train_total + val_total,
        train_records=train_total,
        val_records=val_total,
        unmatched=unmatched_total,
        failed_games=tuple(failed),
    )


_SUMMED = ("record_count", "train_records", "val_records", "unmatched_commands")
_CONCATENATED = ("shards", "uncatalogued_games", "failed_games")


def merge_parts(out_dir: Path) -> dict[str, object]:
    """Join ``manifest.part*.json`` into ``manifest.json`` and delete the parts."""
    paths = sorted(out_dir.glob("manifest.part*.json"))
    if not paths:
        raise FileNotFoundError(f"no manifest parts in {out_dir}")
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    fixed = {k: v for k, v in parts[0].items() if k not in _SUMMED + _CONCATENATED}
    for other in parts[1:]:
        if any(other[k] != v for k, v in fixed.items()):
            raise ValueError(f"manifest parts in {out_dir} disagree on {sorted(fixed)}")
    merged = dict(fixed)
    for key in _SUMMED:
        merged[key] = sum(p[key] for p in parts)
    for key in _CONCATENATED:
        merged[key] = [x for p in parts for x in p[key]]
    merged["shards"] = sorted(merged["shards"], key=lambda s: s["file"])
    merged["uncatalogued_games"] = sorted(set(merged["uncatalogued_games"]))
    (out_dir / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    for p in paths:
        p.unlink()
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build imitation training shards.")
    parser.add_argument("--out", type=Path, default=Path("data/datasets/imitation"))
    parser.add_argument("--limit", type=int, default=None, help="first N clean games")
    parser.add_argument("--part", default=None, metavar="I/N",
                        help="build only shards with index I mod N (run N of these, then --merge)")
    parser.add_argument("--merge", action="store_true", help="join the parts' manifests")
    args = parser.parse_args(argv)
    if args.merge:
        merged = merge_parts(args.out)
        print(f"{len(merged['shards'])} shards, {merged['record_count']} records")
        return
    part = None
    if args.part is not None:
        i, n = (int(x) for x in args.part.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--part {args.part}: need 0 <= I < N")
        part = (i, n)
    stats = build(args.out, args.limit, part=part)
    print(
        f"{stats.shard_count} shards, {stats.record_count} records "
        f"({stats.train_records} train / {stats.val_records} val), "
        f"{stats.unmatched} unmatched commands, {len(stats.failed_games)} failed games"
    )
    for failure in stats.failed_games[:10]:
        print("  failed:", failure)


if __name__ == "__main__":
    main()
