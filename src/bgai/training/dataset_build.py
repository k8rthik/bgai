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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    }


def build(
    out_dir: Path,
    limit: int | None = None,
    raw_dir: Path | None = None,
    game_ids: Sequence[str] | None = None,
) -> ShardStats:
    """Extract ``game_ids`` (default: every clean corpus game, optionally
    the first ``limit`` of them) into shards under ``out_dir``.
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
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ShardStats(
        shard_count=len(shards),
        record_count=train_total + val_total,
        train_records=train_total,
        val_records=val_total,
        unmatched=unmatched_total,
        failed_games=tuple(failed),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build imitation training shards.")
    parser.add_argument("--out", type=Path, default=Path("data/datasets/imitation"))
    parser.add_argument("--limit", type=int, default=None, help="first N clean games")
    args = parser.parse_args(argv)
    stats = build(args.out, args.limit)
    print(
        f"{stats.shard_count} shards, {stats.record_count} records "
        f"({stats.train_records} train / {stats.val_records} val), "
        f"{stats.unmatched} unmatched commands, {len(stats.failed_games)} failed games"
    )
    for failure in stats.failed_games[:10]:
        print("  failed:", failure)


if __name__ == "__main__":
    main()
