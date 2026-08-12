"""Compaction of ingest parts into the canonical corpus files.

``moves``/``deltas`` hold many rows per game, so merging must dedupe at
the *game* level. Row-level ``unique(subset=["game_id"])`` silently
collapses each game to one row -- a 29M-row corpus became 76k.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bgai.data.ingest import compact


def _write_part(parts: Path, index: int, rows: list[tuple[str, int]]) -> None:
    frame = pl.DataFrame(
        {"game_id": [g for g, _ in rows], "row": [r for _, r in rows]}
    )
    for name in ("moves", "deltas", "meta"):
        frame.write_parquet(parts / f"{name}-{index:04d}.parquet")


def test_compact_preserves_every_row_of_a_game(tmp_path: Path) -> None:
    parts, out = tmp_path / "parts", tmp_path / "out"
    parts.mkdir()
    out.mkdir()
    _write_part(parts, 0, [("g1", 1), ("g1", 2), ("g1", 3), ("g2", 1)])
    _write_part(parts, 1, [("g3", 1), ("g3", 2)])

    compact(parts, out)

    merged = pl.read_parquet(out / "moves.parquet")
    assert merged.height == 6, "multi-row games must survive compaction"
    assert merged.filter(pl.col("game_id") == "g1").height == 3
    assert set(merged["game_id"].unique().to_list()) == {"g1", "g2", "g3"}


def test_compact_keeps_canonical_rows_when_a_game_reappears(tmp_path: Path) -> None:
    """Existing canonical rows win over parts for the same game -- whole
    game at a time, so a re-parsed game never half-replaces the old one."""
    parts, out = tmp_path / "parts", tmp_path / "out"
    parts.mkdir()
    out.mkdir()
    pl.DataFrame({"game_id": ["g1", "g1"], "row": [10, 20]}).write_parquet(
        out / "moves.parquet"
    )
    _write_part(parts, 0, [("g1", 99), ("g2", 1)])

    compact(parts, out)

    merged = pl.read_parquet(out / "moves.parquet")
    g1 = merged.filter(pl.col("game_id") == "g1")
    assert sorted(g1["row"].to_list()) == [10, 20], "canonical g1 must be untouched"
    assert merged.filter(pl.col("game_id") == "g2").height == 1
