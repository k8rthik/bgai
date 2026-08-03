"""Build normalized parquet datasets from crawled game logs.

Outputs under ``data/datasets/``:
- ``moves.parquet``  — one row per parsed command (game_id, row, seq, faction,
  kind, verb, args). The training/dataset layer consumes this, never raw text.
- ``deltas.parquet`` — one row per ledger command row: snellman's recorded
  per-resource value+delta snapshots (the engine-correctness oracle).
- ``games_meta.parquet`` — per game: options, players, factions, final VPs.

Run: ``python -m bgai.data.build_moves [raw_games_dir] [out_dir]``
Fails loudly if any command in any game is unparseable.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import orjson
import polars as pl

from bgai.data.ledger_parser import parse_commands

MOVES_SCHEMA = {
    "game_id": pl.Utf8, "row": pl.Int32, "seq": pl.Int16, "faction": pl.Utf8,
    "kind": pl.Utf8, "verb": pl.Utf8, "loc": pl.Utf8, "loc2": pl.Utf8,
    "building": pl.Utf8, "tile": pl.Utf8, "cult": pl.Utf8, "color": pl.Utf8,
    "target": pl.Utf8, "reason": pl.Utf8, "res1": pl.Utf8, "res2": pl.Utf8,
    "n1": pl.Int32, "n2": pl.Int32,
}

_RESOURCE_KEYS = ("VP", "C", "W", "P")


def _delta_row(game_id: str, row_idx: int, faction: str, row: dict) -> dict:
    out: dict = {"game_id": game_id, "row": row_idx, "faction": faction}
    for key in _RESOURCE_KEYS:
        cell = row.get(key) or {}
        out[f"{key.lower()}_value"] = cell.get("value")
        out[f"{key.lower()}_delta"] = cell.get("delta")
    out["pw"] = (row.get("PW") or {}).get("value")
    out["cult"] = (row.get("CULT") or {}).get("value")
    return out


def parse_game(game_id: str, game: dict) -> tuple[list[dict], list[dict], dict]:
    """Returns (move_rows, delta_rows, meta_row); raises on unparseable commands."""
    moves: list[dict] = []
    deltas: list[dict] = []
    for row_idx, row in enumerate(game.get("ledger", [])):
        commands = row.get("commands")
        if not commands:
            continue
        faction = row.get("faction", "")
        parsed, unknown = parse_commands(commands)
        if unknown:
            raise ValueError(f"{game_id} row {row_idx}: unparseable {unknown!r}")
        moves.extend(
            {
                "game_id": game_id, "row": row_idx, "seq": seq, "faction": faction,
                "kind": p.kind.value, "verb": p.verb, "loc": p.loc, "loc2": p.loc2,
                "building": p.building, "tile": p.tile, "cult": p.cult,
                "color": p.color, "target": p.target, "reason": p.reason,
                "res1": p.res1, "res2": p.res2, "n1": p.n1, "n2": p.n2,
            }
            for seq, p in enumerate(parsed)
        )
        deltas.append(_delta_row(game_id, row_idx, faction, row))

    factions = game.get("factions") or {}
    players = game.get("players") or []
    meta = {
        "game_id": game_id,
        "options": sorted((game.get("options") or {}).keys()),
        "player_count": len(players),
        "factions": sorted(factions.keys()),
        "final_vp": {f: (v or {}).get("VP") for f, v in factions.items()},
    }
    return moves, deltas, meta


def build(raw_dir: Path, out_dir: Path) -> None:
    all_moves: list[dict] = []
    all_deltas: list[dict] = []
    all_meta: list[dict] = []
    paths = sorted(raw_dir.glob("*.json.gz"))
    for i, path in enumerate(paths):
        game_id = path.name.removesuffix(".json.gz")
        game = orjson.loads(gzip.decompress(path.read_bytes()))
        moves, deltas, meta = parse_game(game_id, game)
        all_moves.extend(moves)
        all_deltas.extend(deltas)
        all_meta.append(
            {**meta, "options": ",".join(meta["options"]),
             "factions": ",".join(meta["factions"]),
             "final_vp": orjson.dumps(meta["final_vp"]).decode()}
        )
        if (i + 1) % 500 == 0:
            print(f"  parsed {i + 1}/{len(paths)} games", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    moves_df = pl.DataFrame(all_moves, schema=MOVES_SCHEMA)
    moves_df.write_parquet(out_dir / "moves.parquet")
    pl.DataFrame(all_deltas).write_parquet(out_dir / "deltas.parquet")
    pl.DataFrame(all_meta).write_parquet(out_dir / "games_meta.parquet")
    print(f"games: {len(all_meta)}, moves: {moves_df.height}, "
          f"decisions: {moves_df.filter(pl.col('kind') == 'decision').height}")
    print(f"wrote {out_dir}/moves.parquet, deltas.parquet, games_meta.parquet")


if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/games")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/datasets")
    build(raw, out)
