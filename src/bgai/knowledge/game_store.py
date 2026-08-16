"""Persist traced agent games in the corpus's own parquet schema.

A ``TracedGame`` (knowledge/vp_decompose.trace_game) becomes three
tables shaped exactly like ``data/datasets/{moves,deltas,games_meta}``,
so every corpus analysis -- VP decomposition, town-tile distributions,
seat-position stats -- runs unchanged over agent games by pointing the
same polars queries at ``data/agent_games/``. No sequential re-tracing
to answer a new stats question: trace once, query forever.

Layout (append-friendly): each recording batch writes one part file per
table under ``data/agent_games/{moves,deltas,meta}/``; read with
``pl.scan_parquet("data/agent_games/moves/*.parquet")``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from bgai.knowledge.vp_decompose import TracedGame

DEFAULT_STORE = Path("data/agent_games")

_MOVE_FIELDS = (
    "loc", "loc2", "building", "tile", "cult", "color", "target",
    "reason", "res1", "res2",
)


def traced_to_tables(
    traced: TracedGame, game_id: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(moves, deltas, meta) rows for one finished traced game."""
    seats = tuple(traced.final_vp)
    moves_rows = []
    deltas_rows = []
    for row, step in enumerate(traced.steps):
        cmd = step.cmd
        moves_rows.append(
            {
                "game_id": game_id,
                "row": row,
                "seq": 0,
                "faction": step.faction,
                "kind": cmd.kind.name.lower(),
                "verb": cmd.verb,
                **{f: getattr(cmd, f, None) for f in _MOVE_FIELDS},
                "n1": cmd.n1,
                "n2": cmd.n2,
            }
        )
        for faction in seats:
            pre = step.pre.factions[faction]
            post = step.post.factions[faction]
            if (
                post.vp == pre.vp
                and post.coins == pre.coins
                and post.workers == pre.workers
                and post.priests == pre.priests
            ):
                continue
            deltas_rows.append(
                {
                    "game_id": game_id,
                    "row": row,
                    "faction": faction,
                    "vp_value": post.vp,
                    "vp_delta": post.vp - pre.vp,
                    "c_value": post.coins,
                    "c_delta": post.coins - pre.coins,
                    "w_value": post.workers,
                    "w_delta": post.workers - pre.workers,
                    "p_value": post.priests,
                    "p_delta": post.priests - pre.priests,
                    "pw": None,
                    "cult": None,
                }
            )
    moves_schema = {
        "game_id": pl.String, "row": pl.Int32, "seq": pl.Int16,
        "faction": pl.String, "kind": pl.String, "verb": pl.String,
        **{f: pl.String for f in _MOVE_FIELDS},
        "n1": pl.Int32, "n2": pl.Int32,
    }
    deltas_schema = {
        "game_id": pl.String, "row": pl.Int64, "faction": pl.String,
        "vp_value": pl.Int64, "vp_delta": pl.Int64,
        "c_value": pl.Int64, "c_delta": pl.Int64,
        "w_value": pl.Int64, "w_delta": pl.Int64,
        "p_value": pl.Int64, "p_delta": pl.Int64,
        "pw": pl.String, "cult": pl.String,
    }
    moves = pl.DataFrame(moves_rows, schema=moves_schema)
    deltas = pl.DataFrame(deltas_rows, schema=deltas_schema)
    meta = pl.DataFrame(
        [
            {
                "game_id": game_id,
                "options": "",
                "player_count": len(seats),
                "factions": ",".join(seats),
                "final_vp": json.dumps(dict(traced.final_vp)),
                "setup_game_id": traced.setup_game_id,
            }
        ]
    )
    return moves, deltas, meta


def append_batch(
    games: list[tuple[TracedGame, str]], store: Path = DEFAULT_STORE, batch_name: str = "batch"
) -> None:
    """Write one part file per table for a batch of traced games."""
    all_moves, all_deltas, all_meta = [], [], []
    for traced, game_id in games:
        m, d, g = traced_to_tables(traced, game_id)
        all_moves.append(m)
        all_deltas.append(d)
        all_meta.append(g)
    for sub, frames in (("moves", all_moves), ("deltas", all_deltas), ("meta", all_meta)):
        out_dir = store / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.concat(frames).write_parquet(out_dir / f"{batch_name}.parquet")
