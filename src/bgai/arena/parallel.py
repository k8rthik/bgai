"""Run an arena series across processes.

An MCTS game costs ~50 s, so ranking four agents over enough tables to
mean anything is hours on one core. Tables are independent once each
draws its setup from ``table_setup_rng``, so they shard cleanly.

Workers build their own agents from the spec string rather than
receiving constructed ones: agents hold torch modules, and shipping
those through pickle for every task is both slow and fragile. Each
worker pays the checkpoint load once, in an initializer.

Ratings are *not* merged from the shards -- TrueSkill updates are
order-dependent, so partial ratings do not combine. The merged game
results are re-rated once, in table order, which reproduces the
single-process ranking exactly.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Sequence

from bgai.arena.ratings import new_ratings, update
from bgai.arena.series import SeriesResult, run_series
from bgai.arena.sim import GameResult

_WORKER_STATE: dict[str, object] = {}


def _init_worker(spec: str, raw_dir: str) -> None:
    from bgai.arena.run import _build_agents

    agents = _build_agents(spec)
    _WORKER_STATE["agents"] = agents
    _WORKER_STATE["raw_dir"] = Path(raw_dir)


def _run_tables(
    args: tuple[tuple[str, ...], int, Sequence[int]]
) -> list[tuple[int, tuple[GameResult, ...]]]:
    """Returns each table's games tagged with its index.

    The tag is what lets the caller restore table order before rating;
    without it the merged list is shard-major and TrueSkill -- which is
    order-dependent -- produces a different ranking than a sequential run.
    """
    base_seats, seed, tables = args
    agents = _WORKER_STATE["agents"]
    out: list[tuple[int, tuple[GameResult, ...]]] = []
    for table in tables:
        series = run_series(
            agents=agents,  # type: ignore[arg-type]
            base_seats=base_seats,
            n_tables=0,
            seed=seed,
            raw_dir=_WORKER_STATE["raw_dir"],  # type: ignore[arg-type]
            table_indices=[table],
        )
        out.append((table, series.results))
    return out


def _chunk(n_tables: int, workers: int) -> list[list[int]]:
    """Round-robin so every shard mixes early and late tables -- a
    contiguous split would hand one worker a run of unusually long games
    and leave the rest idle at the end."""
    shards: list[list[int]] = [[] for _ in range(workers)]
    for table in range(n_tables):
        shards[table % workers].append(table)
    return [s for s in shards if s]


def run_series_parallel(
    spec: str,
    base_seats: tuple[str, ...],
    n_tables: int,
    seed: int,
    workers: int,
    raw_dir: Path = Path("data/raw/games"),
) -> SeriesResult:
    shards = _chunk(n_tables, max(1, workers))
    tasks = [(base_seats, seed, tables) for tables in shards]
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=len(tasks), initializer=_init_worker, initargs=(spec, str(raw_dir))
    ) as pool:
        collected = pool.map(_run_tables, tasks)

    # Restore table order before rating. TrueSkill is order-dependent, so
    # rating the shard-major concatenation would rank agents differently
    # from a sequential run over the same games.
    tagged = sorted(
        (entry for shard in collected for entry in shard), key=lambda e: e[0]
    )
    by_table: list[GameResult] = [g for _table, games in tagged for g in games]
    from bgai.arena.run import _build_agents

    ratings = new_ratings(_build_agents(spec))
    for result in by_table:
        ratings = update(ratings, result)
    n_errors = sum(1 for r in by_table if r.error is not None)
    return SeriesResult(results=tuple(by_table), ratings=ratings, n_errors=n_errors)
