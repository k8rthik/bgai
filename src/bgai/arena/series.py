"""Multi-table arena orchestration (plan Task 9).

One *table* = one corpus-sampled setup played once per mirrored seat
rotation (4 games). Per-game rngs are derived deterministically from
(seed, table, rotation) so a series reproduces exactly and no game's
randomness depends on how earlier games consumed the stream.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import trueskill

from bgai.agents.base import Agent
from bgai.arena.ratings import new_ratings, update
from bgai.arena.rotation import seat_rotations
from bgai.arena.setups import sample_setup
from bgai.arena.sim import GameResult, run_game

_DEFAULT_RAW_DIR = Path("data/raw/games")


@dataclass(frozen=True)
class SeriesResult:
    results: tuple[GameResult, ...]
    ratings: Mapping[str, trueskill.Rating]
    n_errors: int


def table_setup_rng(seed: int, table: int) -> random.Random:
    """The rng that draws one table's setup.

    Keyed on ``(seed, table)`` rather than advanced through a single
    stream, so table N's setup no longer depends on tables 0..N-1 having
    been drawn first. That independence is what lets tables run in
    separate processes and still reproduce a single-process series.
    """
    return random.Random(f"{seed}:setups:{table}")


def run_series(
    agents: Mapping[str, Agent],
    base_seats: tuple[str, ...],
    n_tables: int,
    seed: int,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    table_indices: Sequence[int] | None = None,
) -> SeriesResult:
    """Run ``n_tables`` tables, or only ``table_indices`` of them.

    A shard's games are identical to the same tables' games in a full
    run; ratings from a shard are partial and meant to be recomputed over
    the merged results.
    """
    unknown = set(base_seats) - set(agents)
    if unknown:
        raise ValueError(f"base_seats reference unknown agents: {sorted(unknown)}")
    tables = range(n_tables) if table_indices is None else table_indices
    ratings = new_ratings(agents)
    results: list[GameResult] = []
    for table in tables:
        setup = sample_setup(table_setup_rng(seed, table), raw_dir)
        for rotation_index, rotation in enumerate(seat_rotations(base_seats)):
            seats = {
                faction: agents[rotation[seat]]
                for seat, faction in enumerate(setup.factions)
            }
            game_rng = random.Random(f"{seed}:{table}:{rotation_index}")
            result = run_game(setup, seats, game_rng)
            results.append(result)
            ratings = update(ratings, result)
    n_errors = sum(1 for r in results if r.error is not None)
    return SeriesResult(results=tuple(results), ratings=ratings, n_errors=n_errors)
