"""Sharded arena execution.

MCTS costs ~50s a game, so a series long enough to rank agents runs for
hours on one core. Tables are independent, so they shard across
processes -- but only once each table's setup is drawn from its own rng
rather than from one stream shared with every earlier table.
"""

from __future__ import annotations

import random

from bgai.arena.series import run_series, table_setup_rng


def test_table_setup_rng_is_independent_of_table_order() -> None:
    """The property sharding needs: table N's setup depends on (seed, N)
    alone, so a worker can draw it without replaying tables 0..N-1."""
    a = table_setup_rng(seed=5, table=3).random()
    b = table_setup_rng(seed=5, table=3).random()
    assert a == b
    assert table_setup_rng(seed=5, table=4).random() != a
    assert table_setup_rng(seed=6, table=3).random() != a


def test_table_indices_selects_a_shard(monkeypatch) -> None:
    """Running tables [2,3] must produce the same games as those tables
    produce inside a full 0..3 run."""
    from bgai.agents import GreedyAgent, RandomAgent

    agents = {"a": RandomAgent(name="a"), "b": GreedyAgent(name="b")}
    seats = ("a", "b", "a", "b")

    full = run_series(agents=agents, base_seats=seats, n_tables=4, seed=11)
    shard = run_series(
        agents=agents, base_seats=seats, n_tables=4, seed=11, table_indices=[2, 3]
    )
    assert len(shard.results) == len(full.results) // 2
    fingerprint = lambda r: (r.setup_game_id, dict(r.vps), dict(r.ranks), r.error)
    full_tail = [fingerprint(r) for r in full.results[len(full.results) // 2 :]]
    shard_all = [fingerprint(r) for r in shard.results]
    assert shard_all == full_tail


def test_seat_rotations_still_cover_every_seat(monkeypatch) -> None:
    from bgai.agents import GreedyAgent, RandomAgent

    agents = {"a": RandomAgent(name="a"), "b": GreedyAgent(name="b")}
    series = run_series(
        agents=agents, base_seats=("a", "b", "a", "b"), n_tables=1, seed=3
    )
    assert len(series.results) == 4, "one table is four mirrored rotations"
