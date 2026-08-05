"""Series orchestration (plan 2026-08-04-arena-baselines, Task 9)."""

from __future__ import annotations

from bgai.agents import RandomAgent
from bgai.arena.series import run_series


def test_series_is_deterministic_and_complete() -> None:
    agents = {"r1": RandomAgent("r1"), "r2": RandomAgent("r2")}
    kwargs = dict(agents=agents, base_seats=("r1", "r2", "r1", "r2"), n_tables=2, seed=17)
    a = run_series(**kwargs)
    b = run_series(**kwargs)
    assert len(a.results) == 2 * 4  # tables x rotations
    assert [r.vps for r in a.results] == [r.vps for r in b.results]
    assert a.n_errors == 0, [r.error for r in a.results if r.error]
    assert set(a.ratings) == {"r1", "r2"}
    # each rotation of a table replays the SAME sampled setup
    assert len({r.setup_game_id for r in a.results[:4]}) == 1
