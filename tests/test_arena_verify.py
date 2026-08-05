"""Phase 4 master-plan gate (slow): greedy >> random over >= 200 games.

50 corpus-sampled tables x 4 mirrored rotations, base seats
(greedy, random, greedy, random). Gate criteria:
(1) zero errored games;
(2) greedy's mean placement beats random's by >= 0.4 ranks;
(3) TrueSkill conservative(greedy) > conservative(random).
"""

from __future__ import annotations

import pytest

from bgai.agents import GreedyAgent, RandomAgent
from bgai.arena.ratings import conservative
from bgai.arena.series import run_series


@pytest.mark.slow
def test_greedy_beats_random_over_200_games() -> None:
    series = run_series(
        agents={"greedy": GreedyAgent(name="greedy"), "random": RandomAgent(name="random")},
        base_seats=("greedy", "random", "greedy", "random"),
        n_tables=50,
        seed=20260804,
    )
    assert series.n_errors == 0, [r.error for r in series.results if r.error]
    mean_rank = {}
    for name in ("greedy", "random"):
        ranks = [
            r.ranks[f] for r in series.results for f, a in r.seats.items() if a == name
        ]
        assert len(ranks) == 400  # 200 games x 2 seats each
        mean_rank[name] = sum(ranks) / len(ranks)
    assert mean_rank["greedy"] + 0.4 <= mean_rank["random"], mean_rank
    assert conservative(series.ratings["greedy"]) > conservative(series.ratings["random"])
