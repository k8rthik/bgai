"""Master-plan arena gates (slow).

Phase 4: greedy >> random over >= 200 games -- 50 corpus-sampled tables
x 4 mirrored rotations, base seats (greedy, random, greedy, random).
Criteria: (1) zero errored games; (2) greedy's mean placement beats
random's by >= 0.4 ranks; (3) conservative(greedy) > conservative(random).

Phase 5: the trained imitation net beats BOTH baselines in mirrored
play. Skips with a clear message until a checkpoint exists.
"""

from __future__ import annotations

from pathlib import Path

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


CHECKPOINT = Path("data/checkpoints/imitation/checkpoint.pt")


@pytest.mark.slow
@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"no trained checkpoint at {CHECKPOINT} (run bgai.training.train first)",
)
def test_imitation_beats_greedy_and_random() -> None:
    """Phase 5 gate: the imitation net must beat both Phase 4 baselines
    in mirrored play (master plan Phase 5 verify clause).
    """
    from bgai.agents.imitation import ImitationAgent

    agents = {
        "imitation": ImitationAgent(CHECKPOINT, name="imitation"),
        "greedy": GreedyAgent(name="greedy"),
        "random": RandomAgent(name="random"),
    }
    series = run_series(
        agents=agents,
        base_seats=("imitation", "greedy", "imitation", "random"),
        n_tables=25,
        seed=20260805,
    )
    assert series.n_errors == 0, [r.error for r in series.results if r.error]
    mean_rank = {}
    for name in agents:
        ranks = [r.ranks[f] for r in series.results for f, a in r.seats.items() if a == name]
        mean_rank[name] = sum(ranks) / len(ranks)
    assert mean_rank["imitation"] + 0.15 <= mean_rank["greedy"], mean_rank
    assert mean_rank["imitation"] + 0.5 <= mean_rank["random"], mean_rank
