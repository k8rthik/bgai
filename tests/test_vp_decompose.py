"""VP-source decomposition (knowledge/vp_decompose.py).

The load-bearing property is conservation: every VP a faction finishes
with must be attributed to exactly one source bucket. ``decompose``
asserts it internally per faction; these tests drive whole games through
the tracer so the assert runs against real engine behaviour, then check
the buckets are individually sane.
"""

from __future__ import annotations

import random

import pytest

from bgai.agents import GreedyAgent, RandomAgent
from bgai.arena.setups import sample_setup
from bgai.knowledge.vp_decompose import aggregate, decompose, trace_game


def _play(seed: int, agent_cls=RandomAgent):
    rng = random.Random(seed)
    setup = sample_setup(rng)
    seats = {f: agent_cls(name=f"a{i}") for i, f in enumerate(setup.factions)}
    return trace_game(setup, seats, rng)


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_every_vp_is_attributed_exactly_once(seed: int) -> None:
    traced = _play(seed)
    assert traced.error is None, traced.error
    buckets = decompose(traced)  # internal assert: buckets sum to final-20
    assert set(buckets) == set(traced.final_vp)


def test_bucket_signs_and_endgame_presence() -> None:
    traced = _play(3, agent_cls=GreedyAgent)
    assert traced.error is None, traced.error
    buckets = decompose(traced)
    for faction, sources in buckets.items():
        assert sources.get("leech", 0) <= 0, (faction, sources)
        assert sources.get("endgame_network", 0) >= 0
        assert sources.get("endgame_cult", 0) >= 0
    # someone must have a network -- network endgame VP can't all be zero
    assert any(s.get("endgame_network", 0) > 0 for s in buckets.values())


def test_aggregate_means_over_games() -> None:
    games = [decompose(_play(s)) for s in (1, 2)]
    means = aggregate(games)
    for sources in means.values():
        assert all(isinstance(v, float) for v in sources.values())
