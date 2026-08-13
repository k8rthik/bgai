"""Batched leaf evaluation for MCTS.

Search runs one network forward pass per simulation, so a 512-sim move
costs 512 sequential passes of a 3.8M-parameter net -- which is why a
120-game 512-sim series takes over an hour. Evaluating leaves in batches
is the throughput fix.

Correctness bar: a batched evaluation must return exactly what the
one-at-a-time path returns. Candidate sets are ragged (mean 23, max 216),
so the batch has to pad and mask without letting padding leak into the
priors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bgai.agents.mcts import MCTSAgent
from bgai.arena.setups import load_setup
from bgai.arena.driver import SimState, advance, decision, new_game

CKPT = Path("data/checkpoints/snap/simplex_best.pt")


@pytest.fixture(scope="module")
def agent() -> MCTSAgent:
    if not CKPT.exists():
        pytest.skip(f"no checkpoint at {CKPT}")
    return MCTSAgent(checkpoint_path=CKPT, device="cpu", simulations=8)


def _positions(n: int) -> list[SimState]:
    """A few distinct positions from one game's opening."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    sim = new_game(setup)
    out = []
    while len(out) < n:
        pending = decision(sim)
        if pending is None:
            break
        out.append(sim)
        faction, offer = pending
        sim = advance(sim, offer[0])
    return out


def test_batched_matches_one_at_a_time(agent: MCTSAgent) -> None:
    sims = _positions(6)
    items = []
    for s in sims:
        pending = decision(s)
        if pending is None:
            continue
        faction, offer = pending
        items.append((s.game, faction, offer))
    assert len(items) >= 3, "need several positions to make batching meaningful"

    singles = [agent._evaluate(g, f, o) for g, f, o in items]
    batched = agent._evaluate_many(items)

    assert len(batched) == len(singles)
    for (p1, v1), (p2, v2) in zip(singles, batched):
        assert p1.shape == p2.shape, "ragged candidate sets must not be truncated"
        np.testing.assert_allclose(p1, p2, atol=1e-4)
        np.testing.assert_allclose(v1, v2, atol=1e-4)


def test_priors_stay_normalised_per_position(agent: MCTSAgent) -> None:
    """Padding must not leak probability mass into shorter candidate sets."""
    sims = _positions(5)
    items = [(s.game, *decision(s)) for s in sims if decision(s) is not None]
    for priors, _ in agent._evaluate_many(items):
        assert priors.sum() == pytest.approx(1.0, abs=1e-4)


def test_empty_batch_is_allowed(agent: MCTSAgent) -> None:
    assert agent._evaluate_many([]) == []


def test_batch_of_one_matches_sequential_search(agent: MCTSAgent) -> None:
    """Batching changes *when* leaves are evaluated, not what search does.
    At batch=1 there is no virtual-loss divergence, so the chosen move
    must be identical to the sequential path."""
    import random

    setup = load_setup("4pLeague_S10_D1L1_G1")
    sim = new_game(setup)
    faction, offer = decision(sim)

    agent.leaf_batch = 1
    a = agent.choose_sim(sim, faction, offer, random.Random(0))
    agent.leaf_batch = 0  # sequential
    b = agent.choose_sim(sim, faction, offer, random.Random(0))
    assert a == b


def test_virtual_loss_makes_concurrent_descents_diverge(agent: MCTSAgent) -> None:
    """Without it, every descent follows the same PUCT-optimal path to the
    same leaf and batching buys nothing."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    sim = new_game(setup)
    faction, offer = decision(sim)
    root, _ = agent._expand(sim)

    pending = agent._descend_batch(root, 6)
    actions = [action for _path, _node, action, _sim in pending]
    assert len(set(actions)) > 1, "virtual loss must spread descents across moves"


def test_batched_search_returns_a_legal_move(agent: MCTSAgent) -> None:
    import random

    setup = load_setup("4pLeague_S10_D1L1_G1")
    sim = new_game(setup)
    faction, offer = decision(sim)
    agent.leaf_batch = 8
    move = agent.choose_sim(sim, faction, offer, random.Random(1))
    assert move in offer
