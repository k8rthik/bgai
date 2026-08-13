"""Prior-based move pruning.

Mean branching is 23 candidates (max 216), so 512 simulations spread
across every legal move leaves the tree barely two levels deep -- and a
6-round game needs to see further than that. Restricting expansion to
the top-k moves by prior spends the same budget deeper.

The risk is pruning the best move away, so the policy has to be good
enough that it lands in the top k: top1 is 58% and top3 84%, so top-8 of
23 retains the human choice the overwhelming majority of the time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import decision, new_game
from bgai.arena.setups import load_setup

CKPT = Path("data/checkpoints/snap/simplex_best.pt")


@pytest.fixture(scope="module")
def agent() -> MCTSAgent:
    if not CKPT.exists():
        pytest.skip(f"no checkpoint at {CKPT}")
    return MCTSAgent(checkpoint_path=CKPT, device="cpu", simulations=64)


def _root(agent: MCTSAgent):
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    root, _ = agent._expand(sim)
    return root, offer


def test_top_k_zero_leaves_every_move_selectable(agent: MCTSAgent) -> None:
    agent.top_k = 0
    root, offer = _root(agent)
    chosen = {agent._select(root) for _ in range(len(offer) * 3)}
    assert len(offer) > 4
    # with no pruning nothing is structurally excluded
    assert agent._allowed(root) is None


def test_top_k_restricts_to_the_highest_priors(agent: MCTSAgent) -> None:
    agent.top_k = 5
    root, offer = _root(agent)
    allowed = agent._allowed(root)
    assert allowed is not None and len(allowed) == 5
    best5 = set(np.argsort(-root.priors)[:5].tolist())
    assert set(allowed.tolist()) == best5


def test_selection_never_leaves_the_allowed_set(agent: MCTSAgent) -> None:
    agent.top_k = 4
    root, _ = _root(agent)
    allowed = set(agent._allowed(root).tolist())
    for _ in range(60):
        action = agent._select(root)
        assert action in allowed
        root.visits[action] += 1
        root.total_visits += 1


def test_top_k_larger_than_the_offer_is_a_noop(agent: MCTSAgent) -> None:
    root, offer = _root(agent)
    agent.top_k = len(offer) + 10
    assert agent._allowed(root) is None
