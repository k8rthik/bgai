"""PUCT selection without per-call numpy overhead.

``_select`` runs ~540k times per self-play game on arrays of ~23
entries -- far below the size where numpy pays for itself. The scalar
path must pick exactly what the vectorised formulation picks, including
tie-breaking (lowest index) and the unvisited-edge fill (mean own-seat Q
over visited edges, 0.25 when nothing is visited).
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent, _Node
from bgai.arena.driver import decision, new_game
from bgai.arena.setups import load_setup
from bgai.training.model import ModelConfig, PolicyValueNet


def _reference_select(agent: MCTSAgent, node: _Node) -> int:
    """The original vectorised formulation, kept verbatim as the oracle."""
    seat = node.sim.game.setup.factions.index(node.faction)
    visits = node.visits
    q = np.zeros(len(node.offer), dtype=np.float32)
    visited = visits > 0
    q[visited] = node.action_value[visited, seat] / visits[visited]
    if visited.any():
        q[~visited] = q[visited].mean()
    else:
        q[:] = 0.25
    u = agent.c_puct * node.priors * math.sqrt(max(node.total_visits, 1)) / (1 + visits)
    score = q + u
    allowed = agent._allowed(node)
    if allowed is not None:
        masked = np.full(len(node.offer), -np.inf, dtype=np.float32)
        masked[allowed] = score[allowed]
        score = masked
    return int(np.argmax(score))


def _agent(**kwargs) -> MCTSAgent:
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    return MCTSAgent(net=net, c_puct=2.5, **kwargs)


def _root(agent: MCTSAgent) -> _Node:
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    assert decision(sim) is not None
    root, _ = agent._expand(sim)
    assert root is not None
    return root


def _randomize(node: _Node, rng: random.Random) -> None:
    n = len(node.offer)
    for i in range(n):
        node.visits[i] = rng.choice([0, 0, 1, 2, 5, 17])
        node.action_value[i] = [rng.random() * node.visits[i] for _ in range(4)]
    node.total_visits = int(node.visits.sum())


def test_fast_select_matches_reference_on_random_stats() -> None:
    rng = random.Random(3)
    for top_k in (0, 4, 8):
        agent = _agent(top_k=top_k)
        root = _root(agent)
        for _ in range(200):
            _randomize(root, rng)
            assert agent._select(root) == _reference_select(agent, root)


def test_fast_select_breaks_exact_ties_on_lowest_index() -> None:
    agent = _agent(top_k=0)
    root = _root(agent)
    root.priors[:] = 1.0 / len(root.offer)
    root.priors_list = root.priors.tolist()
    root.visits[:] = 0
    root.action_value[:] = 0.0
    root.total_visits = 0
    assert agent._select(root) == 0 == _reference_select(agent, root)


def test_unvisited_edges_inherit_mean_visited_q() -> None:
    agent = _agent(top_k=0)
    root = _root(agent)
    root.priors[:] = 1.0 / len(root.offer)
    root.priors_list = root.priors.tolist()
    seat = root.seat
    root.visits[:] = 0
    root.action_value[:] = 0.0
    root.visits[0] = 4
    root.action_value[0, seat] = 0.4  # Q 0.1: every unvisited edge fills at 0.1
    root.total_visits = 4
    # edge 0 is visited (u shrinks by 1/5); an unvisited edge with equal
    # prior and the same Q wins on exploration, lowest index first
    assert agent._select(root) == 1 == _reference_select(agent, root)


def test_external_visit_edits_are_seen() -> None:
    """Tests and tree reuse poke ``visits``/``action_value`` directly; the
    fast path must read the live arrays, not a stale copy."""
    agent = _agent(top_k=3)
    root = _root(agent)
    first = agent._select(root)
    root.visits[first] += 10
    root.total_visits += 10
    assert agent._select(root) != first
    assert agent._select(root) == _reference_select(agent, root)
