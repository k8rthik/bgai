"""Resolved leaves are not re-expanded.

A child that turned out to be terminal (or a zero-move engine edge, or
an engine rejection) used to be stored as ``None`` -- indistinguishable
from "never expanded" -- so every later selection of that edge re-ran
``advance`` and ``decision`` on the identical state. The value backed
up is deterministic, so remembering it changes nothing in the search
and removes the repeated engine work.
"""

from __future__ import annotations

import random

import numpy as np
import torch

import bgai.agents.mcts as mcts_module
from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import load_setup
from bgai.engine.tm.state import Phase
from bgai.training.model import ModelConfig, PolicyValueNet


def _agent(**kwargs) -> MCTSAgent:
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    return MCTSAgent(net=net, **kwargs)


def _near_end_position():
    """Walk a game with a cheap policy until round 6, so terminal leaves
    are within reach of a shallow search."""
    rng = random.Random(4)
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    while True:
        pending = decision(sim)
        assert pending is not None
        _faction, offer = pending
        if sim.game.round >= 6 and sim.game.phase is Phase.ACTIONS and len(offer) > 1:
            passes = [m for m in offer if m.verb == "pass"]
            if passes and all(
                sim.game.factions[f].passed for f in sim.game.setup.factions if f != _faction
            ):
                return sim
        # prefer passing to reach the end quickly
        choice = next((m for m in offer if m.verb == "pass"), rng.choice(offer))
        sim = advance(sim, choice, current=pending)


def test_terminal_child_is_advanced_once(monkeypatch) -> None:
    sim = _near_end_position()
    calls: list[tuple[int, object]] = []
    real_advance = mcts_module.advance

    def counting_advance(parent_sim, choice, **kwargs):
        calls.append((id(parent_sim), choice))
        return real_advance(parent_sim, choice, **kwargs)

    monkeypatch.setattr(mcts_module, "advance", counting_advance)
    for leaf_batch in (0, 4):
        agent = _agent(simulations=40, leaf_batch=leaf_batch, top_k=0)
        root = agent.search(sim)
        assert root is not None
        terminal_edges = [a for a, (kind, _v) in root.leaf_values.items() if kind == "terminal"]
        assert terminal_edges, "a pass with everyone else passed must end the game"
        for a in terminal_edges:
            assert root.children[a] is None
            assert root.visits[a] >= 2  # re-selected...
            # ...but the engine ran that edge exactly once
            assert calls.count((id(root.sim), root.offer[a])) == 1
        calls.clear()


def test_cached_leaf_value_equals_fresh_value() -> None:
    sim = _near_end_position()
    agent = _agent(simulations=40, leaf_batch=4, top_k=0)
    root = agent.search(sim)
    assert root is not None
    for action, (kind, value) in root.leaf_values.items():
        if kind != "terminal":
            continue
        child = advance(root.sim, root.offer[action], current=(root.faction, root.offer))
        assert decision(child) is None
        assert np.array_equal(value, agent._terminal_value(child))
