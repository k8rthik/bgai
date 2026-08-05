"""max^n MCTS (Phase 6)."""

from __future__ import annotations

import random

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import decision, new_game
from bgai.arena.sim import run_game
from bgai.engine.tm.setup import load_setup
from bgai.training.model import ModelConfig, PolicyValueNet


def _agent(**kwargs) -> MCTSAgent:
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    return MCTSAgent(net=net, **kwargs)


def test_search_returns_a_legal_move_and_visits_add_up() -> None:
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    agent = _agent(simulations=16)
    choice = agent.choose_sim(sim, faction, offer, random.Random(0))
    assert choice in offer

    root, _ = agent._expand(sim)
    for _ in range(16):
        agent._simulate(root)
    assert int(root.visits.sum()) == 16 == root.total_visits


def test_value_vectors_are_absolute_seat_order() -> None:
    """Every node's vector must be re-based to the setup's seat order, or
    components would mean different players at different depths."""
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    agent = _agent(simulations=4)
    root, value = agent._expand(sim)
    assert value.shape == (4,)
    seats = sim.game.setup.factions
    # the mover's own component must land in the mover's absolute slot
    faction, offer = decision(sim)
    _, rel_value = agent._evaluate(sim.game, faction, offer)
    assert np.isclose(value[seats.index(faction)], rel_value[0])


def test_deeper_search_changes_the_choice_distribution() -> None:
    """Sanity: search must actually do something beyond the raw prior."""
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    shallow = _agent(simulations=1)
    deep = _agent(simulations=48)
    root_s, _ = shallow._expand(sim)
    for _ in range(1):
        shallow._simulate(root_s)
    root_d, _ = deep._expand(sim)
    for _ in range(48):
        deep._simulate(root_d)
    assert int(root_d.visits.max()) > int(root_s.visits.max())
    assert (root_d.visits > 0).sum() > 1, "search should explore more than one edge"


def test_mcts_plays_a_full_game_through_the_arena() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    agent = _agent(simulations=4)
    result = run_game(setup, {f: agent for f in setup.factions}, random.Random(2))
    assert result.error is None, result.error
    assert result.decisions > 50


def test_default_temperature_is_argmax_over_visits() -> None:
    """Shares a root cause with D5.6: the earlier sampling default was
    compensating for the driver's free-action loop, not for a flaw in
    argmax selection."""
    import inspect

    signature = inspect.signature(MCTSAgent.__init__)
    assert signature.parameters["temperature"].default == 0.0

    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    agent = _agent(simulations=8)
    picks = {
        agent.choose_sim(sim, faction, offer, random.Random(i)).loc for i in range(5)
    }
    assert len(picks) == 1, "argmax over visits must be deterministic"
