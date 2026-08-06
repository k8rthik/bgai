"""max^n MCTS (Phase 6)."""

from __future__ import annotations

import random

import numpy as np
import torch

from bgai.agents.leaf_eval import computed_value
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


def test_default_value_blend_w_is_zero_and_leaves_expand_unchanged() -> None:
    """D6.9 diagnostic A must be default-off: nothing changes until a
    sweep result justifies otherwise."""
    import inspect

    signature = inspect.signature(MCTSAgent.__init__)
    assert signature.parameters["value_blend_w"].default == 0.0

    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    unblended = _agent(simulations=4)
    blended_off = _agent(simulations=4, value_blend_w=0.0)
    _, v1 = unblended._expand(sim)
    _, v2 = blended_off._expand(sim)
    np.testing.assert_array_equal(v1, v2)


def test_value_blend_w_one_replaces_the_leaf_value_with_the_engine_projection() -> None:
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    agent = _agent(simulations=4, value_blend_w=1.0)
    _, value = agent._expand(sim)
    np.testing.assert_allclose(value, computed_value(sim.game))


def test_value_blend_w_half_moves_the_leaf_value_toward_the_engine_projection() -> None:
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    unblended = _agent(simulations=4, value_blend_w=0.0)
    half = _agent(simulations=4, value_blend_w=0.5)
    _, learned = unblended._expand(sim)
    _, mixed = half._expand(sim)
    expected = 0.5 * learned + 0.5 * computed_value(sim.game)
    np.testing.assert_allclose(mixed, expected, rtol=1e-5)
