"""ImitationAgent (plan 2026-08-04-imitation-phase5, Task 9)."""

from __future__ import annotations

import random

import torch

from bgai.agents.imitation import ImitationAgent
from bgai.arena.sim import run_game
from bgai.engine.tm.legal import legal_moves
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, active_faction
from bgai.training.model import ModelConfig, PolicyValueNet


def _agent(temperature: float = 0.0) -> ImitationAgent:
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    return ImitationAgent(net=net, temperature=temperature)


def test_picks_from_offer_and_is_deterministic_at_zero_temperature() -> None:
    state = start_setup(GameState.initial(load_setup("4pLeague_S10_D1L1_G1")))
    faction = active_faction(state)
    offer = legal_moves(state)
    agent = _agent()
    picks = [agent.choose(state, faction, offer, random.Random(i)) for i in range(3)]
    assert all(p in offer for p in picks)
    assert len({p.raw + str(p.loc) for p in picks}) == 1


def test_temperature_sampling_stays_within_offer() -> None:
    state = start_setup(GameState.initial(load_setup("4pLeague_S10_D1L1_G1")))
    faction = active_faction(state)
    offer = legal_moves(state)
    agent = _agent(temperature=1.0)
    for seed in range(5):
        assert agent.choose(state, faction, offer, random.Random(seed)) in offer


def test_untrained_agent_completes_a_full_game() -> None:
    """The agent must satisfy the arena contract even before training."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    agent = _agent()
    result = run_game(setup, {f: agent for f in setup.factions}, random.Random(3))
    assert result.error is None, result.error
