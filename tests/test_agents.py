"""Agent protocol + baseline agents (plan 2026-08-04-arena-baselines, Tasks 1/6)."""

from __future__ import annotations

import random

from bgai.agents import GreedyAgent, RandomAgent
from bgai.engine.tm.legal import legal_moves
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, active_faction


def test_random_agent_picks_from_offer_deterministically() -> None:
    offer = (cmd("pass", tile="BON1"), cmd("pass", tile="BON2"), cmd("pass", tile="BON3"))
    agent = RandomAgent()
    picks_a = [agent.choose(None, "witches", offer, random.Random(7)) for _ in range(5)]
    picks_b = [agent.choose(None, "witches", offer, random.Random(7)) for _ in range(5)]
    assert picks_a == picks_b
    assert all(p in offer for p in picks_a)


def test_random_agent_has_name() -> None:
    assert RandomAgent().name == "random"
    assert RandomAgent(name="rnd2").name == "rnd2"


def test_greedy_prefers_building_over_junk() -> None:
    """On a real mid-setup state greedy must pick a dwelling placement
    (adds a weighted building) over anything else on offer."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    faction = active_faction(state)
    offer = legal_moves(state)
    build_offers = tuple(m for m in offer if m.verb == "build")
    assert build_offers
    pick = GreedyAgent().choose(state, faction, offer, random.Random(0))
    assert pick in build_offers


def test_greedy_is_deterministic() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    faction = active_faction(state)
    offer = legal_moves(state)
    picks = {GreedyAgent().choose(state, faction, offer, random.Random(i)).loc for i in range(3)}
    assert len(picks) == 1
