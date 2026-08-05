"""Agent protocol + baseline agents (plan 2026-08-04-arena-baselines, Tasks 1/6)."""

from __future__ import annotations

import random

from bgai.agents import RandomAgent
from bgai.engine.tm.legal_shared import cmd


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
