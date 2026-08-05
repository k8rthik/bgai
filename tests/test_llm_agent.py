"""LLM track: serialization, provider abstraction, ladder rungs L0-L2.

Everything here runs with MockProvider -- no API key, no spend. The
plumbing of each rung is verified offline; only the *measurement* of a
rung's playing strength needs credentials.
"""

from __future__ import annotations

import random

from bgai.agents.llm_agent import LLMAgent
from bgai.arena.driver import decision, new_game
from bgai.arena.sim import run_game
from bgai.engine.tm.setup import load_setup
from bgai.llm.provider import Message, MockProvider
from bgai.llm.serialize import neighbors, serialize_offer, serialize_state


def _position():
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    return sim, faction, offer


def test_state_serialization_is_mover_relative_and_complete() -> None:
    sim, faction, _ = _position()
    text = serialize_state(sim.game, faction)
    assert text.splitlines()[0].startswith("Terra Mystica, 4 players, round 0/6")
    assert "YOU (" + faction in text
    other = [f for f in sim.game.setup.factions if f != faction][0]
    assert other in text
    assert "round scoring tiles:" in text
    # a different mover produces a different rendering
    assert serialize_state(sim.game, other) != text


def test_offer_is_numbered_from_zero() -> None:
    _, _, offer = _position()
    lines = serialize_offer(offer).splitlines()
    assert len(lines) == len(offer)
    assert lines[0].startswith("0. ")
    assert lines[-1].startswith(f"{len(offer) - 1}. ")


def test_neighbors_tool_matches_the_board() -> None:
    adj = neighbors("A1")
    assert adj and all(isinstance(h, str) for h in adj)
    assert "A1" not in adj


def test_l0_agent_picks_the_index_the_model_returns() -> None:
    sim, faction, offer = _position()
    agent = LLMAgent(provider=MockProvider(replies=["3"]), level=0)
    assert agent.choose(sim.game, faction, offer, random.Random(0)) == offer[3]
    assert agent.fallbacks == 0


def test_malformed_answers_fall_back_without_illegal_moves() -> None:
    sim, faction, offer = _position()
    agent = LLMAgent(provider=MockProvider(replies=["I would build somewhere nice"]), level=0)
    choice = agent.choose(sim.game, faction, offer, random.Random(0))
    assert choice in offer
    assert agent.fallbacks == 1

    huge = LLMAgent(provider=MockProvider(replies=["9999"]), level=0)
    assert huge.choose(sim.game, faction, offer, random.Random(0)) in offer
    assert huge.fallbacks == 1


def test_l1_prompt_carries_knowledge_and_faction_identity() -> None:
    sim, faction, offer = _position()
    seen: list[str] = []

    def responder(messages):
        seen.append(messages[0].content)
        return "1"

    agent = LLMAgent(provider=MockProvider(responder=responder), level=1)
    agent.choose(sim.game, faction, offer, random.Random(0))
    assert "Terra Mystica essentials" in seen[0]
    assert faction in seen[0]


def test_l2_tool_call_is_answered_by_the_real_engine() -> None:
    sim, faction, offer = _position()
    replies = iter(["TOOL apply 0", "0"])
    transcript: list[Message] = []

    def responder(messages):
        transcript.extend(messages[-1:])
        return next(replies)

    agent = LLMAgent(provider=MockProvider(responder=responder), level=2)
    choice = agent.choose_sim(sim, faction, offer, random.Random(0))
    assert choice == offer[0]
    assert agent.tool_calls == 1
    # the tool's answer is a real post-move state, not a hallucination
    tool_reply = transcript[-1].content
    assert "after build" in tool_reply
    assert "Terra Mystica, 4 players" in tool_reply


def test_llm_agent_completes_a_full_arena_game() -> None:
    """Protocol conformance: a model that always answers "0" must still
    produce a legal, terminating game."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    agent = LLMAgent(provider=MockProvider(replies=["0"]), level=1)
    result = run_game(setup, {f: agent for f in setup.factions}, random.Random(0))
    assert result.error is None, result.error
    assert result.decisions > 50
