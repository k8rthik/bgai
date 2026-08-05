"""Agent protocol + RandomAgent baseline (LLM-harness plan, task 1)."""

from bgai.agents.base import AUX_VERBS, progress_moves
from bgai.agents.random_agent import RandomAgent
from bgai.engine.tm.legal_shared import cmd


def test_aux_verbs_cover_the_endless_exchange_moves():
    assert frozenset({"convert", "burn", "wait"}) == AUX_VERBS


def test_progress_moves_filters_aux_verbs():
    moves = (
        cmd("convert", res1="PW", n1=1, res2="C", n2=1),
        cmd("build", loc="A1"),
        cmd("burn", n1=1),
        cmd("wait"),
    )
    assert tuple(m.verb for m in progress_moves(moves)) == ("build",)


def test_progress_moves_falls_back_when_only_aux():
    moves = (cmd("convert", res1="PW", n1=1, res2="C", n2=1), cmd("wait"))
    assert progress_moves(moves) == moves


def test_random_agent_is_deterministic_per_seed():
    moves = tuple(cmd("build", loc=f"A{i}") for i in range(1, 9))
    picks_a = [RandomAgent(seed=7).choose(None, "nomads", moves) for _ in range(5)]
    picks_b = [RandomAgent(seed=7).choose(None, "nomads", moves) for _ in range(5)]
    assert picks_a == picks_b


def test_random_agent_never_picks_aux_when_progress_exists():
    moves = (cmd("convert", res1="PW", n1=1, res2="C", n2=1), cmd("pass", tile="BON1"))
    agent = RandomAgent(seed=1)
    assert all(agent.choose(None, "nomads", moves).verb == "pass" for _ in range(20))


def test_random_agent_raises_on_empty_moves():
    import pytest

    with pytest.raises(ValueError, match="no legal moves"):
        RandomAgent(seed=1).choose(None, "nomads", ())
