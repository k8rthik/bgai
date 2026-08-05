"""HeuristicAgent priority policy (LLM-harness plan, task 5)."""

import random

import pytest

from bgai.agents.heuristic import HeuristicAgent, _pick
from bgai.agents.random_agent import RandomAgent
from bgai.arena.driver import run_game
from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.legal_shared import cmd

RNG = random.Random(0)


def test_leech_accept_small_decline_large():
    small = (cmd("leech", n1=2, target="nomads"), cmd("decline", target="nomads"))
    large = (cmd("leech", n1=3, target="nomads"), cmd("decline", target="nomads"))
    assert _pick(None, "engineers", small, RNG).verb == "leech"
    assert _pick(None, "engineers", large, RNG).verb == "decline"


def test_sh_upgrade_preferred_over_build():
    moves = (
        cmd("build", loc="A1"),
        cmd("upgrade", loc="B2", building="SH"),
        cmd("upgrade", loc="C3", building="TP"),
    )
    assert _pick(None, "engineers", moves, RNG).building == "SH"


def test_tp_upgrade_preferred_over_plain_build():
    moves = (cmd("build", loc="A1"), cmd("upgrade", loc="C3", building="TP"))
    assert _pick(None, "engineers", moves, RNG).building == "TP"


def test_pass_prefers_highest_coin_bonus_tile():
    moves = (
        cmd("pass", tile="BON1"),  # 2C income
        cmd("pass", tile="BON3"),  # 6C income
        cmd("pass", tile="BON2"),  # 4C income
    )
    assert _pick(None, "engineers", moves, RNG).tile == "BON3"


def test_favor_prefers_fav11():
    moves = (cmd("gain_favor", tile="FAV5"), cmd("gain_favor", tile="FAV11"))
    assert _pick(None, "engineers", moves, RNG).tile == "FAV11"


def test_town_prefers_tw3():
    moves = (cmd("gain_town", tile="TW1", n1=1), cmd("gain_town", tile="TW3", n1=1))
    assert _pick(None, "engineers", moves, RNG).tile == "TW3"


def test_heuristic_game_finishes():
    setup = fresh_setup(seed=11)
    agents = {
        setup.factions[0]: HeuristicAgent(seed=1),
        **{f: RandomAgent(seed=2 + i, name=f"random{i}")
           for i, f in enumerate(setup.factions[1:])},
    }
    result = run_game(setup, agents)
    assert set(result.vp) == set(setup.factions)


@pytest.mark.slow
def test_heuristic_beats_random_on_average():
    heuristic_vps: list[int] = []
    random_vps: list[int] = []
    for g in range(20):
        setup = fresh_setup(seed=300 + g)
        heuristic_seat = setup.factions[g % 4]
        agents = {
            f: HeuristicAgent(seed=g) if f == heuristic_seat else RandomAgent(seed=g * 7 + i)
            for i, f in enumerate(setup.factions)
        }
        result = run_game(setup, agents)
        heuristic_vps.append(result.vp[heuristic_seat])
        random_vps.extend(v for f, v in result.vp.items() if f != heuristic_seat)
    assert sum(heuristic_vps) / len(heuristic_vps) > sum(random_vps) / len(random_vps)
