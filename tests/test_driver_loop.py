"""Driver turn protocol + full game loop (LLM-harness plan, task 4)."""

import pytest

from bgai.agents.random_agent import RandomAgent
from bgai.arena.driver import GameResult, run_game
from bgai.arena.setup_factory import fresh_setup


def _agents(setup, base_seed):
    return {
        f: RandomAgent(seed=base_seed + i, name=f"random{i}")
        for i, f in enumerate(setup.factions)
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_random_game_finishes(seed):
    setup = fresh_setup(seed=seed)
    result = run_game(setup, _agents(setup, seed))
    assert isinstance(result, GameResult)
    assert set(result.vp) == set(setup.factions)
    assert all(v >= 0 for v in result.vp.values())
    assert result.winners
    assert set(result.winners) <= set(setup.factions)
    assert max(result.vp.values()) == result.vp[result.winners[0]]


def test_full_random_game_deterministic():
    setup = fresh_setup(seed=9)
    r1 = run_game(setup, _agents(setup, 9))
    r2 = run_game(setup, _agents(setup, 9))
    assert r1 == r2


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(20))
def test_random_game_sweep(seed):
    setup = fresh_setup(seed=100 + seed)
    result = run_game(setup, _agents(setup, seed))
    assert sum(result.vp.values()) > 4 * 20  # everyone started at 20 VP


def test_scripted_setup_reaches_exact_state():
    """Both setup phases driven by explicit scripted choices (not first-legal):
    snake dwelling order (seat, reverse seat, nomads third), reverse-seat
    bonus picks, then round-1 INCOME -- asserting the exact resulting board.
    """
    from driver_helpers import ScriptedAgent

    from bgai.arena.driver import new_game, next_actor, offered_moves
    from bgai.engine.tm.apply import apply
    from bgai.engine.tm.round_flow import advance_turn
    from bgai.engine.tm.state import Phase

    setup = fresh_setup(seed=5, factions=("nomads", "darklings", "engineers", "mermaids"))
    scripts = {
        "nomads": [("build", "B4", None), ("build", "F3", None), ("build", "G4", None),
                   ("pass", None, "BON9")],
        "darklings": [("build", "B3", None), ("build", "E1", None), ("pass", None, "BON3")],
        "engineers": [("build", "F6", None), ("build", "C5", None), ("pass", None, "BON6")],
        "mermaids": [("build", "D2", None), ("build", "E4", None), ("pass", None, "BON1")],
    }
    agents = {f: ScriptedAgent(script, name=f) for f, script in scripts.items()}

    state = new_game(setup)
    while state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        actor = next_actor(state)
        move = agents[actor].choose(state, actor, offered_moves(state, actor, turn_open=False))
        state = apply(state, actor, move)
        state = advance_turn(state)

    assert state.phase is Phase.INCOME
    assert state.round == 1
    assert sorted(state.factions["nomads"].buildings["D"]) == ["B4", "F3", "G4"]
    assert sorted(state.factions["darklings"].buildings["D"]) == ["B3", "E1"]
    assert sorted(state.factions["engineers"].buildings["D"]) == ["C5", "F6"]
    assert sorted(state.factions["mermaids"].buildings["D"]) == ["D2", "E4"]
    assert state.factions["nomads"].bonus == "BON9"
    assert state.factions["mermaids"].bonus == "BON1"
    assert all(not a._script for a in agents.values())  # every scripted move consumed
