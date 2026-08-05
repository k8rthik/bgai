"""Self-driving simulation loop (plan 2026-08-04-arena-baselines, Task 4)."""

from __future__ import annotations

import random

from bgai.agents import RandomAgent
from bgai.arena.sim import GameResult, run_game
from bgai.engine.tm.setup import load_setup


def _seats(setup, agents):
    return {faction: agents[i % len(agents)] for i, faction in enumerate(setup.factions)}


def test_random_game_runs_to_completion() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    seats = _seats(setup, [RandomAgent("r1"), RandomAgent("r2")])
    result = run_game(setup, seats, random.Random(11))
    assert isinstance(result, GameResult)
    assert result.error is None, result.error
    assert set(result.vps) == set(setup.factions)
    assert sorted(result.ranks.values())[0] == 0
    assert result.decisions < 5000


def test_same_seed_same_result() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    seats = _seats(setup, [RandomAgent("r1"), RandomAgent("r2")])
    a = run_game(setup, seats, random.Random(5))
    b = run_game(setup, seats, random.Random(5))
    assert a.vps == b.vps and a.decisions == b.decisions


def test_five_seeds_three_setups_all_complete() -> None:
    for game_id in ("4pLeague_S10_D1L1_G1", "4pLeague_S1_D1L1_G1", "4pLeague_S1_D1L1_G2"):
        setup = load_setup(game_id)
        seats = _seats(setup, [RandomAgent()])
        for seed in range(5):
            result = run_game(setup, seats, random.Random(seed))
            assert result.error is None, f"{game_id} seed {seed}: {result.error}"
