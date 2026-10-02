"""Seeded determinism: same seed + same action sequence => identical
trajectory, in-process and across a fresh interpreter.

This is the property everything else rests on. The engine itself iterates
sets in places, so raw legal-move order varies with PYTHONHASHSEED;
``driver.canonical_moves`` is what makes an *index* reproducible, and these
tests are the check that the env has not lost it.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from bgai.env.config import MAX_PLAYERS, MIN_PLAYERS, EnvConfig, RewardMode
from bgai.env.core import Episode

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _trajectory(
    count: int, seed: int, actions: list[int], reward_mode: RewardMode
) -> dict[str, object]:
    """Replay a fixed *raw* action stream, mapped into each live offer.

    The raw stream is deliberately independent of the game, so two runs
    only agree if the offers themselves agree at every step.
    """
    config = EnvConfig(player_count=count, reward_mode=reward_mode)
    episode = Episode.start(seed, config)
    moves: list[str] = []
    factions: list[str] = []
    rewards: list[float] = []
    offer_sizes: list[int] = []
    mask_sums: list[int] = []
    for raw in actions:
        if episode.finished:
            break
        faction, offer = episode.pending()
        obs = episode.observe(faction)
        index = raw % len(offer)
        moves.append(offer[index].raw)
        factions.append(faction)
        offer_sizes.append(len(offer))
        mask_sums.append(obs.n_legal)
        episode, transition = episode.step(index)
        rewards.append(round(sum(transition.rewards.values()), 9))
    return {
        "setup": episode.setup.game_id,
        "factions_lineup": list(episode.setup.factions),
        "bonus_tiles": list(episode.setup.bonus_tiles),
        "moves": moves,
        "movers": factions,
        "offer_sizes": offer_sizes,
        "mask_sums": mask_sums,
        "rewards": rewards,
        "decisions": episode.decisions,
        "vps": episode.vps(),
        "finished": episode.sim.finished,
    }


@pytest.mark.parametrize("count", COUNTS)
def test_same_seed_same_actions_same_trajectory(count: int) -> None:
    actions = [random.Random(99).randrange(1000) for _ in range(400)]
    first = _trajectory(count, 4242, actions, RewardMode.TERMINAL_VP_SHARE)
    second = _trajectory(count, 4242, actions, RewardMode.TERMINAL_VP_SHARE)
    assert first == second
    assert first["decisions"] > 20


@pytest.mark.parametrize("count", COUNTS)
def test_different_seeds_give_different_trajectories(count: int) -> None:
    actions = [random.Random(98).randrange(1000) for _ in range(200)]
    first = _trajectory(count, 1, actions, RewardMode.TERMINAL_VP_SHARE)
    second = _trajectory(count, 2, actions, RewardMode.TERMINAL_VP_SHARE)
    assert first != second


@pytest.mark.parametrize("mode", list(RewardMode))
def test_rewards_are_deterministic_in_every_mode(mode: RewardMode) -> None:
    actions = [random.Random(97).randrange(1000) for _ in range(300)]
    assert _trajectory(4, 77, actions, mode) == _trajectory(4, 77, actions, mode)


def test_aec_env_is_deterministic_under_a_seeded_policy() -> None:
    """Through the whole PettingZoo surface, not just ``Episode``."""
    from bgai.env.aec import raw_env

    def run() -> list[tuple[str, int, float]]:
        env = raw_env(EnvConfig(player_count=4, reward_mode=RewardMode.DENSE_VP))
        env.reset(seed=515)
        rng = random.Random(515)
        log: list[tuple[str, int, float]] = []
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            if terminated or truncated:
                env.step(None)
                continue
            legal = np.flatnonzero(obs["action_mask"])
            choice = int(legal[rng.randrange(len(legal))])
            log.append((agent, choice, round(reward, 9)))
            env.step(choice)
        return log

    assert run() == run()


def test_single_env_is_deterministic_with_a_seeded_opponent() -> None:
    from bgai.env.single import legal_actions, make_single_env

    def run() -> list[tuple[int, float]]:
        env = make_single_env(player_count=4, opponent="greedy")
        obs, _ = env.reset(seed=616)
        rng = random.Random(616)
        log: list[tuple[int, float]] = []
        while True:
            legal = legal_actions(obs)
            choice = int(legal[rng.randrange(len(legal))])
            obs, reward, terminated, truncated, _ = env.step(choice)
            log.append((choice, round(reward, 9)))
            if terminated or truncated:
                return log

    assert run() == run()


_SUBPROCESS = textwrap.dedent(
    """
    import json, random, sys
    from bgai.env.config import EnvConfig, RewardMode
    from bgai.env.core import Episode

    count, seed = int(sys.argv[1]), int(sys.argv[2])
    actions = [random.Random(99).randrange(1000) for _ in range(400)]
    episode = Episode.start(seed, EnvConfig(player_count=count))
    moves, movers, sizes = [], [], []
    for raw in actions:
        if episode.finished:
            break
        faction, offer = episode.pending()
        index = raw % len(offer)
        moves.append(offer[index].raw)
        movers.append(faction)
        sizes.append(len(offer))
        episode, _ = episode.step(index)
    print(json.dumps({
        "setup": episode.setup.game_id,
        "lineup": list(episode.setup.factions),
        "moves": moves,
        "movers": movers,
        "sizes": sizes,
        "vps": episode.vps(),
        "decisions": episode.decisions,
    }))
    """
)


@pytest.mark.parametrize("count", [2, 4])
def test_trajectories_survive_a_fresh_interpreter_and_hash_seed(count: int) -> None:
    """Two subprocesses with *different* PYTHONHASHSEEDs must agree.

    The engine iterates sets while generating legal moves, so without
    ``canonical_moves`` the offer order -- and therefore the meaning of
    every action index -- would depend on hash randomization. This is the
    test that would catch losing that.
    """
    def run(hash_seed: str) -> dict[str, object]:
        environ = dict(os.environ, PYTHONHASHSEED=hash_seed)
        done = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS, str(count), "4242"],
            capture_output=True,
            text=True,
            env=environ,
            check=True,
        )
        return json.loads(done.stdout)

    first, second = run("1"), run("987654")
    assert first == second
    assert first["decisions"] > 20

    # ... and it matches what this process computes
    actions = [random.Random(99).randrange(1000) for _ in range(400)]
    local = _trajectory(count, 4242, actions, RewardMode.TERMINAL_VP_SHARE)
    assert local["moves"] == first["moves"]
    assert local["movers"] == first["movers"]
    assert local["vps"] == first["vps"]


def test_observations_are_deterministic_for_the_same_position() -> None:
    episode = Episode.start(717, EnvConfig(player_count=4))
    rng = random.Random(717)
    for _ in range(30):
        faction, offer = episode.pending()
        first = episode.observe(faction)
        second = episode.observe(faction)
        assert first == second
        episode, _ = episode.step(rng.randrange(len(offer)))
