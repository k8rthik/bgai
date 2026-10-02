"""The PettingZoo AEC environment."""

from __future__ import annotations

import random

import numpy as np
import pytest

from bgai.env.aec import TerraMysticaAECEnv, env as wrapped_env, raw_env
from bgai.env.config import (
    ILLEGAL_ACTION_REWARD,
    MAX_PLAYERS,
    MIN_PLAYERS,
    EnvConfig,
    EnvStateError,
    IllegalActionError,
    IllegalActionPolicy,
    RewardMode,
)

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _play(env: TerraMysticaAECEnv, seed: int) -> dict[str, float]:
    """Random masked play to the end; returns each agent's return."""
    rng = random.Random(seed)
    env.reset(seed=seed)
    returns: dict[str, float] = dict.fromkeys(env.agents, 0.0)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        returns[agent] += reward
        if terminated or truncated:
            env.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        assert len(legal) > 0
        env.step(int(legal[rng.randrange(len(legal))]))
    return returns


@pytest.mark.parametrize("count", COUNTS)
def test_full_games_run_to_termination(count: int) -> None:
    env = raw_env(EnvConfig(player_count=count))
    returns = _play(env, 2026)
    assert env.agents == []
    assert env.episode.sim.finished
    assert env.episode.engine_error is None
    assert len(returns) == count
    assert sum(returns.values()) == pytest.approx(0.0, abs=1e-9)


def test_agent_ids_follow_the_recommended_shape() -> None:
    env = raw_env(EnvConfig(player_count=5))
    assert env.possible_agents == [f"player_{i}" for i in range(5)]
    env.reset(seed=1)
    assert env.agents == env.possible_agents
    assert env.num_agents == 5


def test_agent_and_faction_map_both_ways() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=1)
    for index, agent in enumerate(env.possible_agents):
        faction = env.faction_of(agent)
        assert faction == env.factions[index]
        assert env.agent_of(faction) == agent


def test_seat_identity_is_unavailable_before_reset() -> None:
    env = raw_env(EnvConfig(player_count=4))
    with pytest.raises(EnvStateError, match="reset"):
        env.faction_of("player_0")
    with pytest.raises(EnvStateError, match="not been reset"):
        env.episode


def test_unknown_agent_is_rejected() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=1)
    with pytest.raises(KeyError):
        env.observe("player_9")
    with pytest.raises(KeyError, match="not at this table"):
        env.agent_of("not_a_faction")


def test_spaces_are_the_same_object_every_call() -> None:
    """``api_test`` asserts this -- space seeding depends on it."""
    env = raw_env(EnvConfig(player_count=4))
    for agent in env.possible_agents:
        assert env.observation_space(agent) is env.observation_space(agent)
        assert env.action_space(agent) is env.action_space(agent)


def test_observation_is_in_its_declared_space() -> None:
    env = raw_env(EnvConfig(player_count=3))
    env.reset(seed=4)
    for agent in env.agents:
        space = env.observation_space(agent)
        assert space.contains(env.observe(agent))


def test_agent_selection_follows_the_engines_turn_order() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=5)
    rng = random.Random(5)
    for _ in range(40):
        acting = env.episode.acting_seat()
        assert env.agent_selection == env.agent_of(acting)
        obs = env.observe(env.agent_selection)
        legal = np.flatnonzero(obs["action_mask"])
        env.step(int(legal[rng.randrange(len(legal))]))


def test_only_the_acting_agent_has_a_live_mask() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=6)
    for agent in env.agents:
        mask = env.observe(agent)["action_mask"]
        assert mask.dtype == np.int8
        assert bool(mask.any()) == (agent == env.agent_selection)


def test_last_returns_the_accumulated_reward() -> None:
    env = raw_env(EnvConfig(player_count=2, reward_mode=RewardMode.DENSE_VP))
    returns = _play(env, 7)
    assert any(value != 0.0 for value in returns.values())


def test_illegal_action_terminates_everyone_with_a_penalty() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=8)
    offender = env.agent_selection
    env.step(10_000)
    assert all(env.terminations.values())
    assert env.rewards[offender] == ILLEGAL_ACTION_REWARD
    assert env.infos[offender]["illegal_action"] == 10_000


def test_strict_policy_raises_on_an_illegal_action() -> None:
    env = raw_env(
        EnvConfig(player_count=4, illegal_action=IllegalActionPolicy.RAISE)
    )
    env.reset(seed=8)
    with pytest.raises(IllegalActionError):
        env.step(10_000)


def test_none_action_for_a_live_agent_is_rejected() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=9)
    with pytest.raises(ValueError, match="must be an index, not None"):
        env.step(None)


def test_truncation_budget_ends_the_episode() -> None:
    env = raw_env(EnvConfig(player_count=4, max_decisions=6))
    _play(env, 10)
    assert env.agents == []
    assert env.episode.decisions == 6
    assert not env.episode.sim.finished


def test_reset_with_no_seed_still_produces_a_table() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset()
    assert env.episode.seed >= 0
    assert len(env.factions) == 4


def test_info_carries_faction_and_game_context() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=11)
    info = env.infos[env.agent_selection]
    assert info["faction"] in env.factions
    assert info["seed"] == 11
    assert info["game_id"].startswith("tmenv_4p_")
    assert info["phase"] == env.episode.game.phase.name


def test_state_is_the_seat_zero_global_vector() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=12)
    state = env.state()
    assert np.array_equal(state, env.episode.observe(env.factions[0]).globals)


def test_snapshot_restore_resumes_the_same_game() -> None:
    env = raw_env(EnvConfig(player_count=4))
    env.reset(seed=13)
    rng = random.Random(13)
    for _ in range(35):
        obs = env.observe(env.agent_selection)
        legal = np.flatnonzero(obs["action_mask"])
        env.step(int(legal[rng.randrange(len(legal))]))
    payload = env.snapshot()
    actions = [rng.randrange(1000) for _ in range(10)]

    def continue_from(target: TerraMysticaAECEnv) -> list[int]:
        seen = []
        for raw_action in actions:
            if target.episode.finished:
                break
            obs = target.observe(target.agent_selection)
            legal = np.flatnonzero(obs["action_mask"])
            choice = int(legal[raw_action % len(legal)])
            seen.append(choice)
            target.step(choice)
        return seen

    first = continue_from(env)
    other = raw_env(EnvConfig(player_count=4))
    other.restore(payload)
    assert other.agent_selection is not None
    assert other.episode.pending() is not None
    second = continue_from(other)
    assert first == second
    assert other.episode.vps() == env.episode.vps()


def test_restore_refuses_a_table_of_a_different_size() -> None:
    small = raw_env(EnvConfig(player_count=2))
    small.reset(seed=14)
    big = raw_env(EnvConfig(player_count=5))
    with pytest.raises(ValueError, match="2-player table"):
        big.restore(small.snapshot())


def test_restore_of_a_finished_table_reports_termination() -> None:
    env = raw_env(EnvConfig(player_count=2))
    payload_env = raw_env(EnvConfig(player_count=2))
    _play(payload_env, 15)
    env.restore(payload_env.episode.to_bytes())
    assert all(env.terminations.values())
    assert not any(env.truncations.values())


def test_render_modes_are_declared_and_validated() -> None:
    assert TerraMysticaAECEnv.metadata["name"] == "terra_mystica_v0"
    with pytest.raises(ValueError, match="render_mode"):
        raw_env(EnvConfig(player_count=4), render_mode="rgb_array")
    quiet = raw_env(EnvConfig(player_count=4))
    quiet.reset(seed=16)
    assert quiet.render() is None
    loud = raw_env(EnvConfig(player_count=4), render_mode="ansi")
    loud.reset(seed=16)
    line = loud.render()
    assert line is not None and "act=" in line
    loud.close()


def test_the_wrapped_env_is_playable() -> None:
    """``env()`` adds PettingZoo's assertion wrappers; it must still play."""
    env = wrapped_env(EnvConfig(player_count=2))
    rng = random.Random(17)
    env.reset(seed=17)
    for agent in env.agent_iter():
        obs, _, terminated, truncated, _ = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        env.step(int(legal[rng.randrange(len(legal))]))
    assert env.agents == []
