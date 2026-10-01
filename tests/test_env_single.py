"""The Gymnasium single-agent wrapper and its opponent policies."""

from __future__ import annotations

import random

import numpy as np
import pytest

from bgai.agents.greedy import GreedyAgent
from bgai.agents.random_agent import RandomAgent
from bgai.env.config import (
    ILLEGAL_ACTION_REWARD,
    MAX_PLAYERS,
    MIN_PLAYERS,
    EnvConfig,
    EnvError,
    EnvStateError,
    RewardMode,
)
from bgai.env.opponents import (
    AgentPolicy,
    UnknownOpponentError,
    build_opponent,
    known_opponents,
    random_policy,
)
from bgai.env.single import (
    TerraMysticaSingleEnv,
    legal_actions,
    make_single_env,
)

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _play(env: TerraMysticaSingleEnv, seed: int) -> tuple[float, int, dict]:
    rng = random.Random(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    while True:
        legal = legal_actions(obs)
        assert len(legal) > 0
        obs, reward, terminated, truncated, info = env.step(
            int(legal[rng.randrange(len(legal))])
        )
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, info


@pytest.mark.parametrize("count", COUNTS)
def test_full_games_run_at_every_player_count(count: int) -> None:
    env = make_single_env(player_count=count, opponent="random")
    total, steps, info = _play(env, 2026)
    assert steps > 0
    assert "final_vps" in info
    assert len(info["final_vps"]) == count
    assert env.episode.sim.finished
    assert info["opponent"] == "random"
    assert info["opponent_decisions"] > 0


def test_one_step_is_one_learner_decision() -> None:
    """Opponents run inside ``step``; the learner is always the seat to move
    when ``step`` returns.
    """
    env = make_single_env(player_count=4)
    obs, _ = env.reset(seed=3)
    rng = random.Random(3)
    for _ in range(25):
        assert env.episode.acting_seat() == env.learner_faction
        legal = legal_actions(obs)
        obs, _, terminated, truncated, _ = env.step(
            int(legal[rng.randrange(len(legal))])
        )
        if terminated or truncated:
            break


def test_observation_is_in_its_declared_space() -> None:
    env = make_single_env(player_count=3)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    obs, _, _, _, _ = env.step(int(legal_actions(obs)[0]))
    assert env.observation_space.contains(obs)


def test_learner_sees_only_its_own_mask() -> None:
    env = make_single_env(player_count=4)
    obs, info = env.reset(seed=5)
    assert info["faction"] == env.learner_faction
    assert obs["action_mask"].sum() == len(env.episode.offer_for(env.learner_faction))


def test_illegal_action_terminates_with_the_penalty() -> None:
    env = make_single_env(player_count=4)
    env.reset(seed=6)
    obs, reward, terminated, truncated, info = env.step(10_000)
    assert terminated and not truncated
    assert reward == ILLEGAL_ACTION_REWARD
    assert info["illegal_action"] == 10_000
    with pytest.raises(EnvError, match="finished episode"):
        env.step(0)


def test_episode_access_before_reset_is_an_error() -> None:
    env = make_single_env(player_count=4)
    with pytest.raises(EnvStateError, match="not been reset"):
        env.episode


def test_learner_seat_is_validated() -> None:
    with pytest.raises(ValueError, match="learner_seat must be in"):
        TerraMysticaSingleEnv(EnvConfig(player_count=2), learner_seat=2)


def test_fixed_seat_stays_put_and_rotation_moves() -> None:
    fixed = make_single_env(player_count=4, learner_seat=1)
    seats = []
    for seed in range(4):
        fixed.reset(seed=seed)
        seats.append(fixed._seat_index)
    assert seats == [1, 1, 1, 1]

    rotating = make_single_env(player_count=4, learner_seat=0, rotate_seats=True)
    seats = []
    for seed in range(5):
        rotating.reset(seed=seed)
        seats.append(rotating._seat_index)
    assert seats == [0, 1, 2, 3, 0]


def test_dense_mode_pays_vp_earned_during_opponent_turns() -> None:
    """A neighbour's build hands the learner power, and the scoring phase
    pays everyone: that VP is real and must reach the learner's return.
    """
    env = make_single_env(player_count=4, reward_mode=RewardMode.DENSE_VP)
    total, _, info = _play(env, 7)
    final = info["final_vps"][env.learner_faction]
    assert total != 0.0
    # the dense terms telescope to (final - 20)/100 plus the terminal share
    from bgai.env.reward import dense_return_bound, terminal_rewards

    expected = dense_return_bound(final) + terminal_rewards(
        RewardMode.DENSE_VP, info["final_vps"]
    )[env.learner_faction]
    assert total == pytest.approx(expected, abs=1e-6)


def test_terminal_mode_pays_only_at_the_end() -> None:
    env = make_single_env(player_count=2)
    obs, _ = env.reset(seed=8)
    rng = random.Random(8)
    while True:
        legal = legal_actions(obs)
        obs, reward, terminated, truncated, _ = env.step(
            int(legal[rng.randrange(len(legal))])
        )
        if terminated or truncated:
            assert reward != 0.0
            break
        assert reward == 0.0


def test_truncation_budget_ends_the_episode() -> None:
    env = make_single_env(player_count=4, max_decisions=8)
    _, _, info = _play(env, 9)
    assert env.episode.decisions == 8
    assert not env.episode.sim.finished


def test_snapshot_restore_round_trips() -> None:
    env = make_single_env(player_count=4)
    obs, _ = env.reset(seed=10)
    rng = random.Random(10)
    for _ in range(12):
        obs, _, _, _, _ = env.step(int(legal_actions(obs)[rng.randrange(1)]))
    payload = env.snapshot()
    other = make_single_env(player_count=4)
    other.restore(payload)
    assert other.episode.sim == env.episode.sim
    small = make_single_env(player_count=2)
    with pytest.raises(ValueError, match="4-player table"):
        small.restore(payload)


def test_render_modes() -> None:
    quiet = make_single_env(player_count=4)
    quiet.reset(seed=11)
    assert quiet.render() is None
    loud = make_single_env(player_count=4, render_mode="ansi")
    loud.reset(seed=11)
    assert "learner=" in (loud.render() or "")
    loud.close()
    with pytest.raises(ValueError, match="unknown render_mode"):
        make_single_env(player_count=4, render_mode="rgb_array")


# --------------------------------------------------------------------------
# opponents
# --------------------------------------------------------------------------


def test_known_opponents_are_what_build_accepts() -> None:
    assert set(known_opponents()) == {"random", "greedy", "imitation", "mcts"}
    for spec in ("random", "greedy"):
        assert build_opponent(spec).name == spec


def test_unknown_opponent_is_named() -> None:
    with pytest.raises(UnknownOpponentError, match="unknown opponent 'bogus'"):
        build_opponent("bogus")


def test_checkpoint_opponents_require_a_checkpoint() -> None:
    for spec in ("imitation", "mcts"):
        with pytest.raises(UnknownOpponentError, match="needs a checkpoint path"):
            build_opponent(spec)


def test_checkpoint_opponents_refuse_non_four_player_tables(tmp_path) -> None:
    """``training.encode_state`` is pinned at four seats; saying so beats
    producing silently wrong numbers at 2/3/5 players.
    """
    for spec in ("imitation", "mcts"):
        with pytest.raises(UnknownOpponentError, match="4-player only"):
            build_opponent(spec, checkpoint=tmp_path / "x.pt", player_count=5)


def test_greedy_opponent_beats_random_in_the_learner_seat() -> None:
    """A sanity check that opponent policies actually play: the one-ply
    greedy baseline should out-VP uniform random over a handful of games
    (the arena measures this properly at n=200; here it only has to be
    directionally alive).
    """
    greedy_vps: list[int] = []
    random_vps: list[int] = []
    for seed in range(4):
        for name, bucket in (("greedy", greedy_vps), ("random", random_vps)):
            env = make_single_env(player_count=4, opponent="random")
            policy = build_opponent(name)
            rng = random.Random(seed)
            env.reset(seed=100 + seed)
            policy.begin_episode()
            while True:
                episode = env.episode
                faction, offer = episode.pending()
                index = policy(episode.sim, faction, offer, rng)
                _, _, terminated, truncated, info = env.step(index)
                if terminated or truncated:
                    break
            bucket.append(info["final_vps"][env.learner_faction])
    assert sum(greedy_vps) / len(greedy_vps) > sum(random_vps) / len(random_vps)


def test_agent_policy_rejects_an_object_that_cannot_choose() -> None:
    with pytest.raises(TypeError, match="neither choose nor choose_sim"):
        AgentPolicy(object())


def test_agent_policy_finds_the_index_of_the_returned_move() -> None:
    env = make_single_env(player_count=4)
    env.reset(seed=12)
    policy = AgentPolicy(RandomAgent())
    episode = env.episode
    faction, offer = episode.pending()
    index = policy(episode.sim, faction, offer, random.Random(0))
    assert 0 <= index < len(offer)


def test_agent_policy_detects_an_out_of_offer_move() -> None:
    class Cheater:
        name = "cheater"

        def choose(self, state, faction, offer, rng):
            from bgai.engine.tm.legal_shared import cmd

            return cmd("bridge", loc="Z99", loc2="Z98")

    env = make_single_env(player_count=4)
    env.reset(seed=13)
    episode = env.episode
    faction, offer = episode.pending()
    with pytest.raises(EnvError, match="outside its offer"):
        AgentPolicy(Cheater())(episode.sim, faction, offer, random.Random(0))


def test_agent_policy_forwards_tree_hooks() -> None:
    class Tracked(GreedyAgent):
        def __init__(self) -> None:
            super().__init__()
            self.resets = 0
            self.notes: list[str] = []

        def reset_tree(self) -> None:
            self.resets += 1

        def note_advance(self, choice) -> None:
            self.notes.append(choice.verb)

    agent = Tracked()
    policy = AgentPolicy(agent, name="tracked")
    policy.begin_episode()
    from bgai.engine.tm.legal_shared import cmd

    policy.observe_move(cmd("done"))
    assert agent.resets == 1
    assert agent.notes == ["done"]


def test_random_policy_hooks_are_no_ops() -> None:
    policy = random_policy("r")
    policy.begin_episode()
    from bgai.engine.tm.legal_shared import cmd

    assert policy.observe_move(cmd("done")) is None


def test_legal_actions_matches_the_mask() -> None:
    env = make_single_env(player_count=4)
    obs, _ = env.reset(seed=14)
    assert np.array_equal(
        legal_actions(obs), np.flatnonzero(obs["action_mask"])
    )
