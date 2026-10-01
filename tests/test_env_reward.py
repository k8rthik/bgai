"""Reward formulation: every mode is a function of engine VP alone, and
each one is centred so an average finish earns zero.
"""

from __future__ import annotations

import random

import pytest

from bgai.arena.sim import _competition_ranks
from bgai.env.config import (
    DENSE_TERMINAL_WEIGHT,
    DENSE_VP_SCALE,
    MAX_PLAYERS,
    MIN_PLAYERS,
    VP_START,
    EnvConfig,
    RewardMode,
)
from bgai.env.core import Episode
from bgai.env.reward import (
    competition_ranks,
    dense_return_bound,
    dense_step_rewards,
    terminal_rewards,
    zero_rewards,
)

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def test_competition_ranks_match_the_arenas_own_rule() -> None:
    """The env's copy must not drift from ``arena.sim``'s."""
    cases = [
        {"a": 100, "b": 90, "c": 80, "d": 70},
        {"a": 100, "b": 100, "c": 80, "d": 70},
        {"a": 50, "b": 50, "c": 50, "d": 50},
        {"a": 10, "b": 90},
    ]
    for vps in cases:
        assert competition_ranks(vps) == _competition_ranks(vps)


@pytest.mark.parametrize("mode", list(RewardMode))
def test_every_terminal_mode_sums_to_zero(mode: RewardMode) -> None:
    """Centred rewards: a seat finishing exactly average earns 0, and the
    table's rewards sum to 0 whatever the mode.
    """
    vps = {"a": 120, "b": 100, "c": 90, "d": 70}
    rewards = terminal_rewards(mode, vps)
    assert sum(rewards.values()) == pytest.approx(0.0, abs=1e-9)


def test_vp_share_is_scale_free() -> None:
    small = terminal_rewards(RewardMode.TERMINAL_VP_SHARE, {"a": 60, "b": 40})
    big = terminal_rewards(RewardMode.TERMINAL_VP_SHARE, {"a": 600, "b": 400})
    assert small == pytest.approx(big)
    assert small["a"] == pytest.approx(0.1)


def test_rank_mode_spans_plus_one_to_minus_one() -> None:
    rewards = terminal_rewards(
        RewardMode.TERMINAL_RANK, {"a": 100, "b": 90, "c": 80, "d": 70}
    )
    assert rewards["a"] == pytest.approx(1.0)
    assert rewards["d"] == pytest.approx(-1.0)
    assert rewards["b"] > rewards["c"]


def test_rank_mode_splits_a_tie_down_the_middle() -> None:
    rewards = terminal_rewards(
        RewardMode.TERMINAL_RANK, {"a": 100, "b": 100, "c": 80, "d": 70}
    )
    assert rewards["a"] == rewards["b"]
    # the two tied seats share places 0 and 1, i.e. mean place 0.5
    assert rewards["a"] == pytest.approx(1.0 - 2.0 * 0.5 / 3.0)


def test_win_mode_pays_the_winner_and_splits_ties() -> None:
    outright = terminal_rewards(RewardMode.TERMINAL_WIN, {"a": 100, "b": 90, "c": 1})
    assert outright["a"] == pytest.approx(1.0 - 1 / 3)
    assert outright["b"] == pytest.approx(-1 / 3)
    shared = terminal_rewards(RewardMode.TERMINAL_WIN, {"a": 100, "b": 100})
    assert shared == pytest.approx({"a": 0.0, "b": 0.0})


def test_dense_mode_terminal_term_is_the_weighted_share() -> None:
    vps = {"a": 120, "b": 80}
    share = terminal_rewards(RewardMode.TERMINAL_VP_SHARE, vps)
    dense = terminal_rewards(RewardMode.DENSE_VP, vps)
    for seat in vps:
        assert dense[seat] == pytest.approx(DENSE_TERMINAL_WEIGHT * share[seat])


def test_dense_step_rewards_are_zero_outside_dense_mode() -> None:
    seats = ("a", "b")
    before, after = {"a": 20, "b": 20}, {"a": 25, "b": 20}
    for mode in RewardMode:
        rewards = dense_step_rewards(mode, before, after, seats)
        if mode is RewardMode.DENSE_VP:
            assert rewards["a"] == pytest.approx(5 / DENSE_VP_SCALE)
        else:
            assert rewards == zero_rewards(seats)


def test_terminal_rewards_need_at_least_one_seat() -> None:
    with pytest.raises(ValueError, match="at least one seat"):
        terminal_rewards(RewardMode.TERMINAL_VP_SHARE, {})


def test_zero_total_vp_does_not_divide_by_zero() -> None:
    assert terminal_rewards(RewardMode.TERMINAL_VP_SHARE, {"a": 0, "b": 0}) == {
        "a": 0.0,
        "b": 0.0,
    }


@pytest.mark.parametrize("count", COUNTS)
def test_terminal_modes_pay_nothing_before_the_end(count: int) -> None:
    config = EnvConfig(player_count=count, reward_mode=RewardMode.TERMINAL_VP_SHARE)
    episode = Episode.start(55, config)
    rng = random.Random(count)
    while True:
        _, offer = episode.pending()
        episode, transition = episode.step(rng.randrange(len(offer)))
        if transition.done:
            break
        assert all(value == 0.0 for value in transition.rewards.values())
    assert transition.terminated
    assert sum(transition.rewards.values()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("count", [2, 4])
def test_dense_returns_telescope_to_absolute_final_vp(count: int) -> None:
    """The module docstring claims the dense terms sum to
    ``(final_vp - 20)/scale``. That is a testable statement, so it is tested.
    """
    config = EnvConfig(player_count=count, reward_mode=RewardMode.DENSE_VP)
    episode = Episode.start(77, config)
    rng = random.Random(count)
    totals = dict.fromkeys(episode.seats, 0.0)
    while True:
        _, offer = episode.pending()
        episode, transition = episode.step(rng.randrange(len(offer)))
        for seat, value in transition.rewards.items():
            totals[seat] += value
        if transition.done:
            break
    final = episode.vps()
    terminal = terminal_rewards(RewardMode.DENSE_VP, final)
    for seat in episode.seats:
        expected = dense_return_bound(final[seat]) + terminal[seat]
        assert totals[seat] == pytest.approx(expected, abs=1e-6)
        assert dense_return_bound(final[seat]) == pytest.approx(
            (final[seat] - VP_START) / DENSE_VP_SCALE
        )


def test_unhandled_reward_mode_is_loud() -> None:
    class Fake:
        pass

    with pytest.raises(ValueError, match="unhandled reward mode"):
        terminal_rewards(Fake(), {"a": 1})  # type: ignore[arg-type]
