"""Outcome statistics: win rate and mean placement.

The arena reported only TrueSkill mu-3sigma, which is a latent skill
estimate rather than a result. "Did it win more games" is the question,
and it needs the placements the games already produced.
"""

from __future__ import annotations

import pytest

from bgai.arena.outcomes import agent_outcomes


class _Result:
    """Minimal stand-in for GameResult (seats: faction -> agent name)."""

    def __init__(self, seats, ranks, vps, error=None):
        self.seats, self.ranks, self.vps, self.error = seats, ranks, vps, error


def _game(winner_agent: str, loser_agent: str) -> _Result:
    return _Result(
        seats={"a": winner_agent, "b": loser_agent, "c": loser_agent, "d": winner_agent},
        ranks={"a": 0, "b": 1, "c": 2, "d": 3},
        vps={"a": 100, "b": 90, "c": 80, "d": 70},
    )


def test_win_rate_counts_first_places_per_seat() -> None:
    games = [_game("x", "y"), _game("x", "y")]
    out = agent_outcomes(games)
    # x holds seats a (rank 0) and d (rank 3); y holds b and c
    assert out["x"]["seat_games"] == 4
    assert out["x"]["wins"] == 2
    assert out["x"]["win_rate"] == pytest.approx(0.5)
    assert out["y"]["wins"] == 0


def test_mean_placement_uses_the_full_ranking() -> None:
    out = agent_outcomes([_game("x", "y")])
    # x: ranks 0 and 3 -> mean 1.5 (0-based); y: ranks 1 and 2 -> 1.5
    assert out["x"]["mean_place"] == pytest.approx(1.5)
    assert out["y"]["mean_place"] == pytest.approx(1.5)


def test_mean_vp_is_reported() -> None:
    out = agent_outcomes([_game("x", "y")])
    assert out["x"]["mean_vp"] == pytest.approx((100 + 70) / 2)
    assert out["y"]["mean_vp"] == pytest.approx((90 + 80) / 2)


def test_errored_games_are_excluded() -> None:
    bad = _game("x", "y")
    bad.error = "engine rejection"
    out = agent_outcomes([_game("x", "y"), bad])
    assert out["x"]["seat_games"] == 2, "an aborted game is not an outcome"


def test_no_games_yields_empty() -> None:
    assert agent_outcomes([]) == {}
