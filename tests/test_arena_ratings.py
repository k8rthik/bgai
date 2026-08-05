"""TrueSkill ratings wrapper (plan 2026-08-04-arena-baselines, Task 8)."""

from __future__ import annotations

from bgai.arena.ratings import (
    conservative,
    faction_placements,
    new_ratings,
    seat_placements,
    update,
)
from bgai.arena.sim import GameResult


def _result(
    ranks: dict[str, int], seats: dict[str, str], error: str | None = None
) -> GameResult:
    return GameResult(
        setup_game_id="g",
        seats=seats,
        vps={f: 100 - r for f, r in ranks.items()},
        ranks=ranks,
        decisions=100,
        error=error,
    )


SEATS = {"witches": "greedy", "nomads": "random", "engineers": "greedy", "darklings": "random"}
RANKS = {"witches": 0, "engineers": 1, "nomads": 2, "darklings": 3}


def test_update_moves_winner_up_loser_down_immutably() -> None:
    before = new_ratings(["greedy", "random"])
    after = update(before, _result(RANKS, SEATS))
    assert after["greedy"].mu > before["greedy"].mu
    assert after["random"].mu < before["random"].mu
    assert before["greedy"].mu == new_ratings(["greedy"])["greedy"].mu  # input untouched


def test_errored_game_is_noop() -> None:
    before = new_ratings(["greedy", "random"])
    after = update(before, _result(RANKS, SEATS, error="boom"))
    assert after == before


def test_conservative_orders_by_mu_minus_3sigma() -> None:
    rating = new_ratings(["a"])["a"]
    assert conservative(rating) == rating.mu - 3 * rating.sigma


def test_placement_tables() -> None:
    results = [_result(RANKS, SEATS), _result(RANKS, SEATS, error="skip me")]
    by_faction = faction_placements(results)
    assert by_faction[("greedy", "witches")] == (1, 0.0)
    assert by_faction[("random", "darklings")] == (1, 3.0)
    by_seat = seat_placements(results)
    # seat order == GameResult.seats insertion order
    assert by_seat[("greedy", 0)] == (1, 0.0)  # witches seat
    assert by_seat[("random", 1)] == (1, 2.0)  # nomads seat
