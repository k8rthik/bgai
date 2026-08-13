"""Win rate and placement, per agent.

TrueSkill's mu-3sigma ranks competitors but is a latent skill estimate,
not a result: it cannot be read as "won this often". These are the plain
outcome statistics the games already contain.

Counting is per *seat*, not per game. An agent may hold several seats at
one table (a two-agent series puts each on two of four), so a game
contributes one outcome per seat it occupies -- which also makes the
numbers comparable across series with different agent counts.

``mean_place`` is 0-based competition rank (0 = won), matching
``GameResult.ranks``; lower is better. With four seats, chance is 1.5.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def agent_outcomes(results: Iterable[Any]) -> dict[str, dict[str, float]]:
    """``{agent: {seat_games, wins, win_rate, mean_place, mean_vp}}``.

    Games that aborted (``error`` set) are skipped -- an engine rejection
    is not a sporting result and rating updates already ignore them.
    """
    seat_games: dict[str, int] = {}
    wins: dict[str, int] = {}
    place_sum: dict[str, int] = {}
    vp_sum: dict[str, int] = {}

    for result in results:
        if getattr(result, "error", None) is not None:
            continue
        seats: Mapping[str, str] = result.seats
        for faction, agent in seats.items():
            rank = int(result.ranks[faction])
            seat_games[agent] = seat_games.get(agent, 0) + 1
            wins[agent] = wins.get(agent, 0) + (1 if rank == 0 else 0)
            place_sum[agent] = place_sum.get(agent, 0) + rank
            vp_sum[agent] = vp_sum.get(agent, 0) + int(result.vps[faction])

    return {
        agent: {
            "seat_games": n,
            "wins": wins[agent],
            "win_rate": wins[agent] / n,
            "mean_place": place_sum[agent] / n,
            "mean_vp": vp_sum[agent] / n,
        }
        for agent, n in sorted(seat_games.items())
        if n
    }
