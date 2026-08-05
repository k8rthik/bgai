"""TrueSkill ratings + placement covariate tables (plan Task 8).

Agent ratings are global (one rating per agent name); faction and seat
strength are handled by the arena's mirrored rotation plus the explicit
per-faction / per-seat placement tables below (master plan asymmetry
requirement 3: faction strength is never conflated with agent strength).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import trueskill

from bgai.arena.sim import GameResult

Ratings = Mapping[str, trueskill.Rating]

_ENV = trueskill.TrueSkill()


def new_ratings(agent_names: Iterable[str]) -> dict[str, trueskill.Rating]:
    return {name: _ENV.create_rating() for name in agent_names}


def conservative(rating: trueskill.Rating) -> float:
    """The standard conservative sort key, mu - 3*sigma."""
    return rating.mu - 3 * rating.sigma


def update(ratings: Ratings, result: GameResult) -> dict[str, trueskill.Rating]:
    """Rate one game as 4 single-player teams; returns a NEW dict, never
    mutates. Errored games are skipped unchanged. When one agent name
    occupies several seats its seats are still separate TrueSkill teams;
    the post-game ratings are merged conservatively (keep the seat with
    the lowest sigma).
    """
    new = dict(ratings)
    if result.error is not None:
        return new
    factions = list(result.seats)
    teams = [{i: ratings[result.seats[f]]} for i, f in enumerate(factions)]
    ranks = [result.ranks[f] for f in factions]
    rated = _ENV.rate(teams, ranks=ranks)
    seen: set[str] = set()
    for i, faction in enumerate(factions):
        name = result.seats[faction]
        candidate = rated[i][i]
        if name not in seen:
            new[name] = candidate
            seen.add(name)
        elif candidate.sigma < new[name].sigma:
            new[name] = candidate
    return new


def faction_placements(
    results: Iterable[GameResult],
) -> dict[tuple[str, str], tuple[int, float]]:
    """``(agent, faction) -> (n_games, mean_rank)`` over clean games."""
    sums: dict[tuple[str, str], list[int]] = {}
    for result in results:
        if result.error is not None:
            continue
        for faction, agent in result.seats.items():
            entry = sums.setdefault((agent, faction), [0, 0])
            entry[0] += 1
            entry[1] += result.ranks[faction]
    return {key: (n, total / n) for key, (n, total) in sums.items()}


def seat_placements(
    results: Iterable[GameResult],
) -> dict[tuple[str, int], tuple[int, float]]:
    """``(agent, seat_index) -> (n_games, mean_rank)`` over clean games.
    Seat index is position in ``GameResult.seats`` insertion order,
    which ``run_game`` builds in ``setup.factions`` seat order.
    """
    sums: dict[tuple[str, int], list[int]] = {}
    for result in results:
        if result.error is not None:
            continue
        for seat, (faction, agent) in enumerate(result.seats.items()):
            entry = sums.setdefault((agent, seat), [0, 0])
            entry[0] += 1
            entry[1] += result.ranks[faction]
    return {key: (n, total / n) for key, (n, total) in sums.items()}
