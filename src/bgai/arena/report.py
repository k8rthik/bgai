"""Aggregate match outcomes into per-agent win rate / VP / TrueSkill reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import trueskill

from bgai.arena.match import MatchOutcome

_DRAW_PROBABILITY = 0.02  # shared-win ties are rare but real (snellman shares wins)


@dataclass(frozen=True)
class AgentReport:
    games: int
    wins: int
    win_rate: float
    mean_vp: float
    trueskill_mu: float
    trueskill_sigma: float


@dataclass(frozen=True)
class ArenaReport:
    agents: Mapping[str, AgentReport]
    games: int


def summarize(outcomes: Sequence[MatchOutcome]) -> ArenaReport:
    """Per agent *name*: games participated, games where one of its seats
    won (shared wins count), mean VP over its seats, and TrueSkill from
    per-seat rating groups ranked by VP (an agent on several seats gets
    the average of its updated seat ratings).
    """
    env = trueskill.TrueSkill(draw_probability=_DRAW_PROBABILITY)
    ratings: dict[str, trueskill.Rating] = {}
    games: dict[str, int] = {}
    wins: dict[str, int] = {}
    vp_sum: dict[str, int] = {}
    vp_n: dict[str, int] = {}

    for outcome in outcomes:
        by_faction = outcome.agent_by_faction
        vp = outcome.result.vp
        factions = list(by_faction)
        names = [by_faction[f] for f in factions]
        for name in set(names):
            ratings.setdefault(name, env.create_rating())
            games[name] = games.get(name, 0) + 1
        for faction, name in by_faction.items():
            vp_sum[name] = vp_sum.get(name, 0) + vp[faction]
            vp_n[name] = vp_n.get(name, 0) + 1
        for name in set(names):
            if any(w in by_faction and by_faction[w] == name for w in outcome.result.winners):
                wins[name] = wins.get(name, 0) + 1

        # VP rank per seat (0 = best; ties share a rank).
        ordered = sorted(set(vp.values()), reverse=True)
        ranks = [ordered.index(vp[f]) for f in factions]
        groups = [(ratings[name],) for name in names]
        rated = env.rate(groups, ranks=ranks)
        updated: dict[str, list[trueskill.Rating]] = {}
        for name, (rating,) in zip(names, rated, strict=True):
            updated.setdefault(name, []).append(rating)
        for name, group in updated.items():
            mu = sum(r.mu for r in group) / len(group)
            sigma = sum(r.sigma for r in group) / len(group)
            ratings[name] = trueskill.Rating(mu=mu, sigma=sigma)

    agents = {
        name: AgentReport(
            games=games[name],
            wins=wins.get(name, 0),
            win_rate=wins.get(name, 0) / games[name],
            mean_vp=vp_sum[name] / vp_n[name],
            trueskill_mu=ratings[name].mu,
            trueskill_sigma=ratings[name].sigma,
        )
        for name in sorted(games)
    }
    return ArenaReport(agents=agents, games=len(outcomes))


def to_json(report: ArenaReport) -> str:
    payload = {
        "games": report.games,
        "agents": {
            name: {
                "games": a.games,
                "wins": a.wins,
                "win_rate": a.win_rate,
                "mean_vp": a.mean_vp,
                "trueskill_mu": a.trueskill_mu,
                "trueskill_sigma": a.trueskill_sigma,
            }
            for name, a in report.agents.items()
        },
    }
    return json.dumps(payload, indent=2)


def format_table(report: ArenaReport) -> str:
    header = (
        f"{'agent':<14} {'games':>5} {'wins':>5} {'win%':>6} "
        f"{'meanVP':>7} {'mu':>6} {'sigma':>6}"
    )
    lines = [header, "-" * len(header)]
    ranked = sorted(report.agents.items(), key=lambda kv: -kv[1].trueskill_mu)
    for name, a in ranked:
        lines.append(
            f"{name:<14} {a.games:>5} {a.wins:>5} {a.win_rate:>6.1%} "
            f"{a.mean_vp:>7.1f} {a.trueskill_mu:>6.2f} {a.trueskill_sigma:>6.2f}"
        )
    return "\n".join(lines)
