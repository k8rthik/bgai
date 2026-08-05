"""Self-contained HTML arena report (plan Task 10).

Sections: header counts, TrueSkill ratings (sorted by the conservative
mu - 3*sigma key), per-faction and per-seat mean-placement tables (the
master plan's covariate reporting), mean VP per agent, and a verbatim
error/anomaly list -- fuzzer findings must be visible, never swallowed.
Everything dynamic is escaped; no external assets.
"""

from __future__ import annotations

import html
from pathlib import Path

from bgai.arena.ratings import conservative, faction_placements, seat_placements
from bgai.arena.series import SeriesResult

_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a2e; }
table { border-collapse: collapse; margin: 0.75rem 0 1.5rem; }
th, td { border: 1px solid #c5c9d6; padding: 0.3rem 0.7rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
h2 { margin-top: 1.6rem; }
.err { color: #8b1e1e; font-family: monospace; }
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render_report(series: SeriesResult) -> str:
    agents = sorted(series.ratings, key=lambda a: -conservative(series.ratings[a]))
    clean = [r for r in series.results if r.error is None]

    rating_rows = []
    games_per_agent = {
        a: sum(1 for r in clean for name in r.seats.values() if name == a) for a in agents
    }
    for agent in agents:
        rating = series.ratings[agent]
        rating_rows.append(
            [
                agent,
                f"{rating.mu:.2f}",
                f"{rating.sigma:.2f}",
                f"{conservative(rating):.2f}",
                str(games_per_agent[agent]),
            ]
        )

    by_faction = faction_placements(series.results)
    factions = sorted({f for a, f in by_faction})
    faction_rows = [
        [faction]
        + [
            f"{by_faction[(agent, faction)][1]:.2f} ({by_faction[(agent, faction)][0]})"
            if (agent, faction) in by_faction
            else "—"
            for agent in agents
        ]
        for faction in factions
    ]

    by_seat = seat_placements(series.results)
    seat_rows = [
        [f"seat {seat}"]
        + [
            f"{by_seat[(agent, seat)][1]:.2f} ({by_seat[(agent, seat)][0]})"
            if (agent, seat) in by_seat
            else "—"
            for agent in agents
        ]
        for seat in sorted({s for a, s in by_seat})
    ]

    vp_rows = []
    for agent in agents:
        vps = [r.vps[f] for r in clean for f, name in r.seats.items() if name == agent]
        vp_rows.append([agent, f"{sum(vps) / len(vps):.1f}" if vps else "—"])

    problems: list[str] = []
    for r in series.results:
        if r.error is not None:
            problems.append(f"{r.setup_game_id}: {r.error}")
        problems.extend(f"{r.setup_game_id}: {a}" for a in r.anomalies)
    problems_html = (
        "".join(f"<li class='err'>{html.escape(p)}</li>" for p in problems)
        if problems
        else "<li>none</li>"
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>bgai arena report</title>
<style>{_STYLE}</style></head><body>
<h1>bgai arena report</h1>
<p>{len(series.results)} games ({len(clean)} clean, {series.n_errors} errored)</p>
<h2>TrueSkill ratings</h2>
{_table(["agent", "μ", "σ", "μ−3σ", "games"], rating_rows)}
<h2>Mean placement by faction (0 = winner)</h2>
{_table(["faction"] + agents, faction_rows)}
<h2>Mean placement by seat</h2>
{_table(["seat"] + agents, seat_rows)}
<h2>Mean final VP</h2>
{_table(["agent", "mean VP"], vp_rows)}
<h2>Errors and anomalies</h2>
<ul>{problems_html}</ul>
</body></html>
"""


def write_report(series: SeriesResult, path: Path) -> None:
    path.write_text(render_report(series), encoding="utf-8")
