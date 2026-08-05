"""HTML arena report (plan 2026-08-04-arena-baselines, Task 10)."""

from __future__ import annotations

from bgai.arena.ratings import new_ratings
from bgai.arena.report import render_report
from bgai.arena.series import SeriesResult
from bgai.arena.sim import GameResult


def _result(ranks, seats, error=None):
    return GameResult(
        setup_game_id="4pLeague_S1_D1L1_G1",
        seats=seats,
        vps={f: 100 - r for f, r in ranks.items()},
        ranks=ranks,
        decisions=100,
        error=error,
    )


def test_report_contains_sections_and_escapes_errors() -> None:
    seats = {"witches": "greedy", "nomads": "random", "engineers": "greedy", "darklings": "random"}
    ranks = {"witches": 0, "engineers": 1, "nomads": 2, "darklings": 3}
    series = SeriesResult(
        results=(_result(ranks, seats), _result(ranks, seats, error="<boom> & bust")),
        ratings=new_ratings(["greedy", "random"]),
        n_errors=1,
    )
    html = render_report(series)
    assert "greedy" in html and "random" in html
    assert "witches" in html
    assert "μ" in html
    assert "&lt;boom&gt; &amp; bust" in html  # escaped, never raw
    assert "<boom>" not in html
