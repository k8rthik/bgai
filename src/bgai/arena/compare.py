"""Head-to-head verdicts from saved arena runs.

    uv run python -m bgai.arena.compare --baseline leg5b data/evals/scale/*.json

A *run* is one ``bgai.arena.run --json`` file: one seed, N tables, every
game with its seats, ranks and VPs. This module turns a set of runs into
one row per candidate checkpoint, judged against a common baseline.

Statistics, and why these:

- ``share``: candidate wins / (candidate + baseline wins). The number C18
  and the scaling-curve evals were read with; 0.5 means even. Its ``z``
  treats wins as independent, which slightly overstates certainty.
- ``win_diff`` / ``place_diff`` / ``vp_diff``: candidate minus baseline,
  computed PER TABLE and then averaged. A table is one setup played in
  all 4 mirrored seat rotations, so the table is the paired unit: setup
  luck cancels inside it, and its standard error (sd over tables /
  sqrt(T)) is the honest one. ``place_diff`` is baseline minus candidate
  mean place, so positive always means "candidate better".
- Seeds pool by concatenating tables (each (seed, table) is distinct),
  never by averaging per-seed percentages.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROTATIONS_PER_TABLE = 4

Run = Mapping[str, Any]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def series_to_run(series: Any, seed: int, agents_spec: str, meta: Mapping[str, Any]) -> dict:
    """A ``SeriesResult`` as a JSON-ready run. Results arrive in table
    order, ``ROTATIONS_PER_TABLE`` per table (``run_series`` and
    ``run_series_parallel`` both guarantee it), which is what tags each
    game with its table."""
    results = tuple(series.results)
    if len(results) % ROTATIONS_PER_TABLE:
        raise ValueError(
            f"{len(results)} results is not a multiple of {ROTATIONS_PER_TABLE} rotations"
        )
    games = [
        {
            "table": index // ROTATIONS_PER_TABLE,
            "rotation": index % ROTATIONS_PER_TABLE,
            "setup_game_id": getattr(r, "setup_game_id", None),
            "seats": dict(r.seats),
            "ranks": {f: int(v) for f, v in r.ranks.items()},
            "vps": {f: int(v) for f, v in r.vps.items()},
            "error": r.error,
        }
        for index, r in enumerate(results)
    ]
    agents = list(dict.fromkeys(a for g in games for a in g["seats"].values()))
    return {
        "seed": seed,
        "tables": len(results) // ROTATIONS_PER_TABLE,
        "agents": agents,
        "agents_spec": agents_spec,
        "n_errors": sum(1 for g in games if g["error"] is not None),
        "meta": dict(meta),
        "games": games,
    }


def load_run(path: Path) -> dict:
    """Read and minimally validate one run file."""
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("seed", "games"):
        if key not in run:
            raise ValueError(f"{path}: not an arena run (missing {key!r})")
    for game in run["games"]:
        missing = {"table", "seats", "ranks", "vps"} - set(game)
        if missing:
            raise ValueError(f"{path}: game missing {sorted(missing)}")
    return run


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _mean_se(values: Sequence[float]) -> tuple[float | None, float | None]:
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def _z(mean: float | None, se: float | None) -> float | None:
    if mean is None or se is None or se == 0:
        return None
    return mean / se


def _per_table(runs: Iterable[Run], agents: tuple[str, ...]) -> tuple[dict, int]:
    """``{(seed, table): {agent: [seats, wins, place_sum, vp_sum]}}`` and
    the number of errored games skipped."""
    tables: dict[tuple[int, int], dict[str, list[int]]] = {}
    errors = 0
    for run in runs:
        for game in run["games"]:
            if game.get("error") is not None:
                errors += 1
                continue
            cell = tables.setdefault((run["seed"], game["table"]),
                                     {a: [0, 0, 0, 0] for a in agents})
            for faction, agent in game["seats"].items():
                if agent not in cell:
                    continue
                rank = int(game["ranks"][faction])
                cell[agent] = [
                    cell[agent][0] + 1,
                    cell[agent][1] + (rank == 0),
                    cell[agent][2] + rank,
                    cell[agent][3] + int(game["vps"][faction]),
                ]
    return tables, errors


def _totals(tables: Mapping, agent: str) -> dict[str, float]:
    seats = sum(t[agent][0] for t in tables.values())
    wins = sum(t[agent][1] for t in tables.values())
    return {
        "seat_games": seats,
        "wins": wins,
        "win_rate": wins / seats if seats else 0.0,
        "mean_place": sum(t[agent][2] for t in tables.values()) / seats if seats else 0.0,
        "mean_vp": sum(t[agent][3] for t in tables.values()) / seats if seats else 0.0,
    }


def head_to_head(runs: Sequence[Run], candidate: str, baseline: str) -> dict:
    """Pooled candidate-vs-baseline statistics over ``runs`` (see module doc)."""
    seen = {a for run in runs for g in run["games"] for a in g["seats"].values()}
    for agent in (candidate, baseline):
        if agent not in seen:
            raise ValueError(f"agent {agent!r} does not appear in these runs")
    tables, errors = _per_table(runs, (candidate, baseline))
    both = {k: t for k, t in tables.items() if t[candidate][0] and t[baseline][0]}

    def diffs(index: int, sign: int) -> list[float]:
        return [
            sign * (t[candidate][index] / t[candidate][0] - t[baseline][index] / t[baseline][0])
            for t in both.values()
        ]

    win_diff, win_se = _mean_se(diffs(1, +1))
    place_diff, place_se = _mean_se(diffs(2, -1))
    vp_diff, vp_se = _mean_se(diffs(3, +1))
    cand, base = _totals(tables, candidate), _totals(tables, baseline)
    decisive = cand["wins"] + base["wins"]
    share = cand["wins"] / decisive if decisive else None
    return {
        candidate: cand,
        baseline: base,
        "seeds": sorted({run["seed"] for run in runs}),
        "tables": len(both),
        "games": sum(1 for run in runs for g in run["games"] if g.get("error") is None),
        "errors": errors,
        "share": share,
        "share_z": _z(share - 0.5, math.sqrt(0.25 / decisive)) if share is not None else None,
        "win_diff": win_diff, "win_diff_se": win_se, "win_diff_z": _z(win_diff, win_se),
        "place_diff": place_diff, "place_diff_se": place_se,
        "place_diff_z": _z(place_diff, place_se),
        "vp_diff": vp_diff, "vp_diff_se": vp_se,
    }


def compare_runs(runs: Sequence[Run], baseline: str) -> list[dict]:
    """One row per candidate (the non-baseline agent of each run), in
    first-seen order, each pooled over all of that candidate's runs."""
    groups: dict[str, list[Run]] = {}
    for run in runs:
        agents = list(run.get("agents") or
                      dict.fromkeys(a for g in run["games"] for a in g["seats"].values()))
        if baseline not in agents:
            raise ValueError(f"run seed={run.get('seed')} has no agent {baseline!r}: {agents}")
        others = [a for a in agents if a != baseline]
        if len(others) != 1:
            raise ValueError(f"run seed={run.get('seed')} must pit ONE candidate "
                             f"against {baseline!r}, has {others}")
        groups.setdefault(others[0], []).append(run)
    return [
        {"candidate": cand, "baseline": baseline, "meta": dict(group[0].get("meta") or {}),
         **head_to_head(group, cand, baseline)}
        for cand, group in groups.items()
    ]


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Markdown table; win/place differences are paired by table (± 1 SE)."""
    lines = [
        "| candidate | games trained | seeds | games | cand win% | base win% | share | z "
        "| win diff (pp) | z | place diff | z | VP diff |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cand, base = r[r["candidate"]], r[r["baseline"]]
        trained = r["meta"].get("approx_games_trained")
        wd, wse = r["win_diff"], r["win_diff_se"]
        lines.append(
            f"| {r['candidate']} | {_fmt(trained, ',')} | {len(r['seeds'])} | {r['games']} "
            f"| {cand['win_rate'] * 100:.1f} | {base['win_rate'] * 100:.1f} "
            f"| {_fmt(r['share'], '.3f')} | {_fmt(r['share_z'], '+.2f')} "
            f"| {_fmt(None if wd is None else wd * 100, '+.1f')}"
            f" ± {_fmt(None if wse is None else wse * 100, '.1f')} "
            f"| {_fmt(r['win_diff_z'], '+.2f')} "
            f"| {_fmt(r['place_diff'], '+.3f')} | {_fmt(r['place_diff_z'], '+.2f')} "
            f"| {_fmt(r['vp_diff'], '+.1f')} |"
        )
    return "\n".join(lines)


def render_brief(rows: Sequence[Mapping[str, Any]]) -> str:
    """One plain line per candidate -- sized for a phone notification."""
    lines = []
    for r in rows:
        n = len(r["seeds"])
        wd, wse = r["win_diff"], r["win_diff_se"]
        diff = "-" if wd is None else f"{wd * 100:+.1f}"
        if wse is not None:
            diff += f"±{wse * 100:.1f}pp (z {_fmt(r['win_diff_z'], '+.1f')})"
        lines.append(
            f"{r['candidate']} vs {r['baseline']}: share {_fmt(r['share'], '.3f')}"
            f" (z {_fmt(r['share_z'], '+.1f')}), win diff {diff},"
            f" {r['games']} games, {n} seed{'s' if n != 1 else ''}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Candidate-vs-baseline verdicts.")
    parser.add_argument("runs", nargs="+", type=Path, help="arena --json run files")
    parser.add_argument("--baseline", required=True, help="baseline agent label")
    parser.add_argument("--json", type=Path, default=None, help="also write rows here")
    parser.add_argument("--brief", action="store_true", help="one line per candidate")
    args = parser.parse_args(argv)
    try:
        rows = compare_runs([load_run(p) for p in args.runs], args.baseline)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"compare: {exc}") from exc
    print(render_brief(rows) if args.brief else render_table(rows))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
