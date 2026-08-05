"""Arena CLI (plan Task 9).

    uv run python -m bgai.arena.run --tables 50 --seed 20260804 \\
        --agents random,greedy --report /tmp/arena.html

Builds the named baseline agents, alternates them across the 4 base
seats, runs ``n_tables`` corpus-sampled tables x 4 mirrored rotations,
writes the HTML report, and prints a one-line summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bgai.agents import GreedyAgent, RandomAgent
from bgai.agents.base import Agent
from bgai.arena.ratings import conservative
from bgai.arena.report import write_report
from bgai.arena.series import run_series

_AGENT_FACTORIES = {"random": RandomAgent, "greedy": GreedyAgent}


def _build_agents(spec: str) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for name in spec.split(","):
        name = name.strip()
        factory = _AGENT_FACTORIES.get(name)
        if factory is None:
            raise SystemExit(
                f"unknown agent {name!r} (available: {sorted(_AGENT_FACTORIES)})"
            )
        agents[name] = factory(name=name)
    if not agents:
        raise SystemExit("no agents given")
    return agents


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a mirrored arena series.")
    parser.add_argument("--tables", type=int, default=50, help="setups; 4 games each")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", default="random,greedy", help="comma-separated names")
    parser.add_argument("--report", type=Path, default=None, help="HTML report path")
    args = parser.parse_args(argv)

    agents = _build_agents(args.agents)
    names = list(agents)
    base_seats = tuple(names[i % len(names)] for i in range(4))
    series = run_series(
        agents=agents, base_seats=base_seats, n_tables=args.tables, seed=args.seed
    )
    if args.report is not None:
        write_report(series, args.report)
    summary = ", ".join(
        f"{name}: μ−3σ={conservative(series.ratings[name]):.2f}"
        for name in sorted(series.ratings, key=lambda n: -conservative(series.ratings[n]))
    )
    print(
        f"{len(series.results)} games, {series.n_errors} errors | {summary}"
        + (f" | report: {args.report}" if args.report else "")
    )


if __name__ == "__main__":
    main()
