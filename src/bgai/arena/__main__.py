"""Bot-vs-bot arena batches: `python -m bgai.arena --agents a,b,c,d --games N`."""

from __future__ import annotations

import argparse
from pathlib import Path

from bgai.arena.match import build_agent, rotation, run_match
from bgai.arena.report import format_table, summarize, to_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a bot-vs-bot TM arena batch.")
    parser.add_argument(
        "--agents", required=True,
        help="4 comma-separated agent names (registry: random*, heuristic*)",
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write report JSON here")
    args = parser.parse_args(argv)

    names = tuple(part.strip() for part in args.agents.split(",") if part.strip())
    if len(names) != 4:
        parser.error(f"--agents needs exactly 4 names, got {len(names)}")
    for name in names:
        build_agent(name, seed=0)  # validate against the registry up front

    outcomes = []
    for spec in rotation(names, games=args.games, base_seed=args.seed):
        outcome = run_match(spec)
        outcomes.append(outcome)
        vp = ", ".join(f"{f}={v}" for f, v in outcome.result.vp.items())
        print(f"game seed={spec.seed}: {vp} -> winners {outcome.result.winners}")

    report = summarize(outcomes)
    print()
    print(format_table(report))
    if args.out is not None:
        args.out.write_text(to_json(report))
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
