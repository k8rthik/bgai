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
from bgai.arena.outcomes import agent_outcomes
from bgai.arena.ratings import conservative
from bgai.arena.report import write_report
from bgai.arena.series import run_series

_AGENT_FACTORIES = {"random": RandomAgent, "greedy": GreedyAgent}

# net-backed agents are imported lazily: torch must not be a hard
# dependency of the arena (see bgai.agents.__init__)
_NET_AGENTS = frozenset({"imitation", "mcts"})


def _coerce(value: str) -> object:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_spec(spec: str) -> tuple[str, str, dict[str, object]]:
    """``name[@label][:k=v,...]`` -> ``(name, label, options)``.

    The label lets one agent type enter a series more than once -- two
    MCTS agents on different checkpoints need distinct rating keys or the
    series treats them as the same competitor.
    """
    head, _, tail = spec.strip().partition(":")
    name, _, label = head.partition("@")
    name, label = name.strip(), (label.strip() or name.strip())
    if name not in _AGENT_FACTORIES and name not in _NET_AGENTS:
        raise ValueError(
            f"unknown agent {name!r} "
            f"(available: {sorted(set(_AGENT_FACTORIES) | _NET_AGENTS)})"
        )
    options: dict[str, object] = {}
    for chunk in filter(None, (c.strip() for c in tail.split(","))):
        key, sep, value = chunk.partition("=")
        if not sep:
            raise ValueError(f"malformed option {chunk!r} in {spec!r} -- expected k=v")
        options[key.strip()] = _coerce(value.strip())
    return name, label, options


def _build_net_agent(name: str, label: str, options: dict[str, object]) -> Agent:
    from pathlib import Path as _Path

    checkpoint = options.get("ckpt")
    if checkpoint is None:
        raise ValueError(f"{name!r} needs ckpt=<path> (agent {label!r})")
    device = str(options.get("device", "cpu"))
    if name == "imitation":
        from bgai.agents.imitation import ImitationAgent

        return ImitationAgent(
            checkpoint_path=_Path(str(checkpoint)),
            name=label,
            temperature=float(options.get("temperature", 0.0)),
            device=device,
        )
    from bgai.agents.mcts import MCTSAgent

    return MCTSAgent(
        checkpoint_path=_Path(str(checkpoint)),
        name=label,
        simulations=int(options.get("sims", 64)),
        c_puct=float(options.get("c_puct", 1.5)),
        temperature=float(options.get("temperature", 0.0)),
        max_depth=int(options.get("max_depth", 24)),
        value_blend_w=float(options.get("blend", 0.0)),
        leaf_batch=int(options.get("batch", 0)),
        top_k=int(options.get("top_k", 0)),
        late_sims=int(options.get("late", 0)),
        root_choice=str(options.get("root", "visits")),
        device=device,
    )


def _split_specs(spec: str) -> list[str]:
    """Split on commas that separate agents, not the ones inside options.

    ``greedy,mcts:ckpt=a.pt,sims=64`` is two agents, not three: a comma
    after a ``k=v`` chunk continues that agent's option list.
    """
    specs: list[str] = []
    for chunk in filter(None, (c.strip() for c in spec.split(","))):
        # a continuation is a bare k=v; anything carrying ':' or '@' names
        # a new agent, even though it also contains '='
        is_option = "=" in chunk and ":" not in chunk and "@" not in chunk
        if specs and is_option:
            specs[-1] = f"{specs[-1]},{chunk}"
        else:
            specs.append(chunk)
    return specs


def _build_agents(spec: str) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for one in _split_specs(spec):
        try:
            name, label, options = parse_spec(one)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if label in agents:
            raise SystemExit(f"duplicate agent label {label!r} -- use name@label")
        if name in _NET_AGENTS:
            try:
                agents[label] = _build_net_agent(name, label, options)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            agents[label] = _AGENT_FACTORIES[name](name=label)
    if not agents:
        raise SystemExit("no agents given")
    return agents


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a mirrored arena series.")
    parser.add_argument("--tables", type=int, default=50, help="setups; 4 games each")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", default="random,greedy", help="comma-separated names")
    parser.add_argument("--report", type=Path, default=None, help="HTML report path")
    parser.add_argument(
        "--workers", type=int, default=1, help="processes to shard tables across"
    )
    args = parser.parse_args(argv)

    agents = _build_agents(args.agents)
    names = list(agents)
    base_seats = tuple(names[i % len(names)] for i in range(4))
    if args.workers > 1:
        from bgai.arena.parallel import run_series_parallel

        series = run_series_parallel(
            spec=args.agents,
            base_seats=base_seats,
            n_tables=args.tables,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        series = run_series(
            agents=agents, base_seats=base_seats, n_tables=args.tables, seed=args.seed
        )
    if args.report is not None:
        write_report(series, args.report)
    outcomes = agent_outcomes(series.results)
    print(f"{len(series.results)} games, {series.n_errors} errors")
    print(f"{'agent':<12}{'win%':>8}{'mean place':>12}{'mean VP':>10}{'μ−3σ':>9}{'seats':>8}")
    for name in sorted(series.ratings, key=lambda n: -conservative(series.ratings[n])):
        o = outcomes.get(name)
        if o is None:
            continue
        print(
            f"{name:<12}{o['win_rate'] * 100:>7.1f}%{o['mean_place']:>12.2f}"
            f"{o['mean_vp']:>10.1f}{conservative(series.ratings[name]):>9.2f}"
            f"{o['seat_games']:>8}"
        )
    if args.report:
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
