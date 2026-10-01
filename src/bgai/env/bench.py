"""End-to-end TM-Env runs: baselines, throughput, API conformance.

    # random play through the PettingZoo AEC env, all four player counts
    uv run python -m bgai.env.bench --episodes 20 --players 2,3,4,5

    # an existing checkpoint playing the Gymnasium wrapper's learner seat
    uv run python -m bgai.env.bench --episodes 8 --players 4 \
        --learner imitation --checkpoint data/checkpoints/selfplay_leg5b/current.pt

    # PettingZoo api_test + Gymnasium check_env
    uv run python -m bgai.env.bench --conformance --players 2,3,4,5

Everything it prints is a measurement, not a target: episode length, VP
distribution, decisions/sec. VP is directly comparable to the numbers in
the README's absolute-strength table, because it is the engine's own
post-``final_scoring`` VP.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bgai.env.config import EnvConfig, RewardMode
from bgai.env.core import Episode
from bgai.env.opponents import build_opponent

__all__ = ["RunStats", "main", "run_conformance", "run_random", "run_with_policy"]


@dataclass(frozen=True)
class RunStats:
    label: str
    player_count: int
    episodes: int
    completed: int
    errored: int
    decisions_mean: float
    decisions_min: int
    decisions_max: int
    vp_mean: float
    vp_stdev: float
    vp_min: int
    vp_max: int
    table_total_mean: float
    reward_mean: float
    decisions_per_sec: float
    wall_secs: float
    learner_vp_mean: float | None = None
    learner_win_rate: float | None = None

    def line(self) -> str:
        extra = ""
        if self.learner_vp_mean is not None:
            extra = (
                f" learner_vp={self.learner_vp_mean:.1f}"
                f" learner_win={self.learner_win_rate:.3f}"
            )
        return (
            f"{self.label:<26} {self.player_count}p n={self.episodes:<4} "
            f"ok={self.completed} err={self.errored} "
            f"len={self.decisions_mean:.1f}[{self.decisions_min}-{self.decisions_max}] "
            f"vp={self.vp_mean:.1f}+/-{self.vp_stdev:.1f}"
            f"[{self.vp_min}-{self.vp_max}] "
            f"table={self.table_total_mean:.1f} "
            f"{self.decisions_per_sec:.0f} dec/s{extra}"
        )


def _summarize(
    label: str,
    config: EnvConfig,
    episodes: int,
    lengths: list[int],
    vps: list[int],
    totals: list[int],
    rewards: list[float],
    errors: int,
    wall: float,
    learner_vps: list[int] | None = None,
    learner_wins: list[float] | None = None,
) -> RunStats:
    total_decisions = sum(lengths)
    return RunStats(
        label=label,
        player_count=config.player_count,
        episodes=episodes,
        completed=len(lengths),
        errored=errors,
        decisions_mean=statistics.fmean(lengths) if lengths else 0.0,
        decisions_min=min(lengths) if lengths else 0,
        decisions_max=max(lengths) if lengths else 0,
        vp_mean=statistics.fmean(vps) if vps else 0.0,
        vp_stdev=statistics.stdev(vps) if len(vps) > 1 else 0.0,
        vp_min=min(vps) if vps else 0,
        vp_max=max(vps) if vps else 0,
        table_total_mean=statistics.fmean(totals) if totals else 0.0,
        reward_mean=statistics.fmean(rewards) if rewards else 0.0,
        decisions_per_sec=total_decisions / wall if wall > 0 else 0.0,
        wall_secs=wall,
        learner_vp_mean=(
            statistics.fmean(learner_vps) if learner_vps else None
        ),
        learner_win_rate=(
            statistics.fmean(learner_wins) if learner_wins else None
        ),
    )


def run_random(
    config: EnvConfig, episodes: int, seed: int, label: str = "random-vs-random"
) -> RunStats:
    """Uniform-random masked play through the AEC env.

    Goes through the real PettingZoo env (``agent_iter``/``last``/``step``),
    not through ``Episode`` directly, so what it measures is the wrapper.
    """
    from bgai.env.aec import raw_env

    env = raw_env(config)
    rng = random.Random(seed)
    lengths: list[int] = []
    vps: list[int] = []
    totals: list[int] = []
    rewards: list[float] = []
    errors = 0
    start = time.perf_counter()
    for episode_index in range(episodes):
        env.reset(seed=seed + episode_index)
        returns: dict[str, float] = dict.fromkeys(env.agents, 0.0)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            returns[agent] = returns.get(agent, 0.0) + reward
            if terminated or truncated:
                env.step(None)
                continue
            legal = np.flatnonzero(obs["action_mask"])
            env.step(int(legal[rng.randrange(len(legal))]))
        table = env.episode.vps()
        if env.episode.engine_error is not None:
            errors += 1
            continue
        lengths.append(env.episode.decisions)
        vps.extend(table.values())
        totals.append(sum(table.values()))
        rewards.extend(returns.values())
    wall = time.perf_counter() - start
    return _summarize(
        label, config, episodes, lengths, vps, totals, rewards, errors, wall
    )


def run_with_policy(
    config: EnvConfig,
    episodes: int,
    seed: int,
    learner: str,
    checkpoint: Path | None,
    label: str | None = None,
    **agent_kwargs: object,
) -> RunStats:
    """Run an existing ``bgai.agents`` agent in the Gymnasium learner seat.

    Seats rotate across episodes, so the agent plays every seat roughly
    equally -- Terra Mystica is strongly seat-dependent and a fixed seat
    would conflate seat strength with agent strength (D4.3).
    """
    from bgai.env.single import TerraMysticaSingleEnv

    policy = build_opponent(
        learner,
        checkpoint=checkpoint,
        player_count=config.player_count,
        **agent_kwargs,
    )
    opponent = build_opponent("random", player_count=config.player_count)
    env = TerraMysticaSingleEnv(
        config=config, opponent=opponent, learner_seat=0, rotate_seats=True
    )
    rng = random.Random(seed)
    lengths: list[int] = []
    vps: list[int] = []
    totals: list[int] = []
    rewards: list[float] = []
    learner_vps: list[int] = []
    learner_wins: list[float] = []
    errors = 0
    start = time.perf_counter()
    for episode_index in range(episodes):
        obs, _ = env.reset(seed=seed + episode_index)
        policy.begin_episode()
        total_reward = 0.0
        while True:
            episode = env.episode
            faction, offer = episode.pending()  # type: ignore[misc]
            index = policy(episode.sim, faction, offer, rng)
            obs, reward, terminated, truncated, info = env.step(index)
            total_reward += reward
            if terminated or truncated:
                break
        table = env.episode.vps()
        if env.episode.engine_error is not None:
            errors += 1
            continue
        lengths.append(env.episode.decisions)
        vps.extend(table.values())
        totals.append(sum(table.values()))
        rewards.append(total_reward)
        own = table[env.learner_faction]
        learner_vps.append(own)
        best = max(table.values())
        learner_wins.append(1.0 if own == best else 0.0)
    wall = time.perf_counter() - start
    return _summarize(
        label or f"{learner}-vs-random",
        config,
        episodes,
        lengths,
        vps,
        totals,
        rewards,
        errors,
        wall,
        learner_vps=learner_vps,
        learner_wins=learner_wins,
    )


def run_conformance(player_counts: tuple[int, ...]) -> list[dict[str, object]]:
    """PettingZoo ``api_test`` + Gymnasium ``check_env``, per player count."""
    import warnings

    from gymnasium.utils.env_checker import check_env
    from pettingzoo.test import api_test

    from bgai.env.aec import env as wrapped_env
    from bgai.env.aec import raw_env
    from bgai.env.single import make_single_env

    results: list[dict[str, object]] = []
    for count in player_counts:
        config = EnvConfig(player_count=count)
        for name, make in (
            ("pettingzoo.api_test(raw_env)", lambda c=config: raw_env(c)),
            ("pettingzoo.api_test(env)", lambda c=config: wrapped_env(c)),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                api_test(make(), num_cycles=30)
            results.append(
                {
                    "check": name,
                    "players": count,
                    "passed": True,
                    "warnings": sorted({str(w.message) for w in caught}),
                }
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(
                make_single_env(player_count=count), skip_render_check=False
            )
        results.append(
            {
                "check": "gymnasium.check_env",
                "players": count,
                "passed": True,
                "warnings": sorted({str(w.message) for w in caught}),
            }
        )
    return results


def _episode_only_throughput(config: EnvConfig, episodes: int, seed: int) -> float:
    """Decisions/sec straight through ``Episode``, no wrapper, for the
    wrapper-overhead line in docs/tm-env.md.
    """
    rng = random.Random(seed)
    decisions = 0
    start = time.perf_counter()
    for index in range(episodes):
        episode = Episode.start(seed + index, config)
        while not episode.finished:
            _, offer = episode.pending()  # type: ignore[misc]
            episode, _ = episode.step(rng.randrange(len(offer)))
            decisions += 1
    return decisions / (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TM-Env baselines and conformance.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--players", default="4", help="comma-separated, e.g. 2,3,4,5")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--reward", default=RewardMode.TERMINAL_VP_SHARE.value,
                        choices=[m.value for m in RewardMode])
    parser.add_argument("--learner", default=None,
                        help="run this agent in the Gymnasium learner seat")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per decision")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--leaf-batch", type=int, default=None)
    parser.add_argument("--conformance", action="store_true")
    parser.add_argument("--no-random", action="store_true",
                        help="skip the random-play baseline")
    parser.add_argument("--raw-throughput", action="store_true",
                        help="also time Episode with no API wrapper")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    counts = tuple(int(p) for p in str(args.players).split(","))
    report: dict[str, object] = {"seed": args.seed, "episodes": args.episodes}

    if args.conformance:
        results = run_conformance(counts)
        report["conformance"] = results
        for row in results:
            print(f"PASS  {row['check']:<30} {row['players']}p")
            for warning in row["warnings"]:  # type: ignore[union-attr]
                print(f"      warn: {warning}")

    runs: list[RunStats] = []
    if not args.no_random:
        for count in counts:
            config = EnvConfig(player_count=count, reward_mode=RewardMode(args.reward))
            runs.append(run_random(config, args.episodes, args.seed))

    if args.learner is not None:
        kwargs = {
            key: value
            for key, value in (
                ("simulations", args.simulations),
                ("top_k", args.top_k),
                ("c_puct", args.c_puct),
                ("max_depth", args.max_depth),
                ("leaf_batch", args.leaf_batch),
            )
            if value is not None
        }
        for count in counts:
            config = EnvConfig(player_count=count, reward_mode=RewardMode(args.reward))
            runs.append(
                run_with_policy(
                    config,
                    args.episodes,
                    args.seed,
                    args.learner,
                    args.checkpoint,
                    **kwargs,
                )
            )

    for stats in runs:
        print(stats.line())
    report["runs"] = [asdict(s) for s in runs]

    if args.raw_throughput:
        raw = {
            str(count): _episode_only_throughput(
                EnvConfig(player_count=count), max(args.episodes // 2, 1), args.seed
            )
            for count in counts
        }
        report["episode_only_decisions_per_sec"] = raw
        for count, rate in raw.items():
            print(f"Episode-only (no API wrapper) {count}p: {rate:.0f} dec/s")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.report}")


if __name__ == "__main__":  # pragma: no cover -- CLI
    main()
