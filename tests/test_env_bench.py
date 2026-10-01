"""The baseline/throughput/conformance runner.

Episode counts here are tiny -- this pins the CLI's plumbing, not any
strength claim. The measured numbers in docs/tm-env.md come from real runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bgai.env.config import MAX_PLAYERS, MIN_PLAYERS, EnvConfig, RewardMode
from bgai.env.bench import (
    RunStats,
    main,
    run_conformance,
    run_random,
    run_with_policy,
)

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


@pytest.mark.parametrize("count", COUNTS)
def test_random_baseline_reports_real_numbers(count: int) -> None:
    stats = run_random(EnvConfig(player_count=count), episodes=2, seed=7)
    assert isinstance(stats, RunStats)
    assert stats.player_count == count
    assert stats.completed == 2
    assert stats.errored == 0
    assert stats.decisions_min <= stats.decisions_mean <= stats.decisions_max
    assert stats.vp_min <= stats.vp_mean <= stats.vp_max
    assert stats.vp_mean > 0
    assert stats.table_total_mean == pytest.approx(stats.vp_mean * count, rel=0.05)
    assert stats.decisions_per_sec > 0
    assert f"{count}p" in stats.line()


def test_terminal_share_returns_sum_to_zero() -> None:
    stats = run_random(
        EnvConfig(player_count=4, reward_mode=RewardMode.TERMINAL_VP_SHARE),
        episodes=2,
        seed=8,
    )
    assert stats.reward_mean == pytest.approx(0.0, abs=1e-9)


def test_greedy_in_the_learner_seat_reports_learner_metrics() -> None:
    stats = run_with_policy(
        EnvConfig(player_count=4), episodes=2, seed=9, learner="greedy", checkpoint=None
    )
    assert stats.label == "greedy-vs-random"
    assert stats.learner_vp_mean is not None and stats.learner_vp_mean > 0
    assert stats.learner_win_rate is not None
    assert 0.0 <= stats.learner_win_rate <= 1.0


def test_conformance_runner_reports_each_check() -> None:
    results = run_conformance((2,))
    checks = {row["check"] for row in results}
    assert checks == {
        "pettingzoo.api_test(raw_env)",
        "pettingzoo.api_test(env)",
        "gymnasium.check_env",
    }
    assert all(row["passed"] for row in results)
    assert all(row["players"] == 2 for row in results)


def test_cli_writes_a_json_report(tmp_path: Path, capsys) -> None:
    report = tmp_path / "bench.json"
    main(
        [
            "--episodes",
            "1",
            "--players",
            "2",
            "--raw-throughput",
            "--report",
            str(report),
        ]
    )
    out = capsys.readouterr().out
    assert "random-vs-random" in out
    assert "Episode-only" in out
    payload = json.loads(report.read_text())
    assert payload["episodes"] == 1
    assert len(payload["runs"]) == 1
    assert "2" in payload["episode_only_decisions_per_sec"]


def test_cli_can_skip_the_random_baseline(capsys) -> None:
    main(["--players", "2", "--no-random", "--conformance"])
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "random-vs-random" not in out
