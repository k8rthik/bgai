"""Learner process: steps only on enough new games, tracks staleness."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from selfplay_helpers import synthetic_records, write_tiny_checkpoint

from bgai.training import selfplay_learner as sl
from bgai.training import shards as sh
from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.selfplay_finetune import FineTuneConfig


def _info(name: str, n_games: int, ckpt_iteration: int, created_at: float, n_records=10):
    return sl.ShardInfo(
        path=Path(name), n_games=n_games, ckpt_iteration=ckpt_iteration,
        created_at=created_at, n_records=n_records, ok=True,
    )


def _write(shards_dir: Path, seq: int, n_games: int, ckpt_iteration: int, n_records: int,
           actor_id: int = 0, created_at: float | None = None) -> Path:
    rng = np.random.default_rng(seq)
    path = shards_dir / sh.shard_name(ckpt_iteration, actor_id, seq)
    sh.write_shard(
        path, synthetic_records(n_records, rng), ckpt_iteration=ckpt_iteration,
        actor_id=actor_id, seed=seq, config_hash="t", n_games=n_games, created_at=created_at,
    )
    return path


# -- pure pieces -----------------------------------------------------------


def test_should_step_needs_games_per_step_new_games() -> None:
    assert not sl.should_step(total_games=5, games_at_last_step=0, games_per_step=6)
    assert sl.should_step(total_games=6, games_at_last_step=0, games_per_step=6)
    assert not sl.should_step(total_games=11, games_at_last_step=6, games_per_step=6)
    assert sl.should_step(total_games=12, games_at_last_step=6, games_per_step=6)


def test_select_window_takes_newest_shards_until_games_covered() -> None:
    infos = [_info("a", 4, 0, 1.0), _info("b", 4, 0, 2.0), _info("c", 4, 1, 3.0),
             _info("d", 2, 1, 4.0)]
    window = sl.select_window(infos, games=8)
    assert [i.path.name for i in window] == ["b", "c", "d"]  # 2 + 4 + 4 >= 8, newest first
    assert [i.path.name for i in sl.select_window(infos, games=100)] == ["a", "b", "c", "d"]
    assert sl.select_window([], games=8) == ()


def test_staleness_stats_weights_by_records() -> None:
    infos = [_info("a", 1, 2, 1.0, n_records=30), _info("b", 1, 3, 2.0, n_records=10)]
    stats = sl.staleness_stats(iteration=3, window=infos)
    assert stats["staleness_hist"] == {"1": 30, "0": 10}
    assert stats["staleness_mean"] == 0.75
    assert stats["staleness_max"] == 1


def test_scan_shards_skips_tmp_and_corrupt_and_is_incremental(tmp_path: Path) -> None:
    d = tmp_path / "shards"
    d.mkdir()
    good = _write(d, 0, n_games=2, ckpt_iteration=0, n_records=5)
    (d / "it0000_a000_000001.tmp").write_bytes(b"partial")
    bad = d / sh.shard_name(0, 0, 2)
    bad.write_bytes(b"corrupt")
    first = sl.scan_shards(d, {})
    assert set(first) == {good.name, bad.name}
    assert first[good.name].ok and first[good.name].n_games == 2
    assert not first[bad.name].ok and first[bad.name].n_games == 0
    # incremental: a second scan reuses known entries (same objects), adds new ones
    newer = _write(d, 3, n_games=1, ckpt_iteration=0, n_records=5)
    second = sl.scan_shards(d, first)
    assert second[good.name] is first[good.name]
    assert set(second) == set(first) | {newer.name}
    assert first is not second and newer.name not in first  # no mutation of the old map
    assert sl.total_games(second) == 3


# -- whole learner ---------------------------------------------------------


def _cfg(run_dir: Path, init: Path, **kw) -> sl.LearnerConfig:
    return sl.LearnerConfig(
        run_dir=run_dir, checkpoint=init, games_per_step=2, window_steps=2,
        finetune=FineTuneConfig(batch_size=8, lr=1e-3), poll_secs=0.05,
        idle_timeout_secs=1.0, **kw,
    )


def test_learner_initializes_current_and_waits_without_enough_games(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    _write(run_dir / "shards", 0, n_games=1, ckpt_iteration=0, n_records=20)
    steps = sl.run_learner(_cfg(run_dir, init, max_iterations=1))
    assert steps == 0  # idle timeout, one game < games_per_step
    current = load_checkpoint(run_dir / "current.pt")
    assert current["iteration"] == 0
    assert current["config"]["model_config"]["hidden"] == 64
    assert not (run_dir / "metrics.jsonl").exists()


def test_learner_steps_once_enough_games_arrive_and_logs_staleness(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    t0 = time.time() - 3600.0
    _write(run_dir / "shards", 0, n_games=1, ckpt_iteration=0, n_records=20, created_at=t0)
    _write(run_dir / "shards", 1, n_games=1, ckpt_iteration=0, n_records=20,
           created_at=t0 + 1800.0)
    before = load_checkpoint(init)["model"]
    steps = sl.run_learner(_cfg(run_dir, init, max_iterations=1))
    assert steps == 1
    current = load_checkpoint(run_dir / "current.pt")
    assert current["iteration"] == 1
    assert (run_dir / "iter_0001.pt").exists()
    import torch

    assert any(not torch.equal(before[k], v) for k, v in current["model"].items())
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["iteration"] == 1 and row["new_games"] == 2 and row["window_games"] == 2
    assert row["window_records"] == 40
    assert row["staleness_hist"] == {"0": 40} and row["staleness_mean"] == 0.0
    assert row["games_per_hour"] > 0
    assert {"policy", "value", "kl", "total"} <= set(row)


def test_learner_second_step_uses_window_and_reports_staleness(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    d = run_dir / "shards"
    _write(d, 0, n_games=2, ckpt_iteration=0, n_records=20)
    assert sl.run_learner(_cfg(run_dir, init, max_iterations=1)) == 1
    # actors kept playing on iteration 0 while the learner stepped: stale by 1
    _write(d, 1, n_games=1, ckpt_iteration=0, n_records=20)
    _write(d, 2, n_games=1, ckpt_iteration=1, n_records=20)
    assert sl.run_learner(_cfg(run_dir, init, max_iterations=2)) == 1  # total, not "1 more"
    assert load_checkpoint(run_dir / "current.pt")["iteration"] == 2
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2]
    row = rows[1]
    assert row["new_games"] == 2 and row["window_games"] == 4 and row["window_records"] == 60
    assert row["staleness_hist"] == {"1": 40, "0": 20}
    assert row["staleness_mean"] == round(40 / 60, 4)


def test_learner_skips_corrupt_shard_in_window(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    d = run_dir / "shards"
    _write(d, 0, n_games=2, ckpt_iteration=0, n_records=20)
    d.mkdir(exist_ok=True)
    (d / sh.shard_name(0, 0, 1)).write_bytes(b"corrupt")
    assert sl.run_learner(_cfg(run_dir, init, max_iterations=1)) == 1
    assert (d / sh.shard_name(0, 0, 1)).exists(), "learner never deletes shards"
