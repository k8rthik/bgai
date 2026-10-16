"""Regression tests for the actor/learner review findings (1-9)."""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from selfplay_helpers import synthetic_records, write_tiny_checkpoint

from bgai.training import selfplay_actor as sa
from bgai.training import selfplay_launcher as launcher
from bgai.training import selfplay_learner as sl
from bgai.training import selfplay_train as st
from bgai.training import shards as sh
from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.selfplay import SelfPlayConfig
from bgai.training.selfplay_finetune import FineTuneConfig
from bgai.training.stop_signal import StopSignal

FAST = SelfPlayConfig(simulations=2, top_k=3, max_depth=4, leaf_batch=2, fast_prior=0.9)
REPO = Path(__file__).resolve().parents[1]


def _rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def _future_stop(run_dir: Path) -> Path:
    """A STOP file that is unambiguously newer than any process start."""
    stop = run_dir / "STOP"
    stop.touch()
    future = time.time_ns() + 60_000_000_000
    os.utime(stop, ns=(future, future))
    return stop


def _write_shard(d: Path, seq: int, n_games: int, n_records: int = 20, **meta) -> Path:
    path = d / sh.shard_name(0, 0, seq)
    sh.write_shard(
        path, synthetic_records(n_records, np.random.default_rng(seq)),
        **{"ckpt_iteration": 0, "actor_id": 0, "seed": seq, "config_hash": "t",
           "n_games": n_games, **meta},
    )
    return path


def _learner_cfg(run_dir: Path, init: Path, **kw) -> sl.LearnerConfig:
    return sl.LearnerConfig(
        run_dir=run_dir, checkpoint=init, games_per_step=2, window_steps=2,
        finetune=FineTuneConfig(batch_size=8, lr=1e-3), poll_secs=0.05,
        idle_timeout_secs=1.0, **kw,
    )


# -- 1: pipelined resume keeps mode + generation settings -------------------


def _pipelined_cfg(tmp_path: Path, **kw) -> st.SelfPlayTrainConfig:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    base = dict(
        checkpoint=init, out=tmp_path / "run", iterations=1, games_per_iteration=4,
        workers=2, actors=2, simulations=8, top_k=4, max_depth=8, leaf_batch=4,
        batch_size=32, lr=1e-3, games_per_shard=2, idle_timeout_secs=120.0,
        mode="pipelined", c_puct=2.5,
    )
    return st.SelfPlayTrainConfig(**{**base, **kw})


def test_pipelined_resume_restores_mode_and_generation_settings(tmp_path: Path) -> None:
    cfg = _pipelined_cfg(tmp_path)
    assert launcher.run_pipelined(cfg, actor_stop_timeout=120.0) == 1
    resumed = st.resumed_config(cfg.out, {"extra_iterations": 1})
    assert resumed.mode == "pipelined"
    assert (resumed.simulations, resumed.c_puct, resumed.games_per_iteration) == (8, 2.5, 4)
    assert (resumed.workers, resumed.actors, resumed.games_per_shard) == (2, 2, 2)
    assert resumed.iterations == 2 and resumed.anchor == cfg.checkpoint
    assert launcher.run_pipelined(resumed, actor_stop_timeout=120.0) == 1
    assert (cfg.out / "iter_0002.pt").exists()
    assert not (cfg.out / "iter_02.pt").exists(), "must not continue as a serial loop"
    assert load_checkpoint(cfg.out / "current.pt")["iteration"] == 2


# -- 2: sync script -----------------------------------------------------------


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_sync_script_pushes_shards_pulls_checkpoint_and_mirrors_stop_removal(
    tmp_path: Path,
) -> None:
    remote, local = tmp_path / "remote", tmp_path / "local"
    (remote / "shards").mkdir(parents=True)
    (local / "shards").mkdir(parents=True)
    write_tiny_checkpoint(remote / "current.pt", iteration=3)
    _write_shard(local / "shards", 0, n_games=2)
    (local / "shards" / "it0000_a000_000001.tmp").write_bytes(b"partial")
    (remote / "STOP").touch()
    script = REPO / "tools" / "sync_run_dir.sh"
    env = {**os.environ, "SYNC_MAX_ROUNDS": "1"}
    run = subprocess.run(
        ["bash", str(script), str(remote), str(local), "0"], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert load_checkpoint(local / "current.pt")["iteration"] == 3
    assert (remote / "shards" / sh.shard_name(0, 0, 0)).exists()
    assert not (remote / "shards" / "it0000_a000_000001.tmp").exists()
    assert (local / "STOP").exists()
    # learner host removed STOP: the mirror must drop it too
    (remote / "STOP").unlink()
    run = subprocess.run(
        ["bash", str(script), str(remote), str(local), "0"], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert not (local / "STOP").exists()
    # rsync failures are reported, not swallowed
    run = subprocess.run(
        ["bash", str(script), str(tmp_path / "nowhere"), str(local), "0"], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert "sync:" in run.stderr and "checkpoint" in run.stderr


def test_sync_script_has_no_invalid_rsync_flags() -> None:
    text = (REPO / "tools" / "sync_run_dir.sh").read_text()
    assert "--inplace=no" not in text
    assert "2>/dev/null" not in text


def test_stale_stop_older_than_process_start_is_removed_and_ignored(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP"
    stop_file.touch()
    past = time.time_ns() - 5_000_000_000
    os.utime(stop_file, ns=(past, past))
    stop = StopSignal(tmp_path)
    assert not stop.requested()
    assert not stop_file.exists(), "a stale STOP from a previous run is cleaned up"


def test_actor_ignores_stale_stop_from_a_previous_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_tiny_checkpoint(run_dir / "current.pt", iteration=0)
    stop_file = run_dir / "STOP"
    stop_file.touch()
    past = time.time_ns() - 5_000_000_000
    os.utime(stop_file, ns=(past, past))
    cfg = sa.ActorConfig(run_dir=run_dir, actor_id=0, games_per_shard=1, workers=1,
                         max_games=1, selfplay=FAST)
    assert sa.run_actor(cfg) == 1
    assert not stop_file.exists()


# -- 3: worker signals + drain cannot hang ---------------------------------


def test_worker_init_ignores_sigint_and_sigterm(tmp_path: Path) -> None:
    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        sa._actor_worker_init(str(ckpt), FAST)
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        st._worker_init(str(ckpt), FAST)
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_drain_recovers_from_a_lost_worker_task_and_flushes(tmp_path: Path) -> None:
    """A worker killed mid-task never reports; the drain must notice the
    death, drop the task and still write the buffered games."""
    run_dir = tmp_path / "run"
    (run_dir / "shards").mkdir(parents=True)
    cfg = sa.ActorConfig(run_dir=run_dir, actor_id=1, games_per_shard=8, workers=1,
                         max_games=2, selfplay=FAST)
    done: queue.Queue = queue.Queue()
    rng = np.random.default_rng(0)
    buffered = sa.ShardBuffer(
        ckpt_iteration=0, games=(tuple(synthetic_records(3, rng)),), seeds=(7,), indices=(0,),
    )
    progress = sa._Progress(next_index=2, played=1, prior_games=0, in_flight=1, seq=0,
                            buffer=buffered)
    deaths = iter([0, 1, 0, 0])

    def lost_workers() -> int:
        return next(deaths, 0)

    def submit(p):
        raise AssertionError("nothing should be submitted after max_games")

    t0 = time.monotonic()
    out = sa._drain(cfg, progress, done, submit, "h", StopSignal(run_dir), lost_workers,
                    poll_secs=0.05)
    assert time.monotonic() - t0 < 5.0
    assert out.in_flight == 0 and out.buffer.n_games == 0
    written = [p for p in (run_dir / "shards").iterdir() if sh.is_shard_file(p)]
    assert len(written) == 1 and sh.read_shard_meta(written[0]).n_games == 1


def test_drain_flushes_buffer_when_it_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "shards").mkdir(parents=True)
    cfg = sa.ActorConfig(run_dir=run_dir, actor_id=1, games_per_shard=8, workers=1,
                         selfplay=FAST)
    done: queue.Queue = queue.Queue()
    rng = np.random.default_rng(0)
    buffered = sa.ShardBuffer(
        ckpt_iteration=0, games=(tuple(synthetic_records(3, rng)),), seeds=(7,), indices=(0,),
    )
    progress = sa._Progress(next_index=1, played=1, prior_games=0, in_flight=1, seq=0,
                            buffer=buffered)
    done.put(("ok", sa.WorkerResult(seed=1, game_index=1, ckpt_iteration=0,
                                    records=tuple(synthetic_records(2, rng)),
                                    reloaded=False, secs=0.0)))

    def submit(p):
        raise RuntimeError("pool exploded")

    with pytest.raises(RuntimeError, match="pool exploded"):
        sa._drain(cfg, progress, done, submit, "h", StopSignal(run_dir), lambda: 0)
    written = [p for p in (run_dir / "shards").iterdir() if sh.is_shard_file(p)]
    assert len(written) == 1 and sh.read_shard_meta(written[0]).n_games == 2


# -- 4: learner keeps STOP; actors latch -----------------------------------


def test_learner_does_not_delete_stop_after_honouring_it(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    d = run_dir / "shards"
    _write_shard(d, 0, n_games=2)
    stop_file = _future_stop(run_dir)
    assert sl.run_learner(_learner_cfg(run_dir, init, max_iterations=10)) == 1
    assert _rows(run_dir)[-1]["event"] == "stopped"
    assert stop_file.exists(), "only the run owner (launcher/operator) removes STOP"


def test_stop_signal_latches_once_seen(tmp_path: Path) -> None:
    stop = StopSignal(tmp_path)
    stop_file = _future_stop(tmp_path)
    assert stop.requested()
    stop_file.unlink()
    assert stop.requested(), "a stop seen once stays requested"


# -- 5: max_iterations is the run total ------------------------------------


def test_learner_max_iterations_is_total_across_restarts(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    d = run_dir / "shards"
    _write_shard(d, 0, n_games=2)
    assert sl.run_learner(_learner_cfg(run_dir, init, max_iterations=1)) == 1
    _write_shard(d, 1, n_games=2)
    # total already reached: a bare restart must not run "1 more"
    assert sl.run_learner(_learner_cfg(run_dir, init, max_iterations=1)) == 0
    assert load_checkpoint(run_dir / "current.pt")["iteration"] == 1
    assert sl.run_learner(_learner_cfg(run_dir, init, max_iterations=2)) == 1
    assert load_checkpoint(run_dir / "current.pt")["iteration"] == 2


def test_resumed_learner_config_total_and_extra_iterations(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    _write_shard(run_dir / "shards", 0, n_games=2)
    assert sl.run_learner(_learner_cfg(run_dir, init, max_iterations=3)) == 1
    stored = sl.resumed_learner_config(run_dir, {})
    assert stored.max_iterations == 3 and stored.finetune.batch_size == 8
    extra = sl.resumed_learner_config(run_dir, {"extra_iterations": 2})
    assert extra.max_iterations == 3  # checkpoint iteration 1 + 2
    total = sl.resumed_learner_config(run_dir, {"max_iterations": 7})
    assert total.max_iterations == 7
    argv_cfg = sl.config_from_argv(["--run-dir", str(run_dir), "--extra-iterations", "4"])
    assert argv_cfg.max_iterations == 5 and argv_cfg.finetune.lr == 1e-3
    argv_cfg = sl.config_from_argv(["--run-dir", str(run_dir), "--max-iterations", "9"])
    assert argv_cfg.max_iterations == 9
    with pytest.raises(SystemExit, match="extra-iterations"):
        sl.config_from_argv(["--run-dir", str(run_dir), "--max-iterations", "9",
                             "--extra-iterations", "1"])


def test_launcher_passes_total_iterations_to_learner(tmp_path: Path) -> None:
    cfg = _pipelined_cfg(tmp_path, iterations=5)
    assert launcher.learner_config(cfg).max_iterations == 5
    assert launcher.learner_config(replace(cfg, resume=cfg.out)).max_iterations == 5


# -- 6: exact per-actor game accounting across restarts --------------------


def _own_shards(run_dir: Path, actor_id: int) -> list[sh.ShardMeta]:
    files = sorted(p for p in (run_dir / "shards").iterdir()
                   if sh.is_shard_file(p) and f"_a{actor_id:03d}_" in p.name)
    return [sh.read_shard_meta(p) for p in files]


def test_restart_after_early_flushed_shards_continues_seeds_exactly(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_tiny_checkpoint(run_dir / "current.pt", iteration=0)
    base = dict(run_dir=run_dir, actor_id=3, workers=1, seed=11, selfplay=FAST)
    assert sa.run_actor(sa.ActorConfig(games_per_shard=2, max_games=3, **base)) == 3
    metas = _own_shards(run_dir, 3)
    assert [m.n_games for m in metas] == [2, 1] and metas[-1].game_index_end == 3
    # restart: two more games (max_games is lifetime), indices 3 and 4
    assert sa.run_actor(sa.ActorConfig(games_per_shard=2, max_games=5, **base)) == 2
    metas = _own_shards(run_dir, 3)
    assert sum(m.n_games for m in metas) == 5
    starts = [m.game_index_start for m in metas]
    ends = [m.game_index_end for m in metas]
    assert starts == [0, 2, 3] and ends == [2, 3, 5], "no gap, no replay"
    assert [m.seed for m in metas] == [sa.game_seed(11, 3, i) for i in starts]
    # lifetime max reached: a further restart plays nothing
    assert sa.run_actor(sa.ActorConfig(games_per_shard=2, max_games=5, **base)) == 0


def test_restart_with_different_games_per_shard_never_replays_seeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_tiny_checkpoint(run_dir / "current.pt", iteration=0)
    base = dict(run_dir=run_dir, actor_id=0, workers=1, seed=3, selfplay=FAST)
    assert sa.run_actor(sa.ActorConfig(games_per_shard=2, max_games=3, **base)) == 3
    assert sa.run_actor(sa.ActorConfig(games_per_shard=5, max_games=5, **base)) == 2
    metas = _own_shards(run_dir, 0)
    ranges = sorted((m.game_index_start, m.game_index_end) for m in metas)
    covered = [i for a, b in ranges for i in range(a, b)]
    assert covered == list(range(5)), ranges


def test_absorb_tracks_game_indices() -> None:
    rng = np.random.default_rng(0)
    res = sa.WorkerResult(seed=9, game_index=4, ckpt_iteration=0,
                          records=tuple(synthetic_records(1, rng)), reloaded=False, secs=0.0)
    buf, _ = sa.absorb(sa.ShardBuffer.empty(), res, games_per_shard=8)
    assert buf.indices == (4,) and buf.game_index_end == 5


# -- 7: checkpoint wait honours STOP/SIGTERM -------------------------------


def test_checkpoint_wait_aborts_on_stop(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stop = StopSignal(run_dir)
    threading.Timer(0.3, lambda: _future_stop(run_dir)).start()
    t0 = time.monotonic()
    with pytest.raises(sa.ActorStopped):
        sa._wait_for_checkpoint(run_dir / "current.pt", timeout=60.0, stop=stop)
    assert time.monotonic() - t0 < 10.0


def test_run_actor_exits_promptly_when_stopped_before_any_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = sa.ActorConfig(run_dir=run_dir, actor_id=0, selfplay=FAST, checkpoint_wait_secs=60.0)
    threading.Timer(0.3, lambda: _future_stop(run_dir)).start()
    t0 = time.monotonic()
    assert sa.run_actor(cfg) == 0
    assert time.monotonic() - t0 < 10.0


def test_launcher_gives_actors_a_short_checkpoint_wait(tmp_path: Path) -> None:
    cfg = _pipelined_cfg(tmp_path)
    argv = launcher.actor_argv(cfg, 0, 1)
    wait = float(argv[argv.index("--checkpoint-wait-secs") + 1])
    assert wait <= 300.0


# -- 8: conflicting resume flags are hard errors ---------------------------


def test_resume_cli_conflicts_are_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--checkpoint"):
        st.config_from_argv(["--resume", "r", "--checkpoint", "x.pt"])
    with pytest.raises(SystemExit, match="--out"):
        st.config_from_argv(["--resume", "r", "--out", "o"])
    with pytest.raises(SystemExit, match="--extra-iterations"):
        st.config_from_argv(["--checkpoint", "x.pt", "--out", "o", "--extra-iterations", "2"])
    with pytest.raises(SystemExit, match="--iterations"):
        st.config_from_argv(["--resume", "r", "--iterations", "3", "--extra-iterations", "2"])


# -- 9: malformed shard meta / metrics rows do not crash the learner -------


def test_shard_meta_rejects_mistyped_fields() -> None:
    good = sh.ShardMeta(encoding_version=1, ckpt_iteration=0, actor_id=0, seed=1,
                        config_hash="h", n_games=2, created_at=1.0, n_records=3)
    assert sh.ShardMeta.from_json(good.to_json()) == good
    bad_n_games = json.loads(good.to_json())
    bad_n_games["n_games"] = "2"
    with pytest.raises(sh.ShardError, match="n_games"):
        sh.ShardMeta.from_json(json.dumps(bad_n_games))
    negative = json.loads(good.to_json())
    negative["n_records"] = -1
    with pytest.raises(sh.ShardError, match="n_records"):
        sh.ShardMeta.from_json(json.dumps(negative))
    boolean = json.loads(good.to_json())
    boolean["ckpt_iteration"] = True
    with pytest.raises(sh.ShardError, match="ckpt_iteration"):
        sh.ShardMeta.from_json(json.dumps(boolean))
    with pytest.raises(sh.ShardError):
        sh.ShardMeta.from_json("[1, 2]")


def test_learner_poll_survives_a_mistyped_shard(tmp_path: Path) -> None:
    d = tmp_path / "shards"
    good = _write_shard(d, 0, n_games=2)
    bad = d / sh.shard_name(0, 0, 1)
    with np.load(good) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta_json"]))
    meta["n_games"] = "two"
    arrays["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(bad, **arrays)
    infos = sl.scan_shards(d, {})
    assert infos[good.name].ok and not infos[bad.name].ok
    assert sl.total_games(infos) == 2


def test_resumed_games_skips_malformed_metrics_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"iteration": 1, "total_games": 6},
        {"iteration": 2, "total_games": None},
        {"iteration": 3, "total_games": "lots"},
        "not json at all",
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n"
    )
    cfg = _learner_cfg(run_dir, tmp_path / "init.pt")
    assert sl._resumed_games(cfg) == 6
