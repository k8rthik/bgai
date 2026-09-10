"""Local pipelined launcher: one in-process learner + N actor subprocesses.

``selfplay_train --mode pipelined`` lands here. It reproduces the serial
loop's one-command experience on one machine while exercising exactly
the code a cluster run uses:

- the learner is ``selfplay_learner.run_learner`` in THIS process
  (``games_per_step = --games-per-iteration``, window 2,
  ``max_iterations = --iterations``);
- each actor is ``python -m bgai.training.selfplay_actor`` with
  ``--workers // --actors`` workers, started BEFORE the learner so its
  pool is warm by the time ``current.pt`` appears (actors wait for it);
- when the learner finishes, ``R/STOP`` is created; actors finish their
  in-flight games, flush a last (partial) shard and exit. Anything still
  alive after ``actor_stop_timeout`` is terminated. ``STOP`` is removed
  afterwards so the run directory can be resumed.

Two-host manual mode is the same thing without this file, e.g.::

    # host A (training)
    uv run python -m bgai.training.selfplay_learner --run-dir R \\
        --checkpoint INIT.pt --anchor ANCHOR.pt --games-per-step 96
    # host B..N (generation), against an rsync mirror of R
    uv run python -m bgai.training.selfplay_actor --run-dir R --actor-id K \\
        --games-per-shard 8 --workers 16 --simulations 512 --c-puct 2.5
    # on each generation host, keep R in sync (see tools/sync_run_dir.sh)
    tools/sync_run_dir.sh hostA:/path/to/R R 30

The learner also stops early if every actor process has died (a bad
flag, a missing ``data/``), so a broken launch fails in seconds rather
than idling for ``--idle-timeout-secs``.

Resume: ``selfplay_train --resume R --mode pipelined --extra-iterations N``
restarts actors (they continue their shard sequences) and a learner that
picks numbering up from ``R/current.pt``. ``R/STOP`` during a run stops
the learner at a step boundary (it removes the file), after which this
launcher re-creates it for the actors and removes it again once they
have exited.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.selfplay_learner import CURRENT_CHECKPOINT, LearnerConfig, run_learner
from bgai.training.selfplay_train import SelfPlayTrainConfig
from bgai.training.stop_signal import STOP_FILE
from bgai.training.worker_env import pin_single_thread_env

log = logging.getLogger(__name__)

ACTOR_STOP_TIMEOUT_SECS = 900.0
"""Long enough for a 512-sim game in flight to finish and be sharded."""


def actor_argv(cfg: SelfPlayTrainConfig, actor_id: int, workers: int) -> list[str]:
    sp = cfg.selfplay()
    return [
        sys.executable, "-m", "bgai.training.selfplay_actor",
        "--run-dir", str(cfg.out),
        "--actor-id", str(actor_id),
        "--games-per-shard", str(min(cfg.games_per_shard, cfg.games_per_iteration)),
        "--workers", str(workers),
        "--seed", str(cfg.seed),
        "--simulations", str(sp.simulations),
        "--top-k", str(sp.top_k),
        "--max-depth", str(sp.max_depth),
        "--leaf-batch", str(sp.leaf_batch),
        "--temp-decisions", str(sp.temp_decisions),
        "--fast-prior", str(sp.fast_prior),
        "--c-puct", str(sp.c_puct),
    ]


def _done_iterations(cfg: SelfPlayTrainConfig) -> int:
    current = cfg.out / CURRENT_CHECKPOINT
    if cfg.resume is None or not current.exists():
        return 0
    return int(load_checkpoint(current).get("iteration", 0))


def learner_config(cfg: SelfPlayTrainConfig) -> LearnerConfig:
    """``--iterations`` is the run total; under ``--resume`` the learner
    only runs the remainder (it picks numbering up from ``current.pt``)."""
    return LearnerConfig(
        run_dir=cfg.out, checkpoint=cfg.checkpoint, anchor=cfg.anchor,
        games_per_step=cfg.games_per_iteration, window_steps=2,
        max_iterations=max(0, cfg.iterations - _done_iterations(cfg)),
        finetune=cfg.finetune(), device=cfg.device or "auto",
        poll_secs=2.0, idle_timeout_secs=cfg.idle_timeout_secs,
    )


def workers_per_actor(workers: int, actors: int) -> int:
    if actors < 1:
        raise ValueError("--actors must be >= 1")
    return max(1, workers // actors)


def _start_actors(cfg: SelfPlayTrainConfig) -> list[subprocess.Popen]:
    per_actor = workers_per_actor(cfg.workers, cfg.actors)
    procs = []
    for actor_id in range(cfg.actors):
        argv = actor_argv(cfg, actor_id, per_actor)
        log.info("launcher: starting actor %d (%d workers)", actor_id, per_actor)
        procs.append(subprocess.Popen(argv, cwd=os.getcwd()))
    return procs


def _stop_actors(
    run_dir: Path, procs: Sequence[subprocess.Popen], timeout: float
) -> None:
    stop = run_dir / STOP_FILE
    stop.touch()
    deadline = time.monotonic() + timeout
    try:
        for proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log.warning("launcher: actor pid %d ignored STOP; terminating", proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
    finally:
        stop.unlink(missing_ok=True)


def run_pipelined(
    cfg: SelfPlayTrainConfig, actor_stop_timeout: float = ACTOR_STOP_TIMEOUT_SECS
) -> int:
    """Run the pipelined loop locally; returns learner steps taken."""
    if cfg.infer_device:
        raise ValueError("--infer-device is a serial-mode option (C17: CPU nets win anyway)")
    pin_single_thread_env()
    cfg.out.mkdir(parents=True, exist_ok=True)
    if cfg.resume is None and (cfg.out / CURRENT_CHECKPOINT).exists():
        raise ValueError(
            f"{cfg.out / CURRENT_CHECKPOINT} exists: pass --resume {cfg.out} to continue it"
        )
    (cfg.out / STOP_FILE).unlink(missing_ok=True)
    procs = _start_actors(cfg)

    def all_actors_dead() -> bool:
        return all(p.poll() is not None for p in procs)

    try:
        steps = run_learner(learner_config(cfg), stop_when=all_actors_dead)
    finally:
        _stop_actors(cfg.out, procs, actor_stop_timeout)
    failed = [p.pid for p in procs if p.returncode not in (0, None)]
    if failed:
        log.warning("launcher: actor processes exited non-zero: %s", failed)
    log.info("launcher: done after %d learner steps", steps)
    return steps
