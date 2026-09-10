"""Self-play LEARNER: trains on actor shards, publishes checkpoints.

    uv run python -m bgai.training.selfplay_learner \\
        --run-dir data/checkpoints/rl_run \\
        --checkpoint data/checkpoints/selfplay_leg5b/current.pt \\
        --anchor data/checkpoints/snap/simplex_best.pt \\
        --games-per-step 96 --window-steps 2

Contract with the run directory ``R``
--------------------------------------
- ``R/current.pt``     written here, atomically (``checkpoint_io``), at
  start (iteration 0 = the init weights, or resumed from an existing
  ``current.pt``) and after every step. Actors poll its mtime.
- ``R/iter_NNNN.pt``   one per step, for later evaluation. Same payload.
- ``R/shards/``        read-only here; never deleted, never rewritten.
  Files named ``*.tmp`` (in-progress writes) are ignored; unreadable
  shards are skipped with a warning and remembered so they are not
  re-parsed every poll.
- ``R/metrics.jsonl``  one row per step (see ``_metrics_row``).

Step rule -- identical to the serial loop's iteration: once at least
``games_per_step`` NEW games have landed since the last step, take the
newest shards covering ``window_steps * games_per_step`` games (the
serial loop's "this iteration + previous iteration" window-2 replay) and
run ``selfplay_finetune.fine_tune`` over them (fresh AdamW, one epoch).
Because actors keep playing while the learner trains, some window games
were played by an older checkpoint than the serial loop would have used;
the metrics row's ``staleness_*`` fields (learner iteration minus the
shard's ``ckpt_iteration``, per record) quantify that lag so a pipelined
leg can be judged against the serial baseline.

One CPU learner suffices: measured ~3,000 rec/s at batch 128 (5,000 on
MPS) versus ~285 generating cores' worth of records per second at
window-2, so ``--device`` is a convenience, not a requirement.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from bgai.training import shards as sh
from bgai.training.checkpoint_io import save_atomic
from bgai.training.selfplay import SelfPlayRecord
from bgai.training.selfplay_finetune import (
    FineTuneConfig,
    TrainableNets,
    checkpoint_payload,
    fine_tune,
    load_trainable_nets,
)

log = logging.getLogger(__name__)

CURRENT_CHECKPOINT = "current.pt"
SHARDS_DIR = "shards"
METRICS_FILE = "metrics.jsonl"


@dataclass(frozen=True)
class LearnerConfig:
    run_dir: Path
    checkpoint: Path
    anchor: Path | None = None
    """Frozen KL anchor; defaults to ``checkpoint``. A resumed run must
    pass the original imitation checkpoint (see selfplay_train)."""
    games_per_step: int = 96
    window_steps: int = 2
    max_iterations: int | None = None
    finetune: FineTuneConfig = FineTuneConfig()
    device: str = "cpu"
    poll_secs: float = 5.0
    idle_timeout_secs: float | None = None
    """Exit when no shard has arrived and no step has run for this long
    (None = wait forever). The local launcher sets it so a crashed actor
    cannot hang the run."""


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    n_games: int
    ckpt_iteration: int
    created_at: float
    n_records: int
    ok: bool


@dataclass(frozen=True)
class LearnerState:
    iteration: int
    games_at_last_step: int
    shards: Mapping[str, ShardInfo]
    prev_mark: float | None
    """``created_at`` of the newest shard consumed by the last step."""
    last_progress: float
    steps_done: int = 0


# --------------------------------------------------------------------------
# pure pieces
# --------------------------------------------------------------------------


def should_step(total_games: int, games_at_last_step: int, games_per_step: int) -> bool:
    return total_games - games_at_last_step >= games_per_step


def total_games(shards: Mapping[str, ShardInfo]) -> int:
    return sum(i.n_games for i in shards.values() if i.ok)


def ordered(shards: Mapping[str, ShardInfo]) -> tuple[ShardInfo, ...]:
    """Readable shards, oldest first (created_at, then name)."""
    return tuple(
        sorted((i for i in shards.values() if i.ok), key=lambda i: (i.created_at, i.path.name))
    )


def select_window(shards: Sequence[ShardInfo], games: int) -> tuple[ShardInfo, ...]:
    """Newest shards (input oldest-first) until at least ``games`` games."""
    chosen: list[ShardInfo] = []
    covered = 0
    for info in reversed(shards):
        if covered >= games:
            break
        chosen.append(info)
        covered += info.n_games
    return tuple(reversed(chosen))


def staleness_stats(iteration: int, window: Sequence[ShardInfo]) -> dict:
    """Learner iteration minus generating iteration, weighted per record."""
    hist: dict[str, int] = {}
    total = weighted = 0
    for info in window:
        stale = max(0, iteration - info.ckpt_iteration)
        hist[str(stale)] = hist.get(str(stale), 0) + info.n_records
        total += info.n_records
        weighted += stale * info.n_records
    return {
        "staleness_hist": hist,
        "staleness_mean": round(weighted / total, 4) if total else 0.0,
        "staleness_max": max((iteration - i.ckpt_iteration for i in window), default=0),
    }


def games_per_hour(new: Sequence[ShardInfo], prev_mark: float | None) -> float | None:
    """Generation rate from shard timestamps. With no earlier mark the
    oldest new shard only provides the start time (its games predate it)."""
    if not new:
        return None
    by_time = sorted(new, key=lambda i: i.created_at)
    if prev_mark is None:
        if len(by_time) < 2:
            return None
        games = sum(i.n_games for i in by_time[1:])
        span = by_time[-1].created_at - by_time[0].created_at
    else:
        games = sum(i.n_games for i in by_time)
        span = by_time[-1].created_at - prev_mark
    return round(games / span * 3600.0, 1) if span > 0 else None


def scan_shards(shards_dir: Path, known: Mapping[str, ShardInfo]) -> dict[str, ShardInfo]:
    """New mapping of every shard file in ``shards_dir``; entries already
    in ``known`` are reused (shards are immutable once written)."""
    if not shards_dir.is_dir():
        return dict(known)
    found = {p.name: p for p in shards_dir.iterdir() if sh.is_shard_file(p)}
    out = {name: info for name, info in known.items() if name in found}
    for name, path in sorted(found.items()):
        if name in out:
            continue
        out[name] = _inspect(path)
    return out


def _inspect(path: Path) -> ShardInfo:
    try:
        meta = sh.read_shard_meta(path)
    except sh.ShardError as exc:
        log.warning("learner: skipping unreadable shard %s: %s", path.name, exc)
        return ShardInfo(path=path, n_games=0, ckpt_iteration=0, created_at=0.0,
                         n_records=0, ok=False)
    return ShardInfo(path=path, n_games=meta.n_games, ckpt_iteration=meta.ckpt_iteration,
                     created_at=meta.created_at, n_records=meta.n_records, ok=True)


def load_window(window: Sequence[ShardInfo]) -> list[SelfPlayRecord]:
    records: list[SelfPlayRecord] = []
    for info in window:
        try:
            _, recs = sh.read_shard(info.path)
        except sh.ShardError as exc:
            log.warning("learner: dropping shard %s from window: %s", info.path.name, exc)
            continue
        records.extend(recs)
    return records


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _run_config(cfg: LearnerConfig) -> dict:
    return {
        **{k: v for k, v in asdict(cfg).items() if k != "finetune"},
        **asdict(cfg.finetune),
        "run_dir": str(cfg.run_dir),
        "checkpoint": str(cfg.checkpoint),
        "anchor": str(cfg.anchor) if cfg.anchor else None,
    }


def _init(cfg: LearnerConfig, device: torch.device) -> tuple[TrainableNets, LearnerState]:
    """Build nets; publish ``current.pt`` (fresh iteration 0, or resume)."""
    current = cfg.run_dir / CURRENT_CHECKPOINT
    if current.exists():
        nets = load_trainable_nets(current, cfg.anchor or cfg.checkpoint, device)
        log.info("learner: resuming from %s at iteration %d", current, nets.iteration)
    else:
        nets = load_trainable_nets(cfg.checkpoint, cfg.anchor, device)
        nets = replace(nets, iteration=0)
        save_atomic(checkpoint_payload(nets.net, nets.model_cfg, 0, _run_config(cfg)), current)
        log.info("learner: initialized %s at iteration 0", current)
    now = time.monotonic()
    return nets, LearnerState(
        iteration=nets.iteration, games_at_last_step=_resumed_games(cfg), shards={},
        prev_mark=None, last_progress=now,
    )


def _resumed_games(cfg: LearnerConfig) -> int:
    """Games already consumed by an earlier learner in this run dir."""
    path = cfg.run_dir / METRICS_FILE
    if not path.exists():
        return 0
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return 0
    try:
        return int(json.loads(rows[-1]).get("total_games", 0))
    except (ValueError, TypeError) as exc:
        log.warning("learner: unreadable last metrics row (%s); consuming from 0", exc)
        return 0


def _step(
    cfg: LearnerConfig, nets: TrainableNets, state: LearnerState, device: torch.device
) -> tuple[TrainableNets, LearnerState]:
    shards = ordered(state.shards)
    total = total_games(state.shards)
    window = select_window(shards, cfg.window_steps * cfg.games_per_step)
    new = tuple(i for i in shards if state.prev_mark is None or i.created_at > state.prev_mark)
    t0 = time.perf_counter()
    records = load_window(window)
    load_secs = time.perf_counter() - t0
    if len(records) < cfg.finetune.batch_size:
        log.warning("learner: window has %d records < batch %d; consuming without a step",
                    len(records), cfg.finetune.batch_size)
        return nets, replace(state, games_at_last_step=total, last_progress=time.monotonic())
    t1 = time.perf_counter()
    parts = fine_tune(nets.net, nets.frozen, records, cfg.finetune, device)
    train_secs = time.perf_counter() - t1
    iteration = state.iteration + 1
    payload = checkpoint_payload(nets.net, nets.model_cfg, iteration, _run_config(cfg))
    save_atomic(payload, cfg.run_dir / CURRENT_CHECKPOINT)
    save_atomic(payload, cfg.run_dir / f"iter_{iteration:04d}.pt")
    row = _metrics_row(
        state, iteration, window, new, len(records), total, load_secs, train_secs, parts
    )
    _append_metrics(cfg.run_dir / METRICS_FILE, row)
    log.info("learner: %s", json.dumps(row))
    newest = max((i.created_at for i in window), default=state.prev_mark)
    return replace(nets, iteration=iteration), replace(
        state, iteration=iteration, games_at_last_step=total, prev_mark=newest,
        last_progress=time.monotonic(), steps_done=state.steps_done + 1,
    )


def _metrics_row(state, iteration, window, new, n_records, total, load_secs, train_secs, parts):
    return {
        "iteration": iteration,
        "new_games": total - state.games_at_last_step,
        "total_games": total,
        "window_games": sum(i.n_games for i in window),
        "window_records": n_records,
        "window_shards": len(window),
        "load_secs": round(load_secs, 2),
        "train_secs": round(train_secs, 1),
        "games_per_hour": games_per_hour(new, state.prev_mark),
        **staleness_stats(state.iteration, window),
        "wall_time": time.time(),
        **{k: round(v, 4) for k, v in parts.items()},
    }


def _append_metrics(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _finished(cfg: LearnerConfig, state: LearnerState) -> bool:
    return cfg.max_iterations is not None and state.steps_done >= cfg.max_iterations


def _idle_too_long(cfg: LearnerConfig, state: LearnerState) -> bool:
    if cfg.idle_timeout_secs is None:
        return False
    return time.monotonic() - state.last_progress > cfg.idle_timeout_secs


def run_learner(cfg: LearnerConfig) -> int:
    """Poll, step, publish; returns the number of steps taken."""
    if cfg.games_per_step < 1 or cfg.window_steps < 1:
        raise ValueError("games_per_step and window_steps must be >= 1")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    nets, state = _init(cfg, device)
    shards_dir = cfg.run_dir / SHARDS_DIR
    while not _finished(cfg, state):
        shards = scan_shards(shards_dir, state.shards)
        if len(shards) != len(state.shards):
            state = replace(state, shards=shards, last_progress=time.monotonic())
        if should_step(total_games(shards), state.games_at_last_step, cfg.games_per_step):
            nets, state = _step(cfg, nets, state, device)
            continue
        if _idle_too_long(cfg, state):
            log.warning("learner: no new shards for %.0fs; exiting after %d steps",
                        cfg.idle_timeout_secs, state.steps_done)
            break
        time.sleep(cfg.poll_secs)
    return state.steps_done


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_loss_args(parser: argparse.ArgumentParser) -> None:
    """Loss/optimiser flags shared with ``selfplay_train`` (same defaults)."""
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--winner-pair-weight", type=float, default=3.0)
    parser.add_argument("--win-weight", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)


def finetune_config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    return FineTuneConfig(
        lr=args.lr, seed=args.seed, epochs_per_iteration=args.epochs,
        batch_size=args.batch_size, lambda_kl=args.lambda_kl, rank_weight=args.rank_weight,
        winner_pair_weight=args.winner_pair_weight, win_weight=args.win_weight,
        aux_weight=args.aux_weight,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Self-play learner (training).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="init weights")
    parser.add_argument("--anchor", type=Path, default=None,
                        help="frozen KL anchor (defaults to --checkpoint)")
    parser.add_argument("--games-per-step", type=int, default=96)
    parser.add_argument("--window-steps", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--poll-secs", type=float, default=5.0)
    parser.add_argument("--idle-timeout-secs", type=float, default=None)
    add_loss_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    steps = run_learner(
        LearnerConfig(
            run_dir=args.run_dir, checkpoint=args.checkpoint, anchor=args.anchor,
            games_per_step=args.games_per_step, window_steps=args.window_steps,
            max_iterations=args.max_iterations, finetune=finetune_config_from_args(args),
            device=args.device, poll_secs=args.poll_secs,
            idle_timeout_secs=args.idle_timeout_secs,
        )
    )
    log.info("learner: exit after %d steps", steps)


if __name__ == "__main__":
    main()
