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

Device: ``--device auto`` (default) trains on cuda, else mps, else cpu
while actors stay on CPU; checkpoints are always written with CPU
tensors so CPU actors load them directly. One CPU learner already keeps
up (measured ~3,000 rec/s at batch 128, 5,000 on MPS, versus ~285
generating cores' worth of records per second at window-2), but on a
DGX Spark the serial loop's CPU ``fine_tune`` measured 39 s per 96-game
iteration against ~368 s of generation -- ~10% of the loop, and in
pipelined mode every second the learner spends training is a second of
extra staleness -- so the idle GPU takes it.

Stopping and restarting: ``R/STOP`` (or SIGTERM) ends the learner after
the in-flight step with a ``{"event": "stopped"}`` row. The learner
does NOT remove ``STOP``: actors on the same directory only check it at
game boundaries and must still see it. The run owner cleans up -- the
pipelined launcher once every actor has exited, or the operator; a
``STOP`` older than a process's own start is stale and ignored, and
``tools/sync_run_dir.sh`` mirrors the removal to generation hosts. A
restarted learner in the same ``R`` resumes from ``current.pt`` with
the settings stored in it (explicit flags override), keeps numbering,
runs only up to ``--max-iterations`` (a run TOTAL; ``--extra-iterations``
adds to the checkpoint's iteration) and logs a ``code_changed`` row if
the git sha differs. Every checkpoint
config and metrics row carries ``git_sha``/``code_dirty``/
``encoding_version``/``hostname``.

Rolling upgrade: upgrade actors one at a time (each restart continues
its own shard sequence), then stop and restart the learner last.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch

from bgai.training import shards as sh
from bgai.training.checkpoint_io import load_checkpoint, save_atomic
from bgai.training.cli_util import config_kwargs_from_stored, explicit_cli_args
from bgai.training.run_provenance import CodeProvenance, code_provenance
from bgai.training.selfplay import SelfPlayRecord
from bgai.training.selfplay_finetune import (
    FineTuneConfig,
    TrainableNets,
    checkpoint_payload,
    fine_tune,
    load_trainable_nets,
)
from bgai.training.stop_signal import StopSignal
from bgai.training.train import pick_device

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
    """Run TOTAL: stop once ``current.pt`` reaches this iteration, whether
    the steps happened in this process or an earlier one. The CLI's
    ``--extra-iterations N`` is the additive form (checkpoint + N)."""
    finetune: FineTuneConfig = FineTuneConfig()
    launcher: dict | None = None
    """Full ``selfplay_train`` config of a pipelined run, stored verbatim
    in every checkpoint so ``selfplay_train --resume`` can rebuild the
    launcher (mode, search, actors) and not silently fall back to a
    serial loop. None for a hand-started learner."""
    device: str = "auto"
    """auto = cuda > mps > cpu (``train.pick_device``)."""
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
    prov: CodeProvenance = code_provenance()


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


def _run_config(cfg: LearnerConfig, prov: CodeProvenance) -> dict:
    return {
        **{k: v for k, v in asdict(cfg).items() if k != "finetune"},
        **asdict(cfg.finetune),
        "run_dir": str(cfg.run_dir),
        "checkpoint": str(cfg.checkpoint),
        "anchor": str(cfg.anchor) if cfg.anchor else None,
        **prov.as_dict(),
    }


def resolve_device(cfg: LearnerConfig) -> torch.device:
    return pick_device(cfg.device or "auto")


def _init(cfg: LearnerConfig, device: torch.device) -> tuple[TrainableNets, LearnerState]:
    """Build nets; publish ``current.pt`` (fresh iteration 0, or resume)."""
    prov = code_provenance()
    current = cfg.run_dir / CURRENT_CHECKPOINT
    if current.exists():
        nets = load_trainable_nets(current, cfg.anchor or cfg.checkpoint, device)
        log.info("learner: resuming from %s at iteration %d", current, nets.iteration)
        before = nets.stored_config.get("git_sha")
        if before is not None and before != prov.git_sha:
            _append_metrics(cfg.run_dir / METRICS_FILE, {
                "event": "code_changed", "from": before, "to": prov.git_sha,
                "wall_time": time.time(), **prov.as_dict(),
            })
    else:
        nets = load_trainable_nets(cfg.checkpoint, cfg.anchor, device)
        nets = replace(nets, iteration=0)
        payload = checkpoint_payload(nets.net, nets.model_cfg, 0, _run_config(cfg, prov))
        save_atomic(payload, current)
        log.info("learner: initialized %s at iteration 0 on %s", current, device)
    now = time.monotonic()
    return nets, LearnerState(
        iteration=nets.iteration, games_at_last_step=_resumed_games(cfg), shards={},
        prev_mark=None, last_progress=now, prov=prov,
    )


def _resumed_games(cfg: LearnerConfig) -> int:
    """Games already consumed by an earlier learner in this run dir (the
    newest step row; event rows carry no ``total_games``)."""
    path = cfg.run_dir / METRICS_FILE
    if not path.exists():
        return 0
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            log.warning("learner: unreadable metrics row (%s); skipping it", exc)
            continue
        if "total_games" not in row:
            continue
        total = row["total_games"]
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            log.warning("learner: malformed total_games %r in metrics row; skipping it", total)
            continue
        return total
    return 0


def stored_learner_config(run_dir: Path) -> dict:
    """The learner config block stored in ``run_dir/current.pt`` ({} if none)."""
    current = Path(run_dir) / CURRENT_CHECKPOINT
    if not current.exists():
        return {}
    return dict(load_checkpoint(current).get("config") or {})


def resumed_learner_config(run_dir: Path, overrides: dict) -> LearnerConfig:
    """Config for continuing ``run_dir``: stored settings, then the
    explicitly typed flags in ``overrides``. ``max_iterations`` stays the
    run total; ``extra_iterations`` means checkpoint iteration + N."""
    current = Path(run_dir) / CURRENT_CHECKPOINT
    payload = load_checkpoint(current)
    stored = dict(payload.get("config") or {})
    cfg_kwargs = config_kwargs_from_stored(LearnerConfig, stored, {"checkpoint", "anchor"})
    ft_kwargs = config_kwargs_from_stored(FineTuneConfig, stored, set())
    ft_names = {f.name for f in fields(FineTuneConfig)}
    cfg_names = {f.name for f in fields(LearnerConfig)}
    ft = FineTuneConfig(**{**ft_kwargs, **{k: v for k, v in overrides.items() if k in ft_names}})
    picked = {k: v for k, v in overrides.items() if k in cfg_names}
    extra = overrides.get("extra_iterations")
    if extra is not None:
        if "max_iterations" in overrides:
            raise ValueError("--max-iterations (run total) and --extra-iterations are exclusive")
        picked["max_iterations"] = int(payload.get("iteration", 0)) + int(extra)
    return LearnerConfig(**{**cfg_kwargs, **picked, "run_dir": Path(run_dir), "finetune": ft})


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
    payload = checkpoint_payload(
        nets.net, nets.model_cfg, iteration, _run_config(cfg, state.prov)
    )
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
        **state.prov.as_dict(),
    }


def _append_metrics(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _finished(cfg: LearnerConfig, state: LearnerState) -> bool:
    """``max_iterations`` is the run total, measured on the checkpoint's
    iteration -- a restart after the total was reached does nothing."""
    return cfg.max_iterations is not None and state.iteration >= cfg.max_iterations


def _idle_too_long(cfg: LearnerConfig, state: LearnerState) -> bool:
    if cfg.idle_timeout_secs is None:
        return False
    return time.monotonic() - state.last_progress > cfg.idle_timeout_secs


def run_learner(cfg: LearnerConfig, stop_when: Callable[[], bool] | None = None) -> int:
    """Poll, step, publish; returns the number of steps taken.

    ``stop_when`` is polled between steps (the local launcher passes
    "every actor process has exited")."""
    if cfg.games_per_step < 1 or cfg.window_steps < 1:
        raise ValueError("games_per_step and window_steps must be >= 1")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)
    nets, state = _init(cfg, device)
    shards_dir = cfg.run_dir / SHARDS_DIR
    stop = StopSignal(cfg.run_dir)
    stop.install()
    try:
        while not _finished(cfg, state):
            shards = scan_shards(shards_dir, state.shards)
            if len(shards) != len(state.shards):
                state = replace(state, shards=shards, last_progress=time.monotonic())
            if should_step(total_games(shards), state.games_at_last_step, cfg.games_per_step):
                nets, state = _step(cfg, nets, state, device)
            elif _idle_too_long(cfg, state):
                log.warning("learner: no new shards for %.0fs; exiting after %d steps",
                            cfg.idle_timeout_secs, state.steps_done)
                break
            elif stop_when is not None and stop_when():
                log.warning("learner: stop condition met; exiting after %d steps",
                            state.steps_done)
                break
            else:
                time.sleep(cfg.poll_secs)
            # STOP (file or signal) is honoured only here, at a step boundary
            if stop.requested():
                _stopped(cfg, state, stop)
                break
    finally:
        stop.uninstall()
    return state.steps_done


def _stopped(cfg: LearnerConfig, state: LearnerState, stop: StopSignal) -> None:
    """Log the stop. The STOP file is deliberately left in place: actors
    sharing the directory check it only between games, and the run owner
    (launcher/operator) removes it once they have all exited."""
    log.warning("learner: STOP requested; exiting after %d steps (STOP left for the run "
                "owner to remove)", state.steps_done)
    _append_metrics(cfg.run_dir / METRICS_FILE, {
        "event": "stopped", "iteration": state.iteration, "wall_time": time.time(),
        **state.prov.as_dict(),
    })


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-play learner (training).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="init weights (required unless --run-dir holds a current.pt)")
    parser.add_argument("--anchor", type=Path, default=None,
                        help="frozen KL anchor (defaults to --checkpoint)")
    parser.add_argument("--games-per-step", type=int, default=96)
    parser.add_argument("--window-steps", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="run total (checkpoint iteration to stop at)")
    parser.add_argument("--extra-iterations", type=int, default=None,
                        help="resume only: run this many more steps than the checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--poll-secs", type=float, default=5.0)
    parser.add_argument("--idle-timeout-secs", type=float, default=None)
    add_loss_args(parser)
    return parser


def config_from_argv(argv: list[str] | None) -> LearnerConfig:
    """Fresh run: flags as given. Existing ``current.pt`` in --run-dir:
    stored settings, overridden only by explicitly typed flags."""
    args = build_parser().parse_args(argv)
    explicit = explicit_cli_args(build_parser, argv)
    if "max_iterations" in explicit and "extra_iterations" in explicit:
        raise SystemExit(
            "error: --max-iterations (run total) and --extra-iterations are mutually exclusive"
        )
    if (args.run_dir / CURRENT_CHECKPOINT).exists():
        if "epochs" in explicit:
            explicit["epochs_per_iteration"] = explicit.pop("epochs")
        return resumed_learner_config(args.run_dir, explicit)
    if args.extra_iterations is not None:
        raise SystemExit("error: --extra-iterations needs an existing current.pt in --run-dir")
    if args.checkpoint is None:
        raise SystemExit("error: --checkpoint is required for a fresh run")
    return LearnerConfig(
        run_dir=args.run_dir, checkpoint=args.checkpoint, anchor=args.anchor,
        games_per_step=args.games_per_step, window_steps=args.window_steps,
        max_iterations=args.max_iterations, finetune=finetune_config_from_args(args),
        device=args.device, poll_secs=args.poll_secs,
        idle_timeout_secs=args.idle_timeout_secs,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    steps = run_learner(config_from_argv(argv))
    log.info("learner: exit after %d steps", steps)


if __name__ == "__main__":
    main()
