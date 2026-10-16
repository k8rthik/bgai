"""Self-play training loop: generate -> fine-tune -> evaluate (Phase 6b).

    uv run python -m bgai.training.selfplay_train \\
        --checkpoint data/checkpoints/imitation/checkpoint.pt \\
        --out data/checkpoints/selfplay_v1 \\
        --iterations 4 --games-per-iteration 240 --workers 8

Two modes, same CLI:

``--mode serial`` (default, the original loop). Each iteration:
1. **Generate** self-play games in parallel worker processes (game
   generation is CPU-bound engine work, and each worker holds its own
   copy of a ~5M-parameter net, so processes scale nearly linearly).
2. **Fine-tune** on the fresh games: policy toward the search's visit
   distribution, value toward realised final VP shares, KL toward the
   *frozen* imitation policy.
3. **Checkpoint** and log, so the run is resumable and every iteration's
   model can be evaluated afterwards.

``--mode pipelined``. The same run, decoupled: one learner in this
process plus ``--actors`` actor processes (``selfplay_actor``) that keep
generating while the learner trains, talking only through ``--out`` as
a run directory (``current.pt``, ``shards/``, ``metrics.jsonl``). Exits
after ``--iterations`` learner steps, signalling actors to stop. A
cluster run is the same two programs started by hand on different
hosts with an rsync'ed run directory: see ``selfplay_learner``,
``selfplay_actor`` and ``tools/sync_run_dir.sh``.

Resuming (both modes)::

    uv run python -m bgai.training.selfplay_train --resume OUT --extra-iterations 4

reads ``OUT/current.pt``: iteration, base seed, anchor and every
search/training setting come from the checkpoint's stored config, and a
flag on the command line overrides only if it was explicitly typed.
Numbering continues from ``iteration + 1`` (so ``seed + iteration``
never replays a game and no ``iter_NN.pt`` is overwritten),
``metrics.jsonl`` is appended, ``--iterations`` means the TOTAL for the
run and ``--extra-iterations N`` is the shorthand. A different
``encoding_version`` is refused; a different ``git_sha`` logs a
``code_changed`` row so throughput changes can be attributed. A fresh
run refuses an ``--out`` that already holds a ``current.pt``.

Stopping: create ``OUT/STOP`` (or SIGTERM the process); the in-flight
iteration finishes and saves, a ``{"event": "stopped"}`` row is logged,
the file is removed and the process exits 0.

Why the KL anchor is not optional here: Phase 6 measured that search
over the imitation value head adds exactly nothing (decision D6.7), so
self-play starts from a policy that is the strongest thing we have and a
value head that is the diagnosed weak point. Without the anchor, a few
thousand self-play games can happily drag a policy trained on 1.2M human
decisions somewhere worse. ``lambda_kl`` is the knob: high values mean
"mostly fix the value head, leave the policy alone".

Scale honesty: at 16 simulations and 8 workers this generates roughly
2,000-3,000 games/hour on an 11-core M3 Pro. A published-quality RL run
is 10^5+ games. What runs here is a real but *small* experiment -- big
enough to move the value head measurably, not big enough to claim a
trained-from-scratch RL result.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import time
from dataclasses import asdict, dataclass, fields
from multiprocessing import get_context
from pathlib import Path

import torch

from bgai.training.checkpoint_io import load_checkpoint, save_atomic
from bgai.training.cli_util import config_kwargs_from_stored, explicit_cli_args
from bgai.training.run_provenance import CodeProvenance, code_provenance
from bgai.training.selfplay import SelfPlayConfig, SelfPlayRecord, play_game
from bgai.training.selfplay_finetune import (
    FineTuneConfig,
    TrainableNets,
    checkpoint_payload,
    collate,
    fine_tune,
    load_trainable_nets,
)
from bgai.training.stop_signal import StopSignal
from bgai.training.train import pick_device
from bgai.training.vocab import ENCODING_VERSION
from bgai.training.worker_env import pin_single_thread_env
from bgai.training.worker_pool import shielded_pool

_collate = collate  # backward-compatible name for tools/offline_sweep-style imports

MODES = ("serial", "pipelined")
DEVICES = ("", "auto", "cpu", "cuda", "mps")
_PATH_FIELDS = {"checkpoint", "out", "anchor", "resume"}


@dataclass(frozen=True)
class SelfPlayTrainConfig:
    checkpoint: Path
    out: Path
    anchor: Path | None = None
    """Checkpoint for the frozen KL anchor. Defaults to ``checkpoint``,
    which is correct for a fresh run; ``--resume`` restores it from the
    stored config so the piKL leash never silently re-anchors to the
    drifted policy it exists to restrain."""
    iterations: int = 4
    games_per_iteration: int = 240
    workers: int = 8
    simulations: int = 512
    top_k: int = 8
    max_depth: int = 48
    leaf_batch: int = 16
    temp_decisions: int = 30
    fast_prior: float = 0.95
    c_puct: float = 1.5
    """PUCT constant for generation. Default keeps every existing leg
    comparable; the arena champion config uses 2.5 (decision C16)."""
    lambda_kl: float = 1.0
    rank_weight: float = 1.0
    winner_pair_weight: float = 3.0
    win_weight: float = 0.5
    aux_weight: float = 0.5
    infer_device: str = ""
    """Central inference server device ("mps"/"cuda"; empty = per-worker
    CPU nets, the previous behaviour). Serial mode only."""
    lr: float = 5e-5
    batch_size: int = 128
    epochs_per_iteration: int = 1
    seed: int = 0
    mode: str = "serial"
    actors: int = 1
    """Pipelined mode: actor processes; ``workers`` is split across them."""
    games_per_shard: int = 8
    idle_timeout_secs: float = 3600.0
    """Pipelined mode: learner gives up when no shard arrives for this long."""
    device: str = ""
    """Training device. Empty = the mode's default: serial keeps the net
    on CPU (generation dominates); pipelined picks cuda > mps > cpu."""
    resume: Path | None = None
    """Set by ``--resume``: the run directory being continued."""

    def selfplay(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            simulations=self.simulations,
            top_k=self.top_k,
            max_depth=self.max_depth,
            leaf_batch=self.leaf_batch,
            temp_decisions=self.temp_decisions,
            fast_prior=self.fast_prior,
            c_puct=self.c_puct,
            lambda_kl=self.lambda_kl,
            rank_weight=self.rank_weight,
            winner_pair_weight=self.winner_pair_weight,
            win_weight=self.win_weight,
            lr=self.lr,
        )

    def finetune(self) -> FineTuneConfig:
        return FineTuneConfig(
            lr=self.lr, seed=self.seed, epochs_per_iteration=self.epochs_per_iteration,
            batch_size=self.batch_size, lambda_kl=self.lambda_kl, rank_weight=self.rank_weight,
            winner_pair_weight=self.winner_pair_weight, win_weight=self.win_weight,
            aux_weight=self.aux_weight,
        )


# --------------------------------------------------------------------------
# parallel generation (serial mode)
# --------------------------------------------------------------------------

_WORKER: dict[str, object] = {}

WORKER_GC_THRESHOLDS = (50_000, 50, 1000)
"""``gc.set_threshold`` arguments for a self-play worker process."""


def _worker_init(checkpoint: str, cfg: SelfPlayConfig, infer=None) -> None:
    """One agent per worker. With ``infer`` set, the worker is torch-free:
    it claims a worker id and evaluates through the central server."""
    import gc

    from bgai.agents.mcts import MCTSAgent

    for sig in (signal.SIGINT, signal.SIGTERM):  # the parent's StopSignal decides
        with contextlib.suppress(ValueError):
            signal.signal(sig, signal.SIG_IGN)
    torch.set_num_threads(1)  # workers must not fight over BLAS threads
    # Search allocates millions of short-lived tuples per game and keeps
    # no reference cycles (immutable states, parent-less tree nodes), so
    # the default gen-0 threshold (700) buys nothing but ~600 collections
    # a game that each walk the pinned memos. Measured ~1 s/game
    # (docs/decisions.md C18).
    gc.set_threshold(*WORKER_GC_THRESHOLDS)
    evaluator = None
    if infer is not None:
        from bgai.agents.inference_server import RemoteEvaluator

        id_q, request_q, response_qs = infer
        wid = id_q.get()
        evaluator = RemoteEvaluator(wid, request_q, response_qs[wid])
    _WORKER["agent"] = MCTSAgent(
        checkpoint_path=None if evaluator is not None else Path(checkpoint),
        evaluator=evaluator,
        simulations=cfg.simulations,
        max_depth=cfg.max_depth,
        top_k=cfg.top_k,
        leaf_batch=cfg.leaf_batch,
        c_puct=cfg.c_puct,
        device="cpu",
    )
    _WORKER["cfg"] = cfg


def _worker_game(seed: int) -> list[SelfPlayRecord]:
    import random as _random

    from bgai.arena.setups import sample_setup

    rng = _random.Random(seed)
    setup = sample_setup(rng)
    return play_game(_WORKER["agent"], setup, rng, _WORKER["cfg"])  # type: ignore[arg-type]


def generate_parallel(
    checkpoint: Path,
    games: int,
    workers: int,
    cfg: SelfPlayConfig,
    seed: int,
    infer_device: str = "",
) -> list[SelfPlayRecord]:
    """With ``infer_device`` (e.g. "mps"), a central server owns the net
    on that device and workers evaluate through it -- one GPU pass serves
    every worker's leaf batch (measured flat 1.7 ms to batch 256 on M3
    Pro MPS). The server is restarted per generation call so it always
    serves the current checkpoint."""
    pin_single_thread_env()  # children inherit; must precede the spawn
    ctx = get_context("spawn")
    tasks = [seed * 100_000 + i for i in range(games)]
    server = None
    infer = None
    if infer_device:
        from bgai.agents.inference_server import start_server

        server, request_q, response_qs, stop_event = start_server(
            checkpoint, infer_device, workers
        )
        id_q = ctx.Queue()
        for i in range(workers):
            id_q.put(i)
        infer = (id_q, request_q, response_qs)
    try:
        with shielded_pool(
            ctx,
            processes=workers,
            initializer=_worker_init,
            initargs=(str(checkpoint), cfg, infer),
        ) as pool:
            batches = pool.map(_worker_game, tasks, chunksize=1)
    finally:
        if server is not None:
            stop_event.set()
            server.join(timeout=5)
            if server.is_alive():
                server.terminate()
    return [record for batch in batches for record in batch]


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def resumed_config(run_dir: Path, overrides: dict) -> SelfPlayTrainConfig:
    """Config for continuing ``run_dir``: everything from the stored
    checkpoint config, then ``overrides`` (explicit CLI flags only).
    ``iterations`` in overrides is the run TOTAL; ``extra_iterations``
    adds to the checkpoint's iteration."""
    run_dir = Path(run_dir)
    payload = load_checkpoint(run_dir / "current.pt")
    stored = dict(payload.get("config") or {})
    version = stored.get("encoding_version", ENCODING_VERSION)
    if version != ENCODING_VERSION:
        raise ValueError(
            f"cannot resume {run_dir}: encoding v{version} != code v{ENCODING_VERSION}"
        )
    # Pipelined runs are written by the learner, whose own config block
    # knows nothing about mode/search/actors; the launcher's full config
    # rides along under "launcher" (finding 1). Serial runs store it flat.
    source = stored.get("launcher") or stored
    if not isinstance(source, dict):
        raise ValueError(f"{run_dir}/current.pt has a malformed launcher config block")
    base = config_kwargs_from_stored(SelfPlayTrainConfig, source, _PATH_FIELDS)
    if "checkpoint" not in base:
        raise ValueError(f"{run_dir}/current.pt stores no run config; cannot resume")
    known = {f.name for f in fields(SelfPlayTrainConfig)}
    fixed = _PATH_FIELDS - {"anchor"}  # the run dir defines these
    picked = {k: v for k, v in overrides.items() if k in known and k not in fixed}
    extra = overrides.get("extra_iterations")
    iteration = int(payload.get("iteration", 0))
    merged = {
        **base, **picked,
        "checkpoint": run_dir / "current.pt",
        "out": run_dir,
        "anchor": picked.get("anchor") or base.get("anchor") or base["checkpoint"],
        "resume": run_dir,
    }
    if extra is not None:
        merged["iterations"] = iteration + int(extra)
    return SelfPlayTrainConfig(**merged)


def _load_prev_records(out: Path, iteration: int) -> list[SelfPlayRecord]:
    """The previous iteration's cached games, for the window-2 replay."""
    if iteration < 1:
        return []
    path = out / f"records_{iteration:02d}.pt"
    if not path.exists():
        print(f"WARN: {path} missing; first resumed iteration trains on its own games only",
              flush=True)
        return []
    return torch.load(path, map_location="cpu", weights_only=False)


def _append_row(path: Path, row: dict) -> None:
    print(json.dumps(row), flush=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _log_code_change(metrics: Path, stored: dict, prov: CodeProvenance) -> None:
    before = stored.get("git_sha")
    if before is not None and before != prov.git_sha:
        _append_row(metrics, {"event": "code_changed", "from": before, "to": prov.git_sha,
                              "wall_time": time.time(), **prov.as_dict()})


# --------------------------------------------------------------------------
# serial loop
# --------------------------------------------------------------------------


def saved_config(cfg: SelfPlayTrainConfig, prov: CodeProvenance) -> dict:
    """The run config block stored in every checkpoint (paths as strings,
    provenance stamped); ``resumed_config`` reads it back."""
    return {
        **asdict(cfg),
        **{k: (str(getattr(cfg, k)) if getattr(cfg, k) is not None else None)
           for k in _PATH_FIELDS},
        **prov.as_dict(),
    }


def _start(cfg: SelfPlayTrainConfig, nets: TrainableNets, prov: CodeProvenance) -> int:
    """Publish iteration 0 for a fresh run, or validate the resume point.
    Returns the first iteration to run."""
    current = cfg.out / "current.pt"
    if cfg.resume is None:
        save_atomic(
            checkpoint_payload(nets.net, nets.model_cfg, 0, saved_config(cfg, prov)), current
        )
        return 1
    start = nets.iteration + 1
    if (cfg.out / f"iter_{start:02d}.pt").exists():
        raise ValueError(f"refusing to overwrite {cfg.out}/iter_{start:02d}.pt on resume")
    _log_code_change(cfg.out / "metrics.jsonl", nets.stored_config, prov)
    return start


def _iteration(
    cfg: SelfPlayTrainConfig, nets: TrainableNets, iteration: int,
    prev_records: list[SelfPlayRecord], device: torch.device, saved_config: dict,
) -> tuple[list[SelfPlayRecord], dict]:
    current = cfg.out / "current.pt"
    t0 = time.perf_counter()
    records = generate_parallel(
        current, cfg.games_per_iteration, cfg.workers, cfg.selfplay(),
        cfg.seed + iteration, infer_device=cfg.infer_device,
    )
    gen_secs = time.perf_counter() - t0
    if len(records) < cfg.batch_size:
        raise RuntimeError(f"iteration {iteration} produced only {len(records)} records")

    # Persist the generation before training on it: generation costs
    # ~26 min/iteration, training ~10 s, so cached records make every
    # training-side hyperparameter (rank_weight, lambda_kl, lr,
    # epochs) sweepable offline without regenerating a single game.
    torch.save(records, cfg.out / f"records_{iteration:02d}.pt")

    t1 = time.perf_counter()
    # Window-2 replay buffer: train on this iteration's games plus the
    # previous iteration's. The leg-4 offline sweep found a single
    # consolidation epoch over AGGREGATED records beat sequential
    # fresh-only training on every ordering metric -- sequential
    # training half-forgets each batch as the next arrives.
    parts = fine_tune(nets.net, nets.frozen, records + prev_records, cfg.finetune(), device)
    train_secs = time.perf_counter() - t1

    payload = checkpoint_payload(nets.net, nets.model_cfg, iteration, saved_config)
    save_atomic(payload, current)
    save_atomic(payload, cfg.out / f"iter_{iteration:02d}.pt")
    row = {
        "iteration": iteration,
        "games": cfg.games_per_iteration,
        "records": len(records),
        "gen_secs": round(gen_secs, 1),
        "train_secs": round(train_secs, 1),
        "games_per_hour": round(cfg.games_per_iteration / gen_secs * 3600),
        **{k: round(v, 4) for k, v in parts.items()},
    }
    return records, row


def run(cfg: SelfPlayTrainConfig) -> None:
    prov = code_provenance()
    device = pick_device(cfg.device or "cpu")  # generation dominates; keep the net on CPU
    cfg.out.mkdir(parents=True, exist_ok=True)
    current = cfg.out / "current.pt"
    if cfg.resume is None and current.exists():
        raise ValueError(
            f"{current} exists: pass --resume {cfg.out} to continue it, or choose a new --out"
        )
    nets = load_trainable_nets(cfg.checkpoint, cfg.anchor, device)
    stored = saved_config(cfg, prov)
    metrics_path = cfg.out / "metrics.jsonl"
    start = _start(cfg, nets, prov)
    prev_records = _load_prev_records(cfg.out, start - 1) if cfg.resume is not None else []
    stop = StopSignal(cfg.out)
    stop.install()
    try:
        for iteration in range(start, cfg.iterations + 1):
            prev_records, row = _iteration(cfg, nets, iteration, prev_records, device, stored)
            _append_row(metrics_path, {**row, **prov.as_dict()})
            if stop.requested():
                _append_row(metrics_path, {"event": "stopped", "iteration": iteration,
                                           "wall_time": time.time(), **prov.as_dict()})
                stop.clear()
                break
    finally:
        stop.uninstall()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human-regularized self-play training.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="init weights")
    parser.add_argument("--anchor", type=Path, default=None,
                        help="frozen KL anchor (defaults to --checkpoint; stored in the "
                        "run config so --resume restores it)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="run directory to continue (settings come from its current.pt; "
                        "explicit flags override)")
    parser.add_argument("--extra-iterations", type=int, default=None,
                        help="with --resume: run this many more iterations")
    parser.add_argument("--mode", choices=MODES, default="serial")
    parser.add_argument("--iterations", type=int, default=4,
                        help="total iterations for the run (also under --resume)")
    parser.add_argument("--games-per-iteration", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--actors", type=int, default=1,
                        help="pipelined: actor processes sharing --workers")
    parser.add_argument("--games-per-shard", type=int, default=8)
    parser.add_argument("--idle-timeout-secs", type=float, default=3600.0)
    parser.add_argument("--device", choices=DEVICES, default="",
                        help="training device (empty: serial=cpu, pipelined=auto)")
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--leaf-batch", type=int, default=16)
    parser.add_argument("--temp-decisions", type=int, default=30)
    parser.add_argument("--fast-prior", type=float, default=0.95,
                        help="skip search when the policy is at least this sure (1.0 disables)")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="PUCT exploration constant (arena champion uses 2.5, C16)")
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--winner-pair-weight", type=float, default=3.0)
    parser.add_argument("--win-weight", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--infer-device", default="",
                        help="central inference server device (mps/cuda; empty = CPU nets)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> SelfPlayTrainConfig:
    if args.checkpoint is None or args.out is None:
        raise SystemExit("--checkpoint and --out are required (or use --resume RUN_DIR)")
    known = {f.name for f in fields(SelfPlayTrainConfig)}
    return SelfPlayTrainConfig(**{k: v for k, v in vars(args).items() if k in known})


def config_from_argv(argv: list[str] | None) -> SelfPlayTrainConfig:
    args = build_parser().parse_args(argv)
    explicit = explicit_cli_args(build_parser, argv)
    if args.resume is not None:
        for fixed in ("checkpoint", "out"):
            if fixed in explicit:
                raise SystemExit(
                    f"error: --{fixed} conflicts with --resume (the run directory defines it)"
                )
        if "iterations" in explicit and "extra_iterations" in explicit:
            raise SystemExit(
                "error: --iterations (run total) and --extra-iterations are mutually exclusive"
            )
        return resumed_config(args.resume, explicit)
    if args.extra_iterations is not None:
        raise SystemExit("error: --extra-iterations requires --resume RUN_DIR")
    return config_from_args(args)


def main(argv: list[str] | None = None) -> None:
    cfg = config_from_argv(argv)
    if cfg.mode == "pipelined":
        from bgai.training.selfplay_launcher import run_pipelined

        run_pipelined(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
