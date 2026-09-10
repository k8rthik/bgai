"""Self-play ACTOR: plays games against the learner's latest weights and
writes shards. One half of the decoupled RL loop (the other is
``selfplay_learner``); they share nothing but a run directory.

    uv run python -m bgai.training.selfplay_actor \\
        --run-dir data/checkpoints/rl_run --actor-id 3 --games-per-shard 8 \\
        --workers 4 --simulations 512 --top-k 8 --max-depth 48 --c-puct 2.5

Contract with the run directory ``R``
--------------------------------------
- ``R/current.pt``   learner-owned, written atomically. Read-only here.
  Before dispatching each game the actor stats it; a changed mtime is
  passed to the worker with the task and the worker reloads weights in
  place (``MCTSAgent.reload``) before playing. Validation failures keep
  the previous weights and retry on the next game.
- ``R/shards/``      actor-owned, append-only. Every ``games_per_shard``
  completed games (or sooner, when the checkpoint changes -- shards are
  homogeneous in ``ckpt_iteration``) one
  ``it{ckpt_iteration:04d}_a{actor_id:03d}_{seq:06d}.npz`` is written
  atomically (``shards.write_shard``). ``seq`` continues from the
  highest existing sequence for this actor id, so restarting an actor
  never overwrites a shard.
- ``R/STOP``         anyone may create it; the actor finishes in-flight
  games, flushes its partial buffer and exits. SIGTERM/SIGINT do the
  same (``stop_signal``), so a tmux kill loses nothing. The actor never
  deletes ``STOP`` -- whoever created it owns it.

Actors are individually restartable at any moment: a fresh actor with
the same ``--actor-id`` continues its shard sequence and game index and
loads whatever ``current.pt`` is present. Rolling code upgrade: stop
and restart actors ONE AT A TIME (the others keep the learner fed),
then the learner last (``selfplay_learner`` resumes from ``current.pt``
and logs a ``code_changed`` row); shard/checkpoint formats are
versioned, and a learner refuses shards from a different
``encoding_version``.

Seeds derive from ``(base seed, actor_id, game index)`` via a hash, so
actors never replay each other's setups and a run is reproducible given
its actor ids. A restarted actor resumes its game index from the shard
sequence so it does not replay its own openings either.

Workers: ``--workers W`` spawn subprocesses that each hold one
``MCTSAgent`` and stay alive across games (no per-iteration pool
rebuild, no per-iteration setup-cache reload). The parent keeps exactly
W games in flight through ``apply_async`` callbacks, which is what lets
each dispatch observe the freshest checkpoint. Thread pinning
(``worker_env``) is exported before the pool spawns.

Two-host use: run this on the generation host(s) against an rsync'ed
mirror of ``R`` -- see ``tools/sync_run_dir.sh``. Shards are small
(~69 B/record, ~20 KB/game) and checkpoints are ~45 MB, so a
once-a-minute sync is ample.
"""

from __future__ import annotations

import argparse
import logging
import queue
import time
from dataclasses import dataclass, replace
from hashlib import blake2b
from multiprocessing import get_context
from pathlib import Path

import torch

from bgai.training import shards as sh
from bgai.training.checkpoint_io import checkpoint_mtime_ns, load_checkpoint
from bgai.training.selfplay import SelfPlayConfig, SelfPlayRecord, play_game
from bgai.training.stop_signal import StopSignal
from bgai.training.worker_env import pin_single_thread_env

log = logging.getLogger(__name__)

CURRENT_CHECKPOINT = "current.pt"
SHARDS_DIR = "shards"


@dataclass(frozen=True)
class ActorConfig:
    run_dir: Path
    actor_id: int
    games_per_shard: int = 8
    workers: int = 1
    max_games: int | None = None
    seed: int = 0
    selfplay: SelfPlayConfig = SelfPlayConfig()
    checkpoint_wait_secs: float = 600.0
    """How long to wait for the learner to create ``current.pt``."""


@dataclass(frozen=True)
class WorkerTask:
    seed: int
    checkpoint: str
    mtime_ns: int | None


@dataclass(frozen=True)
class WorkerResult:
    seed: int
    ckpt_iteration: int
    records: tuple[SelfPlayRecord, ...]
    reloaded: bool
    secs: float


@dataclass(frozen=True)
class ShardBuffer:
    """Completed games awaiting a shard write, all from one checkpoint."""

    ckpt_iteration: int | None
    games: tuple[tuple[SelfPlayRecord, ...], ...]
    seeds: tuple[int, ...]

    @classmethod
    def empty(cls) -> ShardBuffer:
        return cls(ckpt_iteration=None, games=(), seeds=())

    @property
    def n_games(self) -> int:
        return len(self.games)

    def records(self) -> list[SelfPlayRecord]:
        return [r for game in self.games for r in game]


def game_seed(base_seed: int, actor_id: int, game_index: int) -> int:
    """Collision-resistant per-game seed from (base, actor, index)."""
    digest = blake2b(f"{base_seed}:{actor_id}:{game_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def absorb(
    buffer: ShardBuffer, result: WorkerResult, games_per_shard: int
) -> tuple[ShardBuffer, tuple[ShardBuffer, ...]]:
    """Fold one finished game into the buffer; returns the new buffer and
    any buffers that are now due for writing (checkpoint changed, or the
    shard is full). Dropped games (no records) are ignored."""
    if not result.records:
        return buffer, ()
    flushed: tuple[ShardBuffer, ...] = ()
    if buffer.n_games and buffer.ckpt_iteration != result.ckpt_iteration:
        flushed = (buffer,)
        buffer = ShardBuffer.empty()
    buffer = ShardBuffer(
        ckpt_iteration=result.ckpt_iteration,
        games=(*buffer.games, result.records),
        seeds=(*buffer.seeds, result.seed),
    )
    if buffer.n_games >= games_per_shard:
        return ShardBuffer.empty(), (*flushed, buffer)
    return buffer, flushed


def next_seq(shards_dir: Path, actor_id: int) -> int:
    """One past the highest shard sequence this actor id has written."""
    if not shards_dir.is_dir():
        return 0
    tag = f"_a{actor_id:03d}_"
    seqs = [
        int(p.stem.rsplit("_", 1)[1])
        for p in shards_dir.iterdir()
        if sh.is_shard_file(p) and tag in p.name
    ]
    return max(seqs) + 1 if seqs else 0


# --------------------------------------------------------------------------
# worker process
# --------------------------------------------------------------------------

_WORKER: dict[str, object] = {}


def _actor_worker_init(checkpoint: str, cfg: SelfPlayConfig) -> None:
    from bgai.agents.mcts import MCTSAgent
    from bgai.training.model import PolicyValueNet, model_config_from_checkpoint

    torch.set_num_threads(1)
    path = Path(checkpoint)
    payload = load_checkpoint(path)
    net = PolicyValueNet(model_config_from_checkpoint(payload))
    net.load_state_dict(payload["model"])
    agent = MCTSAgent(
        net=net,
        simulations=cfg.simulations,
        max_depth=cfg.max_depth,
        top_k=cfg.top_k,
        leaf_batch=cfg.leaf_batch,
        c_puct=cfg.c_puct,
        device="cpu",
    )
    _WORKER.update(
        agent=agent, cfg=cfg, mtime_ns=checkpoint_mtime_ns(path),
        ckpt_iteration=int(payload.get("iteration", 0)),
    )


def _maybe_reload(task: WorkerTask) -> bool:
    if task.mtime_ns is None or task.mtime_ns == _WORKER["mtime_ns"]:
        return False
    agent = _WORKER["agent"]
    try:
        iteration = agent.reload(Path(task.checkpoint))  # type: ignore[attr-defined]
    except Exception as exc:  # corrupt/foreign file: keep playing on old weights
        log.warning("worker: reload of %s failed, keeping old weights: %s", task.checkpoint, exc)
        return False
    _WORKER.update(mtime_ns=task.mtime_ns, ckpt_iteration=iteration)
    return True


def _actor_worker_game(task: WorkerTask) -> WorkerResult:
    import random as _random

    from bgai.arena.setups import sample_setup

    t0 = time.perf_counter()
    reloaded = _maybe_reload(task)
    rng = _random.Random(task.seed)
    setup = sample_setup(rng)
    agent, cfg = _WORKER["agent"], _WORKER["cfg"]
    records = play_game(agent, setup, rng, cfg)  # type: ignore[arg-type]
    return WorkerResult(
        seed=task.seed,
        ckpt_iteration=int(_WORKER["ckpt_iteration"]),  # type: ignore[arg-type]
        records=tuple(records),
        reloaded=reloaded,
        secs=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------
# actor process
# --------------------------------------------------------------------------


def _wait_for_checkpoint(path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        if path.exists():
            try:
                return load_checkpoint(path)
            except ValueError as exc:
                log.warning("actor: %s not loadable yet: %s", path, exc)
        if time.monotonic() > deadline:
            raise TimeoutError(f"no valid checkpoint at {path} after {timeout:.0f}s")
        time.sleep(1.0)


def _write_buffer(cfg: ActorConfig, buffer: ShardBuffer, seq: int, config_hash: str) -> Path:
    assert buffer.ckpt_iteration is not None
    path = cfg.run_dir / SHARDS_DIR / sh.shard_name(buffer.ckpt_iteration, cfg.actor_id, seq)
    meta = sh.write_shard(
        path, buffer.records(), ckpt_iteration=buffer.ckpt_iteration, actor_id=cfg.actor_id,
        seed=buffer.seeds[0], config_hash=config_hash, n_games=buffer.n_games,
    )
    log.info("actor %d: wrote %s (%d games, %d records)", cfg.actor_id, path.name,
             meta.n_games, meta.n_records)
    return path


@dataclass(frozen=True)
class _Progress:
    next_game: int
    played: int
    in_flight: int
    seq: int
    buffer: ShardBuffer


def _stop_requested(cfg: ActorConfig, progress: _Progress, stop: StopSignal) -> bool:
    if cfg.max_games is not None and progress.next_game >= cfg.max_games:
        return True
    return stop.requested()


def run_actor(cfg: ActorConfig) -> int:
    """Play until ``max_games`` or ``R/STOP``; returns games completed."""
    if cfg.workers < 1 or cfg.games_per_shard < 1:
        raise ValueError("workers and games_per_shard must be >= 1")
    env = pin_single_thread_env()
    current = cfg.run_dir / CURRENT_CHECKPOINT
    payload = _wait_for_checkpoint(current, cfg.checkpoint_wait_secs)
    shards_dir = cfg.run_dir / SHARDS_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    seq0 = next_seq(shards_dir, cfg.actor_id)
    chash = sh.config_hash(cfg.selfplay)
    log.info("actor %d: start at ckpt iteration %s, seq %d, env %s", cfg.actor_id,
             payload.get("iteration"), seq0, env)
    progress = _Progress(
        next_game=seq0 * cfg.games_per_shard, played=0, in_flight=0, seq=seq0,
        buffer=ShardBuffer.empty(),
    )
    done: queue.Queue = queue.Queue()
    stop = StopSignal(cfg.run_dir)  # SIGTERM or R/STOP: finish in-flight games, flush
    stop.install()
    ctx = get_context("spawn")
    try:
        with ctx.Pool(
            processes=cfg.workers,
            initializer=_actor_worker_init,
            initargs=(str(current), cfg.selfplay),
        ) as pool:
            submit = _submitter(pool, cfg, current, done)
            while progress.in_flight < cfg.workers and not _stop_requested(cfg, progress, stop):
                progress = submit(progress)
            progress = _drain(cfg, progress, done, submit, chash, stop)
    finally:
        stop.uninstall()
    if progress.buffer.n_games:
        _write_buffer(cfg, progress.buffer, progress.seq, chash)
    return progress.played


def _submitter(pool, cfg: ActorConfig, current: Path, done: queue.Queue):
    def submit(progress: _Progress) -> _Progress:
        task = WorkerTask(
            seed=game_seed(cfg.seed, cfg.actor_id, progress.next_game),
            checkpoint=str(current), mtime_ns=checkpoint_mtime_ns(current),
        )
        pool.apply_async(
            _actor_worker_game, (task,),
            callback=lambda r: done.put(("ok", r)),
            error_callback=lambda e: done.put(("err", e)),
        )
        return replace(progress, next_game=progress.next_game + 1,
                       in_flight=progress.in_flight + 1)

    return submit


def _drain(
    cfg: ActorConfig, progress: _Progress, done: queue.Queue, submit, chash: str,
    stop: StopSignal,
):
    """Consume results, write due shards, keep the pool fed until stop."""
    while progress.in_flight > 0:
        kind, item = done.get()
        progress = replace(progress, in_flight=progress.in_flight - 1)
        if kind == "err":
            log.error("actor %d: worker game failed: %r", cfg.actor_id, item)
        else:
            buffer, due = absorb(progress.buffer, item, cfg.games_per_shard)
            seq = progress.seq
            for full in due:
                _write_buffer(cfg, full, seq, chash)
                seq += 1
            progress = replace(progress, buffer=buffer, seq=seq, played=progress.played + 1)
            log.info("actor %d: game seed %d done in %.1fs (ckpt it %d%s, %d records)",
                     cfg.actor_id, item.seed, item.secs, item.ckpt_iteration,
                     ", reloaded" if item.reloaded else "", len(item.records))
        if not _stop_requested(cfg, progress, stop):
            progress = submit(progress)
    return progress


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_search_args(parser: argparse.ArgumentParser) -> None:
    """Search flags shared with ``selfplay_train`` (same defaults)."""
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--leaf-batch", type=int, default=16)
    parser.add_argument("--temp-decisions", type=int, default=30)
    parser.add_argument("--fast-prior", type=float, default=0.95)
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="PUCT exploration constant (arena champion uses 2.5, C16)")


def selfplay_config_from_args(args: argparse.Namespace) -> SelfPlayConfig:
    return SelfPlayConfig(
        simulations=args.simulations, top_k=args.top_k, max_depth=args.max_depth,
        leaf_batch=args.leaf_batch, temp_decisions=args.temp_decisions,
        fast_prior=args.fast_prior, c_puct=args.c_puct,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Self-play actor (game generation).")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--games-per-shard", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-wait-secs", type=float, default=600.0)
    add_search_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    played = run_actor(
        ActorConfig(
            run_dir=args.run_dir, actor_id=args.actor_id,
            games_per_shard=args.games_per_shard, workers=args.workers,
            max_games=args.max_games, seed=args.seed,
            selfplay=selfplay_config_from_args(args),
            checkpoint_wait_secs=args.checkpoint_wait_secs,
        )
    )
    log.info("actor %d: exit after %d games", args.actor_id, played)


if __name__ == "__main__":
    main()
