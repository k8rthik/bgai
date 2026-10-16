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
  same (``stop_signal``), so a tmux kill loses nothing. The stop is
  LATCHED once seen and honoured even while waiting for the first
  checkpoint. The actor never deletes a live ``STOP`` -- whoever owns
  the run does (the launcher, or the operator); a ``STOP`` older than
  the actor's own start is stale and is removed on sight.

Game accounting is exact, not estimated: every shard records the range
of per-actor game indices it holds (``game_index_start``/``_end``), so a
restarted actor with the same ``--actor-id`` resumes its index from the
highest end it finds, counts ``--max-games`` over its LIFETIME (sum of
``n_games`` across its own shards), and derives seeds from
``(base seed, actor_id, index)`` -- no index is ever played into two
shards, whatever ``--games-per-shard`` was before. A game in flight when
the process dies is lost (never replayed, never duplicated).

Workers: ``--workers W`` spawn subprocesses that each hold one
``MCTSAgent`` and stay alive across games (no per-iteration pool
rebuild, no per-iteration setup-cache reload). The parent keeps exactly
W games in flight through ``apply_async`` callbacks, which is what lets
each dispatch observe the freshest checkpoint. Workers ignore
SIGINT/SIGTERM so a process-group Ctrl-C reaches only the parent's
latched handler; a worker that dies anyway (OOM kill) is detected by the
drain loop, its game is dropped and the pool's replacement worker is
fed. Thread pinning (``worker_env``) is exported before the pool spawns.

Actors are individually restartable at any moment. Rolling code
upgrade: stop and restart actors ONE AT A TIME (the others keep the
learner fed), then the learner last; shard/checkpoint formats are
versioned, and a learner refuses shards from a different
``encoding_version``.

Two-host use: run this on the generation host(s) against an rsync'ed
mirror of ``R`` -- see ``tools/sync_run_dir.sh``. Shards are small
(~69 B/record, ~20 KB/game) and checkpoints are ~45 MB, so a
once-a-minute sync is ample.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import time
from collections.abc import Callable
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
from bgai.training.worker_pool import shielded_pool

log = logging.getLogger(__name__)

CURRENT_CHECKPOINT = "current.pt"
SHARDS_DIR = "shards"
DRAIN_POLL_SECS = 1.0


class ActorStopped(Exception):
    """A stop arrived before the actor could start playing."""


@dataclass(frozen=True)
class ActorConfig:
    run_dir: Path
    actor_id: int
    games_per_shard: int = 8
    workers: int = 1
    max_games: int | None = None
    """Lifetime cap for this actor id, counted across restarts."""
    seed: int = 0
    selfplay: SelfPlayConfig = SelfPlayConfig()
    checkpoint_wait_secs: float = 600.0
    """How long to wait for the learner to create ``current.pt``."""


@dataclass(frozen=True)
class WorkerTask:
    seed: int
    game_index: int
    checkpoint: str
    mtime_ns: int | None


@dataclass(frozen=True)
class WorkerResult:
    seed: int
    game_index: int
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
    indices: tuple[int, ...]

    @classmethod
    def empty(cls) -> ShardBuffer:
        return cls(ckpt_iteration=None, games=(), seeds=(), indices=())

    @property
    def n_games(self) -> int:
        return len(self.games)

    @property
    def game_index_start(self) -> int:
        return min(self.indices) if self.indices else 0

    @property
    def game_index_end(self) -> int:
        return max(self.indices) + 1 if self.indices else 0

    def records(self) -> list[SelfPlayRecord]:
        return [r for game in self.games for r in game]


@dataclass(frozen=True)
class ActorHistory:
    """What this actor id has already contributed to the run directory."""

    prior_games: int
    next_index: int
    seq: int


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
        indices=(*buffer.indices, result.game_index),
    )
    if buffer.n_games >= games_per_shard:
        return ShardBuffer.empty(), (*flushed, buffer)
    return buffer, flushed


def _own_shards(shards_dir: Path, actor_id: int) -> list[Path]:
    if not shards_dir.is_dir():
        return []
    tag = f"_a{actor_id:03d}_"
    return sorted(p for p in shards_dir.iterdir() if sh.is_shard_file(p) and tag in p.name)


def next_seq(shards_dir: Path, actor_id: int) -> int:
    """One past the highest shard sequence this actor id has written."""
    seqs = [int(p.stem.rsplit("_", 1)[1]) for p in _own_shards(shards_dir, actor_id)]
    return max(seqs) + 1 if seqs else 0


def actor_history(shards_dir: Path, actor_id: int) -> ActorHistory:
    """Exact restart point from this actor's own shards. Unreadable
    shards count for the sequence (never reuse a name) but not for games."""
    prior_games = 0
    next_index = 0
    for path in _own_shards(shards_dir, actor_id):
        try:
            meta = sh.read_shard_meta(path)
        except sh.ShardError as exc:
            log.warning("actor %d: ignoring unreadable own shard %s: %s", actor_id,
                        path.name, exc)
            continue
        prior_games += meta.n_games
        next_index = max(next_index, meta.game_index_end)
    return ActorHistory(prior_games=prior_games, next_index=next_index,
                        seq=next_seq(shards_dir, actor_id))


# --------------------------------------------------------------------------
# worker process
# --------------------------------------------------------------------------

_WORKER: dict[str, object] = {}


def ignore_stop_signals() -> None:
    """Pool workers must not die on a process-group Ctrl-C/SIGTERM: the
    parent's latched handler decides when to stop, and a worker killed
    mid-task would otherwise take its game (and the drain loop) with it."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except ValueError:  # not the main thread (in-process tests)
            return


def _actor_worker_init(checkpoint: str, cfg: SelfPlayConfig) -> None:
    from bgai.agents.mcts import MCTSAgent
    from bgai.training.model import PolicyValueNet, model_config_from_checkpoint

    ignore_stop_signals()
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
        game_index=task.game_index,
        ckpt_iteration=int(_WORKER["ckpt_iteration"]),  # type: ignore[arg-type]
        records=tuple(records),
        reloaded=reloaded,
        secs=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------
# actor process
# --------------------------------------------------------------------------


def _wait_for_checkpoint(path: Path, timeout: float, stop: StopSignal) -> dict:
    """Poll for a loadable ``current.pt``; a stop request ends the wait."""
    deadline = time.monotonic() + timeout
    while True:
        if stop.requested():
            raise ActorStopped(f"stop requested while waiting for {path}")
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
        game_index_start=buffer.game_index_start, game_index_end=buffer.game_index_end,
    )
    log.info("actor %d: wrote %s (%d games, %d records, indices %d-%d)", cfg.actor_id,
             path.name, meta.n_games, meta.n_records, meta.game_index_start,
             meta.game_index_end)
    return path


@dataclass(frozen=True)
class _Progress:
    next_index: int
    """Next per-actor game index (seed derives from it)."""
    played: int
    """Games completed this session (with or without records)."""
    prior_games: int
    """Games this actor id had already sharded before this session."""
    in_flight: int
    seq: int
    buffer: ShardBuffer
    dropped: int = 0
    """Games lost this session (worker death / task failure)."""

    @property
    def scheduled(self) -> int:
        """Lifetime games this actor has played or has in flight."""
        return self.prior_games + self.played + self.dropped + self.in_flight


def _stop_requested(cfg: ActorConfig, progress: _Progress, stop: StopSignal) -> bool:
    if cfg.max_games is not None and progress.scheduled >= cfg.max_games:
        return True
    return stop.requested()


def _lost_worker_counter(pool) -> Callable[[], int]:
    """Counts pool workers that disappeared since the last call. The pool
    replaces a dead worker but never re-issues its task, so each death
    is one game that will never report back."""
    known = {p.pid for p in pool._pool}  # noqa: SLF001 -- no public handle

    def lost() -> int:
        nonlocal known
        now = {p.pid for p in pool._pool}  # noqa: SLF001
        gone = len(known - now)
        known = now
        return gone

    return lost


def run_actor(cfg: ActorConfig) -> int:
    """Play until ``max_games`` (lifetime) or a stop; returns games
    completed this session."""
    if cfg.workers < 1 or cfg.games_per_shard < 1:
        raise ValueError("workers and games_per_shard must be >= 1")
    env = pin_single_thread_env()
    stop = StopSignal(cfg.run_dir)  # SIGTERM or R/STOP: finish in-flight games, flush
    stop.install()
    try:
        return _run_actor(cfg, stop, env)
    finally:
        stop.uninstall()


def _run_actor(cfg: ActorConfig, stop: StopSignal, env: dict[str, str]) -> int:
    current = cfg.run_dir / CURRENT_CHECKPOINT
    try:
        payload = _wait_for_checkpoint(current, cfg.checkpoint_wait_secs, stop)
    except ActorStopped as exc:
        log.warning("actor %d: %s; exiting with 0 games", cfg.actor_id, exc)
        return 0
    shards_dir = cfg.run_dir / SHARDS_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    history = actor_history(shards_dir, cfg.actor_id)
    chash = sh.config_hash(cfg.selfplay)
    log.info("actor %d: start at ckpt iteration %s, %d prior games, next index %d, seq %d, "
             "env %s", cfg.actor_id, payload.get("iteration"), history.prior_games,
             history.next_index, history.seq, env)
    progress = _Progress(
        next_index=history.next_index, played=0, prior_games=history.prior_games,
        in_flight=0, seq=history.seq, buffer=ShardBuffer.empty(),
    )
    done: queue.Queue = queue.Queue()
    ctx = get_context("spawn")
    with shielded_pool(
        ctx,
        processes=cfg.workers,
        initializer=_actor_worker_init,
        initargs=(str(current), cfg.selfplay),
    ) as pool:
        submit = _submitter(pool, cfg, current, done)
        while progress.in_flight < cfg.workers and not _stop_requested(cfg, progress, stop):
            progress = submit(progress)
        progress = _drain(cfg, progress, done, submit, chash, stop, _lost_worker_counter(pool))
    return progress.played


def _submitter(pool, cfg: ActorConfig, current: Path, done: queue.Queue):
    def submit(progress: _Progress) -> _Progress:
        task = WorkerTask(
            seed=game_seed(cfg.seed, cfg.actor_id, progress.next_index),
            game_index=progress.next_index,
            checkpoint=str(current), mtime_ns=checkpoint_mtime_ns(current),
        )
        pool.apply_async(
            _actor_worker_game, (task,),
            callback=lambda r: done.put(("ok", r)),
            error_callback=lambda e: done.put(("err", e)),
        )
        return replace(progress, next_index=progress.next_index + 1,
                       in_flight=progress.in_flight + 1)

    return submit


def _absorb_result(cfg: ActorConfig, progress: _Progress, item: WorkerResult, chash: str):
    buffer, due = absorb(progress.buffer, item, cfg.games_per_shard)
    seq = progress.seq
    for full in due:
        _write_buffer(cfg, full, seq, chash)
        seq += 1
    log.info("actor %d: game %d (seed %d) done in %.1fs (ckpt it %d%s, %d records)",
             cfg.actor_id, item.game_index, item.seed, item.secs, item.ckpt_iteration,
             ", reloaded" if item.reloaded else "", len(item.records))
    return replace(progress, buffer=buffer, seq=seq, played=progress.played + 1)


def _drain(
    cfg: ActorConfig, progress: _Progress, done: queue.Queue, submit, chash: str,
    stop: StopSignal, lost_workers: Callable[[], int], poll_secs: float = DRAIN_POLL_SECS,
) -> _Progress:
    """Consume results, write due shards, keep the pool fed until stop.

    Waits with a timeout and asks ``lost_workers`` how many pool workers
    died meanwhile: each death is an in-flight game that will never
    report, so it is written off instead of blocking forever. Whatever
    happens, the buffered games are flushed on the way out."""
    try:
        while progress.in_flight > 0:
            try:
                kind, item = done.get(timeout=poll_secs)
            except queue.Empty:
                lost = lost_workers()
                if lost:
                    log.error("actor %d: %d worker(s) died; dropping their games",
                              cfg.actor_id, lost)
                    progress = replace(progress, in_flight=max(0, progress.in_flight - lost),
                                       dropped=progress.dropped + lost)
            else:
                progress = replace(progress, in_flight=progress.in_flight - 1)
                if kind == "err":
                    log.error("actor %d: worker game failed: %r", cfg.actor_id, item)
                    progress = replace(progress, dropped=progress.dropped + 1)
                else:
                    progress = _absorb_result(cfg, progress, item, chash)
            while progress.in_flight < cfg.workers and not _stop_requested(cfg, progress, stop):
                progress = submit(progress)
    finally:
        if progress.buffer.n_games:
            _write_buffer(cfg, progress.buffer, progress.seq, chash)
            progress = replace(progress, buffer=ShardBuffer.empty(), seq=progress.seq + 1)
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
    parser.add_argument("--max-games", type=int, default=None,
                        help="lifetime cap for this actor id (counted across restarts)")
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
