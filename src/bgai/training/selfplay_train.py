"""Self-play training loop: generate -> fine-tune -> evaluate (Phase 6b).

    uv run python -m bgai.training.selfplay_train \\
        --checkpoint data/checkpoints/imitation/checkpoint.pt \\
        --out data/checkpoints/selfplay_v1 \\
        --iterations 4 --games-per-iteration 240 --workers 8

Each iteration:
1. **Generate** self-play games in parallel worker processes (game
   generation is CPU-bound engine work, and each worker holds its own
   copy of a ~5M-parameter net, so processes scale nearly linearly).
2. **Fine-tune** on the fresh games: policy toward the search's visit
   distribution, value toward realised final VP shares, KL toward the
   *frozen* imitation policy.
3. **Checkpoint** and log, so the run is resumable and every iteration's
   model can be evaluated afterwards.

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
import json
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.model import PolicyValueNet, model_config_from_checkpoint
from bgai.training.selfplay import SelfPlayConfig, SelfPlayRecord, play_game, regularized_loss
from bgai.training.train import pick_device
from bgai.training.vocab import ENCODING_VERSION


@dataclass(frozen=True)
class SelfPlayTrainConfig:
    checkpoint: Path
    out: Path
    anchor: Path | None = None
    """Checkpoint for the frozen KL anchor. Defaults to ``checkpoint``,
    which is correct for a fresh run; a RESTART from a mid-run
    ``current.pt`` must pass the original human-imitation checkpoint
    here, or the piKL leash silently re-anchors to the drifted policy
    it exists to restrain."""
    iterations: int = 4
    games_per_iteration: int = 240
    workers: int = 8
    simulations: int = 512
    top_k: int = 8
    max_depth: int = 48
    leaf_batch: int = 16
    temp_decisions: int = 30
    lambda_kl: float = 1.0
    rank_weight: float = 1.0
    lr: float = 5e-5
    batch_size: int = 128
    epochs_per_iteration: int = 1
    seed: int = 0

    def selfplay(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            simulations=self.simulations,
            top_k=self.top_k,
            max_depth=self.max_depth,
            leaf_batch=self.leaf_batch,
            temp_decisions=self.temp_decisions,
            lambda_kl=self.lambda_kl,
            rank_weight=self.rank_weight,
            lr=self.lr,
        )


# --------------------------------------------------------------------------
# parallel generation
# --------------------------------------------------------------------------

_WORKER: dict[str, object] = {}


def _worker_init(checkpoint: str, cfg: SelfPlayConfig) -> None:
    """One net per worker process, loaded once."""
    from bgai.agents.mcts import MCTSAgent

    torch.set_num_threads(1)  # workers must not fight over BLAS threads
    _WORKER["agent"] = MCTSAgent(
        Path(checkpoint),
        simulations=cfg.simulations,
        max_depth=cfg.max_depth,
        top_k=cfg.top_k,
        leaf_batch=cfg.leaf_batch,
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
    checkpoint: Path, games: int, workers: int, cfg: SelfPlayConfig, seed: int
) -> list[SelfPlayRecord]:
    ctx = get_context("spawn")
    tasks = [seed * 100_000 + i for i in range(games)]
    with ctx.Pool(
        processes=workers, initializer=_worker_init, initargs=(str(checkpoint), cfg)
    ) as pool:
        batches = pool.map(_worker_game, tasks, chunksize=1)
    return [record for batch in batches for record in batch]


# --------------------------------------------------------------------------
# fine-tuning
# --------------------------------------------------------------------------


def _collate(records: list[SelfPlayRecord], device: torch.device) -> dict[str, torch.Tensor]:
    max_cands = max(r.candidates.shape[0] for r in records)
    n = len(records)
    cand = torch.zeros((n, max_cands, MOVE_FIELDS), dtype=torch.long)
    mask = torch.zeros((n, max_cands), dtype=torch.bool)
    visits = torch.zeros((n, max_cands), dtype=torch.float32)
    for i, r in enumerate(records):
        k = r.candidates.shape[0]
        cand[i, :k] = torch.from_numpy(r.candidates.astype(np.int64))
        mask[i, :k] = True
        visits[i, :k] = torch.from_numpy(r.visits)
    return {
        "hex_planes": torch.stack(
            [torch.from_numpy(r.hex_planes.astype(np.float32)) for r in records]
        ).to(device),
        "globals": torch.stack(
            [torch.from_numpy(r.globals.astype(np.float32)) for r in records]
        ).to(device),
        "faction": torch.tensor([r.faction_id for r in records], dtype=torch.long).to(device),
        "candidates": cand.to(device),
        "cand_mask": mask.to(device),
        "visits": visits.to(device),
        "value": torch.stack(
            [torch.from_numpy(np.asarray(r.final_shares, dtype=np.float32)) for r in records]
        ).to(device),
    }


def fine_tune(
    net: PolicyValueNet,
    frozen: PolicyValueNet,
    records: list[SelfPlayRecord],
    cfg: SelfPlayTrainConfig,
    device: torch.device,
) -> dict[str, float]:
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    net.train()
    last: dict[str, float] = {}
    for _ in range(cfg.epochs_per_iteration):
        order = rng.permutation(len(records))
        for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            batch_records = [records[i] for i in order[start : start + cfg.batch_size]]
            batch = _collate(batch_records, device)
            loss, parts = regularized_loss(net, frozen, batch, cfg.lambda_kl, cfg.rank_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            last = parts
    net.eval()
    return last


def run(cfg: SelfPlayTrainConfig) -> None:
    device = pick_device("cpu")  # generation dominates; keep the net on CPU
    cfg.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("checkpoint encoding version mismatch")

    # The checkpoint knows its own architecture (value_simplex above all):
    # loading simplex weights into a bare head silently mis-scales every
    # value MCTS reads. Same reason every checkpoint written below must
    # carry the flag forward -- workers rebuild agents from these files.
    model_cfg = model_config_from_checkpoint(ckpt)
    net = PolicyValueNet(model_cfg)
    net.load_state_dict(ckpt["model"])
    net.to(device)
    anchor_ckpt = ckpt
    if cfg.anchor is not None:
        anchor_ckpt = torch.load(cfg.anchor, map_location=device, weights_only=False)
        if anchor_ckpt.get("encoding_version") != ENCODING_VERSION:
            raise ValueError("anchor checkpoint encoding version mismatch")
    frozen = PolicyValueNet(model_config_from_checkpoint(anchor_ckpt))
    frozen.load_state_dict(anchor_ckpt["model"])
    frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    saved_config = {
        **asdict(cfg),
        "checkpoint": str(cfg.checkpoint),
        "out": str(cfg.out),
        "value_simplex": model_cfg.value_simplex,
    }
    metrics_path = cfg.out / "metrics.jsonl"
    current = cfg.out / "current.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "encoding_version": ENCODING_VERSION,
            "iteration": 0,
            "config": saved_config,
        },
        current,
    )

    for iteration in range(1, cfg.iterations + 1):
        t0 = time.perf_counter()
        records = generate_parallel(
            current, cfg.games_per_iteration, cfg.workers, cfg.selfplay(),
            cfg.seed + iteration,
        )
        gen_secs = time.perf_counter() - t0
        if len(records) < cfg.batch_size:
            raise RuntimeError(f"iteration {iteration} produced only {len(records)} records")

        t1 = time.perf_counter()
        parts = fine_tune(net, frozen, records, cfg, device)
        train_secs = time.perf_counter() - t1

        payload = {
            "model": net.state_dict(),
            "encoding_version": ENCODING_VERSION,
            "iteration": iteration,
            "config": saved_config,
        }
        torch.save(payload, current)
        torch.save(payload, cfg.out / f"iter_{iteration:02d}.pt")
        row = {
            "iteration": iteration,
            "games": cfg.games_per_iteration,
            "records": len(records),
            "gen_secs": round(gen_secs, 1),
            "train_secs": round(train_secs, 1),
            "games_per_hour": round(cfg.games_per_iteration / gen_secs * 3600),
            **{k: round(v, 4) for k, v in parts.items()},
        }
        print(json.dumps(row), flush=True)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Human-regularized self-play training.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=None,
                        help="frozen KL anchor (defaults to --checkpoint; a restart "
                        "from current.pt must pass the original imitation ckpt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--games-per-iteration", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--leaf-batch", type=int, default=16)
    parser.add_argument("--temp-decisions", type=int, default=30)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(
        SelfPlayTrainConfig(
            checkpoint=args.checkpoint,
            anchor=args.anchor,
            out=args.out,
            iterations=args.iterations,
            games_per_iteration=args.games_per_iteration,
            workers=args.workers,
            simulations=args.simulations,
            top_k=args.top_k,
            max_depth=args.max_depth,
            leaf_batch=args.leaf_batch,
            temp_decisions=args.temp_decisions,
            lambda_kl=args.lambda_kl,
            rank_weight=args.rank_weight,
            lr=args.lr,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
