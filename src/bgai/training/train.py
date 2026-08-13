"""Imitation training loop (plan Task 7).

    uv run python -m bgai.training.train --shards data/datasets/imitation \\
        --out data/checkpoints/imitation_v1 --epochs 3

Cluster-portable by construction: plain PyTorch, a config dataclass, no
hard-coded paths, device auto-selection (cuda > mps > cpu), and
checkpoint/resume so a run can move between the local box and a SLURM
node without code changes.

Loss = division-weighted cross-entropy over the legal candidates (the
policy) + MSE on the 4-seat final-VP-share vector (the value), the
latter scaled by ``value_weight``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bgai.training.dataset import ImitationDataset, collate
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.rank_loss import ordering_stats, pairwise_rank_loss
from bgai.training.sampler import ShardBlockSampler
from bgai.training.vocab import ENCODING_VERSION


@dataclass(frozen=True)
class TrainConfig:
    shards: Path
    out: Path
    epochs: int = 3
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    value_weight: float = 0.5
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    max_steps: int | None = None
    log_every: int = 50
    val_every_steps: int | None = None
    """Validate mid-epoch. At 39k steps/epoch on the population corpus,
    once-per-epoch validation is far too coarse to stop on."""
    early_stop_patience: int = 0
    """Stop after this many consecutive validations with no improvement in
    val loss. 0 disables early stopping."""
    shards_per_block: int = 4
    """Shards mixed together per sampling block; bounds resident memory."""
    resume: Path | None = None
    """Checkpoint to continue from (model + optimizer + step)."""
    allow_dirty_out: bool = False
    """Permit writing into a non-empty output directory. Off by default:
    metrics.jsonl appends and checkpoints overwrite, so reusing a directory
    silently mixes runs and destroys the previous one's weights."""
    value_simplex: bool = False
    """Softmax the value head over seats (see ModelConfig.value_simplex).
    Stored in the checkpoint config so agents rebuild the same head."""
    weight_power: float = 1.0
    """Exponent applied to the per-record strength weight at load time.
    The baked-in weights span only 0.6-1.0 (1.67x), so training imitates
    the average population player rather than a strong one -- and the
    agent measures at the 16th percentile of human play. Sharpening here
    avoids rebuilding a 2.5 GB shard set to change the weighting."""
    rank_weight: float = 0.0
    """Weight on the pairwise ranking term over the value vector. MSE
    optimises the share's magnitude; MCTS only uses the ordering, and the
    two diverge (share-MSE 0.00065 with 30% of pairs ordered backwards).
    Reported val loss deliberately excludes this term so it stays
    comparable across runs with different rank weights."""


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _losses(
    net: PolicyValueNet,
    batch: dict[str, torch.Tensor],
    value_weight: float,
    value_simplex: bool = False
    """Softmax the value head over seats (see ModelConfig.value_simplex).
    Stored in the checkpoint config so agents rebuild the same head."""
    weight_power: float = 1.0
    """Exponent applied to the per-record strength weight at load time.
    The baked-in weights span only 0.6-1.0 (1.67x), so training imitates
    the average population player rather than a strong one -- and the
    agent measures at the 16th percentile of human play. Sharpening here
    avoids rebuilding a 2.5 GB shard set to change the weighting."""
    rank_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(total, policy, value, logits)``.

    The value term is reported separately because the population corpus was
    gathered specifically to fix the value head's coverage (C5) -- a
    combined loss cannot answer whether that worked.

    ``rank_weight`` adds a pairwise ranking term over the seat vector.
    MSE on VP share optimises magnitude, but MCTS only uses the value to
    order positions, and the two come apart: share-MSE 0.00065 coexists
    with 30% of seat pairs ordered backwards. Default 0.0 keeps existing
    runs bit-identical.
    """
    logits, value = net(
        batch["hex_planes"],
        batch["globals"],
        batch["faction"],
        batch["candidates"],
        batch["cand_mask"],
    )
    per_sample = torch.nn.functional.cross_entropy(logits, batch["chosen"], reduction="none")
    weights = batch["weight"]
    policy_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
    value_loss = torch.nn.functional.mse_loss(value, batch["value"])
    total = policy_loss + value_weight * value_loss
    if rank_weight:
        total = total + rank_weight * pairwise_rank_loss(value, batch["value"])
    return total, policy_loss, value_loss, logits


def _topk_correct(logits: torch.Tensor, chosen: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    return int((logits.topk(k, dim=-1).indices == chosen[:, None]).any(dim=-1).sum().item())


@torch.no_grad()
def evaluate(
    net: PolicyValueNet, loader: DataLoader, device: torch.device, value_weight: float
) -> dict[str, float]:
    net.eval()
    n = top1 = top3 = 0
    loss_sum = policy_sum = value_sum = 0.0
    winner = pairs_ok = pairs_tot = 0
    for batch in loader:
        batch = _to_device(batch, device)
        loss, policy_loss, value_loss, logits = _losses(net, batch, value_weight)
        bs = batch["chosen"].shape[0]
        n += bs
        loss_sum += loss.item() * bs
        policy_sum += policy_loss.item() * bs
        value_sum += value_loss.item() * bs
        top1 += _topk_correct(logits, batch["chosen"], 1)
        top3 += _topk_correct(logits, batch["chosen"], 3)
        # ordering is what MCTS consumes; MSE alone cannot see it, and
        # rises by design once rank_weight is on
        _, value = net(
            batch["hex_planes"], batch["globals"], batch["faction"],
            batch["candidates"], batch["cand_mask"],
        )
        stats = ordering_stats(value, batch["value"])
        winner += stats["winner_correct"]
        pairs_ok += stats["pairs_correct"]
        pairs_tot += stats["pairs_total"]
    net.train()
    if n == 0:
        nan = float("nan")
        return {
            "loss": nan, "policy": nan, "value": nan, "top1": nan, "top3": nan,
            "winner_top1": nan, "pairwise": nan, "n": 0,
        }
    return {
        "loss": loss_sum / n,
        "policy": policy_sum / n,
        "value": value_sum / n,
        "top1": top1 / n,
        "top3": top3 / n,
        "winner_top1": winner / n,
        "pairwise": pairs_ok / max(pairs_tot, 1),
        "n": n,
    }


def train(cfg: TrainConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in cfg.out.iterdir()]
    if existing and not cfg.allow_dirty_out:
        raise ValueError(
            f"{cfg.out} is not empty ({', '.join(sorted(existing)[:5])}) -- "
            f"metrics.jsonl appends and checkpoints overwrite, so this would "
            f"mix runs and destroy the previous weights. Pick a fresh --out "
            f"or pass --allow-dirty-out."
        )
    metrics_path = cfg.out / "metrics.jsonl"

    train_ds = ImitationDataset(
        cfg.shards, "train", max_cached_shards=cfg.shards_per_block + 2,
        weight_power=cfg.weight_power,
    )
    val_ds = ImitationDataset(cfg.shards, "val", max_cached_shards=cfg.shards_per_block + 2)
    # block sampling rather than a global shuffle: see sampler module --
    # a global shuffle would miss the shard cache on nearly every access
    train_sampler = ShardBlockSampler(
        train_ds.index, shards_per_block=cfg.shards_per_block, seed=cfg.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg.num_workers,
    )

    net = PolicyValueNet(ModelConfig(value_simplex=cfg.value_simplex)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    if cfg.resume is not None:
        state = torch.load(cfg.resume, map_location=device, weights_only=False)
        if state.get("encoding_version") != ENCODING_VERSION:
            raise ValueError(
                f"checkpoint has encoding v{state.get('encoding_version')}, code is "
                f"v{ENCODING_VERSION} -- weights are not compatible"
            )
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = int(state.get("step", 0))
        print(f"resumed from {cfg.resume} at step {step:,}", flush=True)

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"params={sum(p.numel() for p in net.parameters()):,}",
        flush=True,
    )

    last: dict[str, float] = {}
    best_score = float("inf")
    stale = 0

    def selection_score(val: dict[str, float]) -> float:
        """Lower is better. With rank_weight on, val loss rises by design
        (the margin term widens correct gaps and so inflates value MSE),
        so selecting and stopping on it halts a run while the thing it
        optimises -- seat ordering -- is still improving. Select on
        ordering in that case."""
        return -val["pairwise"] if cfg.rank_weight else val["loss"]

    def checkpoint(val: dict[str, float], epoch: int, t0: float) -> dict[str, float]:
        """Validate-log-save. Returns the metrics row just written.

        Writes ``last.pt`` every time and ``best.pt`` only on improvement:
        saving one file unconditionally means the best weights are
        overwritten by whatever happens to validate last, so an early stop
        would keep the *worst* of its final validations.
        """
        row = {"epoch": epoch, "step": step, "secs": time.perf_counter() - t0, **val}
        print(f"[val] {json.dumps(row)}", flush=True)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        blob = {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "config": {**asdict(cfg), "shards": str(cfg.shards), "out": str(cfg.out)},
            "encoding_version": ENCODING_VERSION,
            "step": step,
            "val": val,
        }
        torch.save(blob, cfg.out / "last.pt")
        if selection_score(val) < best_score:
            torch.save(blob, cfg.out / "best.pt")
        return row

    stop = False
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = _to_device(batch, device)
            loss, policy_loss, value_loss, logits = _losses(
                net, batch, cfg.value_weight, cfg.rank_weight
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                acc = _topk_correct(logits, batch["chosen"], 1) / batch["chosen"].shape[0]
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"policy {policy_loss.item():.4f} value {value_loss.item():.4f} "
                    f"batch_top1 {acc:.3f}",
                    flush=True,
                )
            if cfg.val_every_steps and step % cfg.val_every_steps == 0:
                last = checkpoint(evaluate(net, val_loader, device, cfg.value_weight), epoch, t0)
                t0 = time.perf_counter()
                if selection_score(last) < best_score - 1e-4:
                    best_score, stale = selection_score(last), 0
                else:
                    stale += 1
                    if cfg.early_stop_patience and stale >= cfg.early_stop_patience:
                        print(
                            f"early stop: {stale} validations without improvement "
                            f"(best score {best_score:.4f}, "
                            f"{'pairwise' if cfg.rank_weight else 'loss'})",
                            flush=True,
                        )
                        stop = True
                        break
            if cfg.max_steps is not None and step >= cfg.max_steps:
                stop = True
                break
        if not cfg.val_every_steps:
            last = checkpoint(evaluate(net, val_loader, device, cfg.value_weight), epoch, t0)
        if stop:
            break

    # A step-interval schedule leaves the trailing partial interval
    # unvalidated -- without this the final steps are trained and then
    # discarded, and `last.pt` lags the model that actually exists.
    if cfg.val_every_steps and last.get("step") != step:
        last = checkpoint(evaluate(net, val_loader, device, cfg.value_weight), epoch, t0)
    return last


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the imitation policy/value net.")
    parser.add_argument("--shards", type=Path, default=Path("data/datasets/imitation"))
    parser.add_argument("--out", type=Path, default=Path("data/checkpoints/imitation_v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every-steps", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--shards-per-block", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-dirty-out", action="store_true")
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--value-simplex", action="store_true")
    args = parser.parse_args(argv)
    train(
        TrainConfig(
            shards=args.shards,
            out=args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            max_steps=args.max_steps,
            seed=args.seed,
            val_every_steps=args.val_every_steps,
            early_stop_patience=args.early_stop_patience,
            shards_per_block=args.shards_per_block,
            num_workers=args.num_workers,
            resume=args.resume,
            allow_dirty_out=args.allow_dirty_out,
            rank_weight=args.rank_weight,
            weight_power=args.weight_power,
            value_simplex=args.value_simplex,
        )
    )


if __name__ == "__main__":
    main()
