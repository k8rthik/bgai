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
    net: PolicyValueNet, batch: dict[str, torch.Tensor], value_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return policy_loss + value_weight * value_loss, policy_loss, logits


def _topk_correct(logits: torch.Tensor, chosen: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    return int((logits.topk(k, dim=-1).indices == chosen[:, None]).any(dim=-1).sum().item())


@torch.no_grad()
def evaluate(
    net: PolicyValueNet, loader: DataLoader, device: torch.device, value_weight: float
) -> dict[str, float]:
    net.eval()
    n = top1 = top3 = 0
    loss_sum = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        loss, _, logits = _losses(net, batch, value_weight)
        bs = batch["chosen"].shape[0]
        n += bs
        loss_sum += loss.item() * bs
        top1 += _topk_correct(logits, batch["chosen"], 1)
        top3 += _topk_correct(logits, batch["chosen"], 3)
    net.train()
    if n == 0:
        return {"loss": float("nan"), "top1": float("nan"), "top3": float("nan"), "n": 0}
    return {"loss": loss_sum / n, "top1": top1 / n, "top3": top3 / n, "n": n}


def train(cfg: TrainConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.out / "metrics.jsonl"

    train_ds = ImitationDataset(cfg.shards, "train")
    val_ds = ImitationDataset(cfg.shards, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg.num_workers,
    )

    net = PolicyValueNet(ModelConfig()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"params={sum(p.numel() for p in net.parameters()):,}",
        flush=True,
    )

    step = 0
    last: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = _to_device(batch, device)
            loss, policy_loss, logits = _losses(net, batch, cfg.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                acc = _topk_correct(logits, batch["chosen"], 1) / batch["chosen"].shape[0]
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"policy {policy_loss.item():.4f} batch_top1 {acc:.3f}",
                    flush=True,
                )
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        val = evaluate(net, val_loader, device, cfg.value_weight)
        last = {"epoch": epoch, "step": step, "secs": time.perf_counter() - t0, **val}
        print(f"[val] {json.dumps(last)}", flush=True)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(last) + "\n")
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "config": {**asdict(cfg), "shards": str(cfg.shards), "out": str(cfg.out)},
                "encoding_version": ENCODING_VERSION,
                "step": step,
                "val": val,
            },
            cfg.out / "checkpoint.pt",
        )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
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
        )
    )


if __name__ == "__main__":
    main()
