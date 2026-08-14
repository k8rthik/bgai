"""Offline hyperparameter sweep over cached self-play records.

    uv run python -u tools/offline_sweep.py \
        --records "data/checkpoints/selfplay_deep_d/records_*.pt" \
        --checkpoint data/checkpoints/selfplay_deep_c/current.pt \
        --anchor data/checkpoints/snap/simplex_best.pt

Generation costs ~150x training, so every training-side knob
(rank_weight, winner_pair_weight, win_weight, lambda_kl, lr, epochs)
sweeps against IDENTICAL cached games: each config fine-tunes a fresh
copy of the checkpoint on the same records, then scores on the held-out
human validation set (pairwise ordering + winner-top1, the metrics
search consumes) and reports final self-play losses. Rank/ordering
numbers decide; MSE is reported but never selected on (rank_loss.py's
documented dissociation).
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bgai.training.dataset import ImitationDataset, collate
from bgai.training.model import PolicyValueNet, model_config_from_checkpoint
from bgai.training.rank_loss import ordering_stats
from bgai.training.selfplay import SelfPlayRecord  # noqa: F401 -- unpickling
from bgai.training.selfplay_train import SelfPlayTrainConfig, fine_tune


def held_out_ordering(net: PolicyValueNet, max_batches: int = 60) -> dict[str, float]:
    ds = ImitationDataset(Path("data/datasets/imitation_v2"), "val", max_cached_shards=6)
    loader = DataLoader(ds, batch_size=512, collate_fn=collate)
    totals = {"winner_correct": 0, "pairs_correct": 0, "pairs_total": 0, "rows": 0}
    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            _logits, value = net(
                batch["hex_planes"], batch["globals"], batch["faction"],
                batch["candidates"], batch["cand_mask"],
            )
            stats = ordering_stats(value, batch["value"])
            for k in totals:
                totals[k] += stats[k]
    return {
        "pairwise": totals["pairs_correct"] / max(totals["pairs_total"], 1),
        "winner_top1": totals["winner_correct"] / max(totals["rows"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="glob of records_*.pt files")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/offline_sweep.jsonl"))
    parser.add_argument("--rank-weights", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--win-weights", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--lambda-kls", type=float, nargs="+", default=[1.0])
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-5])
    parser.add_argument("--epochs", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    records = []
    for path in sorted(glob.glob(args.records)):
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
    if not records:
        raise SystemExit(f"no records matched {args.records!r}")
    print(f"{len(records)} cached records from {args.records}", flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    anchor_ckpt = torch.load(args.anchor, map_location="cpu", weights_only=False)
    frozen = PolicyValueNet(model_config_from_checkpoint(anchor_ckpt))
    frozen.load_state_dict(anchor_ckpt["model"])
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    grid = list(itertools.product(
        args.rank_weights, args.win_weights, args.lambda_kls, args.lrs, args.epochs
    ))
    print(f"{len(grid)} configs", flush=True)
    device = torch.device("cpu")
    for rank_w, win_w, kl_w, lr, epochs in grid:
        net = PolicyValueNet(model_config_from_checkpoint(ckpt))
        net.load_state_dict(ckpt["model"])
        cfg = SelfPlayTrainConfig(
            checkpoint=args.checkpoint, out=Path("/tmp/unused"),
            rank_weight=rank_w, win_weight=win_w, lambda_kl=kl_w,
            lr=lr, epochs_per_iteration=epochs,
        )
        parts = fine_tune(net, frozen, records, cfg, device)
        ordering = held_out_ordering(net)
        row = {
            "rank_weight": rank_w, "win_weight": win_w, "lambda_kl": kl_w,
            "lr": lr, "epochs": epochs, **ordering,
            **{f"loss_{k}": v for k, v in parts.items()},
        }
        print(json.dumps(row), flush=True)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
