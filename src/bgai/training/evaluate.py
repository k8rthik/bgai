"""Held-out accuracy breakdowns (plan Task 8 reporting).

    uv run python -m bgai.training.evaluate --checkpoint data/checkpoints/imitation_v1/checkpoint.pt

Reports overall val top-1/top-3 plus per-faction and per-verb splits --
the master plan's asymmetry requirement is that accuracy is never
reported as one aggregate number that hides a faction the net cannot
play. Also reports the random-among-candidates floor for context.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bgai.training.dataset import ImitationDataset, collate
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.train import pick_device
from bgai.training.vocab import ENCODING_VERSION, FACTION_NAMES, VERBS


def _load(checkpoint: Path, device: torch.device) -> PolicyValueNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"checkpoint encoding v{ckpt.get('encoding_version')} != code v{ENCODING_VERSION}"
        )
    net = PolicyValueNet(ModelConfig())
    net.load_state_dict(ckpt["model"])
    return net.to(device).eval()


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path, shards: Path, device_name: str = "auto", batch_size: int = 512
) -> dict[str, object]:
    device = pick_device(device_name)
    net = _load(checkpoint, device)
    loader = DataLoader(
        ImitationDataset(shards, "val"), batch_size=batch_size, collate_fn=collate
    )

    totals = [0, 0, 0]  # n, top1, top3
    floor = 0.0
    by_faction: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_verb: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    value_err = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, value = net(
            batch["hex_planes"], batch["globals"], batch["faction"],
            batch["candidates"], batch["cand_mask"],
        )
        chosen = batch["chosen"]
        n_cand = batch["cand_mask"].sum(dim=1)
        hit1 = logits.argmax(dim=-1) == chosen
        k = min(3, logits.shape[1])
        hit3 = (logits.topk(k, dim=-1).indices == chosen[:, None]).any(dim=-1)
        floor += float((1.0 / n_cand.clamp(min=1)).sum().item())
        value_err += float((value - batch["value"]).abs().mean().item()) * chosen.shape[0]

        # the chosen move's verb id is field 0 of its feature row
        verbs = batch["candidates"][torch.arange(chosen.shape[0]), chosen, 0]
        for i in range(chosen.shape[0]):
            totals[0] += 1
            totals[1] += int(hit1[i])
            totals[2] += int(hit3[i])
            f = int(batch["faction"][i])
            by_faction[f][0] += 1
            by_faction[f][1] += int(hit1[i])
            by_faction[f][2] += int(hit3[i])
            v = int(verbs[i]) - 1
            by_verb[v][0] += 1
            by_verb[v][1] += int(hit1[i])
            by_verb[v][2] += int(hit3[i])

    n = totals[0]
    print(f"val decisions: {n}")
    print(f"top1 {totals[1] / n:.4f}  top3 {totals[2] / n:.4f}  "
          f"(random-among-candidates floor: {floor / n:.4f})")
    print(f"value head mean |err| per seat share: {value_err / n:.4f}")

    print("\nper faction (top1 / top3 / n):")
    for f, (fn, f1, f3) in sorted(by_faction.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        print(f"  {FACTION_NAMES[f]:16} {f1 / fn:.3f}  {f3 / fn:.3f}  {fn}")

    print("\nper verb (top1 / top3 / n):")
    for v, (vn, v1, v3) in sorted(by_verb.items(), key=lambda kv: -kv[1][0]):
        name = VERBS[v] if 0 <= v < len(VERBS) else f"?{v}"
        print(f"  {name:16} {v1 / vn:.3f}  {v3 / vn:.3f}  {vn}")

    return {
        "n": n,
        "top1": totals[1] / n,
        "top3": totals[2] / n,
        "floor": floor / n,
        "by_faction": {FACTION_NAMES[f]: (c[1] / c[0], c[0]) for f, c in by_faction.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an imitation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards", type=Path, default=Path("data/datasets/imitation"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    evaluate_checkpoint(args.checkpoint, args.shards, args.device)


if __name__ == "__main__":
    main()
