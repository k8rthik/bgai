"""Policy/value net + dataset loader (plan 2026-08-04-imitation-phase5, Task 6)."""

from __future__ import annotations

from pathlib import Path

import torch

from bgai.training.dataset import ImitationDataset, collate
from bgai.training.dataset_build import build
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.model import ModelConfig, PolicyValueNet


def _batch(n: int = 4, n_cand: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    mask = torch.ones((n, n_cand), dtype=torch.bool)
    mask[:, -2:] = False  # last two are padding
    return {
        "hex_planes": torch.randn(n, 113, HEX_FEAT_DIM),
        "globals": torch.randn(n, GLOBAL_DIM),
        "faction": torch.randint(0, 14, (n,)),
        "candidates": torch.randint(0, 7, (n, n_cand, MOVE_FIELDS)),
        "cand_mask": mask,
    }


def test_forward_shapes_and_masking() -> None:
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32))
    b = _batch()
    logits, value = net(b["hex_planes"], b["globals"], b["faction"], b["candidates"], b["cand_mask"])
    assert logits.shape == (4, 6) and value.shape == (4, 4)
    assert torch.isinf(logits[:, -2:]).all() and (logits[:, -2:] < 0).all()
    assert logits.argmax(dim=-1).max().item() < 4  # never picks padding
    probs = logits.softmax(dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (probs[:, -2:] == 0).all()


def test_overfits_a_tiny_synthetic_task() -> None:
    """Learning sanity: a fixed batch's labels must be memorizable."""
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    b = _batch(n=8, n_cand=5)
    labels = torch.randint(0, 3, (8,))
    first = last = None
    for step in range(80):
        logits, _ = net(
            b["hex_planes"], b["globals"], b["faction"], b["candidates"], b["cand_mask"]
        )
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5, (first, last)


def test_dataset_split_and_collate(tmp_path: Path) -> None:
    build(tmp_path, limit=4)
    train = ImitationDataset(tmp_path, "train")
    val = ImitationDataset(tmp_path, "val")
    assert len(train) + len(val) > 200
    assert len(set(train.index) & set(val.index)) == 0

    items = [train[i] for i in range(8)]
    batch = collate(items)
    assert batch["hex_planes"].shape == (8, 113, HEX_FEAT_DIM)
    assert batch["candidates"].shape[2] == MOVE_FIELDS
    assert batch["cand_mask"].shape == batch["candidates"].shape[:2]
    assert (batch["chosen"] < batch["cand_mask"].sum(dim=1)).all()
    assert torch.allclose(batch["value"].sum(dim=1), torch.ones(8), atol=1e-5)
    assert (batch["weight"] > 0).all()
