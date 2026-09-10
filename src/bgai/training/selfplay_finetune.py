"""Fine-tuning step shared by the serial self-play loop and the learner.

Exactly the semantics the serial loop had: a fresh AdamW per call, one
(or ``epochs``) shuffled pass over the records, gradient clipping at 5,
``regularized_loss`` as the objective. Both entry points (``selfplay_train``
serial mode and ``selfplay_learner``) call ``fine_tune`` so the pipelined
run differs from the serial one ONLY in where the games come from --
which is what the pipelined-vs-serial validation gate (docs/decisions.md)
relies on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.model import ModelConfig, PolicyValueNet, model_config_from_checkpoint
from bgai.training.selfplay import SelfPlayRecord, regularized_loss
from bgai.training.vocab import ENCODING_VERSION


class FineTuneParams(Protocol):
    lr: float
    seed: int
    epochs_per_iteration: int
    batch_size: int
    lambda_kl: float
    rank_weight: float
    winner_pair_weight: float
    win_weight: float
    aux_weight: float


@dataclass(frozen=True)
class FineTuneConfig:
    """The training-side knobs, detached from any run layout."""

    lr: float = 5e-5
    seed: int = 0
    epochs_per_iteration: int = 1
    batch_size: int = 128
    lambda_kl: float = 1.0
    rank_weight: float = 1.0
    winner_pair_weight: float = 3.0
    win_weight: float = 0.5
    aux_weight: float = 0.5


@dataclass(frozen=True)
class TrainableNets:
    net: PolicyValueNet
    frozen: PolicyValueNet
    model_cfg: ModelConfig
    iteration: int
    """Iteration stamped on the checkpoint the weights came from."""
    stored_config: dict
    """The checkpoint's ``config`` block (resume reads provenance from it)."""


def load_trainable_nets(
    checkpoint: Path, anchor: Path | None, device: torch.device
) -> TrainableNets:
    """Build the trainable net (aux head forced on, C11) from ``checkpoint``
    and the frozen KL anchor from ``anchor`` (default: the same file)."""
    ckpt = load_checkpoint(checkpoint, str(device))
    # The checkpoint knows its own architecture (value_simplex above all):
    # loading simplex weights into a bare head silently mis-scales every
    # value MCTS reads. A checkpoint that predates the aux head loads with
    # strict=False and the aux head alone initializes fresh.
    model_cfg = replace(model_config_from_checkpoint(ckpt), aux_head=True)
    net = PolicyValueNet(model_cfg)
    missing, unexpected = net.load_state_dict(ckpt["model"], strict=False)
    if unexpected or any(not k.startswith("aux_head.") for k in missing):
        raise ValueError(
            f"checkpoint mismatch beyond the aux head: missing={missing} "
            f"unexpected={unexpected}"
        )
    net.to(device).eval()
    anchor_ckpt = ckpt if anchor is None else load_checkpoint(anchor, str(device))
    frozen = PolicyValueNet(model_config_from_checkpoint(anchor_ckpt))
    frozen.load_state_dict(anchor_ckpt["model"])
    frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return TrainableNets(
        net=net, frozen=frozen, model_cfg=model_cfg, iteration=int(ckpt.get("iteration", 0)),
        stored_config=dict(ckpt.get("config") or {}),
    )


def checkpoint_payload(
    net: PolicyValueNet, model_cfg: ModelConfig, iteration: int, run_config: dict
) -> dict:
    """The checkpoint layout every consumer (MCTSAgent, arena, actors)
    reads; ``model_config`` pins the full architecture. Tensors are
    copied to CPU so a learner training on cuda/mps writes files that
    CPU actors load without a device round-trip."""
    return {
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "encoding_version": ENCODING_VERSION,
        "iteration": int(iteration),
        "config": {
            **run_config,
            "value_simplex": model_cfg.value_simplex,
            "aux_head": model_cfg.aux_head,
            "model_config": asdict(model_cfg),
        },
    }


def collate(records: list[SelfPlayRecord], device: torch.device) -> dict[str, torch.Tensor]:
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
        **(
            {
                "aux_target": torch.stack(
                    [
                        torch.from_numpy(np.asarray(r.aux_target, dtype=np.float32))
                        for r in records
                    ]
                ).to(device)
            }
            if all(r.aux_target is not None for r in records)
            else {}
        ),
    }


def fine_tune(
    net: PolicyValueNet,
    frozen: PolicyValueNet,
    records: list[SelfPlayRecord],
    cfg: FineTuneParams,
    device: torch.device,
) -> dict[str, float]:
    """One fine-tuning call: fresh AdamW, ``cfg.epochs_per_iteration``
    shuffled passes, returns the last batch's loss parts."""
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    net.train()
    last: dict[str, float] = {}
    for _ in range(cfg.epochs_per_iteration):
        order = rng.permutation(len(records))
        for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            batch_records = [records[i] for i in order[start : start + cfg.batch_size]]
            batch = collate(batch_records, device)
            loss, parts = regularized_loss(
                net, frozen, batch, cfg.lambda_kl, cfg.rank_weight,
                cfg.winner_pair_weight, cfg.win_weight, cfg.aux_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            last = parts
    net.eval()
    return last
