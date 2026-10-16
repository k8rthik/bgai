"""Shared fixtures for the actor/learner self-play tests.

Everything here is synthetic on purpose: a tiny ``PolicyValueNet`` whose
architecture is recorded in the checkpoint (so workers in spawned
processes rebuild it from the file alone) and hand-built
``SelfPlayRecord`` lists with the real dtypes/shapes, so shard and
learner tests never need to play a game or touch ``data/``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bgai.training.checkpoint_io import save_atomic
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.selfplay import SelfPlayRecord
from bgai.training.vocab import ENCODING_VERSION, HEXES

TINY = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0, aux_head=True)


def tiny_net(seed: int = 0, cfg: ModelConfig = TINY) -> PolicyValueNet:
    torch.manual_seed(seed)
    return PolicyValueNet(cfg)


def checkpoint_payload(net: PolicyValueNet, iteration: int) -> dict:
    return {
        "model": net.state_dict(),
        "encoding_version": ENCODING_VERSION,
        "iteration": iteration,
        "config": {
            "value_simplex": net.cfg.value_simplex,
            "aux_head": net.cfg.aux_head,
            "model_config": asdict(net.cfg),
        },
    }


def write_tiny_checkpoint(path: Path, iteration: int = 0, seed: int = 0) -> PolicyValueNet:
    net = tiny_net(seed)
    save_atomic(checkpoint_payload(net, iteration), path)
    return net


def synthetic_records(
    n: int, rng: np.random.Generator, *, with_aux: bool = True, max_cand: int = 9
) -> list[SelfPlayRecord]:
    """Records with ragged candidate sets (1..max_cand) and binary hex planes."""
    out = []
    for _ in range(n):
        k = int(rng.integers(1, max_cand + 1))
        shares = rng.random(4).astype(np.float32)
        shares /= shares.sum()
        out.append(
            SelfPlayRecord(
                hex_planes=rng.integers(0, 2, (len(HEXES), HEX_FEAT_DIM)).astype(np.int8),
                globals=rng.integers(-3, 40, (GLOBAL_DIM,)).astype(np.int16),
                candidates=rng.integers(0, 8, (k, MOVE_FIELDS)).astype(np.int16),
                visits=rng.integers(0, 50, (k,)).astype(np.float32),
                faction_id=int(rng.integers(0, 14)),
                seat=int(rng.integers(0, 4)),
                final_shares=shares,
                aux_target=rng.random(8).astype(np.float32) if with_aux else None,
            )
        )
    return out


def records_equal(a: SelfPlayRecord, b: SelfPlayRecord) -> bool:
    def same(x, y) -> bool:
        if x is None or y is None:
            return x is None and y is None
        return x.dtype == y.dtype and x.shape == y.shape and bool(np.array_equal(x, y))

    return (
        same(a.hex_planes, b.hex_planes)
        and same(a.globals, b.globals)
        and same(a.candidates, b.candidates)
        and same(a.visits, b.visits)
        and a.faction_id == b.faction_id
        and a.seat == b.seat
        and same(a.final_shares, b.final_shares)
        and same(a.aux_target, b.aux_target)
    )
