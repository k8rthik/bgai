"""State torsos for the architecture sweep.

Every torso maps (hex planes [B, H, F], globals [B, G], faction embedding
[B, E_f]) to one state vector [B, embed]; the policy (dot product with
move embeddings) and the value head are shared and unchanged.

- ``mlp``: the original flattened MLP (built inline by ``PolicyValueNet``
  so its parameter names -- ``torso.0`` ... ``torso.7`` -- stay those of
  every checkpoint already on disk).
- ``resmlp``: the same flattened input, ``depth`` pre-norm residual blocks.
  Asks whether the original is too shallow rather than too narrow.
- ``hexgnn``: one token per hex (features + learned position + broadcast
  global context), ``depth`` rounds of mean-neighbour message passing over
  the static board adjacency, mean/max/context readout. Hard-wires the
  board's geometry, which the flattened MLP has to learn from data.
- ``hexformer``: the same hex tokens plus one context token through a
  pre-norm transformer encoder: board structure learned, not given.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import nn

from bgai.engine.tm.board import base_board
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.vocab import HEXES

if TYPE_CHECKING:
    from bgai.training.model import ModelConfig

TORSOS = ("mlp", "resmlp", "hexgnn", "hexformer")


@lru_cache(maxsize=1)
def hex_adjacency() -> torch.Tensor:
    """Directed edge list [2, E] over ``HEXES`` indices (both directions).
    Static board adjacency: bridges are game state and ride in the hex
    features, not the graph."""
    index = {h: i for i, h in enumerate(HEXES)}
    adjacent = base_board().adjacent
    pairs = sorted(
        {(index[a], index[b]) for a in HEXES for b in adjacent.get(a, ()) if b in index and a != b}
    )
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class ResidualMLP(nn.Module):
    def __init__(self, cfg: ModelConfig, in_dim: int) -> None:
        super().__init__()
        h = cfg.hidden
        self.inp = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(),
                nn.Dropout(cfg.dropout), nn.Linear(h, h),
            )
            for _ in range(cfg.depth)
        )
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.embed))

    def forward(self, hex_planes, globals_, faction_vec) -> torch.Tensor:
        x = self.inp(torch.cat([hex_planes.flatten(start_dim=1), globals_, faction_vec], dim=-1))
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


class _HexTokens(nn.Module):
    """Per-hex token = features + learned position + broadcast context."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hex_dim
        self.feat = nn.Linear(HEX_FEAT_DIM, d)
        self.pos = nn.Parameter(torch.randn(len(HEXES), d) * 0.02)
        self.ctx = nn.Sequential(nn.Linear(GLOBAL_DIM + cfg.faction_embed, d), nn.SiLU())

    def forward(self, hex_planes, globals_, faction_vec) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx(torch.cat([globals_, faction_vec], dim=-1))
        tokens = self.feat(hex_planes) + self.pos + ctx.unsqueeze(1)
        return tokens, ctx


def _readout(cfg: ModelConfig, in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.SiLU(),
        nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, cfg.embed),
    )


class HexGNN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hex_dim
        self.tokens = _HexTokens(cfg)
        edges = hex_adjacency()
        self.register_buffer("src", edges[0], persistent=False)
        self.register_buffer("dst", edges[1], persistent=False)
        degree = torch.bincount(edges[1], minlength=len(HEXES)).clamp(min=1).float()
        self.register_buffer("inv_degree", (1.0 / degree).view(1, -1, 1), persistent=False)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)
            )
            for _ in range(cfg.depth)
        )
        self.readout = _readout(cfg, 3 * d)

    def forward(self, hex_planes, globals_, faction_vec) -> torch.Tensor:
        h, ctx = self.tokens(hex_planes, globals_, faction_vec)
        for layer in self.layers:
            agg = torch.zeros_like(h).index_add_(1, self.dst, h[:, self.src]) * self.inv_degree
            h = h + layer(torch.cat([h, agg], dim=-1))
        return self.readout(torch.cat([h.mean(1), h.amax(1), ctx], dim=-1))


class HexFormer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.hex_dim
        self.tokens = _HexTokens(cfg)
        self.ctx_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d, cfg.heads, dim_feedforward=4 * d, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
        self.readout = _readout(cfg, 2 * d)

    def forward(self, hex_planes, globals_, faction_vec) -> torch.Tensor:
        h, ctx = self.tokens(hex_planes, globals_, faction_vec)
        first = self.ctx_token.expand(h.shape[0], -1, -1) + ctx.unsqueeze(1)
        out = self.encoder(torch.cat([first, h], dim=1))
        return self.readout(torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1))


def build_torso(cfg: ModelConfig, state_dim: int) -> nn.Module:
    """A non-default torso (``mlp`` is built inline by ``PolicyValueNet``)."""
    if cfg.torso == "resmlp":
        return ResidualMLP(cfg, state_dim)
    if cfg.torso == "hexgnn":
        return HexGNN(cfg)
    if cfg.torso == "hexformer":
        return HexFormer(cfg)
    raise ValueError(f"unknown torso {cfg.torso!r}; expected one of {TORSOS}")
