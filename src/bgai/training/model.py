"""Faction-conditioned policy/value net (plan Task 6).

Candidate scoring, not a fixed action space: the torso embeds the state
(hex planes + globals + faction embedding) into a single vector, a small
MLP embeds each legal candidate's field ids, and the policy logit for a
candidate is their dot product. Masked positions get -inf before the
softmax, so the policy is defined over exactly the moves the engine
offered -- legality is structural, never learned.

The value head predicts each of the 4 mover-relative seats' share of the
table's final VP (master plan: 4-player value vector, not a scalar).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.vocab import FACTION_NAMES, HEXES

# Every move field's id space (max id + 1); fields share one embedding
# table with per-field offsets, so a "loc" id can never collide with a
# "tile" id.
_FIELD_SIZES = (32, 128, 128, 8, 64, 8, 16, 8, 8, 8, 32, 32)
assert len(_FIELD_SIZES) == MOVE_FIELDS


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 1024
    embed: int = 256
    move_hidden: int = 256
    faction_embed: int = 32
    field_embed: int = 24
    dropout: float = 0.1
    value_simplex: bool = False
    """Softmax the value head over the 4 seats.

    The target is a share vector summing to 1, but the head is a bare
    Linear(...,4): MSE was the only thing keeping its output near share
    space, so adding the ranking loss inflated value MSE 128x. That also
    breaks MCTS, whose _select falls back to q=0.25 and weighs c_puct
    against Q assuming share-space magnitudes.

    Softmax is monotone, so it bounds the scale without undoing the
    ordering the ranking loss buys. Off by default: applying it to a net
    trained without it would flatten near-uniform outputs, so it belongs
    to runs trained (or fine-tuned) with it on."""


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        state_dim = len(HEXES) * HEX_FEAT_DIM + GLOBAL_DIM + c.faction_embed

        self.faction_embed = nn.Embedding(len(FACTION_NAMES), c.faction_embed)
        self.torso = nn.Sequential(
            nn.Linear(state_dim, c.hidden),
            nn.LayerNorm(c.hidden),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.hidden),
            nn.LayerNorm(c.hidden),
            nn.SiLU(),
            nn.Linear(c.hidden, c.embed),
        )

        offsets = torch.tensor(
            [sum(_FIELD_SIZES[:i]) for i in range(MOVE_FIELDS)], dtype=torch.long
        )
        self.register_buffer("field_offsets", offsets)
        self.field_embed = nn.Embedding(sum(_FIELD_SIZES), c.field_embed)
        self.move_mlp = nn.Sequential(
            nn.Linear(MOVE_FIELDS * c.field_embed, c.move_hidden),
            nn.LayerNorm(c.move_hidden),
            nn.SiLU(),
            nn.Linear(c.move_hidden, c.embed),
        )
        self.value_head = nn.Sequential(
            nn.Linear(c.embed, c.move_hidden), nn.SiLU(), nn.Linear(c.move_hidden, 4)
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / (c.embed**0.5)))

    def state_embedding(
        self, hex_planes: torch.Tensor, globals_: torch.Tensor, faction: torch.Tensor
    ) -> torch.Tensor:
        flat = hex_planes.flatten(start_dim=1)
        x = torch.cat([flat, globals_, self.faction_embed(faction)], dim=-1)
        return self.torso(x)

    def move_embedding(self, candidates: torch.Tensor) -> torch.Tensor:
        ids = candidates.clamp(min=0) + self.field_offsets
        emb = self.field_embed(ids).flatten(start_dim=-2)
        return self.move_mlp(emb)

    def forward(
        self,
        hex_planes: torch.Tensor,
        globals_: torch.Tensor,
        faction: torch.Tensor,
        candidates: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_embedding(hex_planes, globals_, faction)
        moves = self.move_embedding(candidates)
        logits = torch.einsum("be,bce->bc", state, moves) * self.logit_scale
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        value = self.value_head(state)
        if self.cfg.value_simplex:
            value = value.softmax(dim=-1)
        return logits, value
