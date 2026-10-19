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
from bgai.training.torsos import build_torso
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
    aux_head: bool = False
    """Auxiliary economy-trajectory head (2026-08-15, C11 fix): predicts
    per-seat [final absolute VP / 150, final town count / 3] from the
    state embedding. Share targets are scale-blind -- a 100-VP and a
    140-VP win have the same share -- so nothing in the value loss ever
    asked the torso to model how BIG the engine being built will get.
    C11's diagnosis (cult-hoarding, town deficit, no surplus economy)
    is that blindness. Only training reads this head; ``forward`` and
    every search/checkpoint consumer are untouched."""
    torso: str = "mlp"
    """State torso (``torsos.TORSOS``). ``mlp`` is the original; the others
    are the 2026-10 architecture sweep's candidates."""
    depth: int = 2
    """Residual blocks (``resmlp``) or message-passing / attention layers
    (``hexgnn``/``hexformer``). Ignored by ``mlp``, whose two layers are fixed."""
    hex_dim: int = 64
    """Per-hex token width for ``hexgnn``/``hexformer``."""
    heads: int = 4
    """Attention heads for ``hexformer``."""
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
        if c.torso != "mlp":
            self.torso = build_torso(c, state_dim)
        else:
            self.torso = self._mlp_torso(c, state_dim)

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
        if c.aux_head:
            # per-seat [abs VP/150, towns/3], mover-relative: 4 seats x 2
            self.aux_head = nn.Sequential(
                nn.Linear(c.embed, c.move_hidden), nn.SiLU(), nn.Linear(c.move_hidden, 8)
            )
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / (c.embed**0.5)))

    @staticmethod
    def _mlp_torso(c: ModelConfig, state_dim: int) -> nn.Sequential:
        """The original torso; its layer indices are every checkpoint's keys."""
        return nn.Sequential(
            nn.Linear(state_dim, c.hidden),
            nn.LayerNorm(c.hidden),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.hidden),
            nn.LayerNorm(c.hidden),
            nn.SiLU(),
            nn.Linear(c.hidden, c.embed),
        )

    def state_embedding(
        self, hex_planes: torch.Tensor, globals_: torch.Tensor, faction: torch.Tensor
    ) -> torch.Tensor:
        faction_vec = self.faction_embed(faction)
        if self.cfg.torso != "mlp":
            return self.torso(hex_planes, globals_, faction_vec)
        flat = hex_planes.flatten(start_dim=1)
        return self.torso(torch.cat([flat, globals_, faction_vec], dim=-1))

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

    def forward_with_aux(
        self,
        hex_planes: torch.Tensor,
        globals_: torch.Tensor,
        faction: torch.Tensor,
        candidates: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``forward`` plus the auxiliary economy predictions (training
        only -- search never calls this)."""
        state = self.state_embedding(hex_planes, globals_, faction)
        moves = self.move_embedding(candidates)
        logits = torch.einsum("be,bce->bc", state, moves) * self.logit_scale
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        value = self.value_head(state)
        if self.cfg.value_simplex:
            value = value.softmax(dim=-1)
        aux = self.aux_head(state) if self.cfg.aux_head else None
        return logits, value, aux


def model_config_from_checkpoint(checkpoint: dict) -> ModelConfig:
    """Rebuild the architecture a checkpoint was trained with.

    ``config["model_config"]`` (the full ``ModelConfig`` as a dict) pins
    every size. Checkpoints written before it existed carry only the two
    flags and get the default sizes -- which is what they were trained at.
    The ``value_simplex``/``aux_head`` flags still win when both are
    present, so a hand-edited flag cannot disagree with the dict silently:
    loading simplex-trained weights into an unconstrained head produces
    plausible-looking wrong values rather than an error.
    """
    stored = (checkpoint or {}).get("config") or {}
    arch = stored.get("model_config") or {}
    if not isinstance(arch, dict):
        raise ValueError("checkpoint config['model_config'] must be a dict")
    known = {k: v for k, v in arch.items() if k in ModelConfig.__dataclass_fields__}
    return ModelConfig(
        **{
            **known,
            "value_simplex": bool(stored.get("value_simplex", known.get("value_simplex", False))),
            "aux_head": bool(stored.get("aux_head", known.get("aux_head", False))),
        }
    )
