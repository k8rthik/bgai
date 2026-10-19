"""Architecture variants for the sweep: every torso maps the same inputs
to the same state embedding, and the original MLP stays byte-compatible
with every checkpoint already on disk."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.model import ModelConfig, PolicyValueNet, model_config_from_checkpoint
from bgai.training.torsos import TORSOS, hex_adjacency
from bgai.training.vocab import FACTION_NAMES, HEXES

SMALL = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0, value_simplex=True)
VARIANTS = {
    "mlp": SMALL,
    "resmlp": replace(SMALL, torso="resmlp", depth=3),
    "hexgnn": replace(SMALL, torso="hexgnn", depth=2, hex_dim=16),
    "hexformer": replace(SMALL, torso="hexformer", depth=2, hex_dim=16, heads=2),
}


def _batch(b: int = 3, c: int = 5) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    hex_planes = (torch.rand(b, len(HEXES), HEX_FEAT_DIM, generator=g) > 0.8).float()
    globals_ = torch.rand(b, GLOBAL_DIM, generator=g)
    faction = torch.randint(0, len(FACTION_NAMES), (b,), generator=g)
    candidates = torch.randint(0, 8, (b, c, MOVE_FIELDS), generator=g)
    mask = torch.ones(b, c, dtype=torch.bool)
    mask[0, -1] = False
    return hex_planes, globals_, faction, candidates, mask


def test_registry_lists_every_variant() -> None:
    assert set(VARIANTS) == set(TORSOS)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_torso_produces_policy_and_simplex_value(name: str) -> None:
    net = PolicyValueNet(VARIANTS[name]).eval()
    logits, value = net(*_batch())
    assert logits.shape == (3, 5) and torch.isinf(logits[0, -1])
    assert value.shape == (3, 4)
    torch.testing.assert_close(value.sum(-1), torch.ones(3))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_torso_trains(name: str) -> None:
    net = PolicyValueNet(VARIANTS[name]).train()
    logits, value = net(*_batch())
    (logits[:, 0].sum() + value.sum()).backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_checkpoint_round_trip_rebuilds_the_variant(name: str) -> None:
    net = PolicyValueNet(VARIANTS[name])
    ckpt = {"model": net.state_dict(), "config": {"model_config": asdict(net.cfg)}}
    cfg = model_config_from_checkpoint(ckpt)
    assert cfg == VARIANTS[name]
    PolicyValueNet(cfg).load_state_dict(ckpt["model"])  # strict


def test_default_mlp_keeps_original_parameter_names() -> None:
    # every checkpoint on disk was written with these keys
    keys = set(PolicyValueNet(SMALL).state_dict())
    assert {"torso.0.weight", "torso.4.weight", "torso.7.weight"} <= keys


def test_existing_champion_base_loads_strictly() -> None:
    path = Path("data/checkpoints/snap/simplex_best.pt")
    if not path.exists():
        pytest.skip("champion base checkpoint not present")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    PolicyValueNet(model_config_from_checkpoint(ckpt)).load_state_dict(ckpt["model"])


def test_hex_adjacency_is_symmetric_and_board_shaped() -> None:
    edges = hex_adjacency()
    assert edges.shape[0] == 2 and edges.shape[1] > len(HEXES)
    pairs = set(map(tuple, edges.t().tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    assert all(a != b for a, b in pairs)


def test_unknown_torso_is_rejected() -> None:
    with pytest.raises(ValueError, match="torso"):
        PolicyValueNet(replace(SMALL, torso="nope"))
