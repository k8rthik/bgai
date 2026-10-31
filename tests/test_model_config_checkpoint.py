"""A checkpoint must rebuild the exact architecture it was trained with.

Before this, ``model_config_from_checkpoint`` restored only the
``value_simplex``/``aux_head`` flags and every net was rebuilt at the
default sizes, so a checkpoint of any other size could not be loaded --
which blocks evaluating an architecture sweep at all.
``config["model_config"]`` (the full ``ModelConfig`` as a dict, the
actor/learner branch's layout) now pins the architecture.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from bgai.training.encode_state import (
    GLOBAL_DIM,
    GLOBAL_DIM_V1,
    GLOBAL_DIM_V2,
    GLOBAL_DIM_V3,
    HEX_FEAT_DIM,
)
from bgai.training.model import ModelConfig, PolicyValueNet, model_config_from_checkpoint
from bgai.training.vocab import HEXES

SMALL = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0)


def _ckpt(cfg: ModelConfig, **extra) -> dict:
    return {"model": PolicyValueNet(cfg).state_dict(),
            "config": {"model_config": asdict(cfg), **extra}}


def test_non_default_sizes_round_trip_and_load() -> None:
    ckpt = _ckpt(SMALL)
    cfg = model_config_from_checkpoint(ckpt)
    assert (cfg.hidden, cfg.embed, cfg.move_hidden) == (64, 32, 32)
    PolicyValueNet(cfg).load_state_dict(ckpt["model"])  # strict: shapes must match


def test_legacy_checkpoint_without_model_config_gets_defaults() -> None:
    cfg = model_config_from_checkpoint({"config": {"value_simplex": True, "aux_head": True}})
    assert cfg == ModelConfig(value_simplex=True, aux_head=True, global_dim=GLOBAL_DIM_V1)


def test_missing_config_entirely_gets_v1_defaults() -> None:
    v1 = ModelConfig(global_dim=GLOBAL_DIM_V1)
    assert model_config_from_checkpoint({}) == v1
    assert model_config_from_checkpoint({"config": None}) == v1


def _forward(net: PolicyValueNet, globals_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = 3, 5
    g = torch.Generator().manual_seed(0)
    return net.eval()(
        torch.rand(n, len(HEXES), HEX_FEAT_DIM, generator=g),
        torch.rand(n, globals_width, generator=g),
        torch.zeros(n, dtype=torch.long),
        torch.randint(0, 8, (n, k, 12), generator=g),
        torch.ones(n, k, dtype=torch.bool),
    )


def _reads_only_its_prefix(net: PolicyValueNet, width: int) -> None:
    torch.manual_seed(0)
    planes = torch.rand(3, len(HEXES), HEX_FEAT_DIM)
    g = torch.rand(3, GLOBAL_DIM)
    perturbed = g.clone()
    perturbed[:, width:] += 1234.0
    rest = (torch.zeros(3, dtype=torch.long), torch.zeros(3, 5, 12, dtype=torch.long),
            torch.ones(3, 5, dtype=torch.bool))
    net.eval()
    a = net(planes, g, *rest)
    b = net(planes, perturbed, *rest)
    assert a[0].shape == (3, 5) and a[1].shape == (3, 4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_pre_v2_checkpoint_restores_v1_width_and_reads_the_current_prefix() -> None:
    v1 = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0, global_dim=GLOBAL_DIM_V1)
    stored = {k: v for k, v in asdict(v1).items() if k != "global_dim"}
    ckpt = {"model": PolicyValueNet(v1).state_dict(), "config": {"model_config": stored}}
    net = PolicyValueNet(model_config_from_checkpoint(ckpt))
    net.load_state_dict(ckpt["model"])
    assert net.cfg.global_dim == GLOBAL_DIM_V1
    _reads_only_its_prefix(net, GLOBAL_DIM_V1)


@pytest.mark.parametrize(("width", "want"), [(GLOBAL_DIM_V2, 382), (GLOBAL_DIM_V3, 496)])
def test_v2_and_v3_checkpoints_keep_their_width_and_read_the_current_prefix(
    width: int, want: int
) -> None:
    assert width == want < GLOBAL_DIM
    cfg = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0, global_dim=width)
    ckpt = {"model": PolicyValueNet(cfg).state_dict(), "config": {"model_config": asdict(cfg)}}
    net = PolicyValueNet(model_config_from_checkpoint(ckpt))
    net.load_state_dict(ckpt["model"])
    assert net.cfg.global_dim == width
    _reads_only_its_prefix(net, width)


def test_fresh_config_reads_the_full_globals_vector() -> None:
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32))
    first = net.torso[0]
    assert first.in_features == len(HEXES) * HEX_FEAT_DIM + GLOBAL_DIM + net.cfg.faction_embed
    assert model_config_from_checkpoint(
        {"config": {"model_config": asdict(net.cfg)}}
    ).global_dim == GLOBAL_DIM


@pytest.mark.parametrize("torso", ["resmlp", "hexgnn", "hexformer"])
def test_every_torso_honours_global_dim(torso: str) -> None:
    cfg = ModelConfig(hidden=32, embed=16, move_hidden=16, hex_dim=16, depth=1, torso=torso,
                      dropout=0.0, global_dim=GLOBAL_DIM_V1)
    logits, value = _forward(PolicyValueNet(cfg), GLOBAL_DIM)
    assert logits.shape == (3, 5) and torch.isfinite(value).all()


def test_top_level_flags_win_over_the_stored_dict() -> None:
    ckpt = _ckpt(ModelConfig(value_simplex=False, aux_head=False),
                 value_simplex=True, aux_head=True)
    cfg = model_config_from_checkpoint(ckpt)
    assert cfg.value_simplex and cfg.aux_head


def test_unknown_keys_in_the_stored_dict_are_ignored() -> None:
    ckpt = {"config": {"model_config": {**asdict(SMALL), "from_the_future": 1}}}
    assert model_config_from_checkpoint(ckpt).hidden == 64


def test_a_non_dict_model_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_config"):
        model_config_from_checkpoint({"config": {"model_config": "big"}})


def test_selfplay_train_checkpoints_store_the_full_architecture() -> None:
    # selfplay_train (serial and pipelined) writes through checkpoint_payload
    from bgai.training.selfplay_finetune import checkpoint_payload

    saved = checkpoint_payload(PolicyValueNet(SMALL), SMALL, 1, {"a": 1})["config"]
    assert saved["model_config"] == asdict(SMALL) and saved["a"] == 1
    assert saved["value_simplex"] == SMALL.value_simplex
    assert model_config_from_checkpoint({"config": saved}) == SMALL


def test_inference_server_style_checkpoint_loads_a_small_net(tmp_path) -> None:
    path = tmp_path / "small.pt"
    torch.save(_ckpt(SMALL), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(model_config_from_checkpoint(loaded))
    net.load_state_dict(loaded["model"])
