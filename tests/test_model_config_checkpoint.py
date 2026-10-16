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

from bgai.training.model import ModelConfig, PolicyValueNet, model_config_from_checkpoint

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
    assert cfg == ModelConfig(value_simplex=True, aux_head=True)


def test_missing_config_entirely_gets_defaults() -> None:
    assert model_config_from_checkpoint({}) == ModelConfig()
    assert model_config_from_checkpoint({"config": None}) == ModelConfig()


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
