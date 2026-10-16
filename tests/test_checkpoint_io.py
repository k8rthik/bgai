"""Atomic checkpoint writes + change detection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from selfplay_helpers import TINY, checkpoint_payload, tiny_net

from bgai.training import checkpoint_io as cio
from bgai.training.model import model_config_from_checkpoint


def test_save_atomic_leaves_no_tmp_and_loads(tmp_path: Path) -> None:
    net = tiny_net()
    path = tmp_path / "current.pt"
    cio.save_atomic(checkpoint_payload(net, 4), path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    payload = cio.load_checkpoint(path)
    assert payload["iteration"] == 4
    assert model_config_from_checkpoint(payload) == TINY


def test_save_atomic_replaces_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "current.pt"
    cio.save_atomic(checkpoint_payload(tiny_net(0), 1), path)
    cio.save_atomic(checkpoint_payload(tiny_net(1), 2), path)
    assert cio.load_checkpoint(path)["iteration"] == 2


def test_save_atomic_failure_does_not_clobber(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "current.pt"
    cio.save_atomic(checkpoint_payload(tiny_net(0), 1), path)

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        cio.save_atomic(checkpoint_payload(tiny_net(1), 2), path)
    assert cio.load_checkpoint(path)["iteration"] == 1
    assert not path.with_suffix(".tmp").exists()


def test_load_checkpoint_validates(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model": {}, "encoding_version": 999}, path)
    with pytest.raises(ValueError, match="encoding"):
        cio.load_checkpoint(path)
    torch.save({"encoding_version": 1}, path)
    with pytest.raises(ValueError, match="model"):
        cio.load_checkpoint(path)
    (tmp_path / "garbage.pt").write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        cio.load_checkpoint(tmp_path / "garbage.pt")


def test_load_if_newer_only_fires_on_mtime_change(tmp_path: Path) -> None:
    path = tmp_path / "current.pt"
    cio.save_atomic(checkpoint_payload(tiny_net(0), 1), path)
    payload, stamp = cio.load_if_newer(path, None)
    assert payload is not None and payload["iteration"] == 1
    again, stamp2 = cio.load_if_newer(path, stamp)
    assert again is None and stamp2 == stamp
    # force a distinct mtime even on coarse filesystems
    os.utime(path, ns=(stamp + 1_000_000, stamp + 1_000_000))
    cio.save_atomic(checkpoint_payload(tiny_net(1), 2), path)
    os.utime(path, ns=(stamp + 2_000_000, stamp + 2_000_000))
    newer, stamp3 = cio.load_if_newer(path, stamp)
    assert newer is not None and newer["iteration"] == 2 and stamp3 != stamp


def test_load_if_newer_missing_file(tmp_path: Path) -> None:
    payload, stamp = cio.load_if_newer(tmp_path / "nope.pt", None)
    assert payload is None and stamp is None
