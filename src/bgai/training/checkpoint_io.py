"""Atomic checkpoint files for the actor/learner split.

Contract
--------
The learner and every actor share ONE file, ``<run>/current.pt``, and
the only synchronisation between them is that file's mtime. That works
only if a reader can never observe a half-written checkpoint, so every
save goes to ``<path>.tmp`` in the same directory and is then
``os.replace``'d over the target -- an atomic rename on every POSIX
filesystem (and NTFS). ``torch.save`` straight onto ``current.pt`` (the
serial loop's old habit) truncates the file first; an actor reloading at
that instant would read garbage or a short file.

Payload layout (unchanged from the serial loop, so ``MCTSAgent`` and the
arena load these exactly as before)::

    {"model": state_dict, "encoding_version": int, "iteration": int,
     "config": {"value_simplex": bool, "aux_head": bool, ...}}

``load_if_newer`` is the actor's poll: it compares the file's current
``st_mtime_ns`` against the stamp from the last load and only pays for
``torch.load`` when it changed. A checkpoint replaced with an identical
mtime (sub-tick rewrite on a coarse filesystem) would be missed; the
learner writes at most once per training step, seconds apart, so this
never occurs in practice, and the next write catches it anyway.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from bgai.training.vocab import ENCODING_VERSION


def validate_checkpoint(payload: Any) -> dict:
    """Raise ``ValueError`` unless ``payload`` looks like one of ours."""
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload is {type(payload).__name__}, not a dict")
    version = payload.get("encoding_version")
    if version != ENCODING_VERSION:
        raise ValueError(f"checkpoint encoding v{version} != code v{ENCODING_VERSION}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint has no 'model' state dict")
    if not isinstance(payload.get("iteration", 0), int):
        raise ValueError("checkpoint 'iteration' must be an int")
    return payload


def save_atomic(payload: dict, path: Path) -> None:
    """Write ``payload`` so that ``path`` is always either the previous
    complete file or the new complete file, never anything in between."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, device: str = "cpu") -> dict:
    """``torch.load`` + ``validate_checkpoint``; every failure is a
    ``ValueError`` naming the file, so callers can catch one type."""
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    except Exception as exc:  # unpickling raises a zoo of types
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    try:
        return validate_checkpoint(payload)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def checkpoint_mtime_ns(path: Path) -> int | None:
    """``st_mtime_ns`` of ``path``, or None when it does not exist."""
    try:
        return os.stat(path).st_mtime_ns
    except FileNotFoundError:
        return None


def load_if_newer(
    path: Path, last_mtime_ns: int | None, device: str = "cpu"
) -> tuple[dict | None, int | None]:
    """``(payload, mtime_ns)`` when ``path`` changed since ``last_mtime_ns``,
    else ``(None, last_mtime_ns)``. A missing file is "unchanged"."""
    stamp = checkpoint_mtime_ns(path)
    if stamp is None or stamp == last_mtime_ns:
        return None, last_mtime_ns
    return load_checkpoint(path, device), stamp
