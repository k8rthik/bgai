"""Observation <-> dict <-> bytes.

An observation is four numpy arrays, so the bytes form is an uncompressed
``.npz`` payload rather than JSON: the arrays are a few kilobytes of small
ints and base64-in-JSON would triple that for nothing. The dict form keeps
the arrays as arrays (so it is cheap and lossless for in-process use) and
``obs_to_jsonable``/``obs_from_jsonable`` give the plain-list form for
anything that has to cross a JSON boundary.

Round-trip contract, pinned in tests/test_env_serialize.py: dtypes and
shapes are preserved exactly, so a decoded observation still satisfies
``observation_space.contains``.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np

from bgai.env.config import EnvError
from bgai.env.observation import (
    GLOBAL_DIM,
    HEX_FEAT_DIM,
    HEXES,
    OBS_VERSION,
    Observation,
)
from bgai.training.encode_move import MOVE_FIELDS

__all__ = [
    "ObservationCodecError",
    "obs_from_bytes",
    "obs_from_dict",
    "obs_from_jsonable",
    "obs_to_bytes",
    "obs_to_dict",
    "obs_to_jsonable",
]

_VERSION_KEY = "obs_version"
_DTYPES: dict[str, Any] = {
    "hex_planes": np.int8,
    "globals": np.int16,
    "candidates": np.int16,
    "action_mask": np.int8,
}


class ObservationCodecError(EnvError, ValueError):
    """A payload that is not a valid encoded observation."""


def obs_to_dict(obs: Observation) -> dict[str, np.ndarray]:
    """The PettingZoo/Gymnasium observation dict for one agent.

    This is also the exact object the envs hand back from ``observe`` /
    ``reset`` / ``step``, so there is one definition of the shape.
    """
    return {
        "observation": {
            "hex_planes": obs.hex_planes,
            "globals": obs.globals,
            "candidates": obs.candidates,
        },
        "action_mask": obs.action_mask,
    }


def obs_from_dict(raw: Mapping[str, Any]) -> Observation:
    """Inverse of :func:`obs_to_dict`; validates shapes and dtypes."""
    if "observation" not in raw or "action_mask" not in raw:
        raise ObservationCodecError(
            f"observation dict needs 'observation' and 'action_mask' keys, "
            f"got {sorted(raw)}"
        )
    inner = raw["observation"]
    missing = {"hex_planes", "globals", "candidates"} - set(inner)
    if missing:
        raise ObservationCodecError(f"observation is missing {sorted(missing)}")
    arrays = {
        "hex_planes": inner["hex_planes"],
        "globals": inner["globals"],
        "candidates": inner["candidates"],
        "action_mask": raw["action_mask"],
    }
    return _build(arrays)


def _build(arrays: Mapping[str, Any]) -> Observation:
    coerced: dict[str, np.ndarray] = {}
    for key, dtype in _DTYPES.items():
        value = np.asarray(arrays[key], dtype=dtype)
        coerced[key] = value
    _check_shape(coerced["hex_planes"], (len(HEXES), HEX_FEAT_DIM), "hex_planes")
    _check_shape(coerced["globals"], (GLOBAL_DIM,), "globals")
    if coerced["candidates"].ndim != 2 or coerced["candidates"].shape[1] != MOVE_FIELDS:
        raise ObservationCodecError(
            f"candidates must be (n, {MOVE_FIELDS}), got "
            f"{coerced['candidates'].shape}"
        )
    if coerced["action_mask"].shape != (coerced["candidates"].shape[0],):
        raise ObservationCodecError(
            f"action_mask shape {coerced['action_mask'].shape} does not match "
            f"candidates rows {coerced['candidates'].shape[0]}"
        )
    return Observation(
        hex_planes=coerced["hex_planes"],
        globals=coerced["globals"],
        candidates=coerced["candidates"],
        action_mask=coerced["action_mask"],
    )


def _check_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ObservationCodecError(
            f"{name} must have shape {shape}, got {array.shape}"
        )


def obs_to_bytes(obs: Observation) -> bytes:
    """A self-describing ``.npz`` payload, version-stamped."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **{_VERSION_KEY: np.array(OBS_VERSION, dtype=np.int32)},
        hex_planes=obs.hex_planes,
        globals=obs.globals,
        candidates=obs.candidates,
        action_mask=obs.action_mask,
    )
    return buffer.getvalue()


def obs_from_bytes(payload: bytes) -> Observation:
    """Inverse of :func:`obs_to_bytes`. Refuses a foreign layout version."""
    try:
        with np.load(io.BytesIO(payload)) as data:
            files = set(data.files)
            if _VERSION_KEY not in files:
                raise ObservationCodecError("payload has no obs_version stamp")
            version = int(data[_VERSION_KEY])
            if version != OBS_VERSION:
                raise ObservationCodecError(
                    f"payload is observation layout v{version}, this code is "
                    f"v{OBS_VERSION}"
                )
            arrays = {key: data[key] for key in _DTYPES if key in files}
    except (ValueError, OSError) as exc:
        raise ObservationCodecError(f"not a valid observation payload: {exc}") from exc
    missing = set(_DTYPES) - set(arrays)
    if missing:
        raise ObservationCodecError(f"payload is missing {sorted(missing)}")
    return _build(arrays)


def obs_to_jsonable(obs: Observation) -> dict[str, Any]:
    """Plain nested lists plus a version stamp, for a JSON boundary."""
    return {
        _VERSION_KEY: OBS_VERSION,
        "hex_planes": obs.hex_planes.tolist(),
        "globals": obs.globals.tolist(),
        "candidates": obs.candidates.tolist(),
        "action_mask": obs.action_mask.tolist(),
    }


def obs_from_jsonable(raw: Mapping[str, Any]) -> Observation:
    """Inverse of :func:`obs_to_jsonable`."""
    version = raw.get(_VERSION_KEY)
    if version != OBS_VERSION:
        raise ObservationCodecError(
            f"payload is observation layout v{version}, this code is v{OBS_VERSION}"
        )
    missing = set(_DTYPES) - set(raw)
    if missing:
        raise ObservationCodecError(f"payload is missing {sorted(missing)}")
    return _build({key: raw[key] for key in _DTYPES})
