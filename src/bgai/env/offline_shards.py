"""Shard I/O for the offline-RL export.

An offline shard is an uncompressed-ragged ``.npz``: fixed-width arrays for
the observation and the transition fields, plus a flat candidate array with
per-row counts (the same ragged trick ``training/dataset_build.py`` uses).

``next_obs`` is an *index* into the same shard, ``next_index``, with -1 for
terminal -- an observation is ~3.3 kB, so storing a second copy would
double a multi-gigabyte export for no information. Per-seat trajectories
never cross a shard boundary because shards are cut on whole games, so the
index is always local. :func:`iter_transitions` materializes the real
``(obs, action, mask, reward, next_obs, done)`` records.

The mask is not stored: it is implied by ``cand_counts``, since a
canonically ordered offer always occupies rows ``0..count-1``. Writing it
out would be a second, divergeable copy of the same fact.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from bgai.env.offline_records import OfflineTransition
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.vocab import FACTION_INDEX

__all__ = ["iter_transitions", "load_shard", "pack_shard"]


def pack_shard(
    transitions: Sequence[OfflineTransition], games: Sequence[str]
) -> dict[str, np.ndarray]:
    """Flatten a shard. ``next_index`` points inside this same shard; -1 is
    terminal. Per-seat trajectories never cross a shard boundary because
    shards are cut on whole games.
    """
    game_index = {g: i for i, g in enumerate(games)}
    row_of = {(t.game_id, t.position): i for i, t in enumerate(transitions)}
    cand_counts = np.array([t.obs.n_legal for t in transitions], dtype=np.int32)
    next_index = np.array(
        [
            -1
            if t.next_position < 0
            else row_of.get((t.game_id, t.next_position), -1)
            for t in transitions
        ],
        dtype=np.int32,
    )
    return {
        "hex_planes": np.stack([t.obs.hex_planes for t in transitions]),
        "globals": np.stack([t.obs.globals for t in transitions]),
        "cand_flat": np.concatenate(
            [t.obs.candidates[: t.obs.n_legal] for t in transitions]
        ),
        "cand_counts": cand_counts,
        "action": np.array([t.action for t in transitions], dtype=np.int32),
        "reward": np.array([t.reward for t in transitions], dtype=np.float32),
        "done": np.array([t.done for t in transitions], dtype=bool),
        "next_index": next_index,
        "seat": np.array([t.seat for t in transitions], dtype=np.int8),
        "faction": np.array(
            [FACTION_INDEX[t.faction] for t in transitions], dtype=np.int8
        ),
        "game": np.array(
            [game_index[t.game_id] for t in transitions], dtype=np.int32
        ),
    }


def load_shard(path: Path) -> dict[str, np.ndarray]:
    """Read one shard's arrays. ``cand_flat`` is ragged; use
    :func:`iter_transitions` for assembled records.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def iter_transitions(
    path: Path, max_candidates: int
) -> Iterator[dict[str, np.ndarray | int | float | bool]]:
    """Yield materialized ``(obs, action, mask, reward, next_obs, done)``
    dicts from a shard, re-padding candidates to ``max_candidates``.
    """
    shard = load_shard(path)
    offsets = np.concatenate([[0], np.cumsum(shard["cand_counts"])]).astype(np.int64)

    def pad(row: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(offsets[row]), int(offsets[row + 1])
        count = end - start
        if count > max_candidates:
            raise ValueError(
                f"row {row} has {count} candidates, more than max_candidates="
                f"{max_candidates}"
            )
        candidates = np.zeros((max_candidates, MOVE_FIELDS), dtype=np.int16)
        candidates[:count] = shard["cand_flat"][start:end]
        mask = np.zeros(max_candidates, dtype=np.int8)
        mask[:count] = 1
        return candidates, mask

    for row in range(len(shard["action"])):
        candidates, mask = pad(row)
        nxt = int(shard["next_index"][row])
        next_obs = None
        if nxt >= 0:
            n_candidates, n_mask = pad(nxt)
            next_obs = {
                "hex_planes": shard["hex_planes"][nxt],
                "globals": shard["globals"][nxt],
                "candidates": n_candidates,
                "action_mask": n_mask,
            }
        yield {
            "obs": {
                "hex_planes": shard["hex_planes"][row],
                "globals": shard["globals"][row],
                "candidates": candidates,
                "action_mask": mask,
            },
            "action": int(shard["action"][row]),
            "mask": mask,
            "reward": float(shard["reward"][row]),
            "next_obs": next_obs,
            "done": bool(shard["done"][row]),
        }


