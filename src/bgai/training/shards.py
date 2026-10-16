"""Self-play shards: the on-disk contract between actors and the learner.

One shard = one compressed ``.npz`` holding a batch of complete
``SelfPlayRecord``s (a handful of games) plus a JSON metadata blob.
Actors write them, the learner reads them, nobody ever modifies or
deletes one -- a run directory is an append-only log of games.

Layout (all arrays share the record axis, n = ``n_records``)::

    hex_planes    (n, 113, 3) uint8   np.packbits of the int8 {0,1} planes
    globals       (n, GLOBAL_DIM) int16
    candidates    (n, max_cand, 12) int16   zero-padded past n_cand
    visits        (n, max_cand) float32     zero-padded past n_cand
    n_cand        (n,) int32                real offer size per record
    faction_id    (n,) int16
    seat          (n,) int8
    final_shares  (n, 4) float32
    aux_target    (n, 8) float32            present only when has_aux
    has_aux       () bool
    meta_json     () str                    ShardMeta as JSON

Sizes measured on leg-5b records: pickle ~3.4 KB/record; stacked
``savez_compressed`` ~150 B/record; with packbits on the hex planes
~69 B/record. Compression costs ~0.14 s per 4k records.

Metadata pins provenance: ``encoding_version`` (refuse to train on a
different state encoding), ``ckpt_iteration`` (which learner checkpoint
played these games -- the learner's staleness metric reads it),
``actor_id``/``seed`` (reproduce a shard), ``config_hash`` (search
config; a run must not silently mix budgets), ``n_games`` and
``created_at`` (throughput accounting).

Writes go through a temp file + ``os.replace`` so a learner listing the
directory never sees a partial shard; anything named ``*.tmp`` is
ignored by ``is_shard_file`` for the same reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import HEX_FEAT_DIM
from bgai.training.selfplay import SelfPlayConfig, SelfPlayRecord
from bgai.training.vocab import ENCODING_VERSION, HEXES

SHARD_FORMAT_VERSION = 1
_SHARD_RE = re.compile(r"^it(\d{4})_a(\d{3})_(\d{6})\.npz$")
_ARRAY_KEYS = (
    "hex_planes", "globals", "candidates", "visits", "n_cand",
    "faction_id", "seat", "final_shares", "has_aux", "meta_json",
)


class ShardError(ValueError):
    """A shard file that cannot be trusted (corrupt, wrong version, wrong
    layout). The learner skips such files with a warning."""


@dataclass(frozen=True)
class ShardMeta:
    encoding_version: int
    ckpt_iteration: int
    actor_id: int
    seed: int
    config_hash: str
    n_games: int
    created_at: float
    n_records: int
    format_version: int = SHARD_FORMAT_VERSION
    game_index_start: int = 0
    game_index_end: int = 0
    """Per-actor game index range [start, end) held by this shard; a
    restarted actor resumes its seed index from the highest end it finds.
    0/0 on converted legacy pickles (no provenance to recover)."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ShardMeta:
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ShardError(f"bad shard metadata: {exc}") from exc
        if not isinstance(raw, dict):
            raise ShardError("bad shard metadata: not a JSON object")
        fields_ = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        try:
            meta = cls(**fields_)
        except TypeError as exc:
            raise ShardError(f"bad shard metadata: {exc}") from exc
        return _validated(meta)


_INT_FIELDS = (
    "encoding_version", "ckpt_iteration", "actor_id", "seed", "n_games", "n_records",
    "format_version", "game_index_start", "game_index_end",
)
_NON_NEGATIVE = tuple(f for f in _INT_FIELDS if f != "seed")


def _validated(meta: ShardMeta) -> ShardMeta:
    """Type/range checks: a malformed metadata blob must surface as a
    ShardError the learner skips, not a TypeError that kills its poll."""
    for name in _INT_FIELDS:
        value = getattr(meta, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShardError(f"bad shard metadata: {name} must be an int, got {value!r}")
    for name in _NON_NEGATIVE:
        if getattr(meta, name) < 0:
            raise ShardError(f"bad shard metadata: {name} must be >= 0")
    if meta.game_index_end < meta.game_index_start:
        raise ShardError("bad shard metadata: game_index_end < game_index_start")
    if not isinstance(meta.config_hash, str):
        raise ShardError("bad shard metadata: config_hash must be a string")
    created = meta.created_at
    if isinstance(created, bool) or not isinstance(created, int | float) or created < 0:
        raise ShardError(f"bad shard metadata: created_at must be a number, got {created!r}")
    return meta


def shard_name(ckpt_iteration: int, actor_id: int, seq: int) -> str:
    return f"it{ckpt_iteration:04d}_a{actor_id:03d}_{seq:06d}.npz"


def is_shard_file(path: Path) -> bool:
    return _SHARD_RE.match(path.name) is not None


def config_hash(cfg: SelfPlayConfig, extra: dict[str, Any] | None = None) -> str:
    """12-hex digest of the search/loss config that produced a shard."""
    blob = json.dumps({**asdict(cfg), **(extra or {})}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------


def _check_record(r: SelfPlayRecord, index: int, has_aux: bool) -> None:
    if r.final_shares is None:
        raise ValueError(f"record {index} has no final_shares (incomplete game?)")
    if (r.aux_target is None) == has_aux:
        raise ValueError(f"record {index} aux_target presence differs from the shard's")
    if r.hex_planes.shape != (len(HEXES), HEX_FEAT_DIM):
        raise ValueError(f"record {index} hex_planes shape {r.hex_planes.shape}")
    if not np.isin(r.hex_planes, (0, 1)).all():
        raise ValueError(f"record {index} hex_planes are not binary; packbits would corrupt")
    if r.candidates.ndim != 2 or r.candidates.shape[1] != MOVE_FIELDS:
        raise ValueError(f"record {index} candidates shape {r.candidates.shape}")
    if r.visits.shape != (r.candidates.shape[0],):
        raise ValueError(f"record {index} visits/candidates length mismatch")


def _stack(records: list[SelfPlayRecord]) -> dict[str, np.ndarray]:
    has_aux = records[0].aux_target is not None
    for i, r in enumerate(records):
        _check_record(r, i, has_aux)
    n = len(records)
    n_cand = np.array([r.candidates.shape[0] for r in records], dtype=np.int32)
    width = int(n_cand.max())
    cand = np.zeros((n, width, MOVE_FIELDS), dtype=np.int16)
    visits = np.zeros((n, width), dtype=np.float32)
    for i, r in enumerate(records):
        cand[i, : n_cand[i]] = r.candidates.astype(np.int16)
        visits[i, : n_cand[i]] = r.visits.astype(np.float32)
    planes = np.stack([r.hex_planes for r in records]).astype(np.uint8)
    arrays = {
        "hex_planes": np.packbits(planes, axis=-1),
        "globals": np.stack([r.globals for r in records]).astype(np.int16),
        "candidates": cand,
        "visits": visits,
        "n_cand": n_cand,
        "faction_id": np.array([r.faction_id for r in records], dtype=np.int16),
        "seat": np.array([r.seat for r in records], dtype=np.int8),
        "final_shares": np.stack([r.final_shares for r in records]).astype(np.float32),
        "has_aux": np.array(has_aux),
    }
    if has_aux:
        arrays["aux_target"] = np.stack([r.aux_target for r in records]).astype(np.float32)
    return arrays


def write_shard(
    path: Path,
    records: list[SelfPlayRecord],
    *,
    ckpt_iteration: int,
    actor_id: int,
    seed: int,
    config_hash: str,
    n_games: int,
    created_at: float | None = None,
    game_index_start: int = 0,
    game_index_end: int = 0,
) -> ShardMeta:
    """Atomically write ``records`` to ``path``; returns the metadata."""
    if not records:
        raise ValueError("refusing to write an empty shard")
    meta = _validated(ShardMeta(
        encoding_version=ENCODING_VERSION,
        ckpt_iteration=int(ckpt_iteration),
        actor_id=int(actor_id),
        seed=int(seed),
        config_hash=str(config_hash),
        n_games=int(n_games),
        created_at=float(created_at if created_at is not None else time.time()),
        n_records=len(records),
        game_index_start=int(game_index_start),
        game_index_end=int(game_index_end),
    ))
    arrays = {**_stack(records), "meta_json": np.array(meta.to_json())}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return meta


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------


def _load_arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            missing = [k for k in keys if k not in z.files]
            if missing:
                raise ShardError(f"{path}: missing arrays {missing}")
            return {k: z[k] for k in keys}
    except ShardError:
        raise
    except Exception as exc:  # zipfile/pickle/OSError zoo
        raise ShardError(f"{path}: unreadable shard: {exc}") from exc


def _meta_from_arrays(path: Path, arrays: dict[str, np.ndarray]) -> ShardMeta:
    meta = ShardMeta.from_json(str(arrays["meta_json"]))
    if meta.encoding_version != ENCODING_VERSION:
        raise ShardError(
            f"{path}: shard encoding v{meta.encoding_version} != code v{ENCODING_VERSION}"
        )
    if meta.format_version != SHARD_FORMAT_VERSION:
        raise ShardError(f"{path}: shard format v{meta.format_version} unsupported")
    return meta


def read_shard_meta(path: Path) -> ShardMeta:
    """Metadata only -- cheap; the learner polls with this."""
    return _meta_from_arrays(path, _load_arrays(path, ("meta_json",)))


def _unpack(arrays: dict[str, np.ndarray], meta: ShardMeta) -> list[SelfPlayRecord]:
    n = meta.n_records
    planes = np.unpackbits(arrays["hex_planes"], axis=-1, count=HEX_FEAT_DIM).astype(np.int8)
    if planes.shape != (n, len(HEXES), HEX_FEAT_DIM):
        raise ShardError(f"hex_planes shape {planes.shape} for {n} records")
    has_aux = bool(arrays["has_aux"])
    aux = arrays.get("aux_target") if has_aux else None
    if has_aux and aux is None:
        raise ShardError("has_aux set but aux_target array missing")
    n_cand = arrays["n_cand"]
    return [
        SelfPlayRecord(
            hex_planes=planes[i],
            globals=arrays["globals"][i],
            candidates=arrays["candidates"][i, : n_cand[i]].copy(),
            visits=arrays["visits"][i, : n_cand[i]].copy(),
            faction_id=int(arrays["faction_id"][i]),
            seat=int(arrays["seat"][i]),
            final_shares=arrays["final_shares"][i],
            aux_target=aux[i] if aux is not None else None,
        )
        for i in range(n)
    ]


def read_shard(path: Path) -> tuple[ShardMeta, list[SelfPlayRecord]]:
    """Exact inverse of ``write_shard`` (every field round-trips bit-for-bit)."""
    arrays = _load_arrays(path, _ARRAY_KEYS)
    meta = _meta_from_arrays(path, arrays)
    if bool(arrays["has_aux"]):
        arrays = {**arrays, **_load_arrays(path, ("aux_target",))}
    try:
        return meta, _unpack(arrays, meta)
    except (IndexError, ValueError) as exc:
        raise ShardError(f"{path}: inconsistent arrays: {exc}") from exc


# --------------------------------------------------------------------------
# legacy pickles
# --------------------------------------------------------------------------


def convert_records_pt(
    src: Path,
    dst: Path,
    *,
    ckpt_iteration: int,
    actor_id: int = 0,
    seed: int = 0,
    config_hash: str = "legacy",
    n_games: int = 0,
) -> ShardMeta:
    """Rewrite a serial-loop ``records_NN.pt`` pickle as a shard.

    The pickle carries no provenance, so ``ckpt_iteration`` (NN - 1 for
    the serial loop: iteration NN's games were played by ``iter_{NN-1}``)
    and the game count must be supplied by the caller."""
    import torch

    records = torch.load(Path(src), map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not all(isinstance(r, SelfPlayRecord) for r in records):
        raise ValueError(f"{src} is not a list of SelfPlayRecord")
    return write_shard(
        dst, records, ckpt_iteration=ckpt_iteration, actor_id=actor_id, seed=seed,
        config_hash=config_hash, n_games=n_games,
    )
