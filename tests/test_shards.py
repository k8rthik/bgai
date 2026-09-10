"""Self-play shard format: compressed, padded, packbits'd, atomic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from selfplay_helpers import records_equal, synthetic_records

from bgai.training import shards as sh
from bgai.training.selfplay import SelfPlayConfig, SelfPlayRecord
from bgai.training.vocab import ENCODING_VERSION


def _write(tmp_path: Path, records, **kw) -> tuple[Path, sh.ShardMeta]:
    path = tmp_path / "shards" / sh.shard_name(3, 1, 0)
    meta = sh.write_shard(
        path, records,
        **{"ckpt_iteration": 3, "actor_id": 1, "seed": 7, "config_hash": "abc", "n_games": 2, **kw},
    )
    return path, meta


def test_round_trip_is_exact_including_ragged_padding(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    records = synthetic_records(40, rng)
    assert len({r.candidates.shape[0] for r in records}) > 3, "test wants ragged rows"
    path, meta = _write(tmp_path, records)
    assert path.exists() and not path.with_suffix(".tmp").exists()
    meta_back, back = sh.read_shard(path)
    assert meta_back == meta
    assert len(back) == len(records)
    for a, b in zip(records, back, strict=True):
        assert records_equal(a, b)


def test_round_trip_without_aux_targets(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    records = synthetic_records(5, rng, with_aux=False)
    path, _ = _write(tmp_path, records)
    _, back = sh.read_shard(path)
    assert all(r.aux_target is None for r in back)
    assert all(records_equal(a, b) for a, b in zip(records, back, strict=True))


def test_metadata_fields(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    path, meta = _write(tmp_path, synthetic_records(3, rng))
    assert meta.encoding_version == ENCODING_VERSION
    assert meta.ckpt_iteration == 3 and meta.actor_id == 1 and meta.seed == 7
    assert meta.config_hash == "abc" and meta.n_games == 2 and meta.n_records == 3
    assert meta.created_at > 0
    assert sh.read_shard_meta(path) == meta


def test_hex_planes_are_bit_packed(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    path, _ = _write(tmp_path, synthetic_records(20, rng))
    with np.load(path) as z:
        packed = z["hex_planes"]
    assert packed.dtype == np.uint8
    assert packed.shape == (20, 113, 3)  # ceil(17 / 8) bytes per hex row
    assert path.stat().st_size < 20 * 1500  # far below the ~3.4 KB/record pickle


def test_non_binary_hex_planes_are_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    r = synthetic_records(1, rng)[0]
    bad = SelfPlayRecord(
        hex_planes=r.hex_planes.copy(), globals=r.globals, candidates=r.candidates,
        visits=r.visits, faction_id=r.faction_id, seat=r.seat,
        final_shares=r.final_shares, aux_target=r.aux_target,
    )
    bad.hex_planes[0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        _write(tmp_path, [bad])


def test_incomplete_records_are_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    r = synthetic_records(1, rng)[0]
    incomplete = SelfPlayRecord(
        hex_planes=r.hex_planes, globals=r.globals, candidates=r.candidates,
        visits=r.visits, faction_id=r.faction_id, seat=r.seat,
    )
    with pytest.raises(ValueError, match="final_shares"):
        _write(tmp_path, [incomplete])
    with pytest.raises(ValueError, match="empty"):
        _write(tmp_path, [])


def test_write_is_atomic_no_tmp_left_and_replaces(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    path, _ = _write(tmp_path, synthetic_records(2, rng))
    first = path.read_bytes()
    _write(tmp_path, synthetic_records(4, rng))
    assert path.read_bytes() != first
    assert not any(p.suffix == ".tmp" for p in path.parent.iterdir())


def test_corrupt_shard_raises_shard_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a zip file")
    with pytest.raises(sh.ShardError):
        sh.read_shard(path)
    with pytest.raises(sh.ShardError):
        sh.read_shard_meta(path)


def test_encoding_version_mismatch_is_a_shard_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    path, _ = _write(tmp_path, synthetic_records(2, rng))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = sh.ShardMeta.from_json(str(arrays["meta_json"]))
    stale = sh.ShardMeta(**{**meta.__dict__, "encoding_version": 999})
    arrays["meta_json"] = np.array(stale.to_json())
    np.savez_compressed(path, **arrays)
    with pytest.raises(sh.ShardError, match="encoding"):
        sh.read_shard(path)


def test_convert_legacy_records_pickle(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    records = synthetic_records(12, rng)
    src = tmp_path / "records_03.pt"
    torch.save(records, src)
    dst = tmp_path / "shards" / "legacy.npz"
    meta = sh.convert_records_pt(src, dst, ckpt_iteration=2)
    assert meta.n_records == 12 and meta.ckpt_iteration == 2
    _, back = sh.read_shard(dst)
    assert all(records_equal(a, b) for a, b in zip(records, back, strict=True))


def test_config_hash_is_stable_and_sensitive() -> None:
    a = sh.config_hash(SelfPlayConfig(simulations=16))
    b = sh.config_hash(SelfPlayConfig(simulations=16))
    c = sh.config_hash(SelfPlayConfig(simulations=32))
    assert a == b and a != c and len(a) == 12


def test_shard_name_layout() -> None:
    assert sh.shard_name(5, 2, 17) == "it0005_a002_000017.npz"
    assert sh.is_shard_file(Path("x/it0005_a002_000017.npz"))
    assert not sh.is_shard_file(Path("x/it0005_a002_000017.tmp"))
    assert not sh.is_shard_file(Path("x/.it0005_a002_000017.npz.tmp"))
