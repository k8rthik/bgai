"""Actor process: reloads weights on checkpoint change, writes shards."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from selfplay_helpers import checkpoint_payload, tiny_net, write_tiny_checkpoint

from bgai.agents.mcts import MCTSAgent
from bgai.training import selfplay_actor as sa
from bgai.training import shards as sh
from bgai.training.checkpoint_io import checkpoint_mtime_ns, save_atomic
from bgai.training.selfplay import SelfPlayConfig
from bgai.training.worker_env import SINGLE_THREAD_ENV, pin_single_thread_env

FAST = SelfPlayConfig(simulations=2, top_k=3, max_depth=4, leaf_batch=2, fast_prior=0.9)


def _bump_mtime(path: Path) -> None:
    """Guarantee a distinct st_mtime_ns even on coarse filesystems."""
    stamp = os.stat(path).st_mtime_ns + 2_000_000
    os.utime(path, ns=(stamp, stamp))


# -- MCTSAgent.reload ------------------------------------------------------


def test_agent_reload_swaps_weights_in_place_and_resets_tree(tmp_path: Path) -> None:
    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0, seed=0)
    agent = MCTSAgent(checkpoint_path=ckpt, simulations=2)
    before = {k: v.clone() for k, v in agent.net.state_dict().items()}
    net_id = id(agent.net)
    save_atomic(checkpoint_payload(tiny_net(1), 3), ckpt)
    agent._reuse_root = object()  # type: ignore[assignment]
    assert agent.reload(ckpt) == 3
    assert id(agent.net) == net_id, "reload must be in place (workers hold the reference)"
    assert agent._reuse_root is None
    changed = any(
        not torch.equal(before[k], v) for k, v in agent.net.state_dict().items()
    )
    assert changed
    assert not agent.net.training


def test_agent_reload_rejects_architecture_mismatch(tmp_path: Path) -> None:
    from bgai.training.model import ModelConfig

    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0)
    agent = MCTSAgent(checkpoint_path=ckpt, simulations=2)
    before = {k: v.clone() for k, v in agent.net.state_dict().items()}
    other = tiny_net(0, ModelConfig(hidden=32, embed=16, move_hidden=16, aux_head=True))
    save_atomic(checkpoint_payload(other, 1), ckpt)
    with pytest.raises(ValueError, match="architecture"):
        agent.reload(ckpt)
    assert all(torch.equal(before[k], v) for k, v in agent.net.state_dict().items())


def test_agent_reload_rejects_bad_encoding_version(tmp_path: Path) -> None:
    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0)
    agent = MCTSAgent(checkpoint_path=ckpt, simulations=2)
    payload = {**checkpoint_payload(tiny_net(1), 1), "encoding_version": 999}
    save_atomic(payload, ckpt)
    with pytest.raises(ValueError, match="encoding"):
        agent.reload(ckpt)


# -- pure helpers ----------------------------------------------------------


def test_game_seed_is_deterministic_and_distinct() -> None:
    assert sa.game_seed(0, 1, 5) == sa.game_seed(0, 1, 5)
    seeds = {sa.game_seed(b, a, g) for b in range(2) for a in range(3) for g in range(50)}
    assert len(seeds) == 2 * 3 * 50


def test_pin_single_thread_env_uses_setdefault(monkeypatch) -> None:
    for key in SINGLE_THREAD_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    pin_single_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "3"  # never overrides an explicit choice
    for key in SINGLE_THREAD_ENV:
        if key != "OMP_NUM_THREADS":
            assert os.environ[key] == "1"


def _result(ckpt_iteration: int, n_records: int = 1) -> sa.WorkerResult:
    rng = np.random.default_rng(ckpt_iteration)
    from selfplay_helpers import synthetic_records

    return sa.WorkerResult(
        seed=1, ckpt_iteration=ckpt_iteration, records=tuple(synthetic_records(n_records, rng)),
        reloaded=False, secs=0.0,
    )


def test_absorb_flushes_at_games_per_shard() -> None:
    buf = sa.ShardBuffer.empty()
    buf, flushed = sa.absorb(buf, _result(0), games_per_shard=2)
    assert flushed == () and buf.n_games == 1
    buf, flushed = sa.absorb(buf, _result(0), games_per_shard=2)
    assert len(flushed) == 1 and flushed[0].n_games == 2 and buf.n_games == 0


def test_absorb_flushes_partial_buffer_when_checkpoint_changes() -> None:
    buf = sa.ShardBuffer.empty()
    buf, _ = sa.absorb(buf, _result(0), games_per_shard=8)
    buf, flushed = sa.absorb(buf, _result(1), games_per_shard=8)
    assert len(flushed) == 1
    assert flushed[0].ckpt_iteration == 0 and flushed[0].n_games == 1
    assert buf.ckpt_iteration == 1 and buf.n_games == 1


def test_absorb_ignores_dropped_games() -> None:
    buf = sa.ShardBuffer.empty()
    buf, flushed = sa.absorb(buf, _result(0, n_records=0), games_per_shard=1)
    assert flushed == () and buf.n_games == 0


def test_next_seq_continues_after_existing_shards(tmp_path: Path) -> None:
    d = tmp_path / "shards"
    d.mkdir()
    (d / sh.shard_name(0, 2, 0)).touch()
    (d / sh.shard_name(3, 2, 7)).touch()
    (d / sh.shard_name(3, 1, 99)).touch()  # another actor
    (d / "junk.tmp").touch()
    assert sa.next_seq(d, actor_id=2) == 8
    assert sa.next_seq(d, actor_id=5) == 0
    assert sa.next_seq(tmp_path / "missing", actor_id=2) == 0


# -- worker reload trigger (in-process) -------------------------------------


def test_worker_reloads_only_when_checkpoint_mtime_changes(tmp_path: Path) -> None:
    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0)
    sa._actor_worker_init(str(ckpt), FAST, 1.5)
    stamp = checkpoint_mtime_ns(ckpt)
    task = sa.WorkerTask(seed=sa.game_seed(0, 0, 0), checkpoint=str(ckpt), mtime_ns=stamp)
    first = sa._actor_worker_game(task)
    assert first.ckpt_iteration == 0 and not first.reloaded
    assert len(first.records) > 0

    _bump_mtime(ckpt)
    save_atomic(checkpoint_payload(tiny_net(1), 2), ckpt)
    _bump_mtime(ckpt)
    new_stamp = checkpoint_mtime_ns(ckpt)
    assert new_stamp != stamp
    second = sa._actor_worker_game(
        sa.WorkerTask(seed=sa.game_seed(0, 0, 1), checkpoint=str(ckpt), mtime_ns=new_stamp)
    )
    assert second.reloaded and second.ckpt_iteration == 2
    third = sa._actor_worker_game(
        sa.WorkerTask(seed=sa.game_seed(0, 0, 2), checkpoint=str(ckpt), mtime_ns=new_stamp)
    )
    assert not third.reloaded and third.ckpt_iteration == 2


def test_worker_keeps_old_weights_when_new_checkpoint_is_bad(tmp_path: Path) -> None:
    ckpt = tmp_path / "current.pt"
    write_tiny_checkpoint(ckpt, iteration=0)
    sa._actor_worker_init(str(ckpt), FAST, 1.5)
    _bump_mtime(ckpt)
    ckpt.write_bytes(b"garbage")
    _bump_mtime(ckpt)
    res = sa._actor_worker_game(
        sa.WorkerTask(seed=3, checkpoint=str(ckpt), mtime_ns=checkpoint_mtime_ns(ckpt))
    )
    assert not res.reloaded and res.ckpt_iteration == 0 and len(res.records) > 0


# -- whole actor, one worker subprocess ------------------------------------


def test_run_actor_writes_shards_and_stops_at_max_games(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_tiny_checkpoint(run_dir / "current.pt", iteration=4)
    cfg = sa.ActorConfig(
        run_dir=run_dir, actor_id=2, games_per_shard=2, workers=1, max_games=3,
        seed=11, selfplay=FAST, c_puct=2.5,
    )
    played = sa.run_actor(cfg)
    assert played == 3
    files = sorted(p.name for p in (run_dir / "shards").iterdir())
    assert files == [sh.shard_name(4, 2, 0), sh.shard_name(4, 2, 1)]
    metas = [sh.read_shard_meta(run_dir / "shards" / f) for f in files]
    assert [m.n_games for m in metas] == [2, 1]  # the partial tail is flushed on stop
    assert all(m.ckpt_iteration == 4 and m.actor_id == 2 for m in metas)
    assert all(m.config_hash == sh.config_hash(FAST, {"c_puct": 2.5}) for m in metas)
    _, records = sh.read_shard(run_dir / "shards" / files[0])
    assert records and all(r.final_shares is not None for r in records)
