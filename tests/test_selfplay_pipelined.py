"""End-to-end pipelined run: 2 actors x 1 worker + in-process learner.

Tiny net, 8 simulations, 4 games per learner step, 2 steps. Proves the
three processes agree on the run-directory contract; strength is not
measured here (see the validation gate in docs/decisions.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from selfplay_helpers import write_tiny_checkpoint

from bgai.training import shards as sh
from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.selfplay_launcher import (
    actor_argv,
    learner_config,
    run_pipelined,
    workers_per_actor,
)
from bgai.training.selfplay_train import SelfPlayTrainConfig, build_parser, config_from_args
from bgai.training.stop_signal import STOP_FILE


def _cfg(tmp_path: Path, **kw) -> SelfPlayTrainConfig:
    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    base = dict(
        checkpoint=init, out=tmp_path / "run", iterations=2, games_per_iteration=4,
        workers=2, actors=2, simulations=8, top_k=4, max_depth=8, leaf_batch=4,
        batch_size=32, lr=1e-3, games_per_shard=2, idle_timeout_secs=120.0,
        mode="pipelined",
    )
    return SelfPlayTrainConfig(**{**base, **kw})


def test_cli_defaults_keep_serial_mode_and_c_puct() -> None:
    args = build_parser().parse_args(["--checkpoint", "a.pt", "--out", "o"])
    cfg = config_from_args(args)
    assert cfg.mode == "serial" and cfg.c_puct == 1.5 and cfg.selfplay().c_puct == 1.5
    args = build_parser().parse_args(
        ["--checkpoint", "a.pt", "--out", "o", "--mode", "pipelined", "--c-puct", "2.5",
         "--actors", "3"]
    )
    cfg = config_from_args(args)
    assert cfg.mode == "pipelined" and cfg.selfplay().c_puct == 2.5 and cfg.actors == 3


def test_launcher_plumbing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, c_puct=2.5)
    argv = actor_argv(cfg, actor_id=1, workers=1)
    assert argv[1:3] == ["-m", "bgai.training.selfplay_actor"]
    assert "--c-puct" in argv and argv[argv.index("--c-puct") + 1] == "2.5"
    assert argv[argv.index("--games-per-shard") + 1] == "2"
    lc = learner_config(cfg)
    assert lc.games_per_step == 4 and lc.window_steps == 2 and lc.max_iterations == 2
    assert lc.finetune.batch_size == 32 and lc.finetune.lr == 1e-3
    assert workers_per_actor(8, 2) == 4 and workers_per_actor(1, 4) == 1


def test_pipelined_run_end_to_end(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    steps = run_pipelined(cfg, actor_stop_timeout=120.0)
    assert steps == 2
    run_dir = cfg.out
    assert load_checkpoint(run_dir / "current.pt")["iteration"] == 2
    assert (run_dir / "iter_0001.pt").exists() and (run_dir / "iter_0002.pt").exists()
    assert not (run_dir / STOP_FILE).exists()
    assert not any(p.suffix == ".tmp" for p in run_dir.iterdir())

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert all(r["new_games"] >= 4 for r in rows)
    assert all("staleness_hist" in r and "total" in r for r in rows)

    shard_files = [p for p in (run_dir / "shards").iterdir() if sh.is_shard_file(p)]
    assert len(shard_files) >= 4
    metas = [sh.read_shard_meta(p) for p in shard_files]
    assert {m.actor_id for m in metas} == {0, 1}
    assert all(m.n_games >= 1 and m.n_records > 0 for m in metas)
    assert sum(m.n_games for m in metas) >= 8
    # every shard's name agrees with its metadata
    for p, m in zip(shard_files, metas, strict=True):
        assert p.name.startswith(f"it{m.ckpt_iteration:04d}_a{m.actor_id:03d}_")


def test_pipelined_resume_continues_numbering_and_refuses_clobber(tmp_path: Path) -> None:
    import pytest

    cfg = _cfg(tmp_path, iterations=1)
    assert run_pipelined(cfg, actor_stop_timeout=120.0) == 1
    with pytest.raises(ValueError, match="resume"):
        run_pipelined(cfg, actor_stop_timeout=120.0)
    from bgai.training.selfplay_train import resumed_config

    resumed = resumed_config(cfg.out, {"extra_iterations": 1})
    assert resumed.mode == "pipelined" and resumed.iterations == 2
    assert learner_config(resumed).max_iterations == 2  # run total
    assert run_pipelined(resumed, actor_stop_timeout=120.0) == 1
    assert load_checkpoint(cfg.out / "current.pt")["iteration"] == 2
    rows = [json.loads(line) for line in (cfg.out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows if "total_games" in r] == [1, 2]
    assert rows[-1]["new_games"] >= 4, "resumed learner counts only games after its last step"
    # actors restarted with the same ids continued their shard sequences
    names = sorted(p.name for p in (cfg.out / "shards").iterdir() if sh.is_shard_file(p))
    seqs_a0 = [int(n.rsplit("_", 1)[1][:6]) for n in names if "_a000_" in n]
    assert seqs_a0 == list(range(len(seqs_a0)))
