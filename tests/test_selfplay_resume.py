"""Resumable runs: numbering, seeds, STOP, provenance (serial + learner)."""

from __future__ import annotations

import json
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
from selfplay_helpers import synthetic_records, tiny_net, write_tiny_checkpoint

from bgai.training import run_provenance as rp
from bgai.training import selfplay_train as st
from bgai.training import shards as sh
from bgai.training.checkpoint_io import load_checkpoint
from bgai.training.cli_util import explicit_cli_args
from bgai.training.selfplay_finetune import checkpoint_payload
from bgai.training.stop_signal import StopSignal
from bgai.training.vocab import ENCODING_VERSION


def _rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.fixture
def fake_generation(monkeypatch):
    """Replace parallel generation with synthetic records; records seeds."""
    seeds: list[int] = []

    def fake(checkpoint, games, workers, cfg, seed, infer_device=""):
        seeds.append(seed)
        return synthetic_records(60, np.random.default_rng(seed))

    monkeypatch.setattr(st, "generate_parallel", fake)
    return seeds


def _cfg(tmp_path: Path, **kw) -> st.SelfPlayTrainConfig:
    init = tmp_path / "init.pt"
    if not init.exists():
        write_tiny_checkpoint(init, iteration=0)
    base = dict(checkpoint=init, out=tmp_path / "run", iterations=2, games_per_iteration=2,
                workers=1, batch_size=16, lr=1e-3, seed=5)
    return st.SelfPlayTrainConfig(**{**base, **kw})


# -- provenance ---------------------------------------------------------------


def test_code_provenance_fields() -> None:
    prov = rp.code_provenance()
    assert prov.encoding_version == ENCODING_VERSION
    assert prov.hostname and isinstance(prov.code_dirty, bool)
    assert prov.git_sha and len(prov.git_sha) >= 4  # short sha or "unknown"
    d = prov.as_dict()
    assert set(d) == {"git_sha", "code_dirty", "encoding_version", "hostname"}


def test_code_provenance_survives_missing_git(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(rp.subprocess, "run", boom)
    prov = rp.code_provenance()
    assert prov.git_sha == "unknown" and prov.code_dirty is False


def test_checkpoint_payload_tensors_live_on_cpu() -> None:
    from selfplay_helpers import TINY

    net = tiny_net()
    payload = checkpoint_payload(net, TINY, 3, {"x": 1})
    assert all(v.device.type == "cpu" for v in payload["model"].values())
    assert payload["config"]["x"] == 1 and payload["iteration"] == 3


# -- serial run: provenance + resume -----------------------------------------


def test_serial_run_stamps_provenance_and_stores_resume_fields(
    tmp_path: Path, fake_generation
) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    rows = _rows(cfg.out)
    assert [r["iteration"] for r in rows] == [1, 2]
    for r in rows:
        assert {"git_sha", "code_dirty", "encoding_version", "hostname"} <= set(r)
    stored = load_checkpoint(cfg.out / "current.pt")["config"]
    assert stored["seed"] == 5 and stored["anchor"] is None
    assert stored["checkpoint"] == str(cfg.checkpoint)
    assert stored["encoding_version"] == ENCODING_VERSION and "git_sha" in stored
    assert fake_generation == [6, 7]


def test_resume_continues_numbering_without_reusing_seeds(
    tmp_path: Path, fake_generation
) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    first_iter1 = (cfg.out / "iter_01.pt").read_bytes()
    resumed = st.resumed_config(cfg.out, overrides={"extra_iterations": 2})
    assert resumed.iterations == 4 and resumed.resume == cfg.out
    assert resumed.anchor == cfg.checkpoint  # the original init, not current.pt
    assert resumed.checkpoint == cfg.out / "current.pt"
    assert resumed.seed == 5 and resumed.lr == 1e-3 and resumed.batch_size == 16
    st.run(resumed)
    assert fake_generation == [6, 7, 8, 9], "seeds must never repeat across a resume"
    rows = _rows(cfg.out)
    assert [r["iteration"] for r in rows if "iteration" in r] == [1, 2, 3, 4]
    assert load_checkpoint(cfg.out / "current.pt")["iteration"] == 4
    assert (cfg.out / "iter_01.pt").read_bytes() == first_iter1, "never overwrite history"
    assert (cfg.out / "iter_04.pt").exists()
    stored = load_checkpoint(cfg.out / "current.pt")["config"]
    assert stored["anchor"] == str(cfg.checkpoint)


def test_resume_iterations_is_total_and_cli_overrides_only_explicit(
    tmp_path: Path, fake_generation
) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    resumed = st.resumed_config(cfg.out, overrides={"iterations": 3, "lr": 2e-3})
    assert resumed.iterations == 3 and resumed.lr == 2e-3 and resumed.batch_size == 16
    st.run(resumed)
    assert fake_generation == [6, 7, 8]
    # already complete: nothing runs, nothing breaks
    st.run(st.resumed_config(cfg.out, overrides={}))
    assert fake_generation == [6, 7, 8]


def test_resume_refuses_encoding_mismatch(tmp_path: Path, fake_generation) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    payload = load_checkpoint(cfg.out / "current.pt")
    payload["config"]["encoding_version"] = ENCODING_VERSION + 1
    torch.save(payload, cfg.out / "current.pt")
    with pytest.raises(ValueError, match="encoding"):
        st.resumed_config(cfg.out, overrides={})


def test_resume_logs_code_changed_when_sha_differs(
    tmp_path: Path, fake_generation, monkeypatch
) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    monkeypatch.setattr(rp, "_git_sha", lambda _root: "deadbee")
    st.run(st.resumed_config(cfg.out, overrides={"extra_iterations": 1}))
    events = [r for r in _rows(cfg.out) if r.get("event") == "code_changed"]
    assert len(events) == 1 and events[0]["to"] == "deadbee"
    assert events[0]["from"] != "deadbee"


def test_fresh_run_refuses_to_clobber_an_existing_run(tmp_path: Path, fake_generation) -> None:
    cfg = _cfg(tmp_path)
    st.run(cfg)
    with pytest.raises(ValueError, match="resume"):
        st.run(cfg)


def test_explicit_cli_args_detects_only_given_flags() -> None:
    given = explicit_cli_args(st.build_parser, ["--resume", "r", "--lr", "1e-3"])
    assert given == {"resume": Path("r"), "lr": 1e-3}
    args = st.build_parser().parse_args(["--resume", "r", "--extra-iterations", "2"])
    assert args.resume == Path("r") and args.extra_iterations == 2


# -- STOP / SIGTERM ----------------------------------------------------------


def test_stop_file_ends_serial_run_after_the_inflight_iteration(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _cfg(tmp_path, iterations=5)
    calls: list[int] = []

    def fake(checkpoint, games, workers, cfgs, seed, infer_device=""):
        calls.append(seed)
        (cfg.out / "STOP").touch()  # arrives mid-iteration
        return synthetic_records(60, np.random.default_rng(seed))

    monkeypatch.setattr(st, "generate_parallel", fake)
    st.run(cfg)
    assert calls == [6], "the in-flight iteration finishes; no new one starts"
    rows = _rows(cfg.out)
    assert rows[0]["iteration"] == 1
    assert rows[-1]["event"] == "stopped" and rows[-1]["iteration"] == 1
    assert load_checkpoint(cfg.out / "current.pt")["iteration"] == 1
    assert not (cfg.out / "STOP").exists()


def test_stop_signal_flags_on_sigterm_and_stop_file(tmp_path: Path) -> None:
    stop = StopSignal(tmp_path)
    assert not stop.requested()
    stop.install()
    signal.raise_signal(signal.SIGTERM)
    assert stop.requested()
    stop.uninstall()
    other = StopSignal(tmp_path)
    (tmp_path / "STOP").touch()
    assert other.requested()
    other.clear()
    assert not (tmp_path / "STOP").exists()


# -- learner: STOP, device, resume config ------------------------------------


def test_learner_honours_stop_between_steps(tmp_path: Path) -> None:
    from bgai.training import selfplay_learner as sl
    from bgai.training.selfplay_finetune import FineTuneConfig

    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    d = run_dir / "shards"
    for seq in range(2):
        sh.write_shard(
            d / sh.shard_name(0, 0, seq), synthetic_records(20, np.random.default_rng(seq)),
            ckpt_iteration=0, actor_id=0, seed=seq, config_hash="t", n_games=2,
        )
    (run_dir / "STOP").touch()
    cfg = sl.LearnerConfig(
        run_dir=run_dir, checkpoint=init, games_per_step=2, window_steps=2,
        finetune=FineTuneConfig(batch_size=8, lr=1e-3), poll_secs=0.05,
        idle_timeout_secs=5.0, max_iterations=10,
    )
    assert sl.run_learner(cfg) == 1  # one step (games were ready), then STOP
    rows = _rows(run_dir)
    assert rows[0]["iteration"] == 1 and {"git_sha", "hostname"} <= set(rows[0])
    assert rows[-1]["event"] == "stopped"
    assert not (run_dir / "STOP").exists()


def test_learner_device_auto_resolves_and_saves_cpu_tensors(tmp_path: Path) -> None:
    from bgai.training import selfplay_learner as sl
    from bgai.training.selfplay_finetune import FineTuneConfig

    init = tmp_path / "init.pt"
    write_tiny_checkpoint(init, iteration=0)
    run_dir = tmp_path / "run"
    cfg = sl.LearnerConfig(
        run_dir=run_dir, checkpoint=init, games_per_step=2, window_steps=2,
        finetune=FineTuneConfig(batch_size=8, lr=1e-3), poll_secs=0.05,
        idle_timeout_secs=0.2, device="auto",
    )
    assert sl.resolve_device(cfg).type in {"cpu", "cuda", "mps"}
    sl.run_learner(cfg)
    payload = load_checkpoint(run_dir / "current.pt")
    assert all(v.device.type == "cpu" for v in payload["model"].values())
    assert payload["config"]["device"] == "auto"
