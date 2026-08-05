"""Headless LLM arena runner (LLM-harness plan, task 11). The
claude-spawning path is exercised only by the slow test at the bottom."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import arena_llm  # noqa: E402


def test_dry_run_writes_valid_configs(tmp_path, monkeypatch):
    out = tmp_path / "run"
    arena_llm.main(
        [
            "--games", "3", "--seed", "100", "--opponents", "greedy",
            "--rungs", "1,2,3", "--out", str(out), "--dry-run",
        ]
    )
    from bgai.mcp.config import load_config

    for g in range(3):
        session_path = out / f"game_{g}" / "session.json"
        assert session_path.exists()
        monkeypatch.setenv("BGAI_TM_SESSION", str(session_path))
        config = load_config()
        assert config.seed == 100 + g
        assert config.llm_faction_index == g % 4
        assert config.rungs == (1, 2, 3)
        mcp_config = json.loads((out / f"game_{g}" / "mcp.json").read_text())
        assert mcp_config["mcpServers"]["tm"]["env"]["BGAI_TM_SESSION"] == str(session_path)


def test_rung4_appends_compendium(tmp_path):
    playbooks = tmp_path / "compendium"
    playbooks.mkdir()
    (playbooks / "nomads-playbook.md").write_text("# Nomads principles\nsand dwellers")
    prompt = arena_llm.pilot_prompt(rungs=(1, 2, 3, 4), compendium_dir=playbooks)
    assert "sand dwellers" in prompt
    assert "{{COMPENDIUM}}" not in prompt
    bare = arena_llm.pilot_prompt(rungs=(1,), compendium_dir=playbooks)
    assert "sand dwellers" not in bare
    assert "{{COMPENDIUM}}" not in bare


def test_aggregate_skips_invalid_games():
    results = [
        {"status": "ok", "llm_faction": "nomads",
         "vp": {"nomads": 90, "a": 50, "b": 60, "c": 70}, "winners": ["nomads"],
         "commands_used": 120, "cost_usd": 1.5, "duration_s": 60.0},
        {"status": "ok", "llm_faction": "a",
         "vp": {"nomads": 90, "a": 50, "b": 60, "c": 70}, "winners": ["nomads"],
         "commands_used": 100, "cost_usd": 2.0, "duration_s": 80.0},
        {"status": "invalid"},
    ]
    summary = arena_llm.aggregate(results)
    assert summary["games_valid"] == 2
    assert summary["games_invalid"] == 1
    assert summary["llm_win_rate"] == pytest.approx(0.5)
    assert summary["llm_mean_vp"] == pytest.approx((90 + 50) / 2)
    assert summary["mean_cost_usd"] == pytest.approx(1.75)


def test_retry_then_invalid(tmp_path, monkeypatch):
    calls = []

    def failing_spawn(game_dir, prompt, args):
        calls.append(game_dir)
        return None  # simulates a game that never writes result.json

    monkeypatch.setattr(arena_llm, "spawn_game", failing_spawn)
    out = tmp_path / "run"
    arena_llm.main(["--games", "1", "--seed", "7", "--out", str(out)])
    assert len(calls) == 3  # 1 try + 2 retries
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games_invalid"] == 1


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")
def test_one_real_headless_game(tmp_path):
    out = tmp_path / "smoke"
    arena_llm.main(["--games", "1", "--seed", "500", "--rungs", "1", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games_valid"] + summary["games_invalid"] == 1
