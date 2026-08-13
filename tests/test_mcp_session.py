"""MCP session + config, rung 1 (LLM-harness plan, task 8)."""

import json

import pytest

from bgai.engine.tm.state import Phase
from bgai.mcp.config import SessionConfig, load_config
from bgai.mcp.render import render_command
from bgai.mcp.session import Session


def _session(**overrides) -> Session:
    defaults = dict(seed=5, llm_faction_index=0, opponents="random", rungs=(1, 2, 3))
    defaults.update(overrides)
    return Session(SessionConfig(**defaults))


def test_start_reaches_llm_decision():
    session = _session()
    text = session.start()
    assert session.state is not None
    assert session.state.phase is Phase.SETUP_DWELLINGS
    assert session.offered()
    assert "legal moves" in text
    assert session.llm_faction in text


def test_submit_legal_move_advances():
    session = _session()
    session.start()
    move = session.offered()[0]
    before_buildings = session.state.factions[session.llm_faction].buildings["D"]
    out = session.submit(render_command(move))
    assert "ERROR" not in out
    after_buildings = session.state.factions[session.llm_faction].buildings["D"]
    assert len(after_buildings) == len(before_buildings) + 1


def test_submit_illegal_move_returns_error_and_keeps_state():
    session = _session()
    session.start()
    state_before = session.state
    out = session.submit("build Z9")
    assert out.startswith("ERROR")
    assert "legal moves" in out
    assert session.state is state_before


def test_submit_unparseable_returns_error():
    session = _session()
    session.start()
    out = session.submit("flarb the wizzles")
    assert out.startswith("ERROR")
    assert "legal moves" in out


def test_full_llm_game_via_session(tmp_path):
    result_path = tmp_path / "result.json"
    session = _session(opponents="greedy", result_path=str(result_path))
    session.start()
    for _ in range(2000):
        if session.state.phase is Phase.FINISHED:
            break
        session.submit(render_command(session.offered()[0]))
    assert session.state.phase is Phase.FINISHED
    payload = json.loads(result_path.read_text())
    assert payload["llm_faction"] == session.llm_faction
    assert set(payload["vp"]) == set(session.state.setup.factions)
    assert payload["winners"]
    assert payload["rungs"] == [1, 2, 3]


def test_all_external_mode_routes_by_next_actor(tmp_path):
    session = _session(llm_faction_index=-1)
    session.start()
    seen = set()
    for _ in range(12):
        actor = session.current_actor()
        seen.add(actor)
        session.submit(render_command(session.offered()[0]))
    assert len(seen) == 4  # setup snake passes through every faction


def test_config_rejects_unknown_keys(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"seed": 1, "wibble": True}))
    monkeypatch.setenv("BGAI_TM_SESSION", str(path))
    with pytest.raises(ValueError, match="wibble"):
        load_config()


def test_config_rejects_bad_types(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"seed": "not-an-int"}))
    monkeypatch.setenv("BGAI_TM_SESSION", str(path))
    with pytest.raises(ValueError, match="seed"):
        load_config()


def test_config_env_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "seed": 42,
                "llm_faction_index": 2,
                "opponents": "greedy",
                "rungs": [1, 2],
                "result_path": str(tmp_path / "r.json"),
                "max_commands": 5000,
            }
        )
    )
    monkeypatch.setenv("BGAI_TM_SESSION", str(path))
    config = load_config()
    assert config.seed == 42
    assert config.llm_faction_index == 2
    assert config.rungs == (1, 2)
    assert config.max_commands == 5000


def test_config_defaults_without_env(monkeypatch):
    monkeypatch.delenv("BGAI_TM_SESSION", raising=False)
    config = load_config()
    assert config.rungs == (1, 2, 3)
    assert config.opponents == "greedy"


def test_config_imitation_requires_checkpoint():
    with pytest.raises(ValueError, match="opponent_ckpt"):
        SessionConfig(opponents="imitation")
    cfg = SessionConfig(opponents="imitation", opponent_ckpt="x.pt")
    assert cfg.opponent_ckpt == "x.pt"


def test_config_rejects_unknown_opponents():
    with pytest.raises(ValueError, match="opponents"):
        SessionConfig(opponents="mcts")


_DEPRECATED_CKPT = "data/checkpoints/imitation/checkpoint.pt"


@pytest.mark.skipif(
    not __import__("pathlib").Path(_DEPRECATED_CKPT).exists(),
    reason="historical checkpoint not present",
)
def test_session_with_imitation_opponents_reaches_llm_decision():
    """A human/LLM seat vs three copies of a historical net: the session
    must reach the external seat's first decision, driven by real
    ImitationAgent moves (loaded once, shared across seats)."""
    session = Session(SessionConfig(opponents="imitation", opponent_ckpt=_DEPRECATED_CKPT))
    text = session.start()
    assert "your" in text.lower() or session.offered()
    bots = session._bots()
    assert len({id(b) for b in bots.values()}) == 1, "net must be shared across seats"
