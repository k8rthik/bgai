"""Rung-3 sandbox branching (LLM-harness plan, task 10)."""

from bgai.engine.tm.state import Phase
from bgai.mcp.branching import (
    branch_create,
    branch_discard,
    branch_list,
    branch_moves_text,
    branch_play,
    branch_state_text,
)
from bgai.mcp.config import SessionConfig
from bgai.mcp.session import Session


def _session() -> Session:
    session = Session(SessionConfig(seed=5, llm_faction_index=0, opponents="random"))
    session.start()
    return session


def _first_branch_move(session: Session, label: str) -> str:
    text = branch_moves_text(session, label)
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("1."))
    return line.split(".", 1)[1].strip()


def test_branch_forks_without_touching_live_state():
    session = _session()
    live = session.state
    out = branch_create(session, "opening-a")
    assert "opening-a" in out
    branch_play(session, "opening-a", _first_branch_move(session, "opening-a"))
    assert session.state is live  # live game untouched
    assert "opening-a" in branch_list(session)


def test_branch_routes_all_seats():
    session = _session()
    branch_create(session, "b")
    actors = []
    for _ in range(3):
        out = branch_play(session, "b", _first_branch_move(session, "b"))
        assert not out.startswith("ERROR"), out
        actors.append(out.splitlines()[0])
    assert len(set(actors)) >= 2  # setup snake passes through several factions


def test_branch_errors():
    session = _session()
    branch_create(session, "x")
    assert branch_create(session, "x").startswith("ERROR")  # duplicate
    for i in range(7):
        branch_create(session, f"fill{i}")
    assert branch_create(session, "overflow").startswith("ERROR")  # cap 8
    assert branch_play(session, "nope", "build A1").startswith("ERROR")  # unknown label
    assert branch_play(session, "x", "gibberish").startswith("ERROR")


def test_branch_discard():
    session = _session()
    branch_create(session, "gone")
    branch_discard(session, "gone")
    assert "gone" not in branch_list(session)
    assert branch_state_text(session, "gone").startswith("ERROR")


def test_branch_state_carries_fork_header():
    session = _session()
    branch_create(session, "hdr")
    assert "branched from: round 0" in branch_state_text(session, "hdr")


def test_branch_plays_to_finished_with_final_vp():
    session = _session()
    branch_create(session, "full")
    for _ in range(3000):
        moves = branch_moves_text(session, "full")
        if "FINISHED" in moves:
            break
        branch_play(session, "full", _first_branch_move(session, "full"))
    state_text = branch_state_text(session, "full")
    assert "FINISHED" in state_text
    assert "VP" in state_text
    assert session.state.phase is Phase.SETUP_DWELLINGS  # live game still at start
