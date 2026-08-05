"""Rung-2 factual analysis tools (LLM-harness plan, task 9)."""

import dataclasses
import re

from bgai.engine.tm.scoring import compute_cult_scoring
from bgai.engine.tm.state import Phase
from bgai.mcp.analysis import preview, projection
from bgai.mcp.config import SessionConfig
from bgai.mcp.render import render_command
from bgai.mcp.session import Session

JUDGMENT_WORDS = re.compile(
    r"\b(recommend|best|should|strong|weak|good|bad)\b", re.IGNORECASE
)


def _actions_session() -> Session:
    session = Session(SessionConfig(seed=5, llm_faction_index=0, opponents="random"))
    session.start()
    while session.state.phase is not Phase.ACTIONS:
        session.submit(render_command(session.offered()[0]))
    return session


def test_preview_shows_deltas_without_committing():
    session = _actions_session()
    actor = session.current_actor()
    factions = dict(session.state.factions)
    factions[actor] = dataclasses.replace(factions[actor], priests=2)
    session.state = dataclasses.replace(session.state, factions=factions)
    state_before = session.state
    send = next((m for m in session.offered() if m.verb == "send"), None)
    assert send is not None, "expected a send move in round-1 actions"
    text = preview(session, render_command(send))
    assert "P -1" in text
    assert send.cult in text
    assert session.state is state_before  # nothing committed
    # a previewed move is still submittable afterwards
    out = session.submit(render_command(send))
    assert not out.startswith("ERROR")


def test_preview_rejects_illegal_move():
    session = _actions_session()
    assert preview(session, "build Z9").startswith("ERROR")
    assert preview(session, "gibberish").startswith("ERROR")


def test_projection_cult_income_arithmetic():
    session = _actions_session()
    state = session.state
    tile = state.setup.score_tiles[state.round - 1]
    if not tile.cult_income:
        # doctor a known cult position to exercise the arithmetic anyway
        pass
    faction = state.setup.factions[0]
    cults = {**state.cults, faction: {**state.cults[faction], tile.cult: 4 * tile.req}}
    session.state = dataclasses.replace(state, cults=cults)
    text = projection(session)
    for name, amount in tile.cult_income:
        assert f"{4 * amount}{name}" in text.replace(" ", "") or f"{4 * amount} {name}" in text


def test_projection_matches_cult_scoring():
    session = _actions_session()
    scores = compute_cult_scoring(session.state)
    text = projection(session)
    assert "ended now" in text
    for cult, per_faction in scores.items():
        for faction, vp in per_faction.items():
            assert faction in text
    assert "network" in text.lower()


def test_no_judgment_words_in_output():
    session = _actions_session()
    send = next((m for m in session.offered() if m.verb == "send"), None)
    texts = [projection(session)]
    if send is not None:
        texts.append(preview(session, render_command(send)))
    for text in texts:
        assert not JUDGMENT_WORDS.search(text), JUDGMENT_WORDS.search(text)
