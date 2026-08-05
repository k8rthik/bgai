"""Public pending-answer API on legal.py (plan 2026-08-04-arena-baselines, Task 2)."""

from __future__ import annotations

from bgai.engine.tm.legal import has_blocking_pending, pending_answer_moves
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState


def test_no_pending_at_setup_start() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    for faction in setup.factions:
        assert pending_answer_moves(state, faction) == ()
        assert not has_blocking_pending(state, faction)
