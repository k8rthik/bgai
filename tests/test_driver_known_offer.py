"""``advance(sim, choice, current=...)``: a caller that already holds the
pending ``(faction, offer)`` (an MCTS node, the self-play loop) must be
able to skip the driver's own re-enumeration without changing the
result. Legal-move enumeration is the single most expensive thing in a
self-play game, and ``advance`` used to recompute it for the parent on
every child expansion.
"""

from __future__ import annotations

import pytest

from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import load_setup
from bgai.engine.tm.apply import EngineError


def _walk(n: int):
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    for _ in range(n):
        pending = decision(sim)
        assert pending is not None
        yield sim, pending
        sim = advance(sim, pending[1][0])


def test_known_offer_matches_recomputed_offer() -> None:
    for sim, pending in _walk(40):
        _faction, offer = pending
        choice = offer[-1]
        plain = advance(sim, choice)
        known = advance(sim, choice, current=pending)
        assert known == plain


def test_known_offer_still_rejects_a_move_outside_it() -> None:
    sim, pending = next(_walk(1))
    faction, offer = pending
    stranger = offer[0]
    bogus = (faction, tuple(m for m in offer if m != stranger))
    with pytest.raises(EngineError):
        advance(sim, stranger, current=bogus)
