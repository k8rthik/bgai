"""Per-state memoisation for pure engine queries.

``GameState``/``FactionState`` are frozen but hold dict fields, so they
are not hashable; the memo keys on object identity and pins the keyed
object so its id cannot be recycled while the entry lives. A fresh
(structurally equal) object is a miss by design: identity is the only
thing that is free to check, and equal states recompute the same answer.
"""

from __future__ import annotations

import dataclasses

from bgai.arena.setups import load_setup
from bgai.engine.tm.memo import state_memo
from bgai.engine.tm.state import GameState


def _state() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))


def test_repeat_call_on_same_object_hits() -> None:
    calls = []

    @state_memo()
    def probe(state: GameState, faction: str) -> tuple[str, int]:
        calls.append(faction)
        return faction, state.round

    s = _state()
    assert probe(s, "witches") == ("witches", 0)
    assert probe(s, "witches") == ("witches", 0)
    assert probe(s, "nomads") == ("nomads", 0)
    assert calls == ["witches", "nomads"]


def test_new_object_misses_even_when_equal() -> None:
    calls = []

    @state_memo()
    def probe(state: GameState) -> int:
        calls.append(1)
        return state.round

    s = _state()
    probe(s)
    probe(dataclasses.replace(s))
    assert len(calls) == 2


def test_bounded_and_identity_safe() -> None:
    """After eviction a recycled id must not return a stale answer."""

    @state_memo(maxsize=4)
    def probe(state: GameState) -> int:
        return state.round

    s = _state()
    assert probe(s) == 0
    for r in range(1, 9):
        probe(dataclasses.replace(s, round=r))  # evicts ``s``
    later = dataclasses.replace(s, round=42)
    assert probe(later) == 42
    assert probe(s) == 0
