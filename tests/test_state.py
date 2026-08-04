"""tests/test_state.py"""

from __future__ import annotations

from dataclasses import replace

from bgai.engine.tm.cults import PRIEST_SLOT_STEPS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import (
    GameState,
    HexState,
    PendingDecision,
    Phase,
    active_faction,
    cult_string,
    with_faction,
)


def test_initial_state_matches_setup_deltas() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert s.phase is Phase.SETUP_DWELLINGS and s.round == 0
    eng = s.factions["engineers"]
    assert (eng.coins, eng.workers, eng.priests, eng.vp) == (10, 2, 0, 20)
    assert eng.power.as_str() == "3/9/0"  # deltas row 24
    dk = s.factions["darklings"]
    assert (dk.coins, dk.workers, dk.priests) == (15, 1, 1)
    assert cult_string(s, "darklings") == "0/1/1/0"  # deltas row 25
    assert cult_string(s, "nomads") == "1/0/1/0"  # deltas row 26
    assert active_faction(s) == "engineers"


def test_pools_initialized() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert s.favors_pool["FAV1"] == 1 and s.favors_pool["FAV5"] == 3
    assert sum(s.towns_pool.values()) >= 10
    assert all(h.building is None for h in s.hexes.values())


def test_turn_order_matches_seat_order() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    s = GameState.initial(setup)
    assert s.turn_order == setup.factions
    assert s.active_index == 0
    assert s.pending == ()


def test_faction_initial_defaults() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    eng = s.factions["engineers"]
    assert eng.priest_pool == 7
    assert eng.shipping == 0
    assert eng.dig_level == 0
    assert eng.teleport_level == 0
    assert eng.favors == () and eng.towns == () and eng.bonus is None
    assert eng.keys == 0 and eng.passed is False
    assert eng.actions_used == frozenset() and eng.cult_blocked == frozenset()
    assert all(v == frozenset() for v in eng.buildings.values())
    assert set(eng.buildings.keys()) == {"D", "TP", "TE", "SH", "SA"}


def test_mermaids_start_at_shipping_level_1() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert s.factions["mermaids"].shipping == 1


def test_cult_10_and_priest_slots_initialized_empty() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert set(s.cult_10.keys()) == {"FIRE", "WATER", "EARTH", "AIR"}
    assert all(v is None for v in s.cult_10.values())
    assert set(s.priest_slots.keys()) == {"FIRE", "WATER", "EARTH", "AIR"}
    for slots in s.priest_slots.values():
        assert slots == (None,) * len(PRIEST_SLOT_STEPS)


def test_bonus_coins_initialized_zero_for_pool() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    s = GameState.initial(setup)
    assert set(s.bonus_coins.keys()) == set(setup.bonus_tiles)
    assert all(v == 0 for v in s.bonus_coins.values())


def test_active_faction_ignores_pending_regardless_of_kind() -> None:
    """``active_faction`` tracks ``turn_order[active_index]`` only -- a
    queued decision for a *different* faction (here, an outstanding
    leech offer for darklings while engineers is still turn_order's
    active seat) must not override it (module docstring, task-13 fix:
    reference-game row 59 replays with exactly this shape -- mermaids
    takes an ordinary turn while engineers/darklings each still have an
    unanswered leech offer queued)."""
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    pending = (PendingDecision(faction="darklings", kind="leech"),)
    s2 = replace(s, pending=pending)
    assert active_faction(s2) == "engineers"


def test_with_faction_returns_new_state_without_mutating_original() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    original = s.factions["engineers"]
    updated = replace(original, coins=999)
    s2 = with_faction(s, "engineers", updated)
    assert s.factions["engineers"].coins == 10
    assert s2.factions["engineers"].coins == 999
    assert s2 is not s


def test_hex_state_shape() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    h = s.hexes["A1"]
    assert isinstance(h, HexState)
    assert h.building is None and h.owner is None
    river = next(h for h in s.hexes.values() if h.color == "white")
    assert river.building is None
