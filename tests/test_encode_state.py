"""State encoder (plan 2026-08-04-imitation-phase5, Task 3)."""

from __future__ import annotations

import numpy as np

from bgai.engine.tm.apply import apply
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.legal import legal_moves
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, active_faction
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM, encode_state
from bgai.training.vocab import HEX_INDEX


def _initial():
    setup = load_setup("4pLeague_S10_D1L1_G1")
    return setup, start_setup(GameState.initial(setup))


def test_shapes_dtypes_and_determinism() -> None:
    _, state = _initial()
    mover = active_faction(state)
    a = encode_state(state, mover)
    b = encode_state(state, mover)
    assert a.hex_planes.shape == (113, HEX_FEAT_DIM) and a.hex_planes.dtype == np.int8
    assert a.globals.shape == (GLOBAL_DIM,) and a.globals.dtype == np.int16
    assert np.array_equal(a.hex_planes, b.hex_planes)
    assert np.array_equal(a.globals, b.globals)


def test_mover_seat_block_holds_own_resources() -> None:
    _, state = _initial()
    mover = active_faction(state)
    enc = encode_state(state, mover)
    data = FACTIONS[mover]
    # seat block 0 is the mover; documented offsets in encode_state
    from bgai.training.encode_state import SEAT_BLOCK, _RES_OFFSET

    block = enc.globals[:SEAT_BLOCK]
    assert block[_RES_OFFSET + 0] == data.coins
    assert block[_RES_OFFSET + 1] == data.workers
    assert block[_RES_OFFSET + 2] == data.priests


def test_encoding_is_mover_relative() -> None:
    setup, state = _initial()
    a = encode_state(state, setup.factions[0])
    b = encode_state(state, setup.factions[1])
    assert not np.array_equal(a.globals, b.globals)


def test_setup_build_flips_exactly_one_hex_row() -> None:
    _, state = _initial()
    mover = active_faction(state)
    before = encode_state(state, mover)
    move = sorted(
        (m for m in legal_moves(state) if m.verb == "build"), key=lambda m: m.loc or ""
    )[0]
    after = encode_state(apply(state, mover, move), mover)
    diff_rows = np.flatnonzero((before.hex_planes != after.hex_planes).any(axis=1))
    assert diff_rows.tolist() == [HEX_INDEX[move.loc]]
