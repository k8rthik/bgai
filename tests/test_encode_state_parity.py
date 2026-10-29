"""The state encoder's v1 prefix must reproduce the original per-element
v1 encoder bit for bit -- the v1 checkpoints were trained on that layout and
read exactly that prefix of every later encoding, so any drift here silently
changes what they see."""

from __future__ import annotations

import random

import numpy as np
import polars as pl

from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import load_setup
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.state import GameState
from bgai.training import encode_state as es
from bgai.training.encode_state import StateEncoding, encode_state
from bgai.training.vocab import (
    BUILDING_INDEX,
    BUILDINGS,
    COLOR_INDEX,
    COLORS,
    CULT_INDEX,
    CULTS4,
    FACTION_INDEX,
    HEXES,
)


def _reference(state: GameState, faction: str) -> StateEncoding:
    """Verbatim copy of the pre-optimisation encoder."""
    seats = state.setup.factions
    pivot = seats.index(faction)
    order = seats[pivot:] + seats[:pivot]
    seat_of = {name: i for i, name in enumerate(order)}

    planes = np.zeros((len(HEXES), es.HEX_FEAT_DIM), dtype=np.int8)
    for row, hex_key in enumerate(HEXES):
        hx = state.hexes[hex_key]
        if hx.color in COLOR_INDEX:
            planes[row, COLOR_INDEX[hx.color]] = 1
        else:
            planes[row, len(COLORS)] = 1
        if hx.building is not None:
            planes[row, len(COLORS) + 1 + BUILDING_INDEX[hx.building]] = 1
        if hx.owner is not None and hx.owner in seat_of:
            planes[row, len(COLORS) + 1 + len(BUILDINGS) + seat_of[hx.owner]] = 1

    g = np.zeros(es.GLOBAL_DIM_V1, dtype=np.int16)
    active = state.turn_order[state.active_index] if state.turn_order else None
    for seat, name in enumerate(order):
        fs = state.factions[name]
        base = seat * es.SEAT_BLOCK
        g[base + FACTION_INDEX[name]] = 1
        g[base + es._RES_OFFSET : base + es._RES_OFFSET + es._N_RES] = (
            fs.coins, fs.workers, fs.priests, fs.priest_pool, fs.vp
        )
        g[base + es._BOWLS_OFFSET : base + es._BOWLS_OFFSET + 3] = (
            fs.power.bowl1, fs.power.bowl2, fs.power.bowl3
        )
        g[base + es._TRACKS_OFFSET : base + es._TRACKS_OFFSET + 2] = (fs.shipping, fs.dig_level)
        g[base + es._SPADES_OFFSET : base + es._SPADES_OFFSET + 2] = (
            fs.spades_available, fs.extra_actions
        )
        g[base + es._FLAGS_OFFSET] = int(fs.passed)
        g[base + es._FLAGS_OFFSET + 1] = int(name == active)
        for cult in CULTS4:
            g[base + es._CULTS_OFFSET + CULT_INDEX[cult]] = state.cults[name][cult]
        g[base + es._KEYS_OFFSET] = fs.keys
        g[base + es._KEYS_OFFSET + 1] = fs.bridges_built
        if fs.bonus is not None:
            g[base + es._BON_OFFSET + es._BON_INDEX[fs.bonus]] = 1
        for fav in fs.favors:
            g[base + es._FAV_OFFSET + es._FAV_INDEX[fav]] += 1
        g[base + es._TOWNS_OFFSET] = len(fs.towns)

    g[es._CTX_ROUND + state.round] = 1
    g[es._CTX_PHASE + es._PHASE_INDEX[state.phase]] = 1
    for r, tile in enumerate(state.setup.score_tiles):
        base = es._CTX_SCORE + r * es._SCORE_TILE_DIM
        if tile.cult in CULT_INDEX:
            g[base + CULT_INDEX[tile.cult]] = 1
        else:
            g[base + len(CULTS4)] = 1
        g[base + len(CULTS4) + 1] = tile.req
    for bon in state.setup.bonus_tiles:
        g[es._CTX_BON_POOL + es._BON_INDEX[bon]] = 1
    for act in state.power_actions_taken:
        if act.startswith("ACT") and act[3:].isdigit():
            g[es._CTX_ACTS + int(act[3:]) - 1] = 1
    g[es._CTX_PENDING] = len(state.pending)
    return StateEncoding(hex_planes=planes, globals=g)


def test_encoder_matches_reference_along_a_random_game() -> None:
    rng = random.Random(11)
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    checked = 0
    for _ in range(400):
        pending = decision(sim)
        if pending is None:
            break
        faction, offer = pending
        for mover in sim.game.setup.factions:
            got = encode_state(sim.game, mover)
            want = _reference(sim.game, mover)
            assert got.hex_planes.dtype == want.hex_planes.dtype
            assert got.globals.dtype == want.globals.dtype
            assert np.array_equal(got.hex_planes, want.hex_planes)
            assert np.array_equal(got.globals[: es.GLOBAL_DIM_V1], want.globals)
            checked += 1
        sim = advance(sim, rng.choice(offer), current=pending)
    assert checked > 400


def test_v1_prefix_matches_reference_on_replayed_corpus_games() -> None:
    moves = pl.read_parquet("data/datasets/moves.parquet")
    deltas = pl.read_parquet("data/datasets/deltas.parquet")
    checked = 0
    for game_id in ("4pLeague_S10_D1L1_G1", "4pLeague_S12_D1L1_G1"):
        states: list[GameState] = []
        result = replay_game(
            game_id, moves, deltas, on_decision=lambda s, f, c, out=states: out.append(s)
        )
        assert result.error is None
        for state in states[::7]:
            for mover in state.setup.factions:
                got = encode_state(state, mover)
                want = _reference(state, mover)
                assert got.globals.shape == (es.GLOBAL_DIM,)
                assert got.globals[: es.GLOBAL_DIM_V1].tobytes() == want.globals.tobytes()
                assert got.hex_planes.tobytes() == want.hex_planes.tobytes()
                checked += 1
    assert checked > 150
