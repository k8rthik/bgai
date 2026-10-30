"""State encoder (plan 2026-08-04-imitation-phase5, Task 3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from bgai.engine.tm.apply import apply
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.legal import legal_moves
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, active_faction
from bgai.engine.tm.terraform import spade_distance
from bgai.training import encode_state as es
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
    from bgai.training.encode_state import _RES_OFFSET, SEAT_BLOCK

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


# -- economy block (encoding v2) ---------------------------------------------


def _economy(globals_: np.ndarray, seat: int) -> dict[str, list[int]]:
    block = globals_[es._ECON_BASE + seat * es.ECON_SEAT_BLOCK :][: es.ECON_SEAT_BLOCK].tolist()
    return {
        "income": block[es._ECON_INCOME : es._ECON_TOWN],
        "town": block[es._ECON_TOWN : es._ECON_SUPPLY],
        "supply": block[es._ECON_SUPPLY : es._ECON_TARGETS],
        "targets": block[es._ECON_TARGETS :],
    }


def _place(state: GameState, faction: str, hex_key: str, building: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _land_chain(n: int) -> tuple[str, ...]:
    board = base_board()
    for start in board.land_hexes():
        chain = [start]
        while len(chain) < n:
            nxt = sorted(
                h for h in board.adjacent[chain[-1]]
                if board.hexes[h].color != RIVER and h not in chain
            )
            if not nxt:
                break
            chain.append(nxt[0])
        if len(chain) == n:
            return tuple(chain)
    raise AssertionError("no land chain")


def _blank() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))


def test_economy_on_a_replayed_round_one_state() -> None:
    states: list[GameState] = []
    replay_game(
        "4pLeague_S10_D1L1_G1",
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
        on_decision=lambda s, f, c: states.append(s),
    )
    state = next(s for s in states if s.round == 1)
    assert state.setup.factions == ("engineers", "darklings", "nomads", "mermaids")
    econ = [_economy(encode_state(state, "engineers").globals, seat) for seat in range(4)]
    # [C, W, P, PW]: dwelling track at 2 dwellings (nomads 3) + the setup bonus tile
    assert econ[0]["income"] == [0, 2, 0, 3]  # engineers' D track 0,1,2 | BON4 3 PW
    assert econ[1]["income"] == [6, 3, 0, 0]  # D track 1,2,3 | BON3 6 C
    assert econ[2]["income"] == [0, 5, 0, 3]  # D track at 3 is 4 W | BON5 1 W, 3 PW
    assert econ[3]["income"] == [2, 3, 0, 0]  # BON1 2 C
    assert [e["supply"][0] for e in econ] == [6, 6, 5, 6]
    assert all(e["supply"][1:] == [4, 3, 1, 1] for e in econ)
    # setup dwellings are lone one-power clusters; base threshold 7
    assert all(e["town"] == [1, 1, 7, 0] for e in econ)
    assert all(sum(e["targets"]) > 0 for e in econ)


def test_building_raises_income_and_lowers_supply() -> None:
    a, b = _land_chain(2)
    state = _blank()
    before = _economy(encode_state(state, "darklings").globals, 0)
    assert before["income"] == [0, 1, 0, 0]  # the dwelling track's 1 W floor
    assert before["supply"] == [8, 4, 3, 1, 1]
    one = _place(state, "darklings", a, "D")
    after_d = _economy(encode_state(one, "darklings").globals, 0)
    assert after_d["income"] == [0, 2, 0, 0]
    assert after_d["supply"] == [7, 4, 3, 1, 1]
    two = _place(one, "darklings", b, "TP")
    after_tp = _economy(encode_state(two, "darklings").globals, 0)
    assert after_tp["income"] == [2, 2, 0, 1]  # first trading post: 2 C, 1 PW
    assert after_tp["supply"] == [7, 3, 3, 1, 1]


def test_economy_blocks_follow_the_mover_relative_seat_order() -> None:
    state = _place(_blank(), "darklings", _land_chain(1)[0], "D")
    own = _economy(encode_state(state, "darklings").globals, 0)
    assert own == _economy(encode_state(state, "engineers").globals, 1)
    assert own != _economy(encode_state(state, "engineers").globals, 0)


def test_town_progress_counts_open_clusters_and_excludes_founded_towns() -> None:
    chain = _land_chain(4)
    state = _blank()
    for hex_key, building in zip(chain, ["D", "TP", "TP", "TE"], strict=True):
        state = _place(state, "darklings", hex_key, building)
    town = _economy(encode_state(state, "darklings").globals, 0)["town"]
    assert town == [7, 4, 7, 1]  # power 1+2+2+2, four structures, threshold 7, one ready
    founded = replace(
        state, founded_towns={**state.founded_towns, "darklings": (frozenset(chain),)}
    )
    assert _economy(encode_state(founded, "darklings").globals, 0)["town"] == [0, 0, 7, 0]
    fs = replace(state.factions["darklings"], favors=("FAV5",))
    fav5 = replace(state, factions={**state.factions, "darklings": fs})
    assert _economy(encode_state(fav5, "darklings").globals, 0)["town"] == [7, 4, 6, 1]


def test_sanctuary_counts_as_two_structures() -> None:
    a, b = _land_chain(2)
    state = _place(_place(_blank(), "darklings", a, "SA"), "darklings", b, "D")
    assert _economy(encode_state(state, "darklings").globals, 0)["town"] == [4, 3, 7, 0]


def test_expansion_targets_bucket_unbuilt_neighbours_by_spade_distance() -> None:
    board = base_board()
    home = _land_chain(1)[0]
    state = _place(_blank(), "darklings", home, "D")  # no shipping, no tunnelling
    neighbours = sorted(h for h in board.adjacent[home] if board.hexes[h].color != RIVER)
    want = [0, 0, 0, 0]
    for h in neighbours:
        want[spade_distance(board.hexes[h].color, "black")] += 1
    assert _economy(encode_state(state, "darklings").globals, 0)["targets"] == want
    blocked = _place(state, "engineers", neighbours[0], "D")
    want[spade_distance(board.hexes[neighbours[0]].color, "black")] -= 1
    assert _economy(encode_state(blocked, "darklings").globals, 0)["targets"] == want


def test_tile_pools_count_what_is_left() -> None:
    state = _blank()
    taken = replace(
        state,
        towns_pool={**state.towns_pool, "TW3": 1},
        favors_pool={**state.favors_pool, "FAV5": 0},
    )
    before, after = (encode_state(s, "darklings").globals for s in (state, taken))
    towns = slice(es._CTX_TOWN_POOL, es._CTX_FAVOR_POOL)
    favors = slice(es._CTX_FAVOR_POOL, GLOBAL_DIM)
    assert before[towns].tolist() == [2, 2, 2, 2, 2, 1, 2, 1]  # TW1..TW8
    assert (before[towns] - after[towns]).tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    fav_ids = sorted(state.favors_pool)
    assert before[favors].tolist() == [state.favors_pool[f] for f in fav_ids]
    assert (before[favors] - after[favors]).tolist() == [
        3 if f == "FAV5" else 0 for f in fav_ids
    ]
