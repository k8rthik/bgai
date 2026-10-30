"""State encoder (plan 2026-08-04-imitation-phase5, Task 3)."""

from __future__ import annotations

import functools
import json
from dataclasses import replace

import numpy as np
import polars as pl

from bgai.data.ledger_parser import ParsedCommand
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


@functools.cache
def _corpus() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
    )


@functools.cache
def _decisions(game_id: str) -> tuple[tuple[GameState, str, ParsedCommand], ...]:
    """Every decision of a replayed corpus game: (state, faction, command)."""
    found: list[tuple[GameState, str, ParsedCommand]] = []
    result = replay_game(
        game_id, *_corpus(), on_decision=lambda s, f, c: found.append((s, f, c))
    )
    assert result.error is None and not result.mismatches
    return tuple(found)


def test_economy_on_a_replayed_round_one_state() -> None:
    state = next(s for s, _, _ in _decisions("4pLeague_S10_D1L1_G1") if s.round == 1)
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
    favors = slice(es._CTX_FAVOR_POOL, es.GLOBAL_DIM_V2)
    assert before[towns].tolist() == [2, 2, 2, 2, 2, 1, 2, 1]  # TW1..TW8
    assert (before[towns] - after[towns]).tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    fav_ids = sorted(state.favors_pool)
    assert before[favors].tolist() == [state.favors_pool[f] for f in fav_ids]
    assert (before[favors] - after[favors]).tolist() == [
        3 if f == "FAV5" else 0 for f in fav_ids
    ]


# -- scoring block (encoding v3) ----------------------------------------------


def _scoring(globals_: np.ndarray, seat: int) -> dict[str, list[int]]:
    block = globals_[es._SCORE_BASE + seat * es.SCORE_SEAT_BLOCK :][: es.SCORE_SEAT_BLOCK].tolist()
    return {
        # network size, network VP, cult VP, resource VP
        "projection": block[: es._SCORE_CULT_RANK],
        "cult_rank": block[es._SCORE_CULT_RANK : es._SCORE_CULT_STEPS],
        "cult_steps": block[es._SCORE_CULT_STEPS : es._SCORE_CONTACT],
        # dwellings touching an opponent, leech power offered to / by the seat
        "contact": block[es._SCORE_CONTACT :],
    }


def test_scoring_block_on_a_replayed_round_four_state() -> None:
    state = next(s for s, _, _ in _decisions("4pLeague_S10_D1L1_G1") if s.round == 4)
    assert state.setup.factions == ("engineers", "darklings", "nomads", "mermaids")
    assert not state.bridges and all(v is None for v in state.cult_10.values())
    blocks = [_scoring(encode_state(state, "engineers").globals, seat) for seat in range(4)]
    # cults F/W/E/A: engineers 0/0/5/0, darklings 2/4/5/0, nomads 4/0/1/0,
    # mermaids 0/2/3/0, nobody holds a key. EARTH: engineers and darklings
    # tie at 5 and split 8+4; mermaids third. Networks 4/4/3/5: mermaids
    # 18, the tied 4s split 12+6, nomads nothing. Resources burn bowl2/2,
    # then bowl3, priests and workers become coins at 3 per VP:
    # engineers 9 C + 5 PW + 1 P + 4 W = 19 -> 6.
    assert [b["projection"] for b in blocks] == [
        [4, 9, 6, 6], [4, 9, 18, 9], [3, 0, 8, 6], [5, 18, 6, 7]
    ]
    assert [b["cult_rank"] for b in blocks] == [
        [3, 3, 1, 3], [1, 0, 1, 3], [0, 3, 3, 3], [3, 1, 2, 3]
    ]
    assert [b["cult_steps"] for b in blocks] == [
        [3, 3, 2, 3], [1, 1, 2, 3], [1, 3, 2, 3], [3, 1, 2, 3]
    ]
    # engineers: D4 and F4 each touch two opponent dwellings; TP E9 touches
    # darklings' SA and nomads' TP; TE E7 touches a mermaids D; TE F6
    # touches SA, TP and two D. Offered to them: 1*2 + 1*2 + 2*2 + 2*1 + 2*4;
    # offered by them: 2 + 2 + 5 + 1 + 7.
    assert [b["contact"] for b in blocks] == [
        [2, 18, 17], [2, 10, 12], [3, 21, 16], [3, 11, 15]
    ]


def _with_cults(state: GameState, positions: dict[str, dict[str, int]]) -> GameState:
    cults = {name: {**state.cults[name], **positions.get(name, {})} for name in state.cults}
    return replace(state, cults=cults)


def _with_keys(state: GameState, faction: str, keys: int) -> GameState:
    fs = replace(state.factions[faction], keys=keys)
    return replace(state, factions={**state.factions, faction: fs})


def test_cult_steps_need_a_key_and_an_unclaimed_top_space() -> None:
    state = _with_cults(
        _blank(), {"darklings": {"FIRE": 8, "WATER": 9, "EARTH": 10, "AIR": 6}}
    )
    keyless = _scoring(encode_state(state, "darklings").globals, 0)["cult_steps"]
    assert keyless == [0, 0, 0, 1]  # 8 and 9 stop at 9 without a key; AIR 6 -> 7
    keyed = _with_keys(state, "darklings", 1)
    assert _scoring(encode_state(keyed, "darklings").globals, 0)["cult_steps"] == [2, 1, 0, 1]
    claimed = replace(
        _with_cults(keyed, {"engineers": {"FIRE": 10}}),
        cult_10={**keyed.cult_10, "FIRE": "engineers"},
    )
    assert _scoring(encode_state(claimed, "darklings").globals, 0)["cult_steps"] == [0, 1, 0, 1]


def test_cult_rank_counts_seats_at_or_above_and_ties_share_no_top() -> None:
    blank = _blank()
    fire_only = {
        name: {cult: 4 if cult == "FIRE" and name in ("darklings", "engineers") else 0
               for cult in blank.cults[name]}
        for name in blank.cults
    }
    state = _with_cults(blank, fire_only)
    fire = es.CULT_INDEX["FIRE"]
    darklings = _scoring(encode_state(state, "darklings").globals, 0)
    engineers = _scoring(encode_state(state, "engineers").globals, 0)
    assert darklings["cult_rank"][fire] == engineers["cult_rank"][fire] == 1
    assert darklings["projection"][2] == engineers["projection"][2] == (8 + 4) // 2
    ahead = _with_cults(state, {"darklings": {"FIRE": 5}})
    assert _scoring(encode_state(ahead, "darklings").globals, 0)["cult_rank"][fire] == 0
    assert _scoring(encode_state(ahead, "engineers").globals, 0)["cult_rank"][fire] == 1


def _non_adjacent_land_pair() -> tuple[str, str]:
    board = base_board()
    a = _land_chain(1)[0]
    b = next(h for h in sorted(board.land_hexes()) if h != a and h not in board.adjacent[a])
    return a, b


def test_contact_counts_bridges_as_adjacency() -> None:
    a, b = _non_adjacent_land_pair()
    state = _place(_place(_blank(), "darklings", a, "D"), "engineers", b, "TP")
    apart = encode_state(state, "darklings").globals
    assert _scoring(apart, 0)["contact"] == [0, 0, 0]
    bridged = encode_state(replace(state, bridges=frozenset({frozenset({a, b})})), "darklings")
    # darklings' D touches a TP: cheap TP upgrade, offered 1 (its D's
    # power) when the TP upgrades, offers 2 when the D upgrades
    assert _scoring(bridged.globals, 0)["contact"] == [1, 1, 2]
    engineers = es._seat_order(state, "darklings").index("engineers")
    assert _scoring(bridged.globals, engineers)["contact"] == [0, 2, 1]


def test_scoring_blocks_follow_the_mover_relative_seat_order() -> None:
    a, b = _land_chain(2)
    state = _place(_place(_blank(), "darklings", a, "D"), "engineers", b, "D")
    own = _scoring(encode_state(state, "darklings").globals, 0)
    assert own == _scoring(encode_state(state, "engineers").globals, 1)
    assert own["contact"] == [1, 1, 1]


def _final_scoring_rows(game_id: str) -> dict[str, dict[str, int]]:
    """Per faction, the ledger's own final scoring: network and cult VP from
    its score_vp rows, resource VP from the score_resources rows' deltas."""
    moves, deltas = (df.filter(pl.col("game_id") == game_id) for df in _corpus())
    out: dict[str, dict[str, int]] = {}
    for row in moves.filter(pl.col("verb") == "score_vp").iter_rows(named=True):
        kind = "network" if row["reason"] == "NETWORK" else "cult"
        out.setdefault(row["faction"], {}).setdefault(kind, 0)
        out[row["faction"]][kind] += row["n1"]
    resources = moves.filter(pl.col("verb") == "score_resources").select("row", "faction")
    for row in resources.join(deltas, on=["row", "faction"]).iter_rows(named=True):
        out.setdefault(row["faction"], {})["resources"] = row["vp_delta"]
    return out


def test_projection_matches_the_ledgers_final_scoring() -> None:
    meta = pl.read_parquet("data/datasets/games_meta.parquet")
    # tunnelling + carpets + the alchemists' 2 C/VP rate; mermaids + carpets
    for game_id in ("00009", "4pLeague_S25_D7L39_G2"):
        state, faction, last_pass = _decisions(game_id)[-1]
        final = apply(state, faction, last_pass)
        assert final.round == 6 and all(fs.passed for fs in final.factions.values())
        ledger = _final_scoring_rows(game_id)
        final_vp = json.loads(meta.filter(pl.col("game_id") == game_id)["final_vp"][0])
        g = encode_state(final, final.setup.factions[0]).globals
        for seat, name in enumerate(final.setup.factions):
            _, network_vp, cult_vp, resource_vp = _scoring(g, seat)["projection"]
            want = ledger.get(name, {})
            assert (network_vp, cult_vp, resource_vp) == (
                want.get("network", 0), want.get("cult", 0), want.get("resources", 0)
            ), (game_id, name)
            assert final.factions[name].vp + network_vp + cult_vp + resource_vp == final_vp[name]
