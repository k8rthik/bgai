"""Town-progress shaping of MCTS leaf values (towns diagnosis, 2026-10-28):
bots found towns as often as humans once a finisher is legal, but reach
finishable positions half as often -- shaping rewards building toward one."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bgai.agents.town_shaping import shape, town_progress
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState

BOARD = base_board()


def _fresh() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))


def _place(state: GameState, faction: str, hex_key: str, building: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _chain(n: int) -> tuple[str, ...]:
    for start in BOARD.land_hexes():
        seen, stack = [start], [start]
        while stack and len(seen) < n:
            node = stack.pop()
            for nb in sorted(BOARD.adjacent[node]):
                if BOARD.hexes[nb].color != RIVER and nb not in seen:
                    seen.append(nb)
                    stack.append(nb)
                    if len(seen) >= n:
                        break
        if len(seen) >= n:
            return tuple(seen[:n])
    raise AssertionError("no chain")


def _with_cluster(buildings: list[str]) -> tuple[GameState, str]:
    state = _fresh()
    faction = state.setup.factions[0]
    for hex_key, building in zip(_chain(len(buildings)), buildings, strict=True):
        state = _place(state, faction, hex_key, building)
    return state, faction


def test_no_buildings_no_progress() -> None:
    state = _fresh()
    assert town_progress(state, state.setup.factions[0]) == 0.0


def test_progress_grows_with_power_and_structures_and_caps_at_one() -> None:
    p2 = town_progress(*_with_cluster(["D", "D"]))
    p4 = town_progress(*_with_cluster(["D", "D", "D", "D"]))
    p6 = town_progress(*_with_cluster(["D", "D", "TP", "TP"]))
    full = town_progress(*_with_cluster(["D", "TP", "TP", "TE", "D"]))
    assert 0 < p2 < p4 < p6 < 1.0
    assert p4 == pytest.approx(4 / 7)  # 4 structures, power 4 of 7
    assert full == 1.0


def test_founding_a_town_never_lowers_the_potential() -> None:
    # regression: the first version scored only open clusters, so founding
    # dropped the bonus ~1 -> 0 and search saw a finished town as a loss
    state, faction = _with_cluster(["D", "TP", "TP", "TE"])
    before = town_progress(state, faction)
    founded = frozenset(_chain(4))
    after_state = replace(state, founded_towns={**state.founded_towns, faction: (founded,)})
    assert town_progress(after_state, faction) == 1.0  # one town, no open cluster
    assert town_progress(after_state, faction) >= before


def test_open_progress_stops_once_town_tiles_run_out_but_towns_still_count() -> None:
    state, faction = _with_cluster(["D", "D", "TP"])
    empty = replace(state, towns_pool={k: 0 for k in state.towns_pool})
    assert town_progress(empty, faction) == 0.0
    founded = replace(empty, founded_towns={**empty.founded_towns, faction: (frozenset(),)})
    assert town_progress(founded, faction) == 1.0


def test_shape_is_identity_at_zero_and_keeps_a_share_vector() -> None:
    state, faction = _with_cluster(["D", "D", "TP", "TP"])
    v = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(shape(v, state, 0.0), v)
    out = shape(v, state, 0.05)
    assert out.sum() == pytest.approx(1.0)
    seat = state.setup.factions.index(faction)
    assert out[seat] > 0.25 and all(out[i] < 0.25 for i in range(4) if i != seat)


def test_agent_and_arena_accept_the_strength() -> None:
    from bgai.agents.mcts import MCTSAgent
    from bgai.training.model import ModelConfig, PolicyValueNet

    net = PolicyValueNet(ModelConfig(hidden=32, embed=16, move_hidden=16, value_simplex=True))
    assert MCTSAgent(net=net, town_shaping=0.02).town_shaping == 0.02
    assert MCTSAgent(net=net).town_shaping == 0.0
