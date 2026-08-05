"""Move featurizer (plan 2026-08-04-imitation-phase5, Task 4)."""

from __future__ import annotations

import numpy as np
import polars as pl

from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState
from bgai.training.encode_move import MOVE_FIELDS, encode_move


def _state():
    return start_setup(GameState.initial(load_setup("4pLeague_S10_D1L1_G1")))


def test_shape_null_handling_and_determinism() -> None:
    state = _state()
    a = encode_move(cmd("pass", tile="BON1"), state, "engineers")
    assert a.shape == (MOVE_FIELDS,) and a.dtype == np.int16
    assert np.array_equal(a, encode_move(cmd("pass", tile="BON1"), state, "engineers"))
    bare = encode_move(cmd("wait"), state, "engineers")
    # every categorical except verb is the reserved null id 0
    assert bare[0] > 0 and (bare[1:] == 0).all()


def test_grey_color_alias_and_relative_target() -> None:
    state = _state()
    a = encode_move(cmd("transform", loc="A1", color="grey"), state, "engineers")
    b = encode_move(cmd("transform", loc="A1", color="gray"), state, "engineers")
    assert np.array_equal(a, b)
    # leech target encodes as mover-relative seat, so the same command
    # encodes differently for different movers
    seats = state.setup.factions
    t = cmd("leech", n1=1, target=seats[0])
    x = encode_move(t, state, seats[1])
    y = encode_move(t, state, seats[2])
    assert not np.array_equal(x, y)


def test_every_corpus_decision_command_encodes() -> None:
    moves_df = pl.read_parquet("data/datasets/moves.parquet")
    deltas_df = pl.read_parquet("data/datasets/deltas.parquet")
    encoded = []

    def hook(state, faction, command):
        encoded.append(encode_move(command, state, faction))

    result = replay_game("4pLeague_S10_D1L1_G1", moves_df, deltas_df, on_decision=hook)
    assert result.error is None
    assert len(encoded) > 100
