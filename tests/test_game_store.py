"""Agent game store (knowledge/game_store.py): corpus-schema round trip."""

from __future__ import annotations

import json
import random

import polars as pl
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.setups import sample_setup
from bgai.knowledge.game_store import append_batch, traced_to_tables
from bgai.knowledge.vp_decompose import trace_game
from bgai.training.model import ModelConfig, PolicyValueNet


def _traced():
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    agent = MCTSAgent(net=net, simulations=2)
    rng = random.Random(11)
    setup = sample_setup(rng)
    return trace_game(setup, {f: agent for f in setup.factions}, rng)


def test_tables_conserve_vp_and_match_corpus_schema(tmp_path) -> None:
    traced = _traced()
    assert traced.error is None
    moves, deltas, meta = traced_to_tables(traced, "agent_test_0000")

    corpus_moves = pl.scan_parquet("data/datasets/moves.parquet").collect_schema()
    assert dict(moves.schema) == dict(corpus_moves)
    corpus_deltas = pl.scan_parquet("data/datasets/deltas.parquet").collect_schema()
    assert dict(deltas.schema) == dict(corpus_deltas)

    # conservation: per-faction vp_delta sums to final - 20
    final = json.loads(meta["final_vp"][0])
    sums = deltas.group_by("faction").agg(pl.col("vp_delta").sum())
    for r in sums.iter_rows(named=True):
        assert r["vp_delta"] == final[r["faction"]] - 20, r

    append_batch([(traced, "agent_test_0000")], store=tmp_path, batch_name="t")
    back = pl.read_parquet(tmp_path / "moves" / "t.parquet")
    assert len(back) == len(moves)
