"""Opt-in transposition tables (docs/decisions.md C18): exact when on,
absent when off."""

from __future__ import annotations

import random

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena import driver
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import load_setup
from bgai.training.model import ModelConfig, PolicyValueNet


def _net() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))


def test_decision_table_is_off_by_default_and_exact_when_on() -> None:
    assert driver._DECISION_TABLE is None
    rng = random.Random(2)
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    walk = []
    for _ in range(80):
        pending = decision(sim)
        walk.append((sim, pending))
        sim = advance(sim, rng.choice(pending[1]), current=pending)
    driver.enable_decision_table(max_entries=50)
    try:
        for s, pending in walk:
            assert decision(s) == pending  # miss path
            assert decision(s) == pending  # hit path (or evicted: still equal)
        assert driver._DECISION_TABLE is not None and len(driver._DECISION_TABLE) <= 50
    finally:
        driver.disable_decision_table()
    assert driver._DECISION_TABLE is None


def test_eval_table_reproduces_the_untabled_search() -> None:
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    plain = MCTSAgent(net=_net(), simulations=48, leaf_batch=8, top_k=6)
    tabled = MCTSAgent(net=_net(), simulations=48, leaf_batch=8, top_k=6, eval_table=10_000)
    assert plain._eval_table is None and tabled._eval_table is not None
    root_a = plain.search(sim)
    root_b = tabled.search(sim)
    assert root_a is not None and root_b is not None
    assert np.array_equal(root_a.visits, root_b.visits)
    # a second, identical search on the tabled agent is served from the table
    tabled.reset_tree()
    before = len(tabled._eval_table)
    root_c = tabled.search(sim)
    assert root_c is not None and np.array_equal(root_c.visits, root_b.visits)
    assert len(tabled._eval_table) == before
