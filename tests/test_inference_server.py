"""Centralized inference: evaluator seam equivalence + server round-trip."""

from __future__ import annotations

import random

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import decision, new_game
from bgai.arena.setups import sample_setup
from bgai.training.model import ModelConfig, PolicyValueNet


def _net():
    torch.manual_seed(0)
    return PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))


class LocalFakeEvaluator:
    """Evaluator that wraps the same net in-process: results must match
    the agent's own torch path bit-for-bit."""

    def __init__(self, net):
        self.net = net

    def __call__(self, hex_np, glob_np, fac_np, cand_np, mask_np):
        with torch.no_grad():
            logits, value = self.net(
                torch.from_numpy(hex_np), torch.from_numpy(glob_np),
                torch.from_numpy(fac_np), torch.from_numpy(cand_np),
                torch.from_numpy(mask_np),
            )
        return logits.softmax(dim=-1).numpy(), value.numpy()


def test_evaluator_path_matches_local_path() -> None:
    net = _net()
    local = MCTSAgent(net=net, simulations=0)
    remote = MCTSAgent(evaluator=LocalFakeEvaluator(net), simulations=0)
    sim = new_game(sample_setup(random.Random(2)))
    pending = decision(sim)
    assert pending is not None
    faction, offer = pending
    lp, lv = local._evaluate(sim.game, faction, offer)
    rp, rv = remote._evaluate(sim.game, faction, offer)
    np.testing.assert_allclose(lp, rp, rtol=1e-5)
    np.testing.assert_allclose(lv, rv, rtol=1e-5)


def test_search_runs_torch_free_with_evaluator() -> None:
    net = _net()
    agent = MCTSAgent(evaluator=LocalFakeEvaluator(net), simulations=8, leaf_batch=4, top_k=4)
    assert agent.net is None
    sim = new_game(sample_setup(random.Random(2)))
    root = agent.search(sim)
    assert root is not None and root.total_visits >= 8


def test_server_round_trip_over_processes(tmp_path) -> None:
    """Real spawn-context server on CPU: two clients, ragged widths."""
    import torch as _torch

    from bgai.agents.inference_server import RemoteEvaluator, start_server
    from bgai.training.vocab import ENCODING_VERSION
    from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM

    net = _net()
    ckpt_path = tmp_path / "srv.pt"
    _torch.save(
        {"model": net.state_dict(), "encoding_version": ENCODING_VERSION,
         "config": {"value_simplex": False, "aux_head": False}},
        ckpt_path,
    )
    proc, req_q, resp_qs, stop = start_server(ckpt_path, "cpu", n_workers=2, max_batch=64)
    try:
        evs = [RemoteEvaluator(i, req_q, resp_qs[i]) for i in range(2)]
        rng = np.random.default_rng(0)
        for i, width in ((0, 5), (1, 9)):
            n = 3
            priors, values = evs[i](
                rng.standard_normal((n, 113, HEX_FEAT_DIM)).astype(np.float32),
                rng.standard_normal((n, GLOBAL_DIM)).astype(np.float32),
                rng.integers(0, 14, n).astype(np.int64),
                rng.integers(0, 7, (n, width, 12)).astype(np.int64),
                np.ones((n, width), dtype=bool),
            )
            assert priors.shape == (n, width) and values.shape == (n, 4)
            np.testing.assert_allclose(priors.sum(axis=1), 1.0, rtol=1e-4)
    finally:
        stop.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
