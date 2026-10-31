"""Root exploration for self-play: Dirichlet noise on the root priors and
root-only prior pruning, both off by default."""

from __future__ import annotations

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import decision, new_game
from bgai.engine.tm.setup import load_setup
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.selfplay import SelfPlayConfig, search_kwargs
from bgai.training.selfplay_train import config_from_argv


def _agent(**kwargs) -> MCTSAgent:
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))
    return MCTSAgent(net=net, c_puct=2.5, simulations=48, top_k=2, max_depth=8, **kwargs)


def _wide_sim():
    """A position with more legal moves than top_k=2."""
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    pending = decision(sim)
    assert pending is not None and len(pending[1]) > 4
    return sim


def test_default_search_keeps_raw_priors_and_top_k() -> None:
    agent = _agent()
    sim = _wide_sim()
    fresh, _ = agent._expand(sim)
    root = agent.search(sim)
    np.testing.assert_allclose(root.priors, fresh.priors)
    assert np.count_nonzero(root.visits) <= 2


def test_noise_mixes_root_priors_only() -> None:
    agent = _agent(root_noise=0.25, root_alpha=0.3)
    sim = _wide_sim()
    fresh, _ = agent._expand(sim)
    root = agent.search(sim)
    assert not np.allclose(root.priors, fresh.priors)
    assert abs(float(root.priors.sum()) - 1.0) < 1e-4
    assert root.priors.min() >= 0.75 * fresh.priors.min() - 1e-6
    for child in root.children.values():
        if child is not None:
            assert not child.noised


def test_open_root_searches_beyond_top_k() -> None:
    agent = _agent(root_top_k=0)
    root = agent.search(_wide_sim())
    assert len(root.allowed_list) == len(root.offer)
    assert np.count_nonzero(root.visits) > 2


def test_cli_threads_root_exploration_to_the_agent(tmp_path) -> None:
    cfg = config_from_argv([
        "--checkpoint", str(tmp_path / "c.pt"), "--out", str(tmp_path / "o"),
        "--root-noise", "0.25", "--root-alpha", "0.15", "--root-top-k", "0",
    ])
    kwargs = search_kwargs(cfg.selfplay())
    assert (kwargs["root_noise"], kwargs["root_alpha"], kwargs["root_top_k"]) == (0.25, 0.15, 0)
    default = search_kwargs(SelfPlayConfig())
    assert (default["root_noise"], default["root_top_k"]) == (0.0, -1)
