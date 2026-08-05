"""Human-regularized self-play (Phase 6b).

Scale note: a real RL run is ~10^5 games (cluster work). These tests
verify the loop is *correct* — records carry usable targets, and the KL
anchor actually restrains the policy — at a scale that runs in seconds.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.setups import sample_setup
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.encode_state import GLOBAL_DIM, HEX_FEAT_DIM
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.selfplay import SelfPlayConfig, play_game, regularized_loss


def _net() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0))


def test_self_play_game_yields_usable_targets() -> None:
    agent = MCTSAgent(net=_net(), simulations=2)
    cfg = SelfPlayConfig(simulations=2)
    records = play_game(agent, sample_setup(random.Random(1)), random.Random(1), cfg)
    assert len(records) > 50
    for r in records[:20]:
        assert r.visits.sum() > 0, "a recorded decision must have search visits"
        assert r.candidates.shape[0] == r.visits.shape[0]
        assert r.final_shares is not None and r.final_shares.shape == (4,)
        assert np.isclose(r.final_shares.sum(), 1.0, atol=1e-5)


def test_final_shares_are_mover_relative() -> None:
    """Seat 0 of a record's share vector must be that record's own mover."""
    agent = MCTSAgent(net=_net(), simulations=2)
    cfg = SelfPlayConfig(simulations=2)
    setup = sample_setup(random.Random(3))
    records = play_game(agent, setup, random.Random(3), cfg)
    by_seat = {r.seat: r.final_shares for r in records}
    assert len(by_seat) > 1
    # two movers at different seats disagree about which component is "mine"
    seats = sorted(by_seat)
    a, b = by_seat[seats[0]], by_seat[seats[1]]
    assert not np.allclose(a, b) or np.allclose(a, b[::-1])


def test_kl_anchor_restrains_the_policy() -> None:
    """With a large lambda the fine-tuned policy must stay near the frozen
    human policy even when the search target disagrees with it."""
    net, frozen = _net(), _net()
    frozen.load_state_dict(net.state_dict())
    frozen.eval()

    n, n_cand = 4, 6
    batch = {
        "hex_planes": torch.randn(n, 113, HEX_FEAT_DIM),
        "globals": torch.randn(n, GLOBAL_DIM),
        "faction": torch.randint(0, 14, (n,)),
        "candidates": torch.randint(0, 7, (n, n_cand, MOVE_FIELDS)),
        "cand_mask": torch.ones((n, n_cand), dtype=torch.bool),
        "value": torch.rand(n, 4),
        # a target that strongly disagrees with whatever the net believes
        "visits": torch.tensor([[10.0, 0, 0, 0, 0, 0]] * n),
    }

    def drift(lambda_kl: float) -> float:
        model = _net()
        model.load_state_dict(net.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(40):
            loss, _ = regularized_loss(model, frozen, batch, lambda_kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            new_logits, _ = model(
                batch["hex_planes"], batch["globals"], batch["faction"],
                batch["candidates"], batch["cand_mask"],
            )
            ref_logits, _ = frozen(
                batch["hex_planes"], batch["globals"], batch["faction"],
                batch["candidates"], batch["cand_mask"],
            )
            return float(
                torch.nn.functional.kl_div(
                    torch.log_softmax(new_logits, -1),
                    torch.log_softmax(ref_logits, -1),
                    reduction="batchmean",
                    log_target=True,
                ).item()
            )

    assert drift(lambda_kl=10.0) < drift(lambda_kl=0.0)


def test_loss_components_are_all_reported() -> None:
    net, frozen = _net(), _net()
    n, n_cand = 2, 4
    batch = {
        "hex_planes": torch.randn(n, 113, HEX_FEAT_DIM),
        "globals": torch.randn(n, GLOBAL_DIM),
        "faction": torch.randint(0, 14, (n,)),
        "candidates": torch.randint(0, 7, (n, n_cand, MOVE_FIELDS)),
        "cand_mask": torch.ones((n, n_cand), dtype=torch.bool),
        "value": torch.rand(n, 4),
        "visits": torch.rand(n, n_cand) + 0.1,
    }
    _, parts = regularized_loss(net, frozen, batch, lambda_kl=1.0)
    assert set(parts) == {"policy", "value", "kl", "total"}
    assert parts["kl"] >= -1e-6


def test_losses_are_finite_with_padded_candidates() -> None:
    """Regression: padding slots carry -inf logits and 0 soft targets, and
    0 * -inf is NaN -- which silently poisoned every self-play batch until
    the terms were masked explicitly."""
    net, frozen = _net(), _net()
    n, n_cand = 3, 6
    mask = torch.ones((n, n_cand), dtype=torch.bool)
    mask[:, 4:] = False  # two padded slots per row
    visits = torch.zeros(n, n_cand)
    visits[:, :4] = torch.rand(n, 4) + 0.1
    batch = {
        "hex_planes": torch.randn(n, 113, HEX_FEAT_DIM),
        "globals": torch.randn(n, GLOBAL_DIM),
        "faction": torch.randint(0, 14, (n,)),
        "candidates": torch.randint(0, 7, (n, n_cand, MOVE_FIELDS)),
        "cand_mask": mask,
        "value": torch.rand(n, 4),
        "visits": visits,
    }
    loss, parts = regularized_loss(net, frozen, batch, lambda_kl=1.0)
    assert torch.isfinite(loss), parts
    assert all(math.isfinite(v) for v in parts.values()), parts
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
