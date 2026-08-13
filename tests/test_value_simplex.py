"""Value head on the simplex.

The value target is a 4-seat share vector summing to 1, but the head is
a bare Linear(...,4) emitting unconstrained reals -- MSE was the only
thing holding it near share space. Adding the ranking loss removed that
implicit constraint and value MSE rose 128x, which also invalidates
MCTS: _select falls back to q=0.25 and weighs c_puct against Q on the
assumption that values live in share space.

Softmax makes the constraint explicit. It is monotone, so it preserves
the ordering the ranking loss buys.
"""

from __future__ import annotations

import torch

from bgai.training.model import ModelConfig, PolicyValueNet


def _net(**kw) -> PolicyValueNet:
    return PolicyValueNet(ModelConfig(**kw))


def _forward(net):
    b = 3
    return net(
        torch.zeros(b, 113, 17),
        torch.zeros(b, 294),
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, 5, 12, dtype=torch.long),
        torch.ones(b, 5, dtype=torch.bool),
    )[1]


def test_simplex_head_sums_to_one_and_is_nonnegative() -> None:
    v = _forward(_net(value_simplex=True))
    assert torch.allclose(v.sum(dim=-1), torch.ones(v.shape[0]), atol=1e-5)
    assert (v >= 0).all()


def test_default_stays_unconstrained_for_existing_checkpoints() -> None:
    v = _forward(_net())
    assert not torch.allclose(v.sum(dim=-1), torch.ones(v.shape[0]), atol=1e-3)


def test_softmax_preserves_ordering() -> None:
    """The whole point: bounding the scale must not undo the ranking."""
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0], [0.0, 1.0, -2.0, 0.25]])
    shares = torch.softmax(logits, dim=-1)
    assert torch.equal(logits.argsort(dim=-1), shares.argsort(dim=-1))
