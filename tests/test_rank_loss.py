"""Pairwise ranking loss for the value head.

The head is trained with MSE on VP share -- a magnitude objective --
but MCTS only ever uses the value vector to *order* positions. Measured
on 102k val records, share-MSE of 0.00065 coexists with 30% of seat
pairs ordered backwards, which is the noise search backs up the tree.
This loss penalises the ordering directly.
"""

from __future__ import annotations

import torch

from bgai.training.rank_loss import pairwise_rank_loss


def test_loss_decays_towards_zero_as_the_correct_margin_grows() -> None:
    """Softplus never reaches 0 at a finite gap, so the property to pin is
    the decay, not an absolute floor."""
    target = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    narrow = pairwise_rank_loss(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), target)
    wide = pairwise_rank_loss(torch.tensor([[40.0, 30.0, 20.0, 10.0]]), target)
    assert wide.item() < narrow.item()
    assert wide.item() < 1e-3


def test_reversed_ordering_is_penalised() -> None:
    target = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    good = pairwise_rank_loss(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), target)
    bad = pairwise_rank_loss(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), target)
    assert bad.item() > good.item()
    # the wrong ordering must cost enough to dominate the gradient
    assert bad.item() > 5 * good.item()


def test_ties_in_the_target_are_ignored() -> None:
    """Seats that finished level carry no ordering information, so any
    predicted order between them must be free."""
    target = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    a = pairwise_rank_loss(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), target)
    b = pairwise_rank_loss(torch.tensor([[4.0, 3.0, 2.0, 1.0]]), target)
    assert a.item() == b.item() == 0.0


def test_margin_matters_not_just_sign() -> None:
    """A correct but barely-separated ordering should score worse than a
    confident one -- search needs the gap, not just the sign."""
    target = torch.tensor([[0.4, 0.1, 0.3, 0.2]])
    confident = pairwise_rank_loss(torch.tensor([[3.0, 0.0, 2.0, 1.0]]), target)
    marginal = pairwise_rank_loss(torch.tensor([[0.3, 0.0, 0.2, 0.1]]), target)
    assert marginal.item() > confident.item()


def test_is_differentiable() -> None:
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    pairwise_rank_loss(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_batches_average_independently() -> None:
    target = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])
    pred = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    both = pairwise_rank_loss(pred, target)
    first = pairwise_rank_loss(pred[:1], target[:1])
    second = pairwise_rank_loss(pred[1:], target[1:])
    assert torch.allclose(both, (first + second) / 2, atol=1e-5)


def test_value_ordering_metrics() -> None:
    """evaluate() must report ordering, not just MSE. With rank_weight on,
    MSE rises by design -- so a run that selects on loss alone stops early
    and saves the worse-ordering checkpoint, which is what happened on the
    first rank fine-tune."""
    from bgai.training.rank_loss import ordering_stats

    pred = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])
    stats = ordering_stats(pred, target)
    assert stats["winner_correct"] == 1  # first row only
    assert stats["pairs_total"] == 12    # 6 unordered pairs per row
    assert stats["pairs_correct"] == 6   # first row all right, second all wrong
    assert stats["rows"] == 2
