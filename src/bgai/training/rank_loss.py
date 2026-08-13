"""Pairwise ranking loss over the value head's seat vector.

The value head is trained to regress each seat's share of the table's
final VP. That is a *magnitude* objective, but MCTS never consumes the
magnitude -- it backs values up the tree to decide which position is
better than which. The two come apart: measured over 102k validation
records, a share-MSE of 0.00065 sits alongside 30% of seat pairs ordered
backwards (pairwise-rank 0.696), and that misordering is the noise
search inherits at every node.

This is the standard RankNet pairwise logistic loss. For every seat pair
whose true shares differ, the predicted difference should carry the same
sign, and the penalty decays smoothly as the correct gap widens -- so a
confidently correct ordering scores better than a barely correct one,
which matters because PUCT compares value gaps, not just their signs.
Tied seats carry no ordering information and are masked out.
"""

from __future__ import annotations

import torch


def pairwise_rank_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean pairwise logistic ranking loss.

    ``predicted``/``target`` are ``(batch, seats)``. Returns a scalar
    averaged over rows, with each row averaged over its own unmasked
    pairs, so rows with many ties do not quietly dominate the batch.
    """
    # (batch, seats, seats) differences: [.., i, j] = value_i - value_j
    pred_diff = predicted.unsqueeze(2) - predicted.unsqueeze(1)
    true_diff = target.unsqueeze(2) - target.unsqueeze(1)

    # upper triangle only -- each unordered pair once, never a seat with
    # itself, or every pair would be counted twice and cancel
    seats = predicted.shape[-1]
    triu = torch.triu(
        torch.ones(seats, seats, dtype=torch.bool, device=predicted.device), diagonal=1
    )
    mask = triu.unsqueeze(0) & (true_diff != 0)

    # sign of the true ordering; loss is softplus(-sign * predicted_gap),
    # which is 0 for a confidently correct gap and grows linearly when wrong
    sign = torch.sign(true_diff)
    per_pair = torch.nn.functional.softplus(-sign * pred_diff)
    per_pair = per_pair * mask

    counts = mask.sum(dim=(1, 2)).clamp(min=1)
    per_row = per_pair.sum(dim=(1, 2)) / counts
    # rows that were entirely tied contribute 0, not NaN
    return per_row.mean()
