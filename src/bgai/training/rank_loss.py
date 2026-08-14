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


def pairwise_rank_loss(
    predicted: torch.Tensor, target: torch.Tensor, winner_pair_weight: float = 1.0
) -> torch.Tensor:
    """Mean pairwise logistic ranking loss.

    ``predicted``/``target`` are ``(batch, seats)``. Returns a scalar
    averaged over rows, with each row averaged over its own unmasked
    pairs, so rows with many ties do not quietly dominate the batch.

    ``winner_pair_weight`` > 1 up-weights every pair involving the row's
    true top seat: TM games are decided at the top of the table (a
    120-118 finish and a 118-120 finish are nearly identical in share
    space and opposite in outcome), so ordering credit is concentrated
    where placement is actually contested. Weighted normalization keeps
    the loss scale comparable across weights.
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

    weights = mask.to(predicted.dtype)
    if winner_pair_weight != 1.0:
        is_winner = torch.nn.functional.one_hot(target.argmax(dim=-1), seats).bool()
        pair_has_winner = is_winner.unsqueeze(2) | is_winner.unsqueeze(1)
        weights = weights * torch.where(
            pair_has_winner, predicted.new_tensor(winner_pair_weight), predicted.new_tensor(1.0)
        )
    per_pair = per_pair * weights

    counts = weights.sum(dim=(1, 2)).clamp(min=1e-6)
    per_row = per_pair.sum(dim=(1, 2)) / counts
    # rows that were entirely tied contribute 0, not NaN
    return per_row.mean()


def ordering_stats(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, int]:
    """Counts for the ordering accuracy MCTS actually consumes.

    Returned as counts rather than rates so a caller can accumulate them
    across batches and divide once. ``winner_correct`` is how often the
    argmax seat matches; ``pairs_correct``/``pairs_total`` cover every
    untied seat pair.

    This exists because validation reported only MSE. With ``rank_weight``
    on, MSE rises by design (the margin term widens correct gaps), so a
    run selecting on loss alone stops while ordering is still improving
    and saves the worse-ordering checkpoint -- exactly what the first
    rank fine-tune did.
    """
    with torch.no_grad():
        winner = int((predicted.argmax(dim=-1) == target.argmax(dim=-1)).sum())
        pred_diff = predicted.unsqueeze(2) - predicted.unsqueeze(1)
        true_diff = target.unsqueeze(2) - target.unsqueeze(1)
        seats = predicted.shape[-1]
        triu = torch.triu(
            torch.ones(seats, seats, dtype=torch.bool, device=predicted.device),
            diagonal=1,
        )
        mask = triu.unsqueeze(0) & (true_diff != 0)
        agree = (torch.sign(pred_diff) == torch.sign(true_diff)) & mask
        return {
            "winner_correct": winner,
            "pairs_correct": int(agree.sum()),
            "pairs_total": int(mask.sum()),
            "rows": int(predicted.shape[0]),
        }
