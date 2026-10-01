"""Reward formulation for TM-Env.

Terra Mystica's own objective is final VP: the player with the most VP
after round 6's final scoring wins. Everything here is a function of that
and nothing else -- no hand-tuned positional bonuses, which the repo's own
measurement history says we cannot trust (docs/decisions.md C10-C13 found
the agent's weakness was *invisible* to VP-greedy metrics).

Four modes, all centred so a seat that finishes exactly average earns 0:

``TERMINAL_VP_SHARE`` (default)
    0 every step; at the end each seat gets ``vp / table_total - 1/n``.
    Scale-free, sums to 0 across seats, and is exactly the target the
    repo's value head already predicts (``training/dataset.py``'s
    ``share``), so a value function transfers between the two.
    *Limitation, measured:* a share target is blind to compounding
    economy -- C12 found the plateau broke only once absolute final VP was
    added as an auxiliary target. Use ``DENSE_VP`` or your own auxiliary
    head if you care about that.

``TERMINAL_RANK``
    Placement mapped linearly onto [+1, -1] (ties share their mean).
    The metric the arena's TrueSkill reports are built on, so an agent
    trained on it optimizes what the existing head-to-heads measure.
    *Limitation:* throws away margin, and C1/C3 is the cautionary tale --
    placement against our own baselines flattered the agent for months
    while its absolute VP was ~65.

``TERMINAL_WIN``
    1 for the winner (split on ties), 0 otherwise, then centred by 1/n.
    Maximum variance, lowest bias; what C16's c_puct result implies you
    want if outright wins are the goal.

``DENSE_VP``
    Per-decision ``delta(own VP) / DENSE_VP_SCALE`` plus
    ``DENSE_TERMINAL_WEIGHT`` x the terminal VP share.
    This is deliberately *not* potential-based shaping with respect to the
    share objective: the dense terms telescope to
    ``(final_vp - 20) / scale``, a monotone affine function of **absolute**
    final VP, so the mode optimizes absolute VP plus share rather than
    share alone. That is the C12 finding turned into a reward: absolute VP
    was the signal that broke the leg-4 plateau. VP is only ever granted
    by the engine's own scoring, so this adds no rules knowledge.
    *Cost if wrong:* it rewards VP that does not convert into placement
    (the C11 cult-town-hoarding failure mode), so it should be measured
    against ``TERMINAL_VP_SHARE``, not assumed better.

Credit assignment: a seat's dense reward lands on the step *that seat*
took, never on another seat's step. ``step_rewards`` therefore returns a
full per-seat vector each time and the AEC env accumulates it, which is
how PettingZoo expects simultaneous reward emission to work.
"""

from __future__ import annotations

from collections.abc import Mapping

from bgai.env.config import (
    DENSE_TERMINAL_WEIGHT,
    DENSE_VP_SCALE,
    VP_START,
    RewardMode,
)

__all__ = [
    "competition_ranks",
    "dense_step_rewards",
    "terminal_rewards",
    "zero_rewards",
]


def zero_rewards(seats: tuple[str, ...]) -> dict[str, float]:
    return {seat: 0.0 for seat in seats}


def competition_ranks(vps: Mapping[str, int]) -> dict[str, int]:
    """0-based competition ranking, ties sharing the better rank.

    Same rule as ``arena.sim._competition_ranks`` (duplicated rather than
    imported so the env never depends on the arena's private helpers, and
    pinned equal to it in tests/test_env_reward.py).
    """
    ordered = sorted(vps, key=lambda f: -vps[f])
    ranks: dict[str, int] = {}
    for index, faction in enumerate(ordered):
        if index == 0 or vps[faction] != vps[ordered[index - 1]]:
            ranks[faction] = index
        else:
            ranks[faction] = ranks[ordered[index - 1]]
    return ranks


def _vp_share(vps: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(vps.values()))
    n = len(vps)
    if total <= 0.0:  # pathological, but never divide by zero at a boundary
        return {seat: 0.0 for seat in vps}
    return {seat: vps[seat] / total - 1.0 / n for seat in vps}


def _rank_reward(vps: Mapping[str, int]) -> dict[str, float]:
    n = len(vps)
    if n == 1:
        return {seat: 0.0 for seat in vps}
    ranks = competition_ranks(vps)
    # tied seats share the mean of the places they occupy
    by_rank: dict[int, list[str]] = {}
    for seat, rank in ranks.items():
        by_rank.setdefault(rank, []).append(seat)
    out: dict[str, float] = {}
    for rank, tied in by_rank.items():
        places = range(rank, rank + len(tied))
        mean_place = sum(places) / len(tied)
        out.update(dict.fromkeys(tied, 1.0 - 2.0 * mean_place / (n - 1)))
    return out


def _win_reward(vps: Mapping[str, int]) -> dict[str, float]:
    n = len(vps)
    best = max(vps.values())
    winners = [seat for seat in vps if vps[seat] == best]
    payout = 1.0 / len(winners)
    return {
        seat: (payout if seat in winners else 0.0) - 1.0 / n for seat in vps
    }


def terminal_rewards(
    mode: RewardMode, vps: Mapping[str, int]
) -> dict[str, float]:
    """The end-of-episode reward for every seat.

    ``vps`` is the engine's own post-``final_scoring`` VP per faction; this
    function computes nothing about the game, only about those numbers.
    """
    if not vps:
        raise ValueError("terminal_rewards needs at least one seat")
    if mode is RewardMode.TERMINAL_VP_SHARE:
        return _vp_share(vps)
    if mode is RewardMode.TERMINAL_RANK:
        return _rank_reward(vps)
    if mode is RewardMode.TERMINAL_WIN:
        return _win_reward(vps)
    if mode is RewardMode.DENSE_VP:
        share = _vp_share(vps)
        return {seat: DENSE_TERMINAL_WEIGHT * share[seat] for seat in vps}
    raise ValueError(f"unhandled reward mode {mode!r}")


def dense_step_rewards(
    mode: RewardMode,
    before: Mapping[str, int],
    after: Mapping[str, int],
    seats: tuple[str, ...],
) -> dict[str, float]:
    """Per-step reward from a VP delta, or zeros outside ``DENSE_VP``."""
    if mode is not RewardMode.DENSE_VP:
        return zero_rewards(seats)
    return {
        seat: (after[seat] - before[seat]) / DENSE_VP_SCALE for seat in seats
    }


def dense_return_bound(final_vp: int) -> float:
    """The dense part of an episode's return for a seat finishing on
    ``final_vp`` -- exposed so the telescoping claim in this module's
    docstring is a testable statement rather than a comment.
    """
    return (final_vp - VP_START) / DENSE_VP_SCALE
