"""Diagnostic A (docs/decisions.md D6.9): blend the imitation net's
learned leaf value with an engine-computed VP-share projection that is
robust off-distribution by construction.

The learned value head (``bgai.agents.mcts.MCTSAgent._evaluate``) was
trained only on states humans actually reached. MCTS deliberately walks
to states off that distribution and then trusts the head there -- one of
two surviving hypotheses (H1) for why 64-sim max^n search is a tight
zero against the very policy it searches over (D6.5-D6.8). This module
supplies the other half of a leaf value that cannot go stale in that
way: ``bgai.engine.tm.scoring.projected_vp`` is pure arithmetic over
whatever board is in front of it -- current VP, cult standings, network
standings, and resource conversion, exactly the terms ``final_scoring``
would apply if the game ended now -- so it is exact by construction at
every state, on- or off-distribution alike.
"""

from __future__ import annotations

import numpy as np

from bgai.engine.tm.scoring import projected_vp
from bgai.engine.tm.state import GameState


def computed_value(state: GameState) -> np.ndarray:
    """Per-seat projected final-VP share, in absolute seat order
    (``state.setup.factions``) -- the same frame ``MCTSAgent`` backs
    values up in. Falls back to a uniform share only in the degenerate
    case of zero total projected VP (never observed after setup, since
    every faction starts with nonzero VP)."""
    seats = state.setup.factions
    vp = projected_vp(state)
    values = np.array([vp[f] for f in seats], dtype=np.float32)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(seats), 1.0 / len(seats), dtype=np.float32)
    return values / total


def blend(learned: np.ndarray, state: GameState, w: float) -> np.ndarray:
    """``(1 - w) * learned + w * computed_value(state)``, both vectors in
    absolute seat order. ``w=0`` returns ``learned`` unchanged -- the
    control cell of the D6.9 sweep, and today's default behaviour."""
    if w <= 0:
        return learned
    if w >= 1:
        return computed_value(state)
    return (1.0 - w) * learned + w * computed_value(state)
