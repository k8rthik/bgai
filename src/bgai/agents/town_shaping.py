"""Town-progress shaping for MCTS leaf values.

The 2026-10-28 towns diagnosis found that bots found a town about as often
as humans once a town-founding build/upgrade is legal, but reach such
positions half as often: the missing skill is assembling a cluster near the
threshold over several turns, which a value head blind to connectivity
undervalues. Shaping adds, per seat, ``strength * town_progress`` to a
non-terminal leaf's VP-share vector and renormalises, so search prefers
lines that build toward a town. Terminal leaves are never shaped -- towns
are already in the final score.

``town_progress`` is the best unfinished cluster's distance to a town:
``min(1, power / threshold) * min(1, structures / 4)`` (SA counts twice,
threshold 7 or 6 with FAV5), 0 once no town tile is left. Mermaids'
river-skip towns are not modelled here (plain clusters only).
"""

from __future__ import annotations

import numpy as np

from bgai.engine.tm.connectivity import clusters
from bgai.engine.tm.state import GameState
from bgai.engine.tm.towns import _cluster_power_and_count, _town_threshold

TOWN_STRUCTURES = 4


def town_progress(state: GameState, faction: str) -> float:
    if not any(count > 0 for count in state.towns_pool.values()):
        return 0.0
    founded = state.founded_towns.get(faction, ())
    threshold = _town_threshold(state.factions[faction])
    best = 0.0
    for cluster in clusters(state, faction):
        if any(cluster & town for town in founded):
            continue
        power, count = _cluster_power_and_count(state, cluster)
        best = max(best, min(1.0, power / threshold) * min(1.0, count / TOWN_STRUCTURES))
    return best


def shape(value: np.ndarray, state: GameState, strength: float) -> np.ndarray:
    """``value`` is a share vector in ``state.setup.factions`` order."""
    if strength <= 0:
        return value
    phi = np.array([town_progress(state, f) for f in state.setup.factions], dtype=np.float32)
    shaped = value + strength * phi
    return (shaped / shaped.sum()).astype(np.float32)
