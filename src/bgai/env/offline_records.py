"""Record types for the offline-RL export.

Kept in their own module so ``bgai.env.offline`` (which replays games) and
``bgai.env.offline_shards`` (which reads and writes them) can both depend on
the shape without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgai.env.observation import Observation

__all__ = ["ExportStats", "OfflineTransition"]


@dataclass(frozen=True)
class OfflineTransition:
    """One logged human decision as an RL transition."""

    obs: Observation
    action: int
    reward: float
    next_obs: Observation | None
    done: bool
    faction: str
    seat: int
    game_id: str
    position: int
    """Index of this decision within its game's decision stream -- the
    stable key the shard writer builds ``next_index`` from (an identity or
    array comparison would be fragile; a position is not)."""
    next_position: int = -1

    @property
    def mask(self) -> np.ndarray:
        """The legal mask -- the observation's own, by construction."""
        return self.obs.action_mask


@dataclass(frozen=True)
class ExportStats:
    shard_count: int
    record_count: int
    games: int
    failed_games: tuple[str, ...]
    unmatched: int
    skipped_single_candidate: int
