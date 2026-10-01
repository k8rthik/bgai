"""Gymnasium space definitions for TM-Env observations and actions.

Imported lazily by the env modules so ``bgai.env.observation`` /
``bgai.env.codec`` stay usable without gymnasium installed (the core
package does not depend on it; it lives in the dev group).

The observation is the PettingZoo masked-action convention --
``{"observation": ..., "action_mask": ...}`` -- because that is what
``pettingzoo.test.api_test`` and the standard masked-policy wrappers look
for. ``action_mask`` is ``int8`` specifically because
``gymnasium.spaces.Discrete.sample(mask)`` asserts that dtype.
"""

from __future__ import annotations

from gymnasium import spaces

from bgai.env.config import (
    CANDIDATE_MAX,
    CANDIDATE_MIN,
    GLOBAL_MAX,
    GLOBAL_MIN,
    HEX_PLANE_DTYPE_MAX,
    HEX_PLANE_DTYPE_MIN,
)
from bgai.env.observation import GLOBAL_DIM, HEX_FEAT_DIM, HEXES
from bgai.training.encode_move import MOVE_FIELDS

__all__ = ["action_space", "inner_observation_space", "observation_space"]


def inner_observation_space(max_candidates: int) -> spaces.Dict:
    """The ``"observation"`` sub-space: board, globals, candidate features."""
    return spaces.Dict(
        {
            "hex_planes": spaces.Box(
                low=HEX_PLANE_DTYPE_MIN,
                high=HEX_PLANE_DTYPE_MAX,
                shape=(len(HEXES), HEX_FEAT_DIM),
                dtype="int8",
            ),
            "globals": spaces.Box(
                low=GLOBAL_MIN, high=GLOBAL_MAX, shape=(GLOBAL_DIM,), dtype="int16"
            ),
            "candidates": spaces.Box(
                low=CANDIDATE_MIN,
                high=CANDIDATE_MAX,
                shape=(max_candidates, MOVE_FIELDS),
                dtype="int16",
            ),
        }
    )


def observation_space(max_candidates: int) -> spaces.Dict:
    """The full per-agent observation space."""
    return spaces.Dict(
        {
            "observation": inner_observation_space(max_candidates),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(max_candidates,), dtype="int8"
            ),
        }
    )


def action_space(max_candidates: int) -> spaces.Discrete:
    """An index into the canonically ordered offer.

    *Rejected alternative:* a flat, state-independent action space over
    every conceivable ``ParsedCommand``. The engine's move grammar spans
    113 hexes x 22 verbs x building/tile/cult/colour/target/resource/
    amount fields, so a faithful flat space is six figures wide and
    >99.99% masked at every decision. The candidate-index space is the
    shape the repo's own net already scores in (``training/model.py`` dots
    a state embedding with each legal move's embedding), which is why the
    imitation net is a legal player even untrained.

    *Cost:* action index i does not mean the same move in two different
    states, so a policy MUST read ``observation["candidates"]``; an
    index-only policy would be learning noise. Documented in docs/tm-env.md.
    """
    return spaces.Discrete(max_candidates)
