"""TM-Env: standard-API reinforcement-learning environments over the
oracle-verified Terra Mystica engine.

Two wrappers, one engine:

* :mod:`bgai.env.aec` -- a PettingZoo AEC env for the real 2-5 player game.
* :mod:`bgai.env.single` -- a Gymnasium env where one seat learns and the
  rest are filled by a policy from :mod:`bgai.env.opponents`.

Both are thin adapters over :class:`bgai.env.core.Episode`, which is a
thin adapter over ``bgai.arena.driver``. Legal action masks come from
``engine.tm.legal``'s own move generation by way of ``driver.decision``;
there is no second copy of the rules in this package, and the structure
makes one impossible.

``gymnasium`` and ``pettingzoo`` are dev-group dependencies, so the
API-facing modules are imported lazily here: ``bgai.env.config``,
``bgai.env.core``, ``bgai.env.observation``, ``bgai.env.codec``,
``bgai.env.reward`` and ``bgai.env.offline`` all work without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bgai.env.config import (
    MAX_CANDIDATES,
    MAX_PLAYERS,
    MIN_PLAYERS,
    EnvConfig,
    EnvError,
    EnvStateError,
    IllegalActionError,
    OfferOverflowError,
    RewardMode,
    SetupSource,
)
from bgai.env.core import Episode, Transition
from bgai.env.observation import OBS_VERSION, Observation, encode_observation

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from bgai.env.single import TerraMysticaSingleEnv

__all__ = [
    "MAX_CANDIDATES",
    "MAX_PLAYERS",
    "MIN_PLAYERS",
    "OBS_VERSION",
    "Episode",
    "EnvConfig",
    "EnvError",
    "EnvStateError",
    "IllegalActionError",
    "Observation",
    "OfferOverflowError",
    "RewardMode",
    "SetupSource",
    "Transition",
    "aec_env",
    "encode_observation",
    "single_env",
]


def aec_env(*args: Any, **kwargs: Any) -> Any:
    """``bgai.env.aec.env`` -- the wrapped PettingZoo AEC env."""
    from bgai.env.aec import env as _env

    return _env(*args, **kwargs)


def single_env(*args: Any, **kwargs: Any) -> TerraMysticaSingleEnv:
    """``bgai.env.single.make_single_env`` -- the Gymnasium env."""
    from bgai.env.single import make_single_env

    return make_single_env(*args, **kwargs)
