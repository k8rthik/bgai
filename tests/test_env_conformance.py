"""Standard-API conformance: PettingZoo's ``api_test`` and Gymnasium's
``check_env``, for every supported player count.

These are the tests that make "standard-API environment" a claim rather
than a label. ``api_test``'s ``num_cycles`` is kept modest here so the
suite stays fast; ``uv run python -m bgai.env.bench --conformance`` runs the
same checks and prints every warning verbatim.
"""

from __future__ import annotations

import warnings

import gymnasium
import pettingzoo
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test, seed_test

from bgai.env.aec import TerraMysticaAECEnv, env as wrapped_env, raw_env
from bgai.env.config import MAX_CANDIDATES, MAX_PLAYERS, MIN_PLAYERS, EnvConfig
from bgai.env.single import make_single_env
from bgai.env.spaces import action_space, inner_observation_space, observation_space

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))
API_TEST_CYCLES = 25
"""Enough to drive several complete games per player count (a 4-player game
is ~165 decisions) without dominating the suite."""


@pytest.mark.parametrize("count", COUNTS)
def test_pettingzoo_api_test_passes_on_the_raw_env(count: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api_test(raw_env(EnvConfig(player_count=count)), num_cycles=API_TEST_CYCLES)


@pytest.mark.parametrize("count", COUNTS)
def test_pettingzoo_api_test_passes_on_the_wrapped_env(count: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api_test(wrapped_env(EnvConfig(player_count=count)), num_cycles=API_TEST_CYCLES)


@pytest.mark.parametrize("count", COUNTS)
def test_gymnasium_check_env_passes(count: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(
            make_single_env(player_count=count, opponent="random"),
            skip_render_check=False,
        )


def test_the_only_api_test_warnings_are_the_dict_observation_ones() -> None:
    """Both warnings are structural and shared with PettingZoo's own
    ``chess_v6``/``connect_four_v3``: a masked-action env has to use a Dict
    observation space, and ``api_test`` prefers Box/Discrete. Pinned so a
    *new* warning shows up as a failure rather than scrolling past.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        api_test(raw_env(EnvConfig(player_count=4)), num_cycles=API_TEST_CYCLES)
    messages = {str(w.message) for w in caught}
    assert messages == {
        "Observation is not a NumPy array",
        "Observation space for each agent probably should be gymnasium.spaces.box "
        "or gymnasium.spaces.discrete",
    }


def test_the_aec_env_is_a_pettingzoo_aec_env() -> None:
    assert issubclass(TerraMysticaAECEnv, pettingzoo.AECEnv)
    assert isinstance(raw_env(EnvConfig()), pettingzoo.AECEnv)


def test_the_single_env_is_a_gymnasium_env() -> None:
    assert isinstance(make_single_env(), gymnasium.Env)


def test_pettingzoo_seed_test_passes() -> None:
    """PettingZoo's own determinism harness: two envs reset to the same seed
    must produce identical observations, rewards and agent order.
    """
    seed_test(lambda: raw_env(EnvConfig(player_count=4)), num_cycles=10)


# --------------------------------------------------------------------------
# space definitions
# --------------------------------------------------------------------------


def test_action_space_is_an_index_into_the_offer() -> None:
    space = action_space(MAX_CANDIDATES)
    assert isinstance(space, gymnasium.spaces.Discrete)
    assert space.n == MAX_CANDIDATES


def test_action_mask_dtype_is_what_discrete_sample_requires() -> None:
    """``Discrete.sample(mask)`` asserts ``mask.dtype == int8``; if the
    observation drifted to bool or uint8 every masked sampler would break.
    """
    space = observation_space(MAX_CANDIDATES)
    assert space["action_mask"].dtype == "int8"
    sample = space.sample()
    assert action_space(MAX_CANDIDATES).sample(sample["action_mask"]) is not None


def test_observation_space_is_the_masked_action_convention() -> None:
    space = observation_space(MAX_CANDIDATES)
    assert set(space.spaces) == {"observation", "action_mask"}
    inner = inner_observation_space(MAX_CANDIDATES)
    assert set(inner.spaces) == {"hex_planes", "globals", "candidates"}
    assert space["observation"] == inner


def test_observation_space_bounds_are_finite() -> None:
    """``api_test`` warns on infinite bounds and a learner cannot normalize
    against them.
    """
    import numpy as np

    inner = inner_observation_space(MAX_CANDIDATES)
    for key in inner.spaces:
        box = inner[key]
        assert np.isfinite(box.low).all()
        assert np.isfinite(box.high).all()
        assert (box.low < box.high).all()
