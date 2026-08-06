"""tests/test_leaf_eval.py -- D6.9 diagnostic A: the blended leaf value.

``computed_value`` wraps ``scoring.projected_vp`` as a normalized share
vector in absolute seat order; ``blend`` mixes it with a learned value at
weight ``w``. These are pure-arithmetic properties, independent of any
trained checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from bgai.agents.leaf_eval import blend, computed_value
from bgai.engine.tm.scoring import projected_vp
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState

GAME_ID = "4pLeague_S10_D1L1_G1"


def _fresh() -> GameState:
    return GameState.initial(load_setup(GAME_ID))


def test_computed_value_matches_projected_vp_shares() -> None:
    state = _fresh()
    seats = state.setup.factions
    vp = projected_vp(state)
    total = sum(vp.values())
    expected = np.array([vp[f] / total for f in seats], dtype=np.float32)
    np.testing.assert_allclose(computed_value(state), expected, rtol=1e-6)


def test_computed_value_sums_to_one() -> None:
    state = _fresh()
    assert computed_value(state).sum() == pytest.approx(1.0, abs=1e-6)


def test_computed_value_falls_back_to_uniform_when_total_projected_vp_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The degenerate case (module docstring: never observed after setup
    in practice, since every faction starts with nonzero VP) -- exercised
    directly via a stubbed ``projected_vp`` rather than a real state,
    since a genuine all-zero projection cannot occur post-setup."""
    import bgai.agents.leaf_eval as leaf_eval

    state = _fresh()
    monkeypatch.setattr(leaf_eval, "projected_vp", lambda _state: dict.fromkeys(
        state.setup.factions, 0
    ))
    result = computed_value(state)
    np.testing.assert_allclose(result, np.full(4, 0.25, dtype=np.float32))


def test_blend_at_w_zero_returns_learned_unchanged() -> None:
    state = _fresh()
    learned = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = blend(learned, state, w=0.0)
    np.testing.assert_array_equal(result, learned)


def test_blend_at_w_one_returns_computed_value_exactly() -> None:
    state = _fresh()
    learned = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = blend(learned, state, w=1.0)
    np.testing.assert_allclose(result, computed_value(state))


def test_blend_at_w_half_is_the_arithmetic_mean() -> None:
    state = _fresh()
    learned = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = blend(learned, state, w=0.5)
    expected = 0.5 * learned + 0.5 * computed_value(state)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_blend_output_always_sums_to_one() -> None:
    """Both terms are VP-share vectors that sum to 1, so any convex
    combination should too -- a sanity check that the two evaluators
    speak the same units."""
    state = _fresh()
    learned = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    for w in (0.0, 0.25, 0.5, 1.0):
        result = blend(learned, state, w=w)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
