"""Color wheel spade distance tests."""

import pytest

from bgai.engine.tm.factions_data import COLOR_WHEEL
from bgai.engine.tm.terraform import spade_distance


def test_identity_is_zero() -> None:
    for color in COLOR_WHEEL:
        assert spade_distance(color, color) == 0


def test_wheel_neighbors_cost_one() -> None:
    assert spade_distance("yellow", "brown") == 1
    assert spade_distance("yellow", "red") == 1  # wraps around


def test_max_distance_is_three() -> None:
    assert spade_distance("yellow", "blue") == 3
    assert max(
        spade_distance(a, b) for a in COLOR_WHEEL for b in COLOR_WHEEL
    ) == 3


def test_symmetry() -> None:
    for a in COLOR_WHEEL:
        for b in COLOR_WHEEL:
            assert spade_distance(a, b) == spade_distance(b, a)


def test_rejects_river() -> None:
    with pytest.raises(ValueError):
        spade_distance("white", "red")
