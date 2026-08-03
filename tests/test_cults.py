"""Cult track advancement tests (thresholds hand-checked against resources.pm)."""

import pytest

from bgai.engine.tm.cults import advance


def test_simple_advance_no_threshold() -> None:
    result = advance(0, 2, keys_available=0, track_open=True)
    assert (result.new_value, result.power_gained) == (2, 0)


def test_crossing_each_threshold() -> None:
    assert advance(2, 1, 0, True).power_gained == 1   # 3
    assert advance(4, 1, 0, True).power_gained == 2   # 5
    assert advance(6, 1, 0, True).power_gained == 2   # 7
    # One jump across several thresholds accumulates all of them.
    result = advance(2, 5, 0, True)
    assert (result.new_value, result.power_gained) == (7, 5)


def test_reaching_10_with_key() -> None:
    result = advance(9, 1, keys_available=1, track_open=True)
    assert (result.new_value, result.power_gained, result.key_spent) == (10, 3, True)
    assert not result.blocked_at_9


def test_reaching_10_without_key_stops_at_9() -> None:
    result = advance(8, 4, keys_available=0, track_open=True)
    assert (result.new_value, result.key_spent, result.blocked_at_9) == (9, False, True)
    # Threshold power for 3/5/7 was already earned earlier; none granted here.
    assert result.power_gained == 0


def test_occupied_track_caps_at_9_even_with_key() -> None:
    result = advance(7, 5, keys_available=2, track_open=False)
    assert (result.new_value, result.key_spent, result.blocked_at_9) == (9, False, False)


def test_big_jump_from_zero_to_ten() -> None:
    result = advance(0, 10, keys_available=1, track_open=True)
    assert (result.new_value, result.power_gained, result.key_spent) == (10, 8, True)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        advance(11, 1, 0, True)
    with pytest.raises(ValueError):
        advance(3, -1, 0, True)
