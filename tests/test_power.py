"""Power bowl arithmetic tests."""

import pytest

from bgai.engine.tm.power import Power


def test_gain_moves_bowl1_first_then_bowl2() -> None:
    p = Power(5, 7, 0)
    assert p.gain(3) == Power(2, 10, 0)
    assert p.gain(5) == Power(0, 12, 0)
    assert p.gain(7) == Power(0, 10, 2)


def test_gain_beyond_capacity_is_lost() -> None:
    p = Power(0, 2, 10)
    assert p.gain(5) == Power(0, 0, 12)
    assert Power(0, 0, 12).gain(3) == Power(0, 0, 12)


def test_gainable() -> None:
    assert Power(5, 7, 0).gainable() == 17
    assert Power(0, 0, 12).gainable() == 0


def test_burn() -> None:
    assert Power(0, 12, 0).burn(3) == Power(0, 6, 3)
    with pytest.raises(ValueError):
        Power(0, 5, 0).burn(3)


def test_spend_returns_to_bowl1() -> None:
    assert Power(0, 0, 12).spend(4) == Power(4, 0, 8)
    with pytest.raises(ValueError):
        Power(0, 0, 2).spend(3)


def test_total_is_conserved_by_gain_and_spend() -> None:
    p = Power(5, 7, 0)
    assert p.gain(9).total == p.total
    assert Power(0, 0, 12).spend(5).total == 12


def test_burn_destroys_tokens() -> None:
    assert Power(0, 12, 0).burn(6).total == 6


def test_snellman_string_round_trip() -> None:
    assert Power.from_str("5/7/0") == Power(5, 7, 0)
    assert Power(2, 3, 7).as_str() == "2/3/7"
    with pytest.raises(ValueError):
        Power.from_str("5/7")


def test_negative_bowls_rejected() -> None:
    with pytest.raises(ValueError):
        Power(-1, 0, 0)
