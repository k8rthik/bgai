"""Mirrored seat rotation (plan 2026-08-04-arena-baselines, Task 7)."""

from __future__ import annotations

import pytest

from bgai.arena.rotation import seat_rotations


def test_four_cyclic_rotations() -> None:
    rots = seat_rotations(("a", "b", "c", "d"))
    assert rots == (
        ("a", "b", "c", "d"),
        ("b", "c", "d", "a"),
        ("c", "d", "a", "b"),
        ("d", "a", "b", "c"),
    )


def test_each_agent_visits_each_seat_once() -> None:
    rots = seat_rotations(("g", "r", "g", "r"))
    for seat in range(4):
        assert sorted(rot[seat] for rot in rots) == ["g", "g", "r", "r"]


def test_rejects_non_four_seats() -> None:
    with pytest.raises(ValueError):
        seat_rotations(("a", "b"))
