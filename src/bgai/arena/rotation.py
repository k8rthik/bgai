"""Mirrored 4-seat cyclic rotation (plan Task 7)."""

from __future__ import annotations


def seat_rotations(agent_names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """The 4 cyclic rotations of a base seat assignment (master plan:
    mirrored seat/faction rotation -- the same sampled setup is played
    once per rotation, so every agent occupies every seat, and thus
    every faction, exactly as often as it appears in the base
    assignment; seat and faction strength cancel out of agent
    comparisons).
    """
    if len(agent_names) != 4:
        raise ValueError(f"expected 4 seats, got {len(agent_names)}")
    return tuple(
        tuple(agent_names[(seat + shift) % 4] for seat in range(4)) for shift in range(4)
    )
