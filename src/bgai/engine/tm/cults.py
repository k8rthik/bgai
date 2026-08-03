"""Cult track advancement and power-threshold rewards.

Ported from resources.pm ``maybe_gain_power_from_cult`` (lines 190-231) and
cults.pm ``setup_cults`` in jsnell/terra-mystica: crossing 3/5/7 grants
1/2/2 power; reaching 10 requires (and consumes) a town key, grants 3 power,
and locks the space for everyone else. Without a key the faction stops at 9
(and is marked cult-blocked so a later key auto-advances it). Priest slots
per cult: one 3-step slot and three 2-step slots (cults.pm).
"""

from __future__ import annotations

from dataclasses import dataclass

PRIEST_SLOT_STEPS: tuple[int, ...] = (3, 2, 2, 2)


@dataclass(frozen=True)
class CultAdvance:
    """Result of advancing on one cult track."""

    new_value: int
    power_gained: int
    key_spent: bool
    blocked_at_9: bool  # wanted 10 but had no key; caller records cult_blocked


def advance(
    old_value: int,
    steps: int,
    keys_available: int,
    track_open: bool,
) -> CultAdvance:
    """Advance from old_value by steps; track_open=False caps at 9 (10 occupied)."""
    if not 0 <= old_value <= 10:
        raise ValueError(f"bad cult position {old_value}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")

    cap = 10 if track_open else 9
    new_value = min(old_value + steps, cap)

    power = 0
    if old_value <= 2 < new_value:
        power += 1
    if old_value <= 4 < new_value:
        power += 2
    if old_value <= 6 < new_value:
        power += 2

    key_spent = False
    blocked_at_9 = False
    if old_value <= 9 < new_value:
        if keys_available < 1:
            new_value = 9
            blocked_at_9 = True
        else:
            key_spent = True
            power += 3

    return CultAdvance(
        new_value=new_value,
        power_gained=power,
        key_spent=key_spent,
        blocked_at_9=blocked_at_9,
    )
