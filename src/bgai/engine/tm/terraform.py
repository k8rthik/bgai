"""Terraforming color wheel: spade distance between terrain colors.

The seven colors form a cycle (Constants.pm ``@colors``); transforming
terrain costs the shorter way around the wheel, one spade per step.
"""

from __future__ import annotations

from bgai.engine.tm.factions_data import COLOR_WHEEL

_INDEX = {color: i for i, color in enumerate(COLOR_WHEEL)}
_N = len(COLOR_WHEEL)


def spade_distance(from_color: str, to_color: str) -> int:
    """Spades needed to transform from_color into to_color (0-3)."""
    if from_color not in _INDEX or to_color not in _INDEX:
        raise ValueError(f"not terrain colors: {from_color!r} -> {to_color!r}")
    diff = abs(_INDEX[from_color] - _INDEX[to_color])
    return min(diff, _N - diff)
