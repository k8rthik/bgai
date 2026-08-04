"""Per-faction behavior hooks: a no-op default plus a name -> hooks registry.

`factions_data.py` models the *static* rules data (costs, income tracks,
exchange-rate overrides). This module is for the *behavior* that later
tasks need to special-case per faction -- Giants' fixed 2-spade transform
cost (`map.pm` lines 511-513/612-614), Mermaids' river-skip town detection
(`towns.pm` lines 130-150), Engineers' bridge pass-VP (`scoring.pm` lines
166-178), Cultists' leech-effect timing, and similar engine hooks quoted in
each faction's `notes` field in `factions_data.py`.

Every method here is a no-op / identity default. Later tasks give a
faction real behavior by registering a `FactionHooks` subclass instance (or
a dataclass built with `dataclasses.replace`-style method overrides) in
`HOOKS[faction_name]`; factions without an override use the shared
`FactionHooks()` default via `hooks_for`. Method signatures are frozen API
for this task -- later tasks may not change them, only override bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bgai.engine.tm.terraform import spade_distance

if TYPE_CHECKING:
    from bgai.engine.tm.state import GameState


class FactionHooks:
    """No-op default hooks for one faction's special-cased engine behavior."""

    def spade_transform_target(self, state: GameState, faction: str, loc: str, color: str) -> str:
        """Effective terraform target color for `loc` (Giants: always home terrain)."""
        return color

    def spade_transform_cost(
        self, state: GameState, faction: str, from_color: str, to_color: str
    ) -> int:
        """Spade cost to recolor a hex from `from_color` to `to_color`
        (Giants: flat 2 whenever they differ, `map.pm` 610-614 -- see
        `actions_terraform.py`'s `_GiantsHooks` override).
        """
        return spade_distance(from_color, to_color)

    def extra_dig_gain(self, state: GameState, faction: str) -> dict[str, int]:
        """Extra resources granted per spade dug, beyond the dig track (Alchemists SH)."""
        return {}

    def on_stronghold_built(self, state: GameState, faction: str) -> GameState:
        """Fired immediately after `faction`'s stronghold is built."""
        return state

    def on_leech_resolved(self, state: GameState, faction: str, accepted: bool) -> GameState:
        """Fired after a leech offer against `faction`'s build is accepted/declined
        (Cultists' `leech_effect`: gain a cult step if taken, else 1 power).
        """
        return state

    def pass_vp_extra(self, state: GameState, faction: str) -> int:
        """Extra VP granted at pass time, beyond the passed bonus tile (Engineers bridges)."""
        return 0

    def reachable_extra(self, state: GameState, faction: str, loc: str) -> bool:
        """Whether `loc` counts as reachable beyond normal shipping/range rules
        (Dwarves tunneling, Fakirs carpet flight, Mermaids river towns).
        """
        return False


HOOKS: dict[str, FactionHooks] = {}

_DEFAULT_HOOKS = FactionHooks()


def hooks_for(faction: str) -> FactionHooks:
    """Hooks registered for `faction`, or the shared no-op default if none."""
    return HOOKS.get(faction, _DEFAULT_HOOKS)
