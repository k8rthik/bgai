"""Hashable content key for a ``GameState`` (transposition tables).

``GameState``/``FactionState`` are frozen but carry plain dicts, so they
cannot be dict keys themselves. ``game_key`` builds a tuple of every
field's content in a fixed order; two states with equal content get
equal keys (tuple equality, not just equal hashes -- a lookup that hits
compares the full key, so a hash collision can never hand a position
another position's answers).

Coverage is pinned by ``tests/test_state_key.py`` against the dataclass
field lists: adding a field to either state class without extending the
key fails that test, which is the failure mode that matters (a stale key
would merge two different positions).

Cost: ~14 us per state on a mid-game position (half of it the 113 hex
rows), memoised per state object so the driver's decision table and the
search's evaluation table share one computation.
"""

from __future__ import annotations

from collections.abc import Mapping

from bgai.engine.tm.memo import state_memo
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import FactionState, GameState

GAME_FIELDS_COVERED: tuple[str, ...] = (
    "setup",
    "round",
    "phase",
    "turn_order",
    "passed_order",
    "active_index",
    "pending",
    "hexes",
    "bridges",
    "factions",
    "cults",
    "cult_10",
    "priest_slots",
    "favors_pool",
    "towns_pool",
    "bonus_coins",
    "power_actions_taken",
    "founded_towns",
)

FACTION_FIELDS_COVERED: tuple[str, ...] = (
    "name",
    "coins",
    "workers",
    "priests",
    "priest_pool",
    "power",
    "vp",
    "shipping",
    "dig_level",
    "teleport_level",
    "buildings",
    "favors",
    "bonus",
    "towns",
    "keys",
    "passed",
    "actions_used",
    "cult_blocked",
    "bridges_built",
    "spades_available",
    "extra_actions",
    "dropped",
    "teleported_hex",
)


def _sorted_items(mapping: Mapping) -> tuple:
    return tuple(sorted(mapping.items()))


@state_memo(maxsize=64)
def _setup_key(setup: GameSetup) -> tuple:
    return (
        setup.game_id,
        setup.options,
        setup.factions,
        tuple(setup.score_tiles),
        setup.bonus_tiles,
        setup.player_count,
        _sorted_items(setup.dropped_at_row),
    )


def faction_key(fs: FactionState) -> tuple:
    """Content tuple for one faction (``FACTION_FIELDS_COVERED`` order)."""
    return (
        fs.name,
        fs.coins,
        fs.workers,
        fs.priests,
        fs.priest_pool,
        fs.power,
        fs.vp,
        fs.shipping,
        fs.dig_level,
        fs.teleport_level,
        _sorted_items(fs.buildings),
        fs.favors,
        fs.bonus,
        fs.towns,
        fs.keys,
        fs.passed,
        fs.actions_used,
        fs.cult_blocked,
        fs.bridges_built,
        fs.spades_available,
        fs.extra_actions,
        fs.dropped,
        fs.teleported_hex,
    )


@state_memo(maxsize=4096)
def game_key(state: GameState) -> tuple:
    """Content tuple for a whole position (``GAME_FIELDS_COVERED`` order)."""
    return (
        _setup_key(state.setup),
        state.round,
        state.phase,
        state.turn_order,
        state.passed_order,
        state.active_index,
        state.pending,
        tuple(state.hexes.items()),
        state.bridges,
        tuple((name, faction_key(fs)) for name, fs in state.factions.items()),
        tuple((cult, _sorted_items(steps)) for cult, steps in state.cults.items()),
        _sorted_items(state.cult_10),
        _sorted_items(state.priest_slots),
        _sorted_items(state.favors_pool),
        _sorted_items(state.towns_pool),
        _sorted_items(state.bonus_coins),
        state.power_actions_taken,
        tuple((name, towns) for name, towns in state.founded_towns.items()),
    )
