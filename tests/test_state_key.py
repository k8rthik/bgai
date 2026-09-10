"""Content key for transposition tables.

Search reaches the same position through different move orders (a
convert before or after a build, independent free actions, ...): ~18% of
the states a 512-sim self-play game asks legal moves or the net about
are content-repeats of one already seen. Those states are distinct
objects, so identity memos miss; ``game_key`` is a hashable tuple of
every field the engine reads, cheap enough (~14 us) to pay for itself.

The dangerous failure is a *stale* key: a field added to the state that
the key does not cover would make two different positions collide and
hand one the other's legal moves. The coverage test pins the field lists.
"""

from __future__ import annotations

import dataclasses
import random

from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import load_setup
from bgai.engine.tm.state import FactionState, GameState
from bgai.engine.tm.state_key import (
    FACTION_FIELDS_COVERED,
    GAME_FIELDS_COVERED,
    game_key,
)


def _walk(n: int, seed: int = 3):
    rng = random.Random(seed)
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    for _ in range(n):
        pending = decision(sim)
        assert pending is not None
        sim = advance(sim, rng.choice(pending[1]), current=pending)
    return sim


def test_key_covers_every_state_field() -> None:
    assert {f.name for f in dataclasses.fields(GameState)} == set(GAME_FIELDS_COVERED)
    assert {f.name for f in dataclasses.fields(FactionState)} == set(FACTION_FIELDS_COVERED)


def test_equal_content_from_different_objects_shares_a_key() -> None:
    sim = _walk(60)
    a = sim.game
    b = dataclasses.replace(a, hexes=dict(a.hexes), factions=dict(a.factions))
    assert a is not b and a.hexes is not b.hexes
    assert game_key(a) == game_key(b)
    assert hash(game_key(a)) == hash(game_key(b))


def test_any_field_change_changes_the_key() -> None:
    sim = _walk(60)
    a = sim.game
    base = game_key(a)
    fs = a.factions[a.turn_order[0]]
    hex_key = next(k for k, h in a.hexes.items() if h.building is not None)
    variants = {
        "coins": dataclasses.replace(
            a, factions={**a.factions, fs.name: dataclasses.replace(fs, coins=fs.coins + 1)}
        ),
        "hex": dataclasses.replace(
            a, hexes={**a.hexes, hex_key: dataclasses.replace(a.hexes[hex_key], building="TP")}
        ),
        "round": dataclasses.replace(a, round=a.round + 1),
        "pending": dataclasses.replace(a, pending=()),
        "bridges": dataclasses.replace(a, bridges=frozenset({frozenset({"A1", "B1"})})),
        "cults": dataclasses.replace(
            a, cults={**a.cults, fs.name: {**a.cults[fs.name], "FIRE": 9}}
        ),
        "power_actions": dataclasses.replace(a, power_actions_taken=frozenset({"ACT1"})),
        "favors_pool": dataclasses.replace(a, favors_pool={**a.favors_pool, "FAV1": 0}),
        "active_index": dataclasses.replace(a, active_index=(a.active_index + 1) % 4),
    }
    for name, variant in variants.items():
        assert game_key(variant) != base, name


def test_successive_positions_have_distinct_keys() -> None:
    rng = random.Random(9)
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    keys = set()
    for _ in range(120):
        pending = decision(sim)
        assert pending is not None
        keys.add(game_key(sim.game))
        sim = advance(sim, rng.choice(pending[1]), current=pending)
    # a few genuine repeats (e.g. wait/noop loops) are possible; most differ
    assert len(keys) > 100
