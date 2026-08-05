"""Steppable driver invariants (Phase 6, decision D6.1).

MCTS branches positions, so ``advance`` must be a pure function of
(SimState, choice): re-advancing the same choice must reproduce the same
successor, and advancing two different choices from one parent must not
disturb the parent or each other.
"""

from __future__ import annotations

import random

from bgai.arena.driver import advance, decision, new_game
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import Phase


def _setup():
    return load_setup("4pLeague_S10_D1L1_G1")


def test_first_decision_is_a_setup_dwelling_placement() -> None:
    sim = new_game(_setup())
    faction, offer = decision(sim)
    assert sim.game.phase == Phase.SETUP_DWELLINGS
    assert offer and all(m.verb == "build" for m in offer)
    assert faction in _setup().factions


def test_advance_is_pure_and_repeatable() -> None:
    sim = new_game(_setup())
    _, offer = decision(sim)
    a = advance(sim, offer[0])
    b = advance(sim, offer[0])
    assert a.game.hexes == b.game.hexes
    assert a.decisions == b.decisions == sim.decisions + 1


def test_branches_are_independent() -> None:
    sim = new_game(_setup())
    _, offer = decision(sim)
    assert len(offer) >= 2
    before = sim.game.hexes
    left = advance(sim, offer[0])
    right = advance(sim, offer[1])
    assert sim.game.hexes == before, "parent must be untouched by branching"
    assert left.game.hexes != right.game.hexes
    # each branch placed its own dwelling, not the other's
    assert left.game.hexes[offer[0].loc].building == "D"
    assert right.game.hexes[offer[1].loc].building == "D"
    assert right.game.hexes[offer[0].loc].building is None


def test_playthrough_reaches_finished_and_scores() -> None:
    rng = random.Random(5)
    sim = new_game(_setup())
    steps = 0
    while (pending := decision(sim)) is not None:
        _, offer = pending
        sim = advance(sim, offer[rng.randrange(len(offer))])
        steps += 1
        assert steps < 5000
    assert sim.finished
    assert sum(fs.vp for fs in sim.game.factions.values()) > 0


def test_income_is_granted_exactly_once_per_round() -> None:
    """The marker lives on SimState, not GameState -- a marker stashed on
    the shared game object would leak across MCTS branches (and a missing
    one would double-grant income)."""
    rng = random.Random(11)
    sim = new_game(_setup())
    seen_rounds = []
    while (pending := decision(sim)) is not None:
        if sim.game.phase == Phase.ACTIONS and sim.game.round not in seen_rounds:
            seen_rounds.append(sim.game.round)
            total_workers = sum(fs.workers for fs in sim.game.factions.values())
            assert total_workers < 60, f"round {sim.game.round} looks double-granted"
        _, offer = pending
        sim = advance(sim, offer[rng.randrange(len(offer))])
    assert seen_rounds == [1, 2, 3, 4, 5, 6]
