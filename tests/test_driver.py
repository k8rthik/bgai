"""Driver bookkeeping + decision routing (LLM-harness plan, task 3)."""

import dataclasses

from bgai.arena.driver import advance_bookkeeping, live_factions, new_game, next_actor
from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.state import PendingDecision, Phase
from driver_helpers import fast_forward_setup


def test_new_game_starts_setup_dwellings_snake_order():
    state = new_game(fresh_setup(seed=5))
    assert state.phase is Phase.SETUP_DWELLINGS
    assert next_actor(state) == state.turn_order[0]


def test_next_actor_prefers_oldest_blocking_pending():
    state = new_game(fresh_setup(seed=5))
    offer = PendingDecision(
        faction=state.setup.factions[2], kind="leech", amount=1, source=state.setup.factions[0]
    )
    state = dataclasses.replace(state, pending=(offer,))
    assert next_actor(state) == state.setup.factions[2]


def test_next_actor_ignores_marker_pendings():
    state = new_game(fresh_setup(seed=5))
    marker = PendingDecision(faction=state.setup.factions[1], kind="free_d")
    state = dataclasses.replace(state, pending=(marker,))
    assert next_actor(state) == state.turn_order[0]


def test_advance_bookkeeping_grants_income_and_enters_actions():
    state = fast_forward_setup(seed=5)
    assert state.phase is Phase.INCOME
    before = {f: state.factions[f].workers for f in live_factions(state)}
    state = advance_bookkeeping(state)
    assert state.phase is Phase.ACTIONS
    assert any(state.factions[f].workers > before[f] for f in live_factions(state))


def test_advance_bookkeeping_is_noop_during_actions():
    state = advance_bookkeeping(fast_forward_setup(seed=5))
    assert advance_bookkeeping(state) is state
