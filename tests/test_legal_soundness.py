"""Legal-move soundness regressions found by arena random-play fuzzing
(plan 2026-08-04-arena-baselines, Task 4 step 4).

The corpus containment sweep only proves corpus moves are a *subset* of
``legal_moves`` -- it never exercises the extra moves the generator
offers. Random arena play does, and found two classes of offered moves
that ``apply`` rejects (10 of the first 200 fuzzed games):

1. Giants offered ``transform`` to non-home colors ("giants must
   transform to red, not gray") -- their cost hook prices any color pair
   at 2 spades, but their target hook forces red.
2. A faction holding ACTN's ``free_tf`` marker with 0 workers offered
   fold-in ``build`` moves it cannot pay for ("nomads cannot afford
   1 W") -- the marker waives the transform, not the dwelling cost.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.engine.tm.apply import apply
from bgai.engine.tm.legal import legal_moves
from bgai.engine.tm.round_flow import advance_turn, start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, PendingDecision, Phase, active_faction, with_faction


def _through_setup(game_id: str) -> GameState:
    """Deterministically play out both setup phases (first legal pick by
    sorted order) so factions have dwellings on the board; leaves the
    state at Phase.INCOME, round 1.
    """
    state = start_setup(GameState.initial(load_setup(game_id)))
    while state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        faction = active_faction(state)
        want = "build" if state.phase == Phase.SETUP_DWELLINGS else "pass"
        move = sorted(
            (m for m in legal_moves(state) if m.verb == want),
            key=lambda m: (m.loc or "", m.tile or ""),
        )[0]
        state = apply(state, faction, move)
        state = advance_turn(state)
    assert state.phase == Phase.INCOME
    return state


def _in_actions(state: GameState, faction: str) -> GameState:
    return replace(state, phase=Phase.ACTIONS, active_index=state.turn_order.index(faction))


def test_giants_transforms_only_target_home_color() -> None:
    state = _through_setup("4pLeague_S10_D3L3_G3")  # giants seated
    state = _in_actions(state, "giants")
    fs = state.factions["giants"]
    state = with_faction(state, "giants", replace(fs, spades_available=2))
    transforms = [m for m in legal_moves(state) if m.verb == "transform"]
    assert transforms, "giants with 2 spades next to own dwellings must have transforms"
    off_color = [m for m in transforms if m.color != "red"]
    assert off_color == [], f"non-home transform offers: {off_color[:3]}"


def test_free_tf_build_offers_respect_dwelling_cost() -> None:
    state = _through_setup("4pLeague_S10_D1L1_G1")  # nomads seated
    state = _in_actions(state, "nomads")
    fs = state.factions["nomads"]
    state = with_faction(state, "nomads", replace(fs, workers=0))
    state = replace(
        state,
        pending=state.pending + (PendingDecision(faction="nomads", kind="free_tf"),),
    )
    builds = [m for m in legal_moves(state) if m.verb == "build"]
    assert builds == [], f"unaffordable fold-in builds offered with 0 W: {builds[:3]}"
