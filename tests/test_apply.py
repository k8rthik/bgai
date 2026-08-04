"""tests/test_apply.py

Scenarios ported/validated against jsnell/terra-mystica ``src/commands.pm``
(``command_convert``, ``command_send``) and ``src/Game/Factions.pm``
(``initialize_faction``, base ``exchange_rates`` merge with per-faction
overrides).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import (
    HANDLERS,
    EngineError,
    apply,
    pop_pending,
    push_pending,
    register_handler,
)
from bgai.engine.tm.factions.hooks import HOOKS, FactionHooks, hooks_for
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import (
    FactionState,
    GameState,
    PendingDecision,
    active_faction,
    with_faction,
)

GAME_ID = "4pLeague_S10_D1L1_G1"


def _state() -> GameState:
    return GameState.initial(load_setup(GAME_ID))


def _with_power(state: GameState, faction: str, power: Power) -> GameState:
    return with_faction(state, faction, replace(state.factions[faction], power=power))


def _with_priests(state: GameState, faction: str, priests: int) -> GameState:
    return with_faction(state, faction, replace(state.factions[faction], priests=priests))


def _as_active(state: GameState, faction: str) -> GameState:
    """Make `faction` the active faction (turn-order head), independent of
    queue contents, for tests that aren't exercising turn-order itself.
    """
    return replace(state, active_index=state.turn_order.index(faction))


def _alchemists_state() -> GameState:
    base = _state()
    alch = FactionState.initial(FACTIONS["alchemists"])
    factions = dict(base.factions)
    factions["alchemists"] = alch
    return replace(base, factions=factions, turn_order=("alchemists",), active_index=0)


def _cmd(verb: str, kind: Kind = Kind.DECISION, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=kind, raw=verb, **fields)  # type: ignore[arg-type]


def _restore_handler(verb: str, previous: object | None) -> None:
    """Undo a test's temporary `register_handler(verb, stub)` override.

    HANDLERS is process-global and, since Task 8, "leech"/"decline" have
    real registered handlers by the time any test runs (leech.py
    registers them at import time). Restore whatever was there before
    (real handler or nothing) instead of unconditionally `del`-ing, which
    would otherwise permanently break every later test that dispatches a
    leech/decline verb through `apply()`.
    """
    if previous is None:
        HANDLERS.pop(verb, None)
    else:
        HANDLERS[verb] = previous  # type: ignore[assignment]


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------


def test_convert_c_to_vp_base_rate() -> None:
    s = _state()
    cmd = _cmd("convert", n1=3, res1="C", n2=1, res2="VP")
    s2 = apply(s, "engineers", cmd)
    fs = s2.factions["engineers"]
    assert fs.coins == 10 - 3
    assert fs.vp == 20 + 1


def test_convert_w_to_c_base_rate() -> None:
    s = _state()
    cmd = _cmd("convert", n1=2, res1="W", n2=2, res2="C")
    s2 = apply(s, "engineers", cmd)
    fs = s2.factions["engineers"]
    assert fs.workers == 2 - 2
    assert fs.coins == 10 + 2


def test_convert_pw_to_w_base_rate_spends_bowl3() -> None:
    s = _state()
    s = _with_power(s, "engineers", Power(0, 0, 6))
    cmd = _cmd("convert", n1=3, res1="PW", n2=1, res2="W")
    s2 = apply(s, "engineers", cmd)
    fs = s2.factions["engineers"]
    assert fs.power.as_str() == "3/0/3"
    assert fs.workers == 2 + 1


def test_convert_pw_insufficient_bowl3_raises_engine_error_not_value_error() -> None:
    s = _state()  # engineers power starts 3/9/0: bowl3 is empty
    cmd = _cmd("convert", n1=3, res1="PW", n2=1, res2="W")
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_convert_alchemists_c_to_vp_override_rate() -> None:
    s = _alchemists_state()
    cmd = _cmd("convert", n1=2, res1="C", n2=1, res2="VP")
    s2 = apply(s, "alchemists", cmd)
    fs = s2.factions["alchemists"]
    assert fs.coins == 15 - 2
    assert fs.vp == 20 + 1


def test_convert_alchemists_vp_to_c_reverse_rate() -> None:
    s = _alchemists_state()
    cmd = _cmd("convert", n1=1, res1="VP", n2=1, res2="C")
    s2 = apply(s, "alchemists", cmd)
    fs = s2.factions["alchemists"]
    assert fs.vp == 20 - 1
    assert fs.coins == 15 + 1


def test_convert_vp_to_c_not_available_for_non_alchemists() -> None:
    s = _state()
    cmd = _cmd("convert", n1=1, res1="VP", n2=1, res2="C")
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_convert_rate_mismatch_raises() -> None:
    s = _state()
    cmd = _cmd("convert", n1=2, res1="C", n2=1, res2="VP")  # base rate is 3:1, not 2:1
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_convert_insufficient_resources_raises() -> None:
    s = _state()
    cmd = _cmd("convert", n1=30, res1="C", n2=10, res2="VP")
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_convert_w_to_p_without_pending_still_raises() -> None:
    s = _as_active(_state(), "darklings")
    cmd = _cmd("convert", n1=1, res1="W", n2=1, res2="P")
    with pytest.raises(EngineError):
        apply(s, "darklings", cmd)


def test_darklings_sh_convert_w_to_p_pending_enables_1_to_1_conversion() -> None:
    s = _as_active(_state(), "darklings")
    s = push_pending(s, PendingDecision(faction="darklings", kind="convert_w_to_p", amount=3))
    s = with_faction(s, "darklings", replace(s.factions["darklings"], workers=5))
    cmd = _cmd("convert", n1=2, res1="W", n2=2, res2="P")
    s2 = apply(s, "darklings", cmd)
    fs = s2.factions["darklings"]
    assert fs.workers == 5 - 2
    assert fs.priests == 1 + 2  # darklings starts with 1 priest
    # Pending is metered down, not popped: 3 - 2 = 1 remaining.
    assert s2.pending == (PendingDecision(faction="darklings", kind="convert_w_to_p", amount=1),)


def test_darklings_sh_convert_w_to_p_pending_pops_when_fully_consumed() -> None:
    s = _as_active(_state(), "darklings")
    s = push_pending(s, PendingDecision(faction="darklings", kind="convert_w_to_p", amount=3))
    s = with_faction(s, "darklings", replace(s.factions["darklings"], workers=5))
    cmd = _cmd("convert", n1=3, res1="W", n2=3, res2="P")
    s2 = apply(s, "darklings", cmd)
    assert s2.factions["darklings"].workers == 5 - 3
    assert s2.factions["darklings"].priests == 1 + 3
    assert s2.pending == ()


def test_darklings_sh_convert_w_to_p_exceeding_allowance_raises() -> None:
    s = _as_active(_state(), "darklings")
    s = push_pending(s, PendingDecision(faction="darklings", kind="convert_w_to_p", amount=2))
    s = with_faction(s, "darklings", replace(s.factions["darklings"], workers=5))
    cmd = _cmd("convert", n1=3, res1="W", n2=3, res2="P")
    with pytest.raises(EngineError):
        apply(s, "darklings", cmd)
    # No partial mutation: pending and resources untouched.
    assert s.factions["darklings"].workers == 5
    assert s.pending == (PendingDecision(faction="darklings", kind="convert_w_to_p", amount=2),)


def test_darklings_sh_convert_w_to_p_rate_must_stay_1_to_1() -> None:
    s = _as_active(_state(), "darklings")
    s = push_pending(s, PendingDecision(faction="darklings", kind="convert_w_to_p", amount=3))
    s = with_faction(s, "darklings", replace(s.factions["darklings"], workers=5))
    cmd = _cmd("convert", n1=2, res1="W", n2=1, res2="P")  # 2:1, not the allowed 1:1
    with pytest.raises(EngineError):
        apply(s, "darklings", cmd)


# --------------------------------------------------------------------------
# burn
# --------------------------------------------------------------------------


def test_burn_moves_bowls() -> None:
    s = _state()  # engineers power starts 3/9/0
    cmd = _cmd("burn", n1=3)
    s2 = apply(s, "engineers", cmd)
    assert s2.factions["engineers"].power.as_str() == "3/3/3"


def test_burn_too_much_raises() -> None:
    s = _state()
    cmd = _cmd("burn", n1=10)
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


# --------------------------------------------------------------------------
# no-op anchors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["wait", "done", "resign", "annotation", "setup"])
def test_noop_anchor_verbs_leave_state_unchanged(verb: str) -> None:
    s = _state()
    kind = Kind.ANNOTATION if verb == "annotation" else Kind.DECISION
    cmd = _cmd(verb, kind=kind)
    s2 = apply(s, "engineers", cmd)
    assert s2 == s


# --------------------------------------------------------------------------
# send (priest -> cult track)
# --------------------------------------------------------------------------


def test_send_priest_default_takes_biggest_open_slot() -> None:
    s = _as_active(_state(), "darklings")  # darklings starts with 1 priest, FIRE at 0
    cmd = _cmd("send", cult="FIRE", n1=None)
    s2 = apply(s, "darklings", cmd)
    assert s2.priest_slots["FIRE"][0] == "darklings"
    assert s2.cults["darklings"]["FIRE"] == 3
    fs = s2.factions["darklings"]
    assert fs.priests == 0
    assert fs.priest_pool == 6
    assert fs.power.as_str() == "4/8/0"  # crossed the 3-threshold: +1 power


def test_send_priest_explicit_amount_takes_matching_slot() -> None:
    s = _as_active(_state(), "darklings")
    s = _with_priests(s, "darklings", 5)
    cmd = _cmd("send", cult="FIRE", n1=2)
    s2 = apply(s, "darklings", cmd)
    assert s2.priest_slots["FIRE"][0] is None  # 3-slot skipped
    assert s2.priest_slots["FIRE"][1] == "darklings"
    assert s2.cults["darklings"]["FIRE"] == 2
    assert s2.factions["darklings"].priest_pool == 6


def test_send_priest_requested_amount_with_no_matching_open_slot_raises() -> None:
    s = _as_active(_state(), "darklings")
    s = _with_priests(s, "darklings", 5)
    slots = ("engineers", None, None, None)  # 3-slot already taken by someone else
    s = replace(s, priest_slots={**s.priest_slots, "FIRE": slots})
    cmd = _cmd("send", cult="FIRE", n1=3)
    with pytest.raises(EngineError):
        apply(s, "darklings", cmd)
    # No partial mutation on failure.
    assert s.factions["darklings"].priests == 5


def test_send_priest_to_full_track_advances_one_step_and_does_not_occupy_slot() -> None:
    s = _as_active(_state(), "mermaids")
    s = _with_priests(s, "mermaids", 1)
    full_slots = ("engineers", "darklings", "nomads", "mermaids")
    s = replace(s, priest_slots={**s.priest_slots, "FIRE": full_slots})
    cmd = _cmd("send", cult="FIRE", n1=None)
    s2 = apply(s, "mermaids", cmd)
    assert s2.priest_slots["FIRE"] == full_slots  # unchanged: no slot freed up
    assert s2.cults["mermaids"]["FIRE"] == 1  # 1-step bounce, not a slot's step count
    fs = s2.factions["mermaids"]
    assert fs.priests == 0  # priest is still spent
    assert fs.priest_pool == 7  # ...but the pool budget is NOT consumed


def test_send_priest_without_one_in_hand_raises() -> None:
    s = _state()  # engineers starts with 0 priests
    cmd = _cmd("send", cult="FIRE", n1=None)
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


# --------------------------------------------------------------------------
# gain_cult / lose_cult
# --------------------------------------------------------------------------


def test_gain_cult_advances_track_and_grants_threshold_power() -> None:
    s = _state()  # engineers power starts 3/9/0; FIRE starts at 0
    # +3 crosses the 3-threshold for 1 power (gain shifts 1 token bowl1->bowl2).
    cmd = _cmd("gain_cult", kind=Kind.BOOKKEEPING, cult="FIRE", n1=3)
    s2 = apply(s, "engineers", cmd)
    assert s2.cults["engineers"]["FIRE"] == 3
    assert s2.factions["engineers"].power.as_str() == "2/10/0"


def test_gain_cult_pops_matching_cult_choice_pending_at_head() -> None:
    s = _state()
    s = push_pending(s, PendingDecision(faction="darklings", kind="cult_choice"))
    cmd = _cmd("gain_cult", kind=Kind.BOOKKEEPING, cult="WATER", n1=1)
    s2 = apply(s, "darklings", cmd)
    assert s2.pending == ()
    assert s2.cults["darklings"]["WATER"] == 2  # darklings starts WATER=1


def test_lose_cult_retreats_with_no_power_refund() -> None:
    s = _as_active(_state(), "darklings")  # darklings WATER starts at 1
    cmd = _cmd("lose_cult", kind=Kind.BOOKKEEPING, cult="WATER", n1=1)
    s2 = apply(s, "darklings", cmd)
    assert s2.cults["darklings"]["WATER"] == 0
    assert s2.factions["darklings"].power == s.factions["darklings"].power


def test_lose_cult_does_not_go_below_zero() -> None:
    s = _as_active(_state(), "darklings")
    cmd = _cmd("lose_cult", kind=Kind.BOOKKEEPING, cult="WATER", n1=5)
    s2 = apply(s, "darklings", cmd)
    assert s2.cults["darklings"]["WATER"] == 0


# --------------------------------------------------------------------------
# lose_resource / lose_spade / lose_marker / convert_marker
# --------------------------------------------------------------------------


def test_lose_resource_applies_literal_loss() -> None:
    s = _state()
    cmd = _cmd("lose_resource", kind=Kind.BOOKKEEPING, n1=2, res1="C")
    s2 = apply(s, "engineers", cmd)
    assert s2.factions["engineers"].coins == 8


def test_lose_resource_insufficient_raises() -> None:
    s = _state()
    cmd = _cmd("lose_resource", kind=Kind.BOOKKEEPING, n1=99, res1="C")
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_lose_spade_is_a_documented_noop_stub() -> None:
    s = _state()
    cmd = _cmd("lose_spade", kind=Kind.BOOKKEEPING, n1=1)
    s2 = apply(s, "engineers", cmd)
    assert s2 == s


def test_lose_marker_is_a_documented_noop_stub() -> None:
    s = _state()
    cmd = _cmd("lose_marker", kind=Kind.BOOKKEEPING, reason="BRIDGE")
    s2 = apply(s, "engineers", cmd)
    assert s2 == s


def test_convert_marker_is_a_documented_noop_bookkeeping_row() -> None:
    s = _as_active(_state(), "darklings")
    cmd = _cmd("convert_marker", kind=Kind.BOOKKEEPING, n1=3, res1="W", res2="P")
    s2 = apply(s, "darklings", cmd)
    assert s2 == s


# --------------------------------------------------------------------------
# apply() dispatch / queue mechanics
# --------------------------------------------------------------------------


def test_apply_dispatches_unknown_verb_to_engine_error() -> None:
    s = _state()
    cmd = _cmd("frobnicate", kind=Kind.BOOKKEEPING)
    with pytest.raises(EngineError):
        apply(s, "engineers", cmd)


def test_apply_rejects_out_of_turn_non_exempt_move() -> None:
    s = _state()
    assert active_faction(s) == "engineers"
    cmd = _cmd("burn", n1=1)
    with pytest.raises(EngineError):
        apply(s, "darklings", cmd)


def test_apply_allows_wait_out_of_turn() -> None:
    s = _state()
    cmd = _cmd("wait")
    s2 = apply(s, "darklings", cmd)
    assert s2 == s


def test_apply_allows_annotation_out_of_turn() -> None:
    s = _state()
    cmd = _cmd("annotation", kind=Kind.ANNOTATION)
    s2 = apply(s, "darklings", cmd)
    assert s2 == s


def test_apply_allows_leech_answer_out_of_turn_when_queued_for_that_faction() -> None:
    s = _state()
    s = push_pending(
        s,
        PendingDecision(faction="nomads", kind="gain_favor"),
        PendingDecision(faction="darklings", kind="leech"),
    )
    assert active_faction(s) == "nomads"

    # Save/restore rather than `del`: Task 8 (leech.py) registers real
    # "leech"/"decline" handlers at import time, so HANDLERS is no longer
    # empty for these verbs by the time this test runs -- blindly deleting
    # them after the stub override would permanently wipe the real
    # handlers for the rest of the test session.
    prev_leech, prev_decline = HANDLERS.get("leech"), HANDLERS.get("decline")
    register_handler("leech", lambda state, faction, cmd: state)
    register_handler("decline", lambda state, faction, cmd: state)
    try:
        s2 = apply(s, "darklings", _cmd("leech"))
        assert s2 == s
        s3 = apply(s, "darklings", _cmd("decline"))
        assert s3 == s
    finally:
        _restore_handler("leech", prev_leech)
        _restore_handler("decline", prev_decline)


def test_apply_rejects_leech_answer_with_no_queued_offer_for_that_faction() -> None:
    s = _state()
    s = push_pending(s, PendingDecision(faction="nomads", kind="gain_favor"))

    prev_leech = HANDLERS.get("leech")
    register_handler("leech", lambda state, faction, cmd: state)
    try:
        with pytest.raises(EngineError):
            apply(s, "darklings", _cmd("leech"))
    finally:
        _restore_handler("leech", prev_leech)


def test_push_pending_appends_fifo() -> None:
    s = _state()
    a = PendingDecision(faction="engineers", kind="gain_favor")
    b = PendingDecision(faction="darklings", kind="leech")
    s2 = push_pending(s, a, b)
    assert s2.pending == (a, b)


def test_pop_pending_removes_head_by_default() -> None:
    s = _state()
    a = PendingDecision(faction="engineers", kind="gain_favor")
    b = PendingDecision(faction="darklings", kind="leech")
    s2 = push_pending(s, a, b)
    s3 = pop_pending(s2)
    assert s3.pending == (b,)


def test_pop_pending_removes_at_index() -> None:
    s = _state()
    a = PendingDecision(faction="engineers", kind="gain_favor")
    b = PendingDecision(faction="darklings", kind="leech")
    s2 = push_pending(s, a, b)
    s3 = pop_pending(s2, index=1)
    assert s3.pending == (a,)


def test_engine_error_message_carries_game_context() -> None:
    s = _state()
    cmd = ParsedCommand(verb="burn", kind=Kind.DECISION, raw="burn 1", n1=1)
    try:
        apply(s, "darklings", cmd)
        pytest.fail("expected EngineError")
    except EngineError as exc:
        msg = str(exc)
        assert "darklings" in msg
        assert "burn 1" in msg
        assert str(s.round) in msg


# --------------------------------------------------------------------------
# factions/hooks.py
# --------------------------------------------------------------------------


def test_hooks_for_returns_shared_default_instance_when_unregistered() -> None:
    assert "totally_unknown_faction" not in HOOKS
    assert hooks_for("totally_unknown_faction") is hooks_for("also_unknown")


def test_default_faction_hooks_are_noops() -> None:
    hooks = FactionHooks()
    s = _state()
    assert hooks.spade_transform_target(s, "engineers", "A1", "brown") == "brown"
    assert hooks.extra_dig_gain(s, "engineers") == {}
    assert hooks.on_stronghold_built(s, "engineers") == s
    assert hooks.on_leech_resolved(s, "engineers", True) == s
    assert hooks.pass_vp_extra(s, "engineers") == 0
    assert hooks.reachable_extra(s, "engineers", "A1") is False
