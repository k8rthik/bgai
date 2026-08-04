"""tests/test_leech.py

Scenarios validated against jsnell/terra-mystica ``src/resources.pm``
(``note_leech``, ``command_leech``, ``command_decline``,
``cultist_maybe_gain_power``) and ``src/map.pm`` (``compute_leech``) --
see ``leech.py``'s module docstring for the exact citation trail.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.leech import (
    handle_cultist_leech_bonus,
    handle_decline,
    handle_leech,
    has_leechable_neighbor,
    offers_for_build,
    queue_leech,
)
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import (
    FactionState,
    GameState,
    PendingDecision,
    Phase,
    active_faction,
    with_faction,
)

GAME_ID = "4pLeague_S10_D1L1_G1"
BOARD = base_board()


def _state() -> GameState:
    s = GameState.initial(load_setup(GAME_ID))
    return replace(s, round=1, phase=Phase.ACTIONS)


def _triangle() -> tuple[str, str, str]:
    """(anchor, target, neighbor): anchor and neighbor are both directly
    adjacent to target; all three are distinct land hexes."""
    land = set(BOARD.land_hexes())
    for b in BOARD.land_hexes():
        neighbors = sorted(n for n in BOARD.adjacent[b] if n in land)
        if len(neighbors) >= 2:
            return neighbors[0], b, neighbors[1]
    raise AssertionError("no suitable hex triangle found")


ANCHOR, TARGET, NEIGHBOR = _triangle()


def _place(state: GameState, faction: str, hex_key: str, building: str, color: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], color=color, building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(with_faction(state, faction, fs), hexes=hexes)


def _seeded(
    builder: str = "engineers", opponent: str = "darklings", opponent_building: str = "TP"
) -> GameState:
    """``builder`` has an existing D at ANCHOR (its own color); ``opponent``
    has ``opponent_building`` at NEIGHBOR (its own color); TARGET is empty
    and colored for ``builder`` -- ready for ``offers_for_build``.
    """
    s = _state()
    s = _place(s, builder, ANCHOR, "D", FACTIONS[builder].color)
    s = _place(s, opponent, NEIGHBOR, opponent_building, FACTIONS[opponent].color)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color=FACTIONS[builder].color, building=None, owner=None)
    return replace(s, hexes=hexes)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# offers_for_build
# --------------------------------------------------------------------------


def test_offers_for_build_basic_amount_and_source() -> None:
    s = _seeded()
    offers = offers_for_build(s, "engineers", TARGET)
    assert offers == (
        PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers", options=(TARGET,)),
    )


def test_offers_for_build_no_adjacent_opponent_is_empty() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D", FACTIONS["engineers"].color)
    offers = offers_for_build(s, "engineers", TARGET)
    assert offers == ()


def test_offers_for_build_round_zero_is_empty() -> None:
    s = _seeded()
    s = replace(s, round=0)
    assert offers_for_build(s, "engineers", TARGET) == ()


def test_offers_for_build_capped_by_gainable() -> None:
    s = _seeded(opponent_building="SH")  # SH power = 3
    s = with_faction(
        s, "darklings", replace(s.factions["darklings"], power=Power(1, 0, 6))
    )  # gainable=2
    offers = offers_for_build(s, "engineers", TARGET)
    assert offers == (
        PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers", options=(TARGET,)),
    )


def test_offers_for_build_zero_gainable_still_enqueues_an_offer() -> None:
    """Perl's ``note_leech`` queues an offer regardless of ``actual`` being
    0 -- only the raw building-power sum gates whether an offer exists."""
    s = _seeded()
    s = with_faction(s, "darklings", replace(s.factions["darklings"], power=Power(0, 0, 12)))
    offers = offers_for_build(s, "engineers", TARGET)
    assert offers == (
        PendingDecision(faction="darklings", kind="leech", amount=0, source="engineers", options=(TARGET,)),
    )


def test_offers_for_build_multi_opponent_clockwise_order_with_wraparound() -> None:
    """3+ factions, two adjoining: offers enqueue in seat order starting
    *after* the builder, wrapping around the end of ``turn_order`` --
    brief-mandated scenario (Step 2 checklist).

    ``turn_order`` for this game is (engineers, darklings, nomads,
    mermaids). Builder = nomads (index 2): the rotation is
    nomads -> mermaids -> engineers -> darklings. mermaids and engineers
    both have an adjacent building of a different color; darklings does
    not. Expect offers in exactly that order: mermaids, then engineers
    (the wraparound past the end of ``turn_order`` back to index 0).
    """
    s = _state()
    assert s.turn_order == ("engineers", "darklings", "nomads", "mermaids")
    s = _place(s, "mermaids", ANCHOR, "TP", FACTIONS["mermaids"].color)  # power 2
    s = _place(s, "engineers", NEIGHBOR, "SH", FACTIONS["engineers"].color)  # power 3
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(
        hexes[TARGET], color=FACTIONS["nomads"].color, building=None, owner=None
    )
    s = replace(s, hexes=hexes)

    offers = offers_for_build(s, "nomads", TARGET)
    assert [o.faction for o in offers] == ["mermaids", "engineers"]
    assert offers[0].amount == 2
    assert offers[1].amount == 3
    assert all(o.source == "nomads" for o in offers)


def test_offers_for_build_own_color_neighbor_gives_no_offer() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D", FACTIONS["engineers"].color)
    s = _place(s, "engineers", NEIGHBOR, "TP", FACTIONS["engineers"].color)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color=FACTIONS["engineers"].color)
    s = replace(s, hexes=hexes)
    assert offers_for_build(s, "engineers", TARGET) == ()


def test_offers_for_build_excludes_a_dropped_opponent() -> None:
    """``resources.pm``'s ``note_leech`` walks
    ``factions_in_order_from($from, 1)`` -- the ``no_dummy=1`` variant
    that excludes dropped factions -- so a dropped opponent's adjacent
    building generates no offer, even though the color's raw strength is
    still there (``has_leechable_neighbor``, next test)."""
    s = _seeded()
    dropped_fs = replace(s.factions["darklings"], dropped=True)
    s = with_faction(s, "darklings", dropped_fs)
    assert offers_for_build(s, "engineers", TARGET) == ()


# --------------------------------------------------------------------------
# has_leechable_neighbor (task-14 fix: TP-upgrade isolation surcharge)
# --------------------------------------------------------------------------


def test_has_leechable_neighbor_true_for_ordinary_adjacent_opponent() -> None:
    s = _seeded()
    assert has_leechable_neighbor(s, "engineers", TARGET) is True


def test_has_leechable_neighbor_false_with_no_adjacent_opponent() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D", FACTIONS["engineers"].color)
    assert has_leechable_neighbor(s, "engineers", TARGET) is False


def test_has_leechable_neighbor_false_at_round_zero() -> None:
    s = _seeded()
    s = replace(s, round=0)
    assert has_leechable_neighbor(s, "engineers", TARGET) is False


def test_has_leechable_neighbor_stays_true_for_a_dropped_opponents_building() -> None:
    """Task-14 corpus fix: ``4pLeague_S64_D1L1_G6`` row 348, nomads'
    ``upgrade E3 to TP`` -- E3 is directly adjacent to two of *dropped*
    alchemists' still-standing dwellings. Real Perl's ``compute_leech``
    (module docstring's full citation trail) sums a color's building
    strength into ``%this_leech`` before it ever looks up which *living*
    faction currently holds that color, so a dropped opponent's building
    still counts here -- unlike ``offers_for_build``, which correctly
    never queues that dropped faction an actual offer for it (previous
    test). An earlier revision used ``bool(offers_for_build(...))`` for
    the isolation check in ``actions_build.handle_upgrade``, which came
    out wrongly isolated (doubled cost) the moment the dropped faction
    fell out of ``offers_for_build``'s seat-order walk.
    """
    s = _seeded()
    dropped_fs = replace(s.factions["darklings"], dropped=True)
    s = with_faction(s, "darklings", dropped_fs)
    assert offers_for_build(s, "engineers", TARGET) == ()
    assert has_leechable_neighbor(s, "engineers", TARGET) is True


# --------------------------------------------------------------------------
# accept / decline
# --------------------------------------------------------------------------


def test_accept_leech_gains_power_and_pays_vp() -> None:
    s = _seeded()
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=2, target="engineers"))
    fs = s2.factions["darklings"]
    assert fs.power.as_str() == "3/9/0"  # started 5/7/0; gain(2): both P1 tokens move to P2
    assert fs.vp == 20 - 1  # pay n-1 = 1 VP
    assert s2.pending == ()


def test_accept_leech_amount_defaults_to_offer_amount() -> None:
    s = _seeded()
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_leech(s, "darklings", _cmd("leech"))
    assert s2.pending == ()
    assert s2.factions["darklings"].vp == 19


def test_accept_bare_leech_matches_the_sole_offer_regardless_of_requested_amount() -> None:
    """Reference-game row 282: nomads' only outstanding offer is capped
    to amount 1 (module docstring), but the bare ``leech 4`` row (no
    ``from`` clause at all, early-era style) still names the corpus's own
    greedy request, 4. With exactly one queued offer for the faction,
    that's the answer regardless of what ``cmd.n1`` says."""
    s = _seeded()  # offer amount 2 (opponent TP, power 2)
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=4))
    assert s2.pending == ()
    assert s2.factions["darklings"].power.as_str() == "3/9/0"  # capped to the offer's own 2


def test_accept_leech_with_target_matches_regardless_of_requested_amount() -> None:
    """Reference-game row 155: nomads' offer amount is capped to 1 at
    offer-creation time (``gainable()`` had already dropped), but the
    ledger row still reads ``leech 2 from engineers`` -- a *request*, not
    a promise (module docstring's ``_find_leech_pending`` citation).  With
    an explicit ``from X`` clause, the offer's ``source`` alone
    disambiguates; ``cmd.n1`` no longer has to equal the offer's cached
    ``amount`` for the row to be found at all -- ``handle_leech`` still
    caps the *actual* gain down to whatever's affordable.
    """
    s = _seeded()  # offer amount 2 (opponent TP, power 2)
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=5, target="engineers"))
    assert s2.pending == ()
    assert s2.factions["darklings"].power.as_str() == "3/9/0"  # capped to the offer's own 2


def test_decline_leech_is_free() -> None:
    s = _seeded()
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_decline(s, "darklings", _cmd("decline", n1=2, target="engineers"))
    assert s2.pending == ()
    assert s2.factions["darklings"] == s.factions["darklings"]


def test_decline_bare_declines_every_outstanding_offer() -> None:
    s = _state()
    s = replace(
        s,
        pending=(
            PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers"),
            PendingDecision(faction="darklings", kind="leech", amount=3, source="nomads"),
        ),
    )
    s2 = handle_decline(s, "darklings", _cmd("decline"))
    assert s2.pending == ()
    assert s2.factions["darklings"] == s.factions["darklings"]


def test_leech_no_matching_pending_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_leech(s, "darklings", _cmd("leech", n1=2, target="engineers"))


def test_leech_vp_floor_caps_gain() -> None:
    s = _seeded(opponent_building="SH")  # raw power 3
    s = with_faction(s, "darklings", replace(s.factions["darklings"], vp=1))
    s = queue_leech(s, "engineers", TARGET)
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=3, target="engineers"))
    # actual capped to vp+1 = 2; pay 1 VP, leaving 0.
    assert s2.factions["darklings"].vp == 0
    assert s2.factions["darklings"].power.gainable() == 5 * 2 + 7 - 2  # gained 2 worth of steps


# --------------------------------------------------------------------------
# Cultists leech_effect
# --------------------------------------------------------------------------


def _cultists_seeded() -> GameState:
    s = _state()
    factions = dict(s.factions)
    factions["cultists"] = FactionState.initial(FACTIONS["cultists"])
    s = replace(s, factions=factions, turn_order=("cultists", "darklings", "nomads"))
    s = _place(s, "cultists", ANCHOR, "D", FACTIONS["cultists"].color)
    s = _place(s, "darklings", NEIGHBOR, "TP", FACTIONS["darklings"].color)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(
        hexes[TARGET], color=FACTIONS["cultists"].color, building=None, owner=None
    )
    return replace(s, hexes=hexes)


def test_cultists_any_accept_pushes_cult_choice_after_batch_resolves() -> None:
    s = _cultists_seeded()
    s = queue_leech(s, "cultists", TARGET)
    assert any(p.kind == "cultist_leech_watch" for p in s.pending)

    s2 = handle_leech(s, "darklings", _cmd("leech", n1=2, target="cultists"))
    assert s2.pending == (PendingDecision(faction="cultists", kind="cult_choice", amount=1),)


def test_cultists_first_accept_fires_cult_choice_immediately_mid_batch() -> None:
    """Two opponents adjacent to a Cultists build: the FIRST accept must
    push ``cult_choice`` right away, before the second offer resolves --
    not deferred to batch-end (code review Important-2 fix). A second
    accept/decline in the same batch must not push a duplicate.
    """
    s = _state()
    factions = dict(s.factions)
    factions["cultists"] = FactionState.initial(FACTIONS["cultists"])
    s = replace(s, factions=factions, turn_order=("cultists", "darklings", "nomads"))
    s = _place(s, "darklings", ANCHOR, "TP", FACTIONS["darklings"].color)
    s = _place(s, "nomads", NEIGHBOR, "TP", FACTIONS["nomads"].color)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(
        hexes[TARGET], color=FACTIONS["cultists"].color, building=None, owner=None
    )
    s = replace(s, hexes=hexes)

    s = queue_leech(s, "cultists", TARGET)
    assert len([p for p in s.pending if p.kind == "leech"]) == 2
    watch = next(p for p in s.pending if p.kind == "cultist_leech_watch")
    assert watch.amount == 2

    # darklings accepts first -- nomads' offer is still outstanding.
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=2, target="cultists"))
    remaining_leech = [p for p in s2.pending if p.kind == "leech"]
    assert [p.faction for p in remaining_leech] == ["nomads"]
    assert PendingDecision(faction="cultists", kind="cult_choice", amount=1) in s2.pending
    remaining_watch = next(p for p in s2.pending if p.kind == "cultist_leech_watch")
    assert remaining_watch.amount == 1  # batch not fully resolved yet

    # nomads then declines: no second cult_choice, and the watch clears.
    s3 = handle_decline(s2, "nomads", _cmd("decline", n1=2, target="cultists"))
    cult_choices = [p for p in s3.pending if p.kind == "cult_choice"]
    assert cult_choices == [PendingDecision(faction="cultists", kind="cult_choice", amount=1)]
    assert not any(p.kind == "cultist_leech_watch" for p in s3.pending)


def test_cultist_watch_disambiguates_by_hex_key_across_two_simultaneous_batches() -> None:
    """Task-14 fix, corpus ``4pLeague_S20_D1L1_G7`` row 217: Cultists can
    build/upgrade a second time before every offer from an earlier build
    is answered, queuing two simultaneous ``cultist_leech_watch``
    pendings with the identical ``source="cultists"``. Resolving an
    offer from the *second* batch must update the second batch's own
    watch (and, on first accept, push its own ``cult_choice``) -- not
    silently decrement/consume the first, older batch's watch just
    because it happens to come first in the pending list (an earlier
    revision matched by ``source`` alone). The older batch's watch must
    stay untouched.
    """
    s = _state()
    factions = dict(s.factions)
    factions["cultists"] = FactionState.initial(FACTIONS["cultists"])
    s = replace(s, factions=factions, turn_order=("cultists", "darklings"))
    # A1's only opponent-adjacent hex is A2 (a TP, raw power 2); A3 is
    # *also* adjacent to A2 but additionally to A4 (an SH, raw power 3),
    # so batch #1 (A1) and batch #2 (A3) end up with distinct offer
    # amounts (2 vs 5) -- avoiding _find_leech_pending's own documented
    # target/amount disambiguation ambiguity (a separate, pre-existing
    # constraint, not this fix's concern) so ``cmd.n1`` alone reliably
    # picks batch #2's offer below.
    s = _place(s, "darklings", "A2", "TP", FACTIONS["darklings"].color)
    s = _place(s, "darklings", "A4", "SH", FACTIONS["darklings"].color)
    hexes = dict(s.hexes)
    for hex_key in ("A1", "A3"):
        hexes[hex_key] = replace(
            hexes[hex_key], color=FACTIONS["cultists"].color, building=None, owner=None
        )
    s = replace(s, hexes=hexes)

    s = queue_leech(s, "cultists", "A1")  # batch #1, offer amount 2
    s = queue_leech(s, "cultists", "A3")  # batch #2, offer amount 5 -- same source="cultists"
    watches_by_hex = {p.options[0]: p for p in s.pending if p.kind == "cultist_leech_watch"}
    assert set(watches_by_hex) == {"A1", "A3"}
    assert watches_by_hex["A1"].amount == 1
    assert watches_by_hex["A3"].amount == 1

    second_offer = next(p for p in s.pending if p.kind == "leech" and p.options == ("A3",))
    assert second_offer.amount == 5
    s2 = handle_leech(s, "darklings", _cmd("leech", n1=second_offer.amount, target="cultists"))

    assert PendingDecision(faction="cultists", kind="cult_choice", amount=1) in s2.pending
    remaining_watches = {p.options[0]: p for p in s2.pending if p.kind == "cultist_leech_watch"}
    assert set(remaining_watches) == {"A1"}  # batch #2's watch (sole offer) fully resolved, popped
    assert remaining_watches["A1"] == watches_by_hex["A1"]  # batch #1 completely untouched


def test_apply_lets_cultists_answer_cult_choice_mid_batch_through_the_gate() -> None:
    """Regression test (code review round 2, apply()-level; updated
    task-13: ``active_faction`` no longer prefers ``pending``'s head --
    see ``state.py``'s docstring for why). With two opponents adjacent to
    a Cultists build, after the first accept the pending queue is
    [leech(nomads), cultist_leech_watch, cult_choice] -- ``active_faction``
    stays ``cultists`` (``turn_order[active_index]``, untouched by no
    ``advance_turn`` call happening in this handler-level test) throughout.
    Cultists must still be able to submit their ``+CULT`` answer *through
    `apply()`* (not just by calling the handler directly), and doing so
    must consume exactly their ``cult_choice`` pending (not the
    head-of-queue entry), leaving the still-outstanding ``nomads`` leech
    offer untouched.
    """
    s = _state()
    factions = dict(s.factions)
    factions["cultists"] = FactionState.initial(FACTIONS["cultists"])
    cults = {**s.cults, "cultists": dict(FACTIONS["cultists"].cults)}
    s = replace(s, factions=factions, cults=cults, turn_order=("cultists", "darklings", "nomads"))
    s = _place(s, "darklings", ANCHOR, "TP", FACTIONS["darklings"].color)
    s = _place(s, "nomads", NEIGHBOR, "TP", FACTIONS["nomads"].color)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(
        hexes[TARGET], color=FACTIONS["cultists"].color, building=None, owner=None
    )
    s = replace(s, hexes=hexes)
    s = queue_leech(s, "cultists", TARGET)

    # darklings accepts through apply() -- exempt via the queued-leech rule.
    s = apply(s, "darklings", _cmd("leech", n1=2, target="cultists"))
    assert [p.kind for p in s.pending] == ["leech", "cultist_leech_watch", "cult_choice"]
    assert active_faction(s) == "cultists"  # turn_order[active_index], unmoved

    # Before the original fix this raised EngineError("faction acted out of turn").
    s2 = apply(s, "cultists", _cmd("gain_cult", cult="FIRE", n1=1))

    assert s2.cults["cultists"]["FIRE"] == 2  # cultists start FIRE=1
    assert not any(p.kind == "cult_choice" for p in s2.pending)
    # The sibling nomads offer is untouched by cultists jumping the queue
    # to answer their own choice.
    remaining_leech = [p for p in s2.pending if p.kind == "leech"]
    assert [p.faction for p in remaining_leech] == ["nomads"]
    assert active_faction(s2) == "cultists"

    # It is still legitimately cultists' turn (no advance_turn has run) --
    # an ordinary main-track verb from them succeeds, same as real Perl's
    # `$assert_active_faction` (only checks `is_active`, never consults
    # `action_required`/pending -- state.py's docstring). A genuinely
    # out-of-turn faction is still rejected (test_apply.py's
    # test_apply_rejects_out_of_turn_non_exempt_move covers that).
    s3 = apply(s2, "cultists", _cmd("burn", n1=1))
    assert s3.factions["cultists"].power.bowl3 == s2.factions["cultists"].power.bowl3 + 1


def test_cultists_all_decline_clears_the_watch_but_grants_nothing_from_decline_itself() -> None:
    """Task-13 report follow-up (``4pLeague_S10_D1L1_G5`` row 208):
    ``handle_decline`` no longer grants the ``not_taken`` power itself --
    that now comes directly off the ledger's own
    ``"[all opponents declined power]"`` bracket row
    (``test_cultist_leech_bonus_grants_not_taken_power_directly`` below),
    since inferring it from the resolving decline lands it one ledger row
    late (``leech.py``'s module docstring). ``handle_decline`` still walks
    the watch marker to zero and pops it either way."""
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=True))
    )
    s = queue_leech(s, "cultists", TARGET)
    before = s.factions["cultists"].power
    s2 = handle_decline(s, "darklings", _cmd("decline", n1=2, target="cultists"))
    assert s2.pending == ()
    assert s2.factions["cultists"].power == before


def test_cultists_all_decline_without_errata_option_also_grants_nothing() -> None:
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=False))
    )
    s = queue_leech(s, "cultists", TARGET)
    before = s.factions["cultists"].power
    s2 = handle_decline(s, "darklings", _cmd("decline", n1=2, target="cultists"))
    assert s2.pending == ()
    assert s2.factions["cultists"].power == before


def test_cultist_leech_bonus_grants_not_taken_power_directly() -> None:
    """``handle_cultist_leech_bonus`` (``ledger_parser.py``'s
    ``cultist_leech_bonus`` verb): grants ``leech_effect["not_taken"]``
    (``{"PW": 1}``) straight to the row's own faction, independent of any
    pending/watch bookkeeping -- this is what actually lands the +1 PW at
    the ledger's checkpoint (task-13 report follow-up, reference row 208)."""
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=True))
    )
    before = s.factions["cultists"].power
    s2 = handle_cultist_leech_bonus(s, "cultists", _cmd("cultist_leech_bonus"))
    assert s2.factions["cultists"].power == before.gain(1)


def test_cultist_leech_bonus_without_errata_option_is_rejected() -> None:
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=False))
    )
    with pytest.raises(EngineError):
        handle_cultist_leech_bonus(s, "cultists", _cmd("cultist_leech_bonus"))


def test_non_cultists_builder_never_gets_a_watch_marker() -> None:
    s = _seeded()
    s = queue_leech(s, "engineers", TARGET)
    assert not any(p.kind == "cultist_leech_watch" for p in s.pending)
