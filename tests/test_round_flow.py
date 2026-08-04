"""tests/test_round_flow.py

Scenarios validated against jsnell/terra-mystica ``src/acting.pm``
(``setup_order``, turn/phase machinery) and the crawled-corpus empirical
income-split evidence -- see ``round_flow.py``'s module docstring for the
exact citation trail (game ``4pLeague_S10_D1L1_G1``, ledger rows 24-45,
94-97, 100-103, 136-139, 179-182).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.power import Power
from bgai.engine.tm.round_flow import (
    advance_turn,
    begin_actions,
    end_of_round,
    grant_missing_cult_income,
    handle_income_row,
    is_turn_boundary,
    never_starts_action,
    start_setup,
)
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

# rows 28-36 of the reference game, in ledger order.
_REFERENCE_SETUP_ROWS = (
    ("engineers", "E7"),
    ("darklings", "G5"),
    ("nomads", "D3"),
    ("mermaids", "D2"),
    ("mermaids", "D5"),
    ("nomads", "F3"),
    ("darklings", "E10"),
    ("engineers", "F6"),
    ("nomads", "G4"),
)

# A second reference game -- roster engineers/halflings/chaosmagicians/mermaids
# (no Nomads), exercising acting.pm:186-189's *other* branch (a single-dwelling
# faction placed last) instead of the 3-dwelling one. Ledger rows 28-34.
CM_GAME_ID = "4pLeague_S10_D2L1_G4"
_CM_REFERENCE_SETUP_ROWS = (
    ("engineers", "E7"),
    ("halflings", "F5"),
    ("mermaids", "D5"),
    ("mermaids", "E4"),
    ("halflings", "D8"),
    ("engineers", "C5"),
    ("chaosmagicians", "D7"),
)
# rows 35-38: reverse seat order (mermaids, chaosmagicians, halflings, engineers).
_CM_REFERENCE_BONUS_PICKS = (
    ("mermaids", "BON1"),
    ("chaosmagicians", "BON4"),
    ("halflings", "BON5"),
    ("engineers", "BON2"),
)


def _state() -> GameState:
    s = GameState.initial(load_setup(GAME_ID))
    return replace(s, round=1, phase=Phase.ACTIONS)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


def _rich(state: GameState, faction: str, **fields: object) -> GameState:
    fs = replace(state.factions[faction], **fields)  # type: ignore[arg-type]
    return with_faction(state, faction, fs)


# --------------------------------------------------------------------------
# Income
# --------------------------------------------------------------------------


def test_other_income_grants_base_and_building_and_bonus_and_favor_income() -> None:
    s = _state()
    s = _rich(s, "engineers", coins=0, workers=0)
    s = _rich(s, "engineers", bonus="BON3")  # {"C": 6}
    s2 = handle_income_row(s, "engineers", _cmd("other_income_for_faction"))
    fs = s2.factions["engineers"]
    # engineers D income floor is 0 (module docstring: Engineers' D track
    # starts at 0, unlike every other faction's floor-1) -- no dwellings
    # built, so building/base income is 0; only BON3's {C: 6} lands.
    assert fs.coins == 6
    assert fs.workers == 0


def test_other_income_includes_favor_income() -> None:
    s = _state()
    s = _rich(s, "darklings", favors=("FAV7",), coins=0, workers=0, power=Power(1, 0, 0))
    s2 = handle_income_row(s, "darklings", _cmd("other_income_for_faction"))
    fs = s2.factions["darklings"]
    # FAV7 income={"W": 1, "PW": 1}; darklings' D floor also adds nothing
    # (D_STD floor is 1 W -- darklings uses D_STD) so total W = 1 (floor) + 1 (FAV7).
    assert fs.workers == 1 + 1
    assert fs.power == Power(0, 1, 0)  # 1 PW gain moves a bowl1 token to bowl2


def test_cult_income_scales_by_floor_division_7_earth_req_4() -> None:
    """Brief's exact scenario: 7 EARTH, req 4 -> 1 unit. Round 4's tile
    (cult EARTH, req 4, income {"SPADE": 1}) is used directly.
    """
    s = replace(_state(), round=4)
    cults = dict(s.cults)
    cults["nomads"] = {**cults["nomads"], "EARTH": 7}
    s = replace(s, cults=cults)
    before = s.factions["nomads"].spades_available
    s2 = handle_income_row(s, "nomads", _cmd("cult_income_for_faction"))
    assert s2.factions["nomads"].spades_available == before + 1


def test_cult_income_below_req_grants_nothing() -> None:
    s = replace(_state(), round=4)
    cults = dict(s.cults)
    cults["nomads"] = {**cults["nomads"], "EARTH": 3}
    s = replace(s, cults=cults)
    before = s.factions["nomads"].spades_available
    s2 = handle_income_row(s, "nomads", _cmd("cult_income_for_faction"))
    assert s2.factions["nomads"].spades_available == before


def test_grant_missing_cult_income_matches_handle_income_row() -> None:
    """``grant_missing_cult_income`` (``replay.py``'s public entry point
    for a faction whose ``cult_income_for_faction`` ledger row is
    genuinely absent, task-14 fix) must compute *exactly* the same grant
    as a real ``cult_income_for_faction`` row would via
    ``handle_income_row`` -- it is the same underlying
    ``_grant_cult_income``, just callable without constructing a
    throwaway ``ParsedCommand``.
    """
    s = replace(_state(), round=4)
    cults = dict(s.cults)
    cults["nomads"] = {**cults["nomads"], "EARTH": 7}
    s = replace(s, cults=cults)
    via_row = handle_income_row(s, "nomads", _cmd("cult_income_for_faction"))
    via_wrapper = grant_missing_cult_income(s, "nomads")
    assert via_wrapper == via_row


def test_cult_income_round_1_spade_destination_is_spades_available() -> None:
    """SCORE1 in this game: cult WATER, req 4, income {"SPADE": 1} --
    "4 WATER -> 1 SPADE" (brief's exact scenario)."""
    s = _state()  # round 1
    cults = dict(s.cults)
    cults["engineers"] = {**cults["engineers"], "WATER": 4}
    s = replace(s, cults=cults)
    s2 = handle_income_row(s, "engineers", _cmd("cult_income_for_faction"))
    assert s2.factions["engineers"].spades_available == 1


def test_cult_income_spade_scores_halflings_unconditional_spade_vp_bonus() -> None:
    """Analogous to ``4pLeague_S10_D1L1_G2`` row 142 (halflings' round-2
    cult income, AIR position 4 req 4, tile income ``{"SPADE": 1}``,
    grants 1 spade): using this fixture's own round-1 tile (WATER, req 4,
    income ``{"SPADE": 1}`` -- the brief's own scenario), a SPADE cult
    income grant must also score halflings' unconditional +1 VP/spade
    special (``factions_data.py`` ``special_gain["SPADE"] = {"VP": 1}``)
    -- ``resources.pm``'s generic per-unit gain loop fires for *any*
    positive SPADE delta, cult income included, not just ``dig``'s
    (``_apply_spade_income_bonus``'s own docstring; task-13 report)."""
    s = _state()
    factions = dict(s.factions)
    factions["halflings"] = FactionState.initial(FACTIONS["halflings"])
    cults = {**s.cults, "halflings": {**FACTIONS["halflings"].cults, "WATER": 4}}
    s = replace(s, factions=factions, cults=cults)
    before_vp = s.factions["halflings"].vp
    s2 = handle_income_row(s, "halflings", _cmd("cult_income_for_faction"))
    assert s2.factions["halflings"].spades_available == 1
    assert s2.factions["halflings"].vp == before_vp + 1


def test_cult_income_temple_scoring_tile_uses_priest_slot_occupancy() -> None:
    """Task-14 fix, corpus ``4pLeague_S12_D1L1_G1`` rows 150/151 (round 2's
    tile there is the optional SCORE9/``temple-scoring-tile``, cult
    ``CULT_P``, req 1, income ``{"C": 2}``): engineers/alchemists each hold
    exactly 1 occupied priest slot (a prior ``send`` that landed on an
    empty step) and are owed 2 C; nomads/witches hold 0 and correctly get
    nothing (sibling test below). ``CULT_P`` is not a real cult track
    (``state.cults`` has no such key) -- ``_cult_p_position`` sources it
    from ``state.priest_slots`` occupancy instead, mirroring
    ``commands.pm``'s ``$faction->{CULT_P}`` counter (incremented once per
    ``send`` that claims a previously-empty slot, ``command_send`` line
    334).
    """
    from bgai.engine.tm.tiles import ScoringTile

    temple_tile = ScoringTile(
        cult="CULT_P", req=1, vp_mode="build", vp=(("TE", 4),), cult_income=(("C", 2),)
    )
    s = _state()
    s = replace(s, setup=replace(s.setup, score_tiles=(temple_tile,) + s.setup.score_tiles[1:]))
    slots = dict(s.priest_slots)
    slots["FIRE"] = ("engineers", None, None, None)
    s = replace(s, priest_slots=slots)
    before = s.factions["engineers"].coins
    s2 = handle_income_row(s, "engineers", _cmd("cult_income_for_faction"))
    assert s2.factions["engineers"].coins == before + 2


def test_cult_income_temple_scoring_tile_grants_nothing_with_no_occupied_slots() -> None:
    from bgai.engine.tm.tiles import ScoringTile

    temple_tile = ScoringTile(
        cult="CULT_P", req=1, vp_mode="build", vp=(("TE", 4),), cult_income=(("C", 2),)
    )
    s = _state()
    s = replace(s, setup=replace(s.setup, score_tiles=(temple_tile,) + s.setup.score_tiles[1:]))
    before = s.factions["nomads"].coins
    s2 = handle_income_row(s, "nomads", _cmd("cult_income_for_faction"))
    assert s2.factions["nomads"].coins == before


def test_all_income_for_faction_grants_both_components() -> None:
    s = replace(_state(), round=4)
    s = _rich(s, "nomads", coins=0, workers=0)
    cults = dict(s.cults)
    cults["nomads"] = {**cults["nomads"], "EARTH": 4}
    s = replace(s, cults=cults)
    s2 = handle_income_row(s, "nomads", _cmd("all_income_for_faction"))
    fs = s2.factions["nomads"]
    assert fs.spades_available == 1  # cult component
    assert fs.workers >= 0  # other-income component ran without error


def test_other_income_matches_a_third_independent_corpus_game() -> None:
    """Game 3 of the Step-1 empirical evidence (module docstring):
    ``4pLeague_S10_D1L1_G2``, darklings' row-148 ``other_income_for_faction``
    -- board D=3/TP=1/TE=1, holding BON10 (``{"PW": 3}``) and FAV9
    (``{"C": 3}``, gained at row 103). Corpus deltas: C +5, W +4, P +1.
    This isolates the favor-income contribution specifically (FAV9's C:3
    is the only source of C besides TP's own C:2).
    """
    s = _state()
    s = _rich(
        s,
        "darklings",
        coins=0,
        workers=0,
        priests=0,
        bonus="BON10",
        favors=("FAV9",),
        buildings={
            **s.factions["darklings"].buildings,
            "D": frozenset({"d1", "d2", "d3"}),
            "TP": frozenset({"tp1"}),
            "TE": frozenset({"te1"}),
        },
    )
    s2 = handle_income_row(s, "darklings", _cmd("other_income_for_faction"))
    fs = s2.factions["darklings"]
    assert fs.coins == 5
    assert fs.workers == 4
    assert fs.priests == 1


def test_income_row_verb_gated_exempt_from_turn_order() -> None:
    """Any faction's income row is applied regardless of whose 'turn' it
    nominally is (module docstring / apply.py's gate exemption)."""
    s = _state()  # active_faction is engineers (turn_order[0])
    assert active_faction(s) == "engineers"
    s2 = apply(s, "mermaids", _cmd("other_income_for_faction"))
    assert s2 is not None  # did not raise


def test_begin_actions_transitions_income_to_actions() -> None:
    s = replace(_state(), phase=Phase.INCOME, active_index=2)
    s2 = begin_actions(s)
    assert s2.phase == Phase.ACTIONS
    assert s2.active_index == 0


def test_begin_actions_rejects_wrong_phase() -> None:
    with pytest.raises(ValueError):
        begin_actions(_state())  # already ACTIONS


def test_begin_actions_skips_a_dropped_faction_at_turn_order_zero() -> None:
    """Task-14 fix: ``begin_actions`` previously always seeded
    ``active_index=0`` unconditionally -- if ``turn_order[0]`` happened
    to be a faction that dropped earlier in the game (e.g. mid the
    *previous* round's ``SETUP_BONUS``/``ACTIONS`` phase), round 1's very
    first ``active_faction`` would wrongly be that dropped faction, since
    ``_advance_actions``'s own dropped-skip logic only ever runs when
    *stepping forward* from an already-valid position, never when
    ``active_index`` is freshly seeded. Corpus: ``4pLeague_S21_D3L1_G4``
    row 47, darklings (seat index 0, dropped mid-``SETUP_BONUS`` at row
    41) wrongly blocked engineers' very first round-1 action.
    """
    s = replace(_state(), phase=Phase.INCOME, active_index=2)
    assert s.turn_order[0] == "engineers"
    s = _rich(s, "engineers", dropped=True)
    s2 = begin_actions(s)
    assert s2.phase == Phase.ACTIONS
    assert active_faction(s2) == "darklings"


def test_begin_actions_clears_teleported_hex_for_every_faction() -> None:
    """The round's first active faction (``turn_order[0]``) never goes
    through ``_advance_actions``'s own reset, so ``begin_actions`` is the
    other half of Perl's ``start_full_move`` reset (``FactionState.
    teleported_hex`` docstring, task-14 fix)."""
    s = replace(_state(), phase=Phase.INCOME, active_index=2)
    s = _rich(s, "engineers", teleported_hex="A1")
    s = _rich(s, "mermaids", teleported_hex="B2")
    s2 = begin_actions(s)
    assert s2.factions["engineers"].teleported_hex is None
    assert s2.factions["mermaids"].teleported_hex is None


# --------------------------------------------------------------------------
# end_of_round
# --------------------------------------------------------------------------


def test_end_of_round_increments_bonus_coins_only_on_untaken_tiles() -> None:
    s = replace(_state(), phase=Phase.CLEANUP)
    s = _rich(s, "engineers", bonus="BON1")
    s2 = end_of_round(s)
    assert s2.bonus_coins["BON1"] == 0  # held -> no accumulation
    for tile in s.setup.bonus_tiles:
        if tile != "BON1":
            assert s2.bonus_coins[tile] == 1


def test_end_of_round_resets_per_round_balances_but_not_spades() -> None:
    """``spades_available`` deliberately survives ``end_of_round`` (that
    function's own docstring, task-13 fix): reference-game rows 96-98
    need a cleanup-granted cult-income spade to still be there for a
    ``transform`` two rows -- and one ``end_of_round`` call -- later."""
    s = replace(_state(), phase=Phase.CLEANUP)
    s = _rich(
        s,
        "engineers",
        actions_used=frozenset({"BON1"}),
        spades_available=2,
        extra_actions=1,
        passed=True,
    )
    s = replace(s, power_actions_taken=frozenset({"ACT4"}))
    s2 = end_of_round(s)
    fs = s2.factions["engineers"]
    assert fs.actions_used == frozenset()
    assert fs.spades_available == 2  # NOT reset
    assert fs.extra_actions == 0
    assert fs.passed is False
    assert s2.power_actions_taken == frozenset()


def test_end_of_round_advances_to_next_income_phase() -> None:
    s = replace(_state(), phase=Phase.CLEANUP, round=2)
    s2 = end_of_round(s)
    assert s2.round == 3
    assert s2.phase == Phase.INCOME
    assert s2.active_index == 0
    assert s2.passed_order == ()


def test_end_of_round_after_round_6_transitions_to_finished() -> None:
    s = replace(_state(), phase=Phase.CLEANUP, round=6)
    s2 = end_of_round(s)
    assert s2.round == 6
    assert s2.phase == Phase.FINISHED


_ALT_PASSED_ORDER = ("mermaids", "nomads", "darklings", "engineers")


def _setup_with_variable_turn_order(enabled: bool) -> GameState:
    base = load_setup(GAME_ID)
    setup = replace(base, options=replace(base.options, variable_turn_order=enabled))
    return GameState.initial(setup)


def test_end_of_round_rotates_seat_order_to_the_first_passer_without_variable_turn_order() -> None:
    """Task-14 correction: without ``variable-turn-order``,
    ``raw_factions_in_order`` is never touched by individual passes
    (``end_of_round``'s own docstring has the full citation trail and
    corpus counter-example, ``4pLeague_S1_D1L1_G1``) -- only *which* seat
    starts next round is chronological (whoever passed first), the other
    3 keep their original seat *positions*, not their pass order. Seats
    here are (engineers, darklings, nomads, mermaids);
    ``_ALT_PASSED_ORDER``'s first entry is mermaids (seat index 3), so
    the correct result rotates seat order to start there: (mermaids,
    engineers, darklings, nomads) -- not ``_ALT_PASSED_ORDER`` itself,
    and not unrotated seat order either."""
    s = _setup_with_variable_turn_order(False)
    s = replace(s, phase=Phase.CLEANUP, passed_order=_ALT_PASSED_ORDER)
    s2 = end_of_round(s)
    assert s2.turn_order == ("mermaids", "engineers", "darklings", "nomads")


def test_end_of_round_uses_passed_order_under_variable_turn_order() -> None:
    s = _setup_with_variable_turn_order(True)
    s = replace(s, phase=Phase.CLEANUP, passed_order=_ALT_PASSED_ORDER)
    s2 = end_of_round(s)
    assert s2.turn_order == _ALT_PASSED_ORDER


def test_end_of_round_rejects_wrong_phase() -> None:
    with pytest.raises(ValueError):
        end_of_round(_state())  # ACTIONS, not CLEANUP


# --------------------------------------------------------------------------
# Setup snake order: reproduces the reference game's ledger rows 28-36
# --------------------------------------------------------------------------


def test_setup_dwellings_snake_order_reproduces_reference_game() -> None:
    s = GameState.initial(load_setup(GAME_ID))
    s = start_setup(s)
    for faction, hex_key in _REFERENCE_SETUP_ROWS:
        assert active_faction(s) == faction
        s = apply(s, faction, _cmd("build", loc=hex_key))
        s = advance_turn(s)
    assert s.phase == Phase.SETUP_BONUS
    assert s.turn_order == tuple(reversed(s.setup.factions))
    assert s.active_index == 0


def test_setup_dwellings_wrong_order_build_raises() -> None:
    s = start_setup(GameState.initial(load_setup(GAME_ID)))
    # first real move should be engineers at E7; try darklings instead.
    with pytest.raises(EngineError):
        apply(s, "darklings", _cmd("build", loc="G5"))


def test_advance_setup_dwellings_skips_every_remaining_slot_of_a_dropped_faction() -> None:
    """``commands.pm``'s ``drop-faction`` handler removes *every*
    remaining ``setup_order`` entry for the dropped faction outright
    (~1597-1599), not just the very next one -- a faction that drops
    before its own forward-order dwelling turn arrives loses *both* its
    slots (forward and reverse) in one shot. Corpus:
    ``4pLeague_S45_D3L4_G1``, darklings drops at row 30, exactly when its
    own forward pick would start; the reverse-order pass a few rows later
    goes straight from swarmlings' own forward pick to swarmlings' own
    reverse pick with no darklings row between (task-14 fix).
    """
    s = GameState.initial(load_setup(GAME_ID))
    s = start_setup(s)
    # order: engineers, darklings, nomads, mermaids, mermaids, nomads,
    # darklings, engineers, nomads (test_setup_dwellings_order_gives_...
    # below pins this exact sequence). Drop darklings right after
    # engineers' first (forward) pick.
    assert active_faction(s) == "engineers"
    s = apply(s, "engineers", _cmd("build", loc="E7"))
    s = advance_turn(s)
    assert active_faction(s) == "darklings"
    s = with_faction(s, "darklings", replace(s.factions["darklings"], dropped=True))
    s = advance_turn(s)
    # darklings' forward slot is gone -- nomads goes next, not darklings.
    assert active_faction(s) == "nomads"


def test_setup_dwellings_order_gives_nomads_a_third_and_earlier_factions_two() -> None:
    from bgai.engine.tm.round_flow import _setup_dwellings_order

    order = _setup_dwellings_order(load_setup(GAME_ID))
    assert order.count("nomads") == 3
    assert order.count("engineers") == 2
    assert order.count("darklings") == 2
    assert order.count("mermaids") == 2
    assert order == (
        "engineers",
        "darklings",
        "nomads",
        "mermaids",
        "mermaids",
        "nomads",
        "darklings",
        "engineers",
        "nomads",
    )


def test_setup_bonus_reverse_order_then_round_1_income() -> None:
    s = GameState.initial(load_setup(GAME_ID))
    s = start_setup(s)
    for faction, hex_key in _REFERENCE_SETUP_ROWS:
        s = apply(s, faction, _cmd("build", loc=hex_key))
        s = advance_turn(s)
    assert s.phase == Phase.SETUP_BONUS
    reverse_seats = tuple(reversed(s.setup.factions))
    tiles = iter(("BON1", "BON5", "BON3", "BON4"))
    for faction in reverse_seats:
        assert active_faction(s) == faction
        s = apply(s, faction, _cmd("pass", tile=next(tiles)))
        s = advance_turn(s)
    assert s.phase == Phase.INCOME
    assert s.round == 1
    assert s.turn_order == s.setup.factions


def test_setup_bonus_transition_bumps_coins_on_every_untaken_tile() -> None:
    """Reference-game row 83: nomads takes BON7 (untouched by any of the
    4 SETUP_BONUS picks -- BON1/BON5/BON3/BON4 above) for exactly 1 C,
    still within round 1's own ACTIONS phase (round 1's own cleanup
    hasn't run yet). Only explained by ``command_start``'s bonus-coin
    bump firing on *every* round transition, including 0 -> 1
    (``_bumped_bonus_coins``'s own docstring; task-13 report)."""
    s = GameState.initial(load_setup(GAME_ID))
    s = start_setup(s)
    for faction, hex_key in _REFERENCE_SETUP_ROWS:
        s = apply(s, faction, _cmd("build", loc=hex_key))
        s = advance_turn(s)
    for faction, tile in zip(
        tuple(reversed(s.setup.factions)), ("BON1", "BON5", "BON3", "BON4"), strict=True
    ):
        s = apply(s, faction, _cmd("pass", tile=tile))
        s = advance_turn(s)
    assert s.phase == Phase.INCOME and s.round == 1
    taken = {"BON1", "BON5", "BON3", "BON4"}
    for tile in s.setup.bonus_tiles:
        assert s.bonus_coins[tile] == (0 if tile in taken else 1)


def test_setup_dwellings_snake_order_reproduces_chaos_magicians_game() -> None:
    """Second, independent corpus game (no Nomads): exercises the
    single-dwelling-faction-placed-last branch of ``acting.pm``'s
    setup_order (module docstring citation lines 186-189) instead of the
    3-dwelling-faction-extra-turn branch the reference game covers.
    Chaos Magicians (``start_dwellings=1``) sit out both the forward and
    reverse passes entirely and place their one dwelling last.
    """
    s = GameState.initial(load_setup(CM_GAME_ID))
    s = start_setup(s)
    for faction, hex_key in _CM_REFERENCE_SETUP_ROWS:
        assert active_faction(s) == faction
        s = apply(s, faction, _cmd("build", loc=hex_key))
        s = advance_turn(s)
    assert s.phase == Phase.SETUP_BONUS
    assert s.turn_order == tuple(reversed(s.setup.factions))
    assert s.active_index == 0
    # chaosmagicians placed exactly once, last, after 6 other builds.
    assert len(s.factions["chaosmagicians"].buildings["D"]) == 1
    assert len(s.factions["engineers"].buildings["D"]) == 2
    assert len(s.factions["halflings"].buildings["D"]) == 2
    assert len(s.factions["mermaids"].buildings["D"]) == 2

    for faction, tile in _CM_REFERENCE_BONUS_PICKS:
        assert active_faction(s) == faction
        s = apply(s, faction, _cmd("pass", tile=tile))
        s = advance_turn(s)
    assert s.phase == Phase.INCOME
    assert s.round == 1
    assert s.turn_order == s.setup.factions


def test_setup_dwellings_order_places_single_dwelling_faction_last() -> None:
    from bgai.engine.tm.round_flow import _setup_dwellings_order

    order = _setup_dwellings_order(load_setup(CM_GAME_ID))
    assert order == (
        "engineers",
        "halflings",
        "mermaids",
        "mermaids",
        "halflings",
        "engineers",
        "chaosmagicians",
    )
    assert order[-1] == "chaosmagicians"
    assert order.count("chaosmagicians") == 1


# --------------------------------------------------------------------------
# advance_turn: ACTIONS phase
# --------------------------------------------------------------------------


def test_advance_turn_skips_passed_factions() -> None:
    s = _state()  # turn_order engineers/darklings/nomads/mermaids, active engineers
    s = _rich(s, "darklings", passed=True)
    s2 = advance_turn(s)
    assert active_faction(s2) == "nomads"


def test_advance_turn_wraps_around() -> None:
    s = replace(_state(), active_index=3)  # mermaids active (last seat)
    s2 = advance_turn(s)
    assert active_faction(s2) == "engineers"


def test_advance_turn_spends_extra_actions_without_moving_to_next_faction() -> None:
    s = _state()
    s = _rich(s, "engineers", extra_actions=2)
    s2 = advance_turn(s)
    assert active_faction(s2) == "engineers"
    assert s2.factions["engineers"].extra_actions == 1
    s3 = advance_turn(s2)
    assert active_faction(s3) == "engineers"
    assert s3.factions["engineers"].extra_actions == 0
    s4 = advance_turn(s3)
    assert active_faction(s4) == "darklings"


def test_advance_turn_clears_cult_blocked_for_the_newly_active_faction() -> None:
    """Task-14 fix, corpus ``4pLeague_S10_D3L1_G4`` row 329/336: a cult
    step capped at 9 for lack of a key does not carry into a *later*
    turn -- Perl's ``start_full_move`` deletes ``$faction->{cult_blocked}``
    every fresh full action (``_start_full_move_reset`` docstring). A key
    gained in a different, later turn has nothing queued to retry once
    this reset has run (contrast ``test_gain_town_retries_a_cult_blocked_
    at_9_once_its_own_key_covers_it`` in ``test_actions_build.py``, whose
    block and retrying key land in the *same* turn/row, never touching
    this reset at all)."""
    s = _state()
    s = _rich(s, "darklings", cult_blocked=frozenset({"FIRE"}))
    s2 = advance_turn(s)  # engineers -> darklings
    assert active_faction(s2) == "darklings"
    assert s2.factions["darklings"].cult_blocked == frozenset()


def test_advance_turn_clears_teleported_hex_for_the_newly_active_faction() -> None:
    """``FactionState.teleported_hex`` docstring, task-14 fix: Perl's
    ``start_full_move`` deletes ``TELEPORT_TO`` every time a faction
    becomes newly active -- a Dwarves/Fakirs tunnel fee paid last turn
    must not exempt this turn's tunnel use on a *different* hex (or even
    the same one, per the corpus: ``4pLeague_S10_D3L2_G6`` row 350 pays
    the same hex's fee again)."""
    s = _state()
    s = _rich(s, "darklings", teleported_hex="A1")
    s2 = advance_turn(s)  # engineers -> darklings
    assert active_faction(s2) == "darklings"
    assert s2.factions["darklings"].teleported_hex is None


def test_advance_turn_clears_teleported_hex_when_spending_an_extra_action() -> None:
    """Same reset, for the ACTC "same faction, fresh full action" case
    (``_advance_actions``'s ``extra_actions`` branch)."""
    s = _state()
    s = _rich(s, "engineers", extra_actions=1, teleported_hex="A1")
    s2 = advance_turn(s)
    assert active_faction(s2) == "engineers"
    assert s2.factions["engineers"].teleported_hex is None


def test_advance_turn_transitions_to_cleanup_once_everyone_passed() -> None:
    s = _state()
    for name in s.turn_order:
        s = _rich(s, name, passed=True)
    s2 = advance_turn(s)
    assert s2.phase == Phase.CLEANUP


def test_advance_turn_is_a_noop_during_income_and_cleanup() -> None:
    s_income = replace(_state(), phase=Phase.INCOME)
    assert advance_turn(s_income) == s_income
    s_cleanup = replace(_state(), phase=Phase.CLEANUP)
    assert advance_turn(s_cleanup) == s_cleanup


# --------------------------------------------------------------------------
# is_turn_boundary / never_starts_action: ACTC compound-turn bundling
# (task-14 fix -- corpus row 308, ``action ACTC; action BON2; gain_cult
# n1=1; pass BON3``, 3 Perl-level full actions bundled into one ledger row)
# --------------------------------------------------------------------------


def test_action_pass_advance_send_are_always_turn_boundaries() -> None:
    s = _state()
    for verb in ("action", "pass", "advance", "send"):
        assert is_turn_boundary(_cmd(verb), s, "engineers", prev_verb="action") is True


def test_transform_and_connect_never_start_a_fresh_action() -> None:
    s = _state()
    for verb in ("transform", "connect"):
        assert is_turn_boundary(_cmd(verb), s, "engineers", prev_verb="action") is False
        assert never_starts_action(verb) is True
    assert never_starts_action("action") is False
    assert never_starts_action("build") is False


def test_build_after_dig_or_transform_is_a_continuation() -> None:
    s = _rich(_state(), "engineers", spades_available=0)
    for prev in ("dig", "transform"):
        assert is_turn_boundary(_cmd("build", loc="A1"), s, "engineers", prev_verb=prev) is False


def test_build_with_no_spade_or_marker_context_is_fresh() -> None:
    # Corpus pattern ``action ACTC; build; build`` -- a bare second build,
    # not preceded by dig/transform and with no leftover spade balance or
    # marker, is a genuinely independent action.
    s = _rich(_state(), "engineers", spades_available=0)
    assert is_turn_boundary(_cmd("build", loc="A1"), s, "engineers", prev_verb="build") is True


def test_build_with_unspent_spades_available_is_a_continuation() -> None:
    # ACT5/ACT6/BON1 grant SPADE directly, no intervening ``transform``/
    # ``dig`` command -- e.g. ``action BON1; build F2`` (corpus row 364).
    s = _rich(_state(), "engineers", spades_available=1)
    assert is_turn_boundary(_cmd("build", loc="A1"), s, "engineers", prev_verb="action") is False


def test_build_with_pending_free_marker_is_a_continuation() -> None:
    s = _state()
    for kind in ("free_d", "free_tf"):
        s2 = replace(s, pending=(PendingDecision(faction="engineers", kind=kind),))
        boundary = is_turn_boundary(_cmd("build", loc="A1"), s2, "engineers", prev_verb="action")
        assert boundary is False


def test_upgrade_with_pending_free_tp_marker_is_a_continuation() -> None:
    s = replace(_state(), pending=(PendingDecision(faction="engineers", kind="free_tp"),))
    cmd = _cmd("upgrade", loc="A1", building="TP")
    assert is_turn_boundary(cmd, s, "engineers", prev_verb="action") is False
