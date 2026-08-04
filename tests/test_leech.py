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
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.leech import handle_decline, handle_leech, offers_for_build, queue_leech
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, PendingDecision, Phase, with_faction

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
        PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers"),
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
        PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers"),
    )


def test_offers_for_build_zero_gainable_still_enqueues_an_offer() -> None:
    """Perl's ``note_leech`` queues an offer regardless of ``actual`` being
    0 -- only the raw building-power sum gates whether an offer exists."""
    s = _seeded()
    s = with_faction(s, "darklings", replace(s.factions["darklings"], power=Power(0, 0, 12)))
    offers = offers_for_build(s, "engineers", TARGET)
    assert offers == (
        PendingDecision(faction="darklings", kind="leech", amount=0, source="engineers"),
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


def test_cultists_all_decline_grants_power_under_errata_option() -> None:
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=True))
    )
    s = queue_leech(s, "cultists", TARGET)
    before = s.factions["cultists"].power
    s2 = handle_decline(s, "darklings", _cmd("decline", n1=2, target="cultists"))
    assert s2.pending == ()
    assert s2.factions["cultists"].power == before.gain(1)


def test_cultists_all_decline_without_errata_option_grants_nothing() -> None:
    s = _cultists_seeded()
    s = replace(
        s, setup=replace(s.setup, options=replace(s.setup.options, errata_cultist_power=False))
    )
    s = queue_leech(s, "cultists", TARGET)
    before = s.factions["cultists"].power
    s2 = handle_decline(s, "darklings", _cmd("decline", n1=2, target="cultists"))
    assert s2.pending == ()
    assert s2.factions["cultists"].power == before


def test_non_cultists_builder_never_gets_a_watch_marker() -> None:
    s = _seeded()
    s = queue_leech(s, "engineers", TARGET)
    assert not any(p.kind == "cultist_leech_watch" for p in s.pending)
