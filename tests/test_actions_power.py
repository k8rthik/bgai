"""tests/test_actions_power.py

Scenarios validated against jsnell/terra-mystica ``src/Game/Constants.pm``
(``%actions``), ``src/commands.pm`` (``command_action``, ``command_build``,
``command_upgrade``, ``command_transform``) and ``src/resources.pm``
(``adjust_resource``) -- see ``actions_power.py``'s module docstring for the
exact citation trail, and the crawled-corpus evidence quoted there.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.actions_build import handle_build, handle_upgrade
from bgai.engine.tm.actions_power import handle_action, handle_lose_marker
from bgai.engine.tm.actions_terraform import handle_transform
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

GAME_ID = "4pLeague_S10_D1L1_G1"
BOARD = base_board()

_EXTRA_FACTIONS = ("auren", "chaosmagicians", "giants", "swarmlings", "witches")


def _state() -> GameState:
    """GAME_ID's real roster (engineers/darklings/nomads/mermaids) plus a
    fresh ``FactionState`` for every faction these tests exercise but that
    isn't in that roster -- mirrors ``test_actions_terraform.py``'s
    ``_state`` pattern.
    """
    s = GameState.initial(load_setup(GAME_ID))
    factions = dict(s.factions)
    cults = dict(s.cults)
    for name in _EXTRA_FACTIONS:
        factions[name] = FactionState.initial(FACTIONS[name])
        cults[name] = dict(FACTIONS[name].cults)
    return replace(s, round=1, phase=Phase.ACTIONS, factions=factions, cults=cults)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


def _rich(state: GameState, faction: str, **fields: object) -> GameState:
    fs = replace(state.factions[faction], **fields)  # type: ignore[arg-type]
    return with_faction(state, faction, fs)


_SH_STUB_HEX = "I9"  # far corner of the board, distinct from ANCHOR/TARGET/_far_hex()


def _with_sh(state: GameState, faction: str, hex_key: str | None = None) -> GameState:
    """Mark ``faction`` as having a built stronghold on a real board hex
    (needed even for tests that don't otherwise care where -- an SH marker
    that isn't a real ``state.hexes`` key breaks ``_maybe_queue_town``'s
    cluster walk in ``handle_build``/``handle_upgrade``).
    """
    hex_key = hex_key or _SH_STUB_HEX
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, "SH": fs.buildings["SH"] | {hex_key}})
    state = with_faction(state, faction, fs)
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(
        hexes[hex_key], color=FACTIONS[faction].color, building="SH", owner=faction
    )
    return replace(state, hexes=hexes)


def _triangle() -> tuple[str, str]:
    """(anchor, target): both land hexes, target directly adjacent to anchor."""
    land = set(BOARD.land_hexes())
    for a in BOARD.land_hexes():
        for t in sorted(BOARD.adjacent[a]):
            if t in land:
                return a, t
    raise AssertionError("no suitable adjacent hex pair found")


ANCHOR, TARGET = _triangle()


def _far_hex() -> str:
    """A land hex not adjacent to ``ANCHOR`` (and thus not reachable through
    it) -- used to prove a marker bypasses the usual reachability check.
    """
    land = set(BOARD.land_hexes())
    for h in sorted(land):
        if h != ANCHOR and h not in BOARD.adjacent.get(ANCHOR, frozenset()):
            return h
    raise AssertionError("no suitable far hex found")


# --------------------------------------------------------------------------
# ACT1-6: the power wheel
# --------------------------------------------------------------------------


def test_act6_costs_6_power_and_gains_2_spades() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 6))
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT6"))
    fs = s2.factions["engineers"]
    assert fs.power == Power(6, 0, 0)
    assert fs.spades_available == 2
    assert "ACT6" in s2.power_actions_taken


def test_act6_insufficient_power_raises() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 5))
    with pytest.raises(EngineError):
        handle_action(s, "engineers", _cmd("action", tile="ACT6"))


def test_act6_blocks_a_second_take_by_any_faction() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 12))
    s = _rich(s, "darklings", power=Power(0, 0, 12))
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT6"))
    with pytest.raises(EngineError):
        handle_action(s2, "darklings", _cmd("action", tile="ACT6"))


def test_act1_pushes_a_bridge_pending() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 3))
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT1"))
    assert any(p.faction == "engineers" and p.kind == "bridge" for p in s2.pending)
    assert s2.factions["engineers"].power == Power(3, 0, 0)


def test_act2_gains_one_priest() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 3))
    before = s.factions["engineers"].priests
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT2"))
    assert s2.factions["engineers"].priests == before + 1


def test_act2_clamps_to_priest_pool_cap_instead_of_raising() -> None:
    """resources.pm ``adjust_resource``'s generic branch (~327-331): a gain
    that would exceed a tracked ``MAX_$type`` is silently clamped, never an
    error. ``priest_pool`` is our engine's live ``MAX_P`` (module
    docstring) -- ACT2 must clamp to it, not raise.
    """
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 3), priests=7, priest_pool=7)
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT2"))
    assert s2.factions["engineers"].priests == 7


def test_act3_gains_2_workers() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 4))
    before = s.factions["engineers"].workers
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT3"))
    assert s2.factions["engineers"].workers == before + 2


def test_act4_gains_7_coins() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 4))
    before = s.factions["engineers"].coins
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT4"))
    assert s2.factions["engineers"].coins == before + 7


def test_act5_adds_1_spade_to_the_balance() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 4))
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT5"))
    assert s2.factions["engineers"].spades_available == 1


def test_power_actions_taken_blocking_is_global_but_per_tile() -> None:
    s = _state()
    s = _rich(s, "engineers", power=Power(0, 0, 8))
    s = _rich(s, "darklings", power=Power(0, 0, 8))
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACT4"))
    # A different tile (ACT3) is unaffected by ACT4's block.
    s3 = handle_action(s2, "darklings", _cmd("action", tile="ACT3"))
    assert s3.factions["darklings"].workers == s.factions["darklings"].workers + 2


# --------------------------------------------------------------------------
# Faction special actions: gating (SH/board-printed, once per round)
# --------------------------------------------------------------------------


def test_faction_special_action_without_stronghold_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_action(s, "auren", _cmd("action", tile="ACTA"))


def test_faction_special_action_wrong_faction_raises() -> None:
    s = _with_sh(_state(), "witches")
    with pytest.raises(EngineError):
        handle_action(s, "witches", _cmd("action", tile="ACTA"))


def test_faction_special_action_blocks_a_second_use_same_round() -> None:
    s = _with_sh(_state(), "auren")
    s2 = handle_action(s, "auren", _cmd("action", tile="ACTA"))
    assert "ACTA" in s2.factions["auren"].actions_used
    with pytest.raises(EngineError):
        handle_action(s2, "auren", _cmd("action", tile="ACTA"))


def test_acte_needs_no_stronghold_and_never_blocks() -> None:
    s = _state()  # engineers: no SH built
    s = _rich(s, "engineers", workers=10)
    s2 = handle_action(s, "engineers", _cmd("action", tile="ACTE"))
    fs = s2.factions["engineers"]
    assert fs.workers == 8  # 10 - 2W
    assert "ACTE" not in fs.actions_used
    assert any(p.faction == "engineers" and p.kind == "bridge" for p in s2.pending)
    # Reusable in the same round -- ACTE is never blocked (dont_block).
    s3 = handle_action(s2, "engineers", _cmd("action", tile="ACTE"))
    assert s3.factions["engineers"].workers == 6


# --------------------------------------------------------------------------
# ACTA (Auren): 2 cult steps, no marker needed
# --------------------------------------------------------------------------


def test_acta_marks_used_and_costs_nothing_leaving_cult_gain_to_the_companion_row() -> None:
    s = _with_sh(_state(), "auren")
    before = s.factions["auren"]
    before_water = s.cults["auren"]["WATER"]
    s2 = handle_action(s, "auren", _cmd("action", tile="ACTA"))
    fs = s2.factions["auren"]
    assert fs.coins == before.coins and fs.workers == before.workers
    assert "ACTA" in fs.actions_used
    assert s2.pending == ()  # no marker -- the ledger's own "+2<cult>" row does the advance
    # The companion `+2<cult>` row (same faction, same batch) applies through
    # the pre-existing gain_cult handler with no gate to satisfy -- Auren
    # isn't in this fixture's real turn_order, so make it the sole active
    # faction to exercise this through the real apply() gate too.
    s2 = replace(s2, turn_order=("auren",), active_index=0)
    s3 = apply(s2, "auren", _cmd("gain_cult", cult="WATER", n1=2))
    assert s3.cults["auren"]["WATER"] == before_water + 2


# --------------------------------------------------------------------------
# ACTG (Giants): 2 free spades, feeding the shared balance
# --------------------------------------------------------------------------


def test_actg_grants_2_spades_to_the_shared_balance() -> None:
    s = _with_sh(_state(), "giants")
    s2 = handle_action(s, "giants", _cmd("action", tile="ACTG"))
    assert s2.factions["giants"].spades_available == 2


# --------------------------------------------------------------------------
# ACTW (Witches' Ride): free dwelling on any green hex, no reachability
# --------------------------------------------------------------------------


def test_actw_pushes_free_d_pending() -> None:
    s = _with_sh(_state(), "witches")
    s2 = handle_action(s, "witches", _cmd("action", tile="ACTW"))
    assert any(p.faction == "witches" and p.kind == "free_d" for p in s2.pending)


def test_actw_then_build_is_free_and_bypasses_reachability() -> None:
    s = _with_sh(_state(), "witches")
    far = _far_hex()
    hexes = dict(s.hexes)
    hexes[far] = replace(hexes[far], color=FACTIONS["witches"].color, building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = handle_action(s, "witches", _cmd("action", tile="ACTW"))
    before = s.factions["witches"]
    s2 = handle_build(s, "witches", _cmd("build", loc=far))
    fs = s2.factions["witches"]
    assert fs.workers == before.workers and fs.coins == before.coins  # free
    assert s2.hexes[far].building == "D"
    assert not any(p.kind == "free_d" for p in s2.pending)  # marker consumed


def test_actw_build_still_requires_home_color() -> None:
    """The Task-8 seam comment's "any color" assumption was wrong -- Perl's
    ``build_color_ok`` is never bypassed by ``FREE_D`` (module docstring).
    """
    s = _with_sh(_state(), "witches")
    far = _far_hex()
    hexes = dict(s.hexes)
    hexes[far] = replace(hexes[far], color="red", building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = handle_action(s, "witches", _cmd("action", tile="ACTW"))
    with pytest.raises(EngineError):
        handle_build(s, "witches", _cmd("build", loc=far))


def test_actw_build_still_triggers_leech() -> None:
    """``note_leech`` is unconditional in ``command_build`` -- a Witches'
    Ride build still queues leech offers for an adjacent opponent building
    of a different color (module docstring).
    """
    s = _with_sh(_state(), "witches")
    s = replace(s, turn_order=(*s.turn_order, "witches"))  # leech seat order needs a real slot
    s = _place(s, "engineers", ANCHOR, "TP")  # different color, directly adjacent to TARGET
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(
        hexes[TARGET], color=FACTIONS["witches"].color, building=None, owner=None
    )
    s = replace(s, hexes=hexes)
    s = handle_action(s, "witches", _cmd("action", tile="ACTW"))
    s2 = handle_build(s, "witches", _cmd("build", loc=TARGET))
    assert s2.hexes[TARGET].building == "D"
    assert any(p.faction == "engineers" and p.kind == "leech" for p in s2.pending)


def _place(
    state: GameState, faction: str, hex_key: str, building: str, color: str | None = None
) -> GameState:
    hexes = dict(state.hexes)
    color = color if color is not None else FACTIONS[faction].color
    hexes[hex_key] = replace(hexes[hex_key], color=color, building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(with_faction(state, faction, fs), hexes=hexes)


# --------------------------------------------------------------------------
# ACTS (Swarmlings): free D->TP upgrade
# --------------------------------------------------------------------------


def test_acts_pushes_free_tp_pending() -> None:
    s = _with_sh(_state(), "swarmlings")
    s2 = handle_action(s, "swarmlings", _cmd("action", tile="ACTS"))
    assert any(p.faction == "swarmlings" and p.kind == "free_tp" for p in s2.pending)


def test_acts_then_upgrade_is_free() -> None:
    s = _with_sh(_state(), "swarmlings")
    s = _place(s, "swarmlings", TARGET, "D")
    s = handle_action(s, "swarmlings", _cmd("action", tile="ACTS"))
    before = s.factions["swarmlings"]
    s2 = handle_upgrade(s, "swarmlings", _cmd("upgrade", loc=TARGET, building="TP"))
    fs = s2.factions["swarmlings"]
    assert fs.workers == before.workers and fs.coins == before.coins
    assert TARGET in fs.buildings["TP"]
    assert not any(p.kind == "free_tp" for p in s2.pending)


# --------------------------------------------------------------------------
# ACTN (Nomads Sandstorm): free transform, direct adjacency, home color
# --------------------------------------------------------------------------


def test_actn_pushes_free_tf_pending() -> None:
    s = _with_sh(_state(), "nomads")
    s2 = handle_action(s, "nomads", _cmd("action", tile="ACTN"))
    assert any(p.faction == "nomads" and p.kind == "free_tf" for p in s2.pending)


def test_actn_then_transform_is_free_home_color_and_needs_direct_adjacency() -> None:
    s = _with_sh(_state(), "nomads")
    s = _place(s, "nomads", ANCHOR, "D")
    off_color = "red" if FACTIONS["nomads"].color != "red" else "blue"
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color=off_color, building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = handle_action(s, "nomads", _cmd("action", tile="ACTN"))
    s2 = handle_transform(s, "nomads", _cmd("transform", loc=TARGET))
    assert s2.hexes[TARGET].color == FACTIONS["nomads"].color
    assert s2.factions["nomads"].spades_available == 0  # not touched
    assert not any(p.kind == "free_tf" for p in s2.pending)


def test_actn_transform_rejects_non_adjacent_hex() -> None:
    s = _with_sh(_state(), "nomads")
    s = _place(s, "nomads", ANCHOR, "D")
    far = _far_hex()
    off_color = "red" if FACTIONS["nomads"].color != "red" else "blue"
    hexes = dict(s.hexes)
    hexes[far] = replace(hexes[far], color=off_color, building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = handle_action(s, "nomads", _cmd("action", tile="ACTN"))
    with pytest.raises(EngineError):
        handle_transform(s, "nomads", _cmd("transform", loc=far))


# --------------------------------------------------------------------------
# ACTC (Chaos Magicians): double turn flag
# --------------------------------------------------------------------------


def test_actc_sets_extra_actions_for_task_11() -> None:
    s = _with_sh(_state(), "chaosmagicians")
    s2 = handle_action(s, "chaosmagicians", _cmd("action", tile="ACTC"))
    assert s2.factions["chaosmagicians"].extra_actions == 2


def test_actc_never_blocks_actions_used() -> None:
    """ACTC (like every other ``ACT[A-Z]`` id) still blocks per-faction --
    only ACTE opts out (module docstring)."""
    s = _with_sh(_state(), "chaosmagicians")
    s2 = handle_action(s, "chaosmagicians", _cmd("action", tile="ACTC"))
    assert "ACTC" in s2.factions["chaosmagicians"].actions_used
    with pytest.raises(EngineError):
        handle_action(s2, "chaosmagicians", _cmd("action", tile="ACTC"))


# --------------------------------------------------------------------------
# BON1/BON2/FAV6: bonus/favor special actions
# --------------------------------------------------------------------------


def test_bon1_requires_holding_the_tile() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_action(s, "engineers", _cmd("action", tile="BON1"))


def test_bon1_grants_a_spade_once_per_round() -> None:
    s = _rich(_state(), "engineers", bonus="BON1")
    s2 = handle_action(s, "engineers", _cmd("action", tile="BON1"))
    assert s2.factions["engineers"].spades_available == 1
    assert "BON1" in s2.factions["engineers"].actions_used
    with pytest.raises(EngineError):
        handle_action(s2, "engineers", _cmd("action", tile="BON1"))


def test_bon2_requires_holding_the_tile_and_marks_used() -> None:
    s = _rich(_state(), "engineers", bonus="BON2")
    before = s.factions["engineers"]
    s2 = handle_action(s, "engineers", _cmd("action", tile="BON2"))
    fs = s2.factions["engineers"]
    assert fs.coins == before.coins and fs.workers == before.workers
    assert "BON2" in fs.actions_used


def test_fav6_requires_holding_the_favor_and_marks_used() -> None:
    s = _rich(_state(), "engineers", favors=("FAV6",))
    s2 = handle_action(s, "engineers", _cmd("action", tile="FAV6"))
    assert "FAV6" in s2.factions["engineers"].actions_used


def test_fav6_without_holding_it_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_action(s, "engineers", _cmd("action", tile="FAV6"))


def test_unknown_action_tile_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_action(s, "engineers", _cmd("action", tile="BOGUS"))


# --------------------------------------------------------------------------
# lose_marker: declining a granted marker
# --------------------------------------------------------------------------


def test_lose_marker_pops_a_matching_pending() -> None:
    s = _with_sh(_state(), "witches")
    s = handle_action(s, "witches", _cmd("action", tile="ACTW"))
    s2 = handle_lose_marker(s, "witches", _cmd("lose_marker", reason="FREE_D"))
    assert not any(p.kind == "free_d" for p in s2.pending)


def test_lose_marker_with_none_outstanding_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_lose_marker(s, "witches", _cmd("lose_marker", reason="FREE_D"))


def test_lose_marker_unknown_reason_raises() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_lose_marker(s, "witches", _cmd("lose_marker", reason="NOT_A_MARKER"))
