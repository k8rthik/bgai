"""tests/test_actions_terraform.py

Scenarios validated against jsnell/terra-mystica ``src/commands.pm``
(``command_dig``, ``command_transform``) and ``src/resources.pm``
(``adjust_resource``'s gain-mode ``maybe_gain_faction_special`` loop) --
see ``actions_terraform.py``'s module docstring for the exact citation
trail and the crawled-corpus evidence behind the Giants/``lose_spade``
calls.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.actions_terraform import handle_dig, handle_lose_spade, handle_transform
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, PendingDecision, Phase, with_faction

GAME_ID = "4pLeague_S10_D1L1_G1"
BOARD = base_board()


_EXTRA_FACTIONS = ("halflings", "alchemists", "giants", "dwarves")


def _state() -> GameState:
    """GAME_ID's real roster (engineers/darklings/nomads/mermaids) plus a
    fresh ``FactionState`` for every faction these tests exercise but that
    isn't in that roster -- mirrors ``test_actions_build.py``'s
    ``_sh_state`` pattern of injecting a faction directly rather than
    needing a game whose real roster happens to include it.
    """
    s = GameState.initial(load_setup(GAME_ID))
    factions = dict(s.factions)
    for name in _EXTRA_FACTIONS:
        factions[name] = FactionState.initial(FACTIONS[name])
    return replace(s, round=1, phase=Phase.ACTIONS, factions=factions)


def _triangle() -> tuple[str, str]:
    """(anchor, target): both land hexes, target directly adjacent to anchor."""
    land = set(BOARD.land_hexes())
    for a in BOARD.land_hexes():
        for t in sorted(BOARD.adjacent[a]):
            if t in land:
                return a, t
    raise AssertionError("no suitable adjacent hex pair found")


ANCHOR, TARGET = _triangle()


def _place(state: GameState, faction: str, hex_key: str, building: str = "D") -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(
        hexes[hex_key], color=FACTIONS[faction].color, building=building, owner=faction
    )
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(with_faction(state, faction, fs), hexes=hexes)


def _set_color(state: GameState, hex_key: str, color: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], color=color, building=None, owner=None)
    return replace(state, hexes=hexes)


def _rich(state: GameState, faction: str, **fields: object) -> GameState:
    fs = replace(state.factions[faction], **fields)  # type: ignore[arg-type]
    return with_faction(state, faction, fs)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# dig
# --------------------------------------------------------------------------


def test_dig_standard_pays_3w_and_gains_one_spade_at_level_0() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    s = _rich(s, "engineers", workers=10)
    before = s.factions["engineers"]
    s2 = handle_dig(s, "engineers", _cmd("dig", n1=1))
    fs = s2.factions["engineers"]
    assert fs.workers == before.workers - 3
    assert fs.spades_available == before.spades_available + 1


def test_dig_uses_cost_at_current_level() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    s = _rich(s, "engineers", dig_level=1, workers=20)
    before = s.factions["engineers"]
    s2 = handle_dig(s, "engineers", _cmd("dig", n1=2))
    fs = s2.factions["engineers"]
    assert fs.workers == before.workers - 2 * 2  # cost[1] = {"W": 2}, amount 2
    assert fs.spades_available == 2


def test_dig_insufficient_workers_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    s = _rich(s, "engineers", workers=0)
    with pytest.raises(EngineError):
        handle_dig(s, "engineers", _cmd("dig", n1=1))


def test_dig_darklings_pays_priest_gains_spade_and_2vp_never_advances_track() -> None:
    s = _state()
    s = _place(s, "darklings", ANCHOR)
    before = s.factions["darklings"]
    s2 = handle_dig(s, "darklings", _cmd("dig", n1=1))
    fs = s2.factions["darklings"]
    assert fs.priests == before.priests - 1
    assert fs.spades_available == before.spades_available + 1
    assert fs.vp == before.vp + 2
    assert fs.dig_level == before.dig_level  # Darklings' track never advances


def test_dig_halflings_gains_extra_vp_per_spade() -> None:
    s = _state()
    s = _place(s, "halflings", ANCHOR)
    s = _rich(s, "halflings", workers=10)
    before = s.factions["halflings"]
    s2 = handle_dig(s, "halflings", _cmd("dig", n1=2))
    fs = s2.factions["halflings"]
    assert fs.spades_available == 2
    assert fs.vp == before.vp + 2  # +1 VP per spade, unconditional


def test_dig_alchemists_no_pw_without_stronghold() -> None:
    s = _state()
    s = _place(s, "alchemists", ANCHOR)
    before = s.factions["alchemists"]
    s2 = handle_dig(s, "alchemists", _cmd("dig", n1=1))
    assert s2.factions["alchemists"].power == before.power


def test_dig_alchemists_gains_2pw_per_spade_with_stronghold() -> None:
    s = _state()
    s = _place(s, "alchemists", ANCHOR)
    fs = s.factions["alchemists"]
    fs = replace(fs, buildings={**fs.buildings, "SH": frozenset({"stub"})})
    s = with_faction(s, "alchemists", fs)
    before = s.factions["alchemists"].power
    s2 = handle_dig(s, "alchemists", _cmd("dig", n1=1))
    assert s2.factions["alchemists"].power == before.gain(2)


def test_dig_location_hint_is_ignored() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    s = _rich(s, "engineers", workers=10)
    s2 = handle_dig(s, "engineers", _cmd("dig", n1=1, loc=TARGET))
    assert s2.factions["engineers"].spades_available == 1
    assert s2.hexes[TARGET].color == BOARD.hexes[TARGET].color  # untouched


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------


def _one_step_neighbor(home_color: str) -> str:
    """A color one wheel-step away from ``home_color`` (some faction's other
    color choice, guaranteed to differ from ``home_color`` itself)."""
    from bgai.engine.tm.factions_data import COLOR_WHEEL

    idx = COLOR_WHEEL.index(home_color)
    return COLOR_WHEEL[(idx + 1) % len(COLOR_WHEEL)]


def test_transform_one_step_consumes_one_spade() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    home = FACTIONS["engineers"].color
    off_color = _one_step_neighbor(home)
    s = _set_color(s, TARGET, off_color)
    s = _rich(s, "engineers", spades_available=1)
    s2 = handle_transform(s, "engineers", _cmd("transform", loc=TARGET, color=home))
    assert s2.hexes[TARGET].color == home
    assert s2.factions["engineers"].spades_available == 0


def test_transform_defaults_to_home_color_when_unspecified() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    home = FACTIONS["engineers"].color
    off_color = _one_step_neighbor(home)
    s = _set_color(s, TARGET, off_color)
    s = _rich(s, "engineers", spades_available=1)
    s2 = handle_transform(s, "engineers", _cmd("transform", loc=TARGET))
    assert s2.hexes[TARGET].color == home


def test_transform_insufficient_spades_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    home = FACTIONS["engineers"].color
    s = _set_color(s, TARGET, _one_step_neighbor(home))
    with pytest.raises(EngineError):
        handle_transform(s, "engineers", _cmd("transform", loc=TARGET, color=home))


def test_transform_occupied_hex_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    s = _place(s, "darklings", TARGET, building="D")
    s = _rich(s, "engineers", spades_available=5)
    with pytest.raises(EngineError):
        handle_transform(s, "engineers", _cmd("transform", loc=TARGET))


def test_transform_river_hex_rejected() -> None:
    river_hex = next(k for k, h in BOARD.hexes.items() if h.color == RIVER)
    s = _state()
    s = _rich(s, "engineers", spades_available=5)
    with pytest.raises(EngineError):
        handle_transform(s, "engineers", _cmd("transform", loc=river_hex))


def test_transform_unreachable_hex_rejected() -> None:
    s = _state()  # engineers has no buildings -> reachable() is empty
    s = _rich(s, "engineers", spades_available=5)
    with pytest.raises(EngineError):
        handle_transform(s, "engineers", _cmd("transform", loc=TARGET))


def test_transform_already_target_color_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    home = FACTIONS["engineers"].color
    s = _set_color(s, TARGET, home)
    s = _rich(s, "engineers", spades_available=5)
    with pytest.raises(EngineError):
        handle_transform(s, "engineers", _cmd("transform", loc=TARGET))


def test_transform_grey_alias_normalized_to_gray() -> None:
    s = _state()
    s = _place(s, "dwarves", ANCHOR)  # dwarves' home color is "gray"
    s = _set_color(s, TARGET, _one_step_neighbor("gray"))
    s = _rich(s, "dwarves", spades_available=1)
    s2 = handle_transform(s, "dwarves", _cmd("transform", loc=TARGET, color="grey"))
    assert s2.hexes[TARGET].color == "gray"


def test_transform_giants_always_costs_two_regardless_of_distance() -> None:
    s = _state()
    s = _place(s, "giants", ANCHOR)
    home = FACTIONS["giants"].color
    assert home == "red"
    s = _set_color(s, TARGET, "black")  # true wheel distance from black to red is 3
    s = _rich(s, "giants", spades_available=2)
    s2 = handle_transform(s, "giants", _cmd("transform", loc=TARGET, color=home))
    assert s2.hexes[TARGET].color == home
    assert s2.factions["giants"].spades_available == 0


def test_transform_giants_default_target_is_home_color() -> None:
    s = _state()
    s = _place(s, "giants", ANCHOR)
    s = _set_color(s, TARGET, "black")
    s = _rich(s, "giants", spades_available=2)
    s2 = handle_transform(s, "giants", _cmd("transform", loc=TARGET))
    assert s2.hexes[TARGET].color == FACTIONS["giants"].color


def test_transform_giants_explicit_non_home_color_rejected() -> None:
    s = _state()
    s = _place(s, "giants", ANCHOR)
    s = _set_color(s, TARGET, "black")
    s = _rich(s, "giants", spades_available=2)
    with pytest.raises(EngineError):
        handle_transform(s, "giants", _cmd("transform", loc=TARGET, color="green"))


def test_transform_absorbs_halflings_pending_and_scores_full_grant_vp() -> None:
    s = _state()
    s = _place(s, "halflings", ANCHOR)
    home = FACTIONS["halflings"].color
    s = _set_color(s, TARGET, _one_step_neighbor(home))
    s = replace(
        s, pending=(PendingDecision(faction="halflings", kind="halflings_spades", amount=3),)
    )
    before_vp = s.factions["halflings"].vp
    s2 = handle_transform(s, "halflings", _cmd("transform", loc=TARGET, color=home))
    fs = s2.factions["halflings"]
    assert fs.vp == before_vp + 3  # whole 3-spade grant scored, not just the 1 used
    assert fs.spades_available == 2  # 3 absorbed, 1 spent on this transform
    assert s2.pending == ()


# --------------------------------------------------------------------------
# lose_spade
# --------------------------------------------------------------------------


def test_lose_spade_decrements_balance() -> None:
    s = _state()
    s = _rich(s, "engineers", spades_available=3)
    s2 = handle_lose_spade(s, "engineers", _cmd("lose_spade", n1=2))
    assert s2.factions["engineers"].spades_available == 1


def test_lose_spade_bare_form_defaults_to_one() -> None:
    s = _state()
    s = _rich(s, "engineers", spades_available=1)
    s2 = handle_lose_spade(s, "engineers", _cmd("lose_spade"))
    assert s2.factions["engineers"].spades_available == 0


def test_lose_spade_insufficient_balance_rejected() -> None:
    s = _state()
    s = _rich(s, "engineers", spades_available=1)
    with pytest.raises(EngineError):
        handle_lose_spade(s, "engineers", _cmd("lose_spade", n1=2))


def test_lose_spade_absorbs_halflings_pending_first() -> None:
    s = _state()
    s = replace(
        s, pending=(PendingDecision(faction="halflings", kind="halflings_spades", amount=3),)
    )
    before_vp = s.factions["halflings"].vp
    s2 = handle_lose_spade(s, "halflings", _cmd("lose_spade", n1=3))
    fs = s2.factions["halflings"]
    assert fs.spades_available == 0
    assert fs.vp == before_vp + 3
    assert s2.pending == ()
