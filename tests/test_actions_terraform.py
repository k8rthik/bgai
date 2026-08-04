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

import polars as pl
import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.actions_terraform import (
    _default_target_color,
    _transform_colors_on_cycle,
    handle_dig,
    handle_lose_spade,
    handle_transform,
)
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

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


def _skip_chain() -> tuple[str, str]:
    """(a, b): two land hexes at ``hex_distance`` 2 via some middle hex, not
    directly adjacent -- Dwarves' tunnel range (``test_actions_build.py``'s
    identically-shaped helper, duplicated here for the same no-cross-test-
    coupling reason this module's sibling test files already document)."""
    land = set(BOARD.land_hexes())
    for mid in land:
        neighbors = [n for n in BOARD.adjacent[mid] if n in land]
        pair = next(
            (
                (a, b)
                for a in neighbors
                for b in neighbors
                if a != b and b not in BOARD.adjacent[a]
            ),
            None,
        )
        if pair:
            return pair
    raise AssertionError("no skip-chain found")


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


def test_dig_scores_current_round_tiles_gain_vp_per_spade() -> None:
    """GAME_ID's round 2 tile (``score_tiles[1]``) is ``SPADE >> 2``, gain
    mode (``tiles.scored_vp``; ``resources.pm``'s generic per-unit
    positive-gain loop, 388-391) -- 3 spades dug nets 3*2 = 6 VP on top of
    the ordinary resource cost/gain. Round 1's tile (``TP >> 3``) does not
    key SPADE at all, so an identical dig one round earlier gains none."""
    s = replace(_state(), round=2)
    s = _place(s, "engineers", ANCHOR)
    s = _rich(s, "engineers", workers=10)
    before = s.factions["engineers"]
    s2 = handle_dig(s, "engineers", _cmd("dig", n1=3))
    assert s2.factions["engineers"].vp == before.vp + 6

    s_round1 = _place(_state(), "engineers", ANCHOR)
    s_round1 = _rich(s_round1, "engineers", workers=10)
    before_round1 = s_round1.factions["engineers"]
    s2_round1 = handle_dig(s_round1, "engineers", _cmd("dig", n1=3))
    assert s2_round1.factions["engineers"].vp == before_round1.vp


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


def test_transform_default_color_under_balance_lands_on_wheel_limited_color() -> None:
    """map.pm ``transform_colors_on_cycle`` (473-493): with fewer spades
    than the true home-distance, a bare ``transform`` doesn't jump straight
    to home -- it lands wherever ``spades_available`` steps actually reach.
    Darklings (home ``black``) starting from ``gray`` is 3 steps by the
    short way; with only 2 banked, walking 2 steps each direction gives
    ``yellow`` (clockwise) vs ``blue`` (counter-clockwise, distance 1 from
    black) -- ``blue`` is closer, so it wins (module docstring / see the
    direct ``_transform_colors_on_cycle`` tests below for the arithmetic).
    """
    s = _state()
    s = _place(s, "darklings", ANCHOR)
    assert FACTIONS["darklings"].color == "black"
    s = _set_color(s, TARGET, "gray")
    s = _rich(s, "darklings", spades_available=2)
    s2 = handle_transform(s, "darklings", _cmd("transform", loc=TARGET))
    assert s2.hexes[TARGET].color == "blue"
    assert s2.factions["darklings"].spades_available == 0  # full balance spent, cost == 2


def test_transform_default_color_zero_balance_is_a_noop_current_color() -> None:
    """map.pm ``transform_colors`` 517-518: with zero spades banked, the
    default target is the current color itself -- so the caller's
    "already this color" check (not an insufficient-spades error) is what
    rejects a bare ``transform`` with nothing to spend.
    """
    s = _state()
    s = _place(s, "engineers", ANCHOR)
    off_color = _one_step_neighbor(FACTIONS["engineers"].color)
    s = _set_color(s, TARGET, off_color)
    with pytest.raises(EngineError, match="already"):
        handle_transform(s, "engineers", _cmd("transform", loc=TARGET))


def test_transform_colors_on_cycle_prefers_closer_direction() -> None:
    """Direct unit test of the ported wheel-walk (map.pm 473-493), matching
    the hand-computed values behind
    ``test_transform_default_color_under_balance_lands_on_wheel_limited_color``.
    """
    preferred, other = _transform_colors_on_cycle("gray", "black", 2)
    assert (preferred, other) == ("blue", "yellow")  # ccw wins: distance 1 < cw's distance 2
    assert _default_target_color("gray", "black", 2) == "blue"


def test_transform_colors_on_cycle_ties_prefer_counter_clockwise() -> None:
    """map.pm 487-492: the direction comparison is a strict ``<``, so an
    exact tie falls through to the ``else`` branch, which returns the
    counter-clockwise result first. On this engine's actual 7-color
    (odd-length) wheel, an exhaustive search over every (home, current,
    spade-count) triple never produces a tie between
    two *distinct* colors -- odd-cycle symmetry makes that mathematically
    impossible here (task-9 fix-2 report) -- so the only real tie is both
    directions fully converging on home once the balance covers even the
    long way around; this test exercises that boundary and confirms the
    branch returns home cleanly rather than erroring.
    """
    # black<->gray: short distance 3, long way 4 -- 4 spades exhausts both.
    preferred, other = _transform_colors_on_cycle("gray", "black", 4)
    assert preferred == other == "black"


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


def test_transform_across_tunnel_pays_teleport_cost_and_gains_vp() -> None:
    """Task-14 fix, corpus ``4pLeague_S10_D3L2_G6`` rows 338/344: a bare
    ``dig 1. transform A10/A12`` (no ``build`` in the same row) on a hex
    reachable only via Dwarves' tunnel never paid the tunnel's own W cost
    or VP gain -- ``handle_build`` already had this wired (task-13 fix),
    but ``map.pm``'s ``transform_cost``/``check_reachable`` (588-600/
    631-632) folds the *same* teleport cost/gain into a bare
    ``command_transform`` too, and this was the missing call site. Each
    reference row was 4 VP low and short the teleport level's own W cost
    on top of the recolor's own spade cost.
    """
    a, b = _skip_chain()
    s = with_faction(_state(), "dwarves", FactionState.initial(FACTIONS["dwarves"]))
    s = _place(s, "dwarves", a)
    home = FACTIONS["dwarves"].color
    s = _set_color(s, b, _one_step_neighbor(home))
    s = _rich(s, "dwarves", spades_available=1)
    before = s.factions["dwarves"]
    s2 = handle_transform(s, "dwarves", _cmd("transform", loc=b, color=home))
    after = s2.factions["dwarves"]
    assert s2.hexes[b].color == home
    assert after.spades_available == before.spades_available - 1  # recolor's own spade cost
    assert after.workers == before.workers - 2  # tunnel cost at teleport_level 0
    assert after.vp == before.vp + 4  # tunnel's flat VP gain
    assert b in after.teleported_hexes


def test_transform_across_tunnel_is_free_for_a_hex_already_teleported_to() -> None:
    """``FactionState.teleported_hexes`` docstring: once paid, a hex is
    free forever after -- guards against double-charging if a later
    command (a second bare ``transform``, or ``handle_build``'s own
    teleport-crossing call) revisits the same hex."""
    a, b = _skip_chain()
    s = with_faction(_state(), "dwarves", FactionState.initial(FACTIONS["dwarves"]))
    s = _place(s, "dwarves", a)
    home = FACTIONS["dwarves"].color
    s = _set_color(s, b, _one_step_neighbor(home))
    s = _rich(s, "dwarves", spades_available=2, teleported_hexes=frozenset({b}))
    before = s.factions["dwarves"]
    s2 = handle_transform(s, "dwarves", _cmd("transform", loc=b, color=home))
    after = s2.factions["dwarves"]
    assert after.workers == before.workers  # no tunnel W charge, already paid
    assert after.vp == before.vp  # no tunnel VP gain, already paid
    assert after.spades_available == before.spades_available - 1  # recolor's own spade cost still applies


def test_giants_corpus_transforms_never_target_non_home_color() -> None:
    """Corpus regression lock for the brief's stricter Giants contract
    (module docstring): every real ``giants transform ... to X`` ledger
    row across the crawled corpus targets ``red`` (their home color) --
    committed as a real test (task-9 fix-2 review) so corpus growth that
    ever surfaces a non-home Giants transform fails loudly here instead of
    silently drifting from the documented justification.
    """
    moves = pl.read_parquet("data/datasets/moves.parquet")
    colors = (
        moves.filter((pl.col("faction") == "giants") & (pl.col("verb") == "transform"))
        .select("color")
        .drop_nulls()
    )
    assert colors.height > 0
    assert set(colors["color"].to_list()) == {"red"}
