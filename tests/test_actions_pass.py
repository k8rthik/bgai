"""tests/test_actions_pass.py

Scenarios validated against jsnell/terra-mystica ``src/commands.pm``
(``command_pass``, ``command_advance``) and ``src/towns.pm``
(``check_mermaid_river_connection_town``) -- see ``actions_pass.py``'s
module docstring for the exact citation trail and the crawled-corpus
evidence (bonus-tile coin accumulation, Mermaids' ``connect`` row) quoted
there.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.actions_pass import handle_advance, handle_connect, handle_pass
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

GAME_ID = "4pLeague_S10_D1L1_G1"
BOARD = base_board()


def _state() -> GameState:
    s = GameState.initial(load_setup(GAME_ID))
    return replace(s, round=1, phase=Phase.ACTIONS)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


def _rich(state: GameState, faction: str, **fields: object) -> GameState:
    fs = replace(state.factions[faction], **fields)  # type: ignore[arg-type]
    return with_faction(state, faction, fs)


def _place(state: GameState, faction: str, hex_key: str, building: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _place_all(state: GameState, faction: str, layout: dict[str, str]) -> GameState:
    for hex_key, building in layout.items():
        state = _place(state, faction, hex_key, building)
    return state


def _mermaid_river_layout() -> tuple[str, str, str, str, str]:
    """(a, a2, river, b, b2): two 2-hex land groups joined only via one river hex.

    Identical algorithm to ``test_towns.py``'s helper of the same name --
    duplicated locally rather than imported (test modules aren't a shared
    library here). ``sorted(...)`` in ``land_neighbors`` makes the search
    deterministic across ``PYTHONHASHSEED``-randomized processes -- see
    that sibling helper's own docstring for why an unsorted
    ``frozenset`` walk here made this file's
    ``test_connect_two_hex_early_era_form_derives_the_river`` flaky
    (task-14 fix): a different, sometimes-genuinely-ambiguous layout on
    every other run.
    """

    def land_neighbors(x: str, exclude: frozenset[str] = frozenset()) -> list[str]:
        return sorted(
            n for n in BOARD.adjacent[x] if BOARD.hexes[n].color != RIVER and n not in exclude
        )

    for river, hex_ in BOARD.hexes.items():
        if hex_.color != RIVER:
            continue
        lands = land_neighbors(river)
        for a in lands:
            for b in lands:
                if a == b or b in BOARD.adjacent[a]:
                    continue
                for a2 in land_neighbors(a, frozenset({b, river})):
                    if a2 in BOARD.adjacent[b]:
                        continue
                    for b2 in land_neighbors(b, frozenset({a, river, a2})):
                        if b2 in BOARD.adjacent[a] or b2 in BOARD.adjacent[a2]:
                            continue
                        return a, a2, river, b, b2
    raise AssertionError("no mermaid river layout found")


def _any_bridgeable_pair() -> tuple[str, str]:
    """Any two legally-bridgeable land hexes (colors irrelevant here -- the
    Engineers pass-VP hook only cares that both ends are gray and built-on,
    which the caller forces directly; no real board pair of *gray* hexes
    is ever directly bridgeable, gray being a scarce, scattered color).
    """
    from bgai.engine.tm.actions_build import _bridgable_pairs

    pair = next(iter(_bridgable_pairs()))
    a, b = tuple(pair)
    return a, b


# --------------------------------------------------------------------------
# pass: pass-VP
# --------------------------------------------------------------------------


def test_bon7_pass_vp_scales_with_trading_posts() -> None:
    """BON7 pass_vp = (("TP", 2),): 2 TP -> 4 VP (brief's exact scenario)."""
    s = _state()
    s = _rich(s, "engineers", bonus="BON7")
    s = _place(s, "engineers", "F6", "TP")
    s = _place(s, "engineers", "E7", "TP")
    before_vp = s.factions["engineers"].vp
    s2 = handle_pass(s, "engineers", _cmd("pass"))
    assert s2.factions["engineers"].vp == before_vp + 4


def test_bon9_pass_vp_scales_with_dwellings() -> None:
    s = _state()
    s = _rich(s, "darklings", bonus="BON9")
    s = _place(s, "darklings", "G5", "D")
    before_vp = s.factions["darklings"].vp
    s2 = handle_pass(s, "darklings", _cmd("pass"))
    assert s2.factions["darklings"].vp == before_vp + 1


def test_bon10_pass_vp_scales_with_shipping_level() -> None:
    s = _state()
    s = _rich(s, "nomads", bonus="BON10", shipping=2)
    before_vp = s.factions["nomads"].vp
    s2 = handle_pass(s, "nomads", _cmd("pass"))
    assert s2.factions["nomads"].vp == before_vp + 6  # 3 VP * 2 levels


def test_fav12_pass_vp_is_count_indexed_not_per_unit() -> None:
    s = _state()
    s = _rich(s, "mermaids", favors=("FAV12",))
    s = _place_all(s, "mermaids", {"D2": "TP", "D5": "TP"})
    before_vp = s.factions["mermaids"].vp
    s2 = handle_pass(s, "mermaids", _cmd("pass"))
    assert s2.factions["mermaids"].vp == before_vp + 3  # table[2] == 3, not 2*per-unit


def test_no_bonus_or_favor_grants_zero_pass_vp() -> None:
    s = _state()
    before_vp = s.factions["engineers"].vp
    s2 = handle_pass(s, "engineers", _cmd("pass"))
    assert s2.factions["engineers"].vp == before_vp


# --------------------------------------------------------------------------
# pass: Engineers' stronghold bridge hook
# --------------------------------------------------------------------------


def _force_gray(state: GameState, hex_key: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], color="gray")
    return replace(state, hexes=hexes)


def test_engineers_gain_3vp_per_bridge_connecting_gray_buildings() -> None:
    a, b = _any_bridgeable_pair()
    s = _state()
    s = _force_gray(s, a)
    s = _force_gray(s, b)
    s = _place(s, "engineers", a, "D")
    s = _place(s, "engineers", b, "D")
    s = _place(s, "engineers", "F6", "SH")  # any hex; only presence matters
    s = replace(s, bridges=s.bridges | {frozenset({a, b})})
    before_vp = s.factions["engineers"].vp
    s2 = handle_pass(s, "engineers", _cmd("pass"))
    assert s2.factions["engineers"].vp == before_vp + 3


def test_engineers_hook_requires_stronghold() -> None:
    a, b = _any_bridgeable_pair()
    s = _state()
    s = _force_gray(s, a)
    s = _force_gray(s, b)
    s = _place(s, "engineers", a, "D")
    s = _place(s, "engineers", b, "D")
    s = replace(s, bridges=s.bridges | {frozenset({a, b})})
    before_vp = s.factions["engineers"].vp
    s2 = handle_pass(s, "engineers", _cmd("pass"))
    assert s2.factions["engineers"].vp == before_vp  # no SH -> no bonus


# --------------------------------------------------------------------------
# pass: bonus-tile swap, accumulated coins, round-6 bare pass
# --------------------------------------------------------------------------


def test_pass_returns_old_tile_and_takes_new_one_with_accumulated_coins() -> None:
    s = _state()
    s = _rich(s, "engineers", bonus="BON1")
    s = replace(s, bonus_coins={**s.bonus_coins, "BON7": 3})
    before_coins = s.factions["engineers"].coins
    s2 = handle_pass(s, "engineers", _cmd("pass", tile="BON7"))
    fs = s2.factions["engineers"]
    assert fs.bonus == "BON7"
    assert fs.coins == before_coins + 3
    assert s2.bonus_coins["BON7"] == 0
    # the old tile (BON1) is simply released, not tracked as "returned" anywhere
    # besides no longer being anyone's `fs.bonus`.
    assert all(other.bonus != "BON1" for other in s2.factions.values())


def test_pass_marks_passed_and_appends_to_passed_order() -> None:
    s = _state()
    s2 = handle_pass(s, "mermaids", _cmd("pass", tile="BON3"))
    assert s2.factions["mermaids"].passed is True
    assert s2.passed_order == (*s.passed_order, "mermaids")


def test_round_6_bare_pass_takes_no_tile() -> None:
    s = replace(_state(), round=6)
    s = _rich(s, "darklings", bonus="BON5")
    before_coins = s.factions["darklings"].coins
    s2 = handle_pass(s, "darklings", _cmd("pass"))
    fs = s2.factions["darklings"]
    assert fs.bonus is None
    assert fs.coins == before_coins  # no tile taken -> no coin payout


def test_pass_clamps_extra_actions_under_strict_chaosmagician_sh() -> None:
    """Task-14 fix, corpus ``4pLeague_S13_D3L4_G3``/``4pLeague_S16_D2L1_G6``
    row ``action ACTC; pass`` (both ``strict-chaosmagician-sh``): passing
    with an unused Chaos Magicians ACTC extra action still banked
    (``FactionState.extra_actions``) must discard it, mirroring
    ``commands.pm``'s own clamp (~771-773: forces ``allowed_actions`` to 1
    right before ``command_pass``'s own decrement) -- otherwise
    ``round_flow._advance_actions`` treats the faction as still owed more
    turns it will never actually take (a passed faction never gets
    another row), stranding ``active_index`` on them for the rest of the
    game."""
    s = _state()
    assert s.setup.options.strict_chaosmagician_sh
    s = with_faction(s, "engineers", replace(s.factions["engineers"], extra_actions=1))
    s2 = handle_pass(s, "engineers", _cmd("pass", tile="BON7"))
    assert s2.factions["engineers"].extra_actions == 0


def test_pass_rejects_a_tile_already_held_by_someone_else() -> None:
    s = _state()
    s = _rich(s, "darklings", bonus="BON7")
    with pytest.raises(EngineError):
        handle_pass(s, "engineers", _cmd("pass", tile="BON7"))


_NON_PASS_PHASES = [Phase.INCOME, Phase.CLEANUP, Phase.SETUP_DWELLINGS, Phase.FINISHED]


@pytest.mark.parametrize("phase", _NON_PASS_PHASES)
def test_pass_rejects_every_phase_other_than_actions_and_setup_bonus(phase: Phase) -> None:
    s = replace(_state(), phase=phase)
    with pytest.raises(EngineError):
        handle_pass(s, "engineers", _cmd("pass", tile="BON7"))


# --------------------------------------------------------------------------
# pass: SETUP_BONUS one-time pick
# --------------------------------------------------------------------------


def test_setup_bonus_pick_sets_bonus_with_no_coins_or_passed_flag() -> None:
    s = GameState.initial(load_setup(GAME_ID))
    s = replace(s, phase=Phase.SETUP_BONUS)
    s2 = handle_pass(s, "mermaids", _cmd("pass", tile="BON1"))
    fs = s2.factions["mermaids"]
    assert fs.bonus == "BON1"
    assert fs.passed is False
    assert s2.passed_order == s.passed_order  # untouched


def test_setup_bonus_pick_rejects_unknown_tile() -> None:
    s = replace(GameState.initial(load_setup(GAME_ID)), phase=Phase.SETUP_BONUS)
    with pytest.raises(EngineError):
        handle_pass(s, "mermaids", _cmd("pass", tile="BON99"))


# --------------------------------------------------------------------------
# advance: ship / dig
# --------------------------------------------------------------------------


def test_advance_ship_pays_cost_and_scores_indexed_vp() -> None:
    s = _state()
    s = _rich(s, "engineers", coins=10, priests=2)
    before_vp = s.factions["engineers"].vp
    s2 = handle_advance(s, "engineers", _cmd("advance", reason="ship"))
    fs = s2.factions["engineers"]
    assert fs.shipping == 1
    assert fs.coins == 6  # 10 - 4
    assert fs.priests == 1  # 2 - 1
    assert fs.vp == before_vp + 2  # SHIP_STD.advance_vp[0]


def test_advance_ship_second_level_uses_next_vp_index() -> None:
    s = _state()
    s = _rich(s, "engineers", coins=10, priests=2, shipping=1)
    before_vp = s.factions["engineers"].vp
    s2 = handle_advance(s, "engineers", _cmd("advance", reason="ship"))
    assert s2.factions["engineers"].shipping == 2
    assert s2.factions["engineers"].vp == before_vp + 3  # SHIP_STD.advance_vp[1]


def test_advance_dig_pays_cost_and_scores_vp() -> None:
    s = _state()
    s = _rich(s, "engineers", coins=10, workers=5, priests=2)
    before_vp = s.factions["engineers"].vp
    s2 = handle_advance(s, "engineers", _cmd("advance", reason="dig"))
    fs = s2.factions["engineers"]
    assert fs.dig_level == 1
    assert fs.coins == 5  # 10 - 5
    assert fs.workers == 3  # 5 - 2
    assert fs.priests == 1
    assert fs.vp == before_vp + 6  # DIG_STD.advance_vp[0]


def test_advance_ship_rejects_faction_with_no_shipping_track() -> None:
    s = _state()
    s = with_faction(s, "engineers", FactionState.initial(FACTIONS["dwarves"]))
    with pytest.raises(EngineError):
        handle_advance(s, "engineers", _cmd("advance", reason="ship"))


def test_advance_dig_rejects_darklings_whose_dig_never_advances() -> None:
    s = _state()
    s = with_faction(s, "engineers", FactionState.initial(FACTIONS["darklings"]))
    with pytest.raises(EngineError):
        handle_advance(s, "engineers", _cmd("advance", reason="dig"))


def test_advance_ship_at_max_level_raises() -> None:
    s = _state()
    max_level = FACTIONS["engineers"].shipping.max_level
    s = _rich(s, "engineers", coins=100, priests=10, shipping=max_level)
    with pytest.raises(EngineError):
        handle_advance(s, "engineers", _cmd("advance", reason="ship"))


def test_advance_insufficient_resources_raises() -> None:
    s = _state()
    s = _rich(s, "engineers", coins=0, priests=0)
    with pytest.raises(EngineError):
        handle_advance(s, "engineers", _cmd("advance", reason="ship"))


# --------------------------------------------------------------------------
# connect: Mermaids river-hex town join
# --------------------------------------------------------------------------


def test_connect_founds_a_river_joined_town() -> None:
    a, a2, river, b, b2 = _mermaid_river_layout()
    s = _state()
    s = _place_all(s, "mermaids", {a: "SH", a2: "D", b: "TE", b2: "D"})
    s2 = handle_connect(s, "mermaids", _cmd("connect", loc=river))
    pendings = [p for p in s2.pending if p.kind == "gain_town" and p.faction == "mermaids"]
    assert len(pendings) == 1
    assert set(pendings[0].source.split(",")) == {a, a2, b, b2}


def test_connect_scopes_the_candidate_to_the_named_river_not_any_touching_river() -> None:
    """Task-13 report follow-up, ``4pLeague_S10_D2L1_G3`` row 363: a plain
    4-hex chain (A3=SH, A4=TE, B2=D, B3=D -- power 3+2+1+1=7, count 4,
    already qualifying on its own) has one end (A3) touching river ``r0``
    and the other end (B3) touching a *different* river ``r2``, which also
    touches a separate mermaids D at C3. Connecting via ``r0`` must found
    exactly the 4-hex chain -- not the 5-hex cluster reachable by joining
    through ``r2`` instead, even though that merged candidate also touches
    ``r0``'s own neighbour hexes (module docstring's "Known limitation",
    now fixed via ``towns.river_town_candidate``)."""
    s = _state()
    s = _place_all(s, "mermaids", {"A3": "SH", "A4": "TE", "B2": "D", "B3": "D", "C3": "D"})
    s2 = handle_connect(s, "mermaids", _cmd("connect", loc="r0"))
    pendings = [p for p in s2.pending if p.kind == "gain_town" and p.faction == "mermaids"]
    assert len(pendings) == 1
    assert set(pendings[0].source.split(",")) == {"A3", "A4", "B2", "B3"}


def test_connect_two_hex_early_era_form_derives_the_river() -> None:
    """Task-14 fix, corpus ``4pLeague_S1_D2L1_G1`` row 258 (an early-era
    ledger encoding): ``connect`` can name the two land hexes on either
    side of the river instead of the river hex itself
    (``ledger_parser.py``'s ``connect`` rule already parsed a second
    ``loc2`` for this; ``handle_connect`` never used it, always treating
    ``cmd.loc`` as the river directly and hard-erroring "not a river
    hex"). ``_river_between`` derives the unique river hex adjacent to
    both, mirroring ``commands.pm``'s fully generic ``command_connect``
    (930-967: finds whichever river is adjacent to *every* hex in the
    given list)."""
    a, a2, river, b, b2 = _mermaid_river_layout()
    s = _state()
    s = _place_all(s, "mermaids", {a: "SH", a2: "D", b: "TE", b2: "D"})
    s2 = handle_connect(s, "mermaids", _cmd("connect", loc=a, loc2=b))
    pendings = [p for p in s2.pending if p.kind == "gain_town" and p.faction == "mermaids"]
    assert len(pendings) == 1
    assert set(pendings[0].source.split(",")) == {a, a2, b, b2}


def test_connect_two_hex_form_resolves_multiple_qualifying_rivers() -> None:
    """Task-14 phase-4 fix, corpus ``4pLeague_S1_D2L1_G1`` row 322
    (``connect B3 C3``): B3 and C3 have *two* river hexes adjacent to both
    (``r2`` and ``r9`` on the real board), not the one
    ``test_connect_two_hex_early_era_form_derives_the_river`` exercises --
    ``_rivers_between`` used to require exactly one candidate and hard-
    erred "no unique river hex connects B3 and C3" the moment a second
    board location repeated this shape. Both candidates here happen to
    have an identical building intersection (only B3/C3 themselves), so
    either resolves to the same 4-hex cluster -- this pins that the
    multi-candidate path picks *a* qualifying cluster rather than raising.
    Neither {B3, C2} nor {C3, D4} qualifies on its own (power 4/3, count
    2/2 -- below the power-7/count-4 threshold); only joining them via
    either r2 or r9 reaches power 7 / count 4.
    """
    s = _state()
    s = _place_all(s, "mermaids", {"B3": "SH", "C2": "D", "C3": "D", "D4": "TE"})
    s2 = handle_connect(s, "mermaids", _cmd("connect", loc="B3", loc2="C3"))
    pendings = [p for p in s2.pending if p.kind == "gain_town" and p.faction == "mermaids"]
    assert len(pendings) == 1
    assert set(pendings[0].source.split(",")) == {"B3", "C2", "C3", "D4"}


def test_connect_rejects_non_mermaids() -> None:
    _a, _a2, river, _b, _b2 = _mermaid_river_layout()
    s = _state()
    with pytest.raises(EngineError):
        handle_connect(s, "darklings", _cmd("connect", loc=river))


def test_connect_rejects_non_river_hex() -> None:
    s = _state()
    land = next(h for h in BOARD.land_hexes())
    with pytest.raises(EngineError):
        handle_connect(s, "mermaids", _cmd("connect", loc=land))


def test_connect_with_no_qualifying_cluster_raises() -> None:
    a, a2, river, b, b2 = _mermaid_river_layout()
    s = _state()  # no buildings placed at all
    with pytest.raises(EngineError):
        handle_connect(s, "mermaids", _cmd("connect", loc=river))
