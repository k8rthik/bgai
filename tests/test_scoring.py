"""tests/test_scoring.py

Scenarios validated against jsnell/terra-mystica ``src/scoring.pm`` (cult/
network tie-split arithmetic, leftover-resource conversion) and the
crawled-corpus reference game ``4pLeague_S10_D1L1_G1``'s round-6
final-scoring block (``moves.parquet``/``deltas.parquet`` rows 361-393) --
see ``scoring.py``'s module docstring for the full citation trail.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import clusters
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.power import Power
from bgai.engine.tm.scoring import (
    _apply_score_resources,
    _score_type_rankings,
    compute_cult_scoring,
    compute_network_scoring,
    final_scoring,
    handle_score_resources,
    handle_score_vp,
    network_size,
)
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

BOARD = base_board()
GAME_ID = "4pLeague_S10_D1L1_G1"


def _fresh() -> GameState:
    return GameState.initial(load_setup(GAME_ID))


def _finished() -> GameState:
    return replace(_fresh(), round=6, phase=Phase.FINISHED)


def _with_faction_fields(state: GameState, faction: str, **fields: object) -> GameState:
    fs = replace(state.factions[faction], **fields)  # type: ignore[arg-type]
    return with_faction(state, faction, fs)


def _with_cults(state: GameState, positions: dict[str, dict[str, int]]) -> GameState:
    new_cults = {f: dict(v) for f, v in state.cults.items()}
    for faction, cults in positions.items():
        new_cults[faction].update(cults)
    return replace(state, cults=new_cults)


def _place(state: GameState, faction: str, hex_key: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building="D", owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, "D": fs.buildings["D"] | {hex_key}})
    return replace(state, hexes=hexes, factions={**state.factions, faction: fs})


def _with_dwarves(state: GameState) -> GameState:
    return with_faction(state, "dwarves", FactionState.initial(FACTIONS["dwarves"]))


def _with_fakirs(state: GameState) -> GameState:
    return with_faction(state, "fakirs", FactionState.initial(FACTIONS["fakirs"]))


def _river_gap() -> tuple[str, str, str]:
    """(land_a, river, land_b): two land hexes joined only via one river hex."""
    for r in (k for k, h in BOARD.hexes.items() if h.color == RIVER):
        lands = [n for n in BOARD.adjacent[r] if BOARD.hexes[n].color != RIVER]
        for a in lands:
            for b in lands:
                if a != b and b not in BOARD.adjacent[a]:
                    return a, r, b
    raise AssertionError("no river gap on base map")


def _land_skip_gap() -> tuple[str, str, str]:
    """(land_a, mid, land_b): a land->land->land chain where the ends are
    not directly adjacent (hex_distance 2) -- for tunnel/carpet range 1.
    """
    for mid in (k for k, h in BOARD.hexes.items() if h.color != RIVER):
        lands = [n for n in BOARD.adjacent[mid] if BOARD.hexes[n].color != RIVER]
        for a in lands:
            for b in lands:
                if a != b and b not in BOARD.adjacent[a]:
                    return a, mid, b
    raise AssertionError("no land skip-chain found")


def _cmd(**fields: object) -> ParsedCommand:
    return ParsedCommand(verb="score_vp", kind=Kind.SCORING, raw="", **fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Tie-split arithmetic (_score_type_rankings / compute_cult_scoring)
# --------------------------------------------------------------------------


def test_two_way_tie_for_first_splits_top_two_points_evenly() -> None:
    # scoring.pm module docstring's worked example: two factions tied 1st
    # on a cult -> (8 + 4) // 2 == 6 each; third distinct level gets 2.
    levels = {"a": 5, "b": 5, "c": 3}
    assert _score_type_rankings(levels, (8, 4, 2)) == {"a": 6, "b": 6, "c": 2}


def test_three_way_tie_for_first_floors_the_remainder() -> None:
    levels = {"a": 5, "b": 5, "c": 5, "d": 1}
    # bucket[5] = 8 + 4 + 2 = 14; count[5] = 3; 14 // 3 == 4 (remainder 2 dropped).
    assert _score_type_rankings(levels, (8, 4, 2)) == {"a": 4, "b": 4, "c": 4}


def test_four_way_tie_for_first_only_pops_top_three_slots_but_splits_among_all_four() -> None:
    # Only 3 points-slots exist, but every faction sharing the top level
    # value shares the pooled total -- not just the ones "popped".
    levels = {"a": 9, "b": 9, "c": 9, "d": 9, "e": 3}
    assert _score_type_rankings(levels, (8, 4, 2)) == {"a": 3, "b": 3, "c": 3, "d": 3}


def test_position_zero_scores_nothing_even_as_the_sole_occupant() -> None:
    levels = {"a": 0, "b": 0}
    assert _score_type_rankings(levels, (8, 4, 2)) == {}


def test_a_faction_outranked_by_more_than_points_len_scores_nothing() -> None:
    # Four distinct nonzero levels, only 3 points slots: the 4th-ranked
    # faction's own level never becomes a bucket key.
    levels = {"a": 4, "b": 3, "c": 2, "d": 1}
    assert _score_type_rankings(levels, (8, 4, 2)) == {"a": 8, "b": 4, "c": 2}


def test_compute_cult_scoring_matches_reference_game_final_cult_rows() -> None:
    """Corpus citation (scoring.py module docstring): reference game's
    real cult positions entering round 6's final scoring (most recent
    ``gain_cult``/``send`` snapshot per faction before ledger row 369,
    ``deltas.parquet`` ``cult`` column) reproduce the exact score_vp rows
    369-383 (F/W/E/A points 8/4/2, position 0 excluded, EARTH's 9/9 tie
    splitting 6/6).
    """
    state = _with_cults(
        _finished(),
        {
            "nomads": {"FIRE": 7, "WATER": 4, "EARTH": 2, "AIR": 4},
            "engineers": {"FIRE": 0, "WATER": 1, "EARTH": 9, "AIR": 7},
            "mermaids": {"FIRE": 2, "WATER": 3, "EARTH": 3, "AIR": 0},
            "darklings": {"FIRE": 3, "WATER": 5, "EARTH": 9, "AIR": 2},
        },
    )
    scores = compute_cult_scoring(state)
    assert scores["FIRE"] == {"nomads": 8, "darklings": 4, "mermaids": 2}
    assert scores["WATER"] == {"darklings": 8, "nomads": 4, "mermaids": 2}
    assert scores["EARTH"] == {"darklings": 6, "engineers": 6, "mermaids": 2}
    assert scores["AIR"] == {"engineers": 8, "nomads": 4, "darklings": 2}


def test_network_tie_split_matches_reference_game_final_network_rows() -> None:
    """Same reference game's NETWORK score_vp rows 385-388: engineers alone
    at the top network size keeps the full 18; nomads/darklings/mermaids
    share a lower tied size, splitting 12 + 6 == 18 three ways -> 6 each.
    """
    levels = {"engineers": 18, "nomads": 6, "darklings": 6, "mermaids": 6}
    assert _score_type_rankings(levels, (18, 12, 6)) == {
        "engineers": 18,
        "nomads": 6,
        "darklings": 6,
        "mermaids": 6,
    }


# --------------------------------------------------------------------------
# network_size / clusters(indirect=True) geometry
# --------------------------------------------------------------------------


def test_network_size_is_one_for_a_single_isolated_building() -> None:
    a, _r, _b = _river_gap()
    s = _place(_finished(), "engineers", a)
    assert network_size(s, "engineers") == 1


def test_network_size_excludes_indirect_reach_without_shipping() -> None:
    a, _r, b = _river_gap()
    s = _place(_place(_finished(), "engineers", a), "engineers", b)
    assert network_size(s, "engineers") == 1  # two separate 1-hex clusters


def test_network_size_includes_shipping_reach() -> None:
    a, _r, b = _river_gap()
    s = _place(_place(_finished(), "engineers", a), "engineers", b)
    s = _with_faction_fields(s, "engineers", shipping=1)
    assert network_size(s, "engineers") == 2


def test_network_size_includes_dwarves_innate_tunnel_range() -> None:
    """Dwarves/Fakirs DO count in final network scoring (scoring.py module
    docstring): they have no shipping track, but their teleport range is
    an unconditional branch of ``find_building_cliques``'s indirect join,
    active from turn one (nonzero base ``TeleportTrack.range``).
    """
    a, _mid, b = _land_skip_gap()
    s = _place(_place(_with_dwarves(_finished()), "dwarves", a), "dwarves", b)
    assert network_size(s, "dwarves") == 2


def test_network_size_excludes_tunnel_reach_for_a_non_teleport_faction() -> None:
    a, _mid, b = _land_skip_gap()
    s = _place(_place(_finished(), "darklings", a), "darklings", b)
    assert network_size(s, "darklings") == 1


def test_network_size_includes_fakirs_innate_carpet_range() -> None:
    a, _mid, b = _land_skip_gap()
    s = _place(_place(_with_fakirs(_finished()), "fakirs", a), "fakirs", b)
    assert network_size(s, "fakirs") == 2


def test_network_size_includes_mermaids_river_skip_even_at_shipping_zero() -> None:
    a, _r, b = _river_gap()
    s = _place(_place(_finished(), "mermaids", a), "mermaids", b)
    s = _with_faction_fields(s, "mermaids", shipping=0)  # isolate the skip mechanism
    assert network_size(s, "mermaids") == 2


def test_network_size_honors_bridges() -> None:
    a, _r, b = _river_gap()
    s = _place(_place(_finished(), "engineers", a), "engineers", b)
    s = replace(s, bridges=frozenset({frozenset({a, b})}))
    assert network_size(s, "engineers") == 2


def test_compute_network_scoring_uses_network_size_across_all_factions() -> None:
    a, _r, b = _river_gap()
    state = _place(_place(_finished(), "engineers", a), "engineers", b)
    state = _with_faction_fields(state, "engineers", shipping=1)
    c = next(
        h
        for h in BOARD.hexes
        if h not in (a, b)
        and BOARD.hexes[h].color != RIVER
        and a not in BOARD.adjacent[h]
        and b not in BOARD.adjacent[h]
    )
    state = _place(state, "darklings", c)
    assert network_size(state, "engineers") == 2
    assert network_size(state, "darklings") == 1
    # engineers (size 2) takes 18; darklings (size 1, next distinct level)
    # takes 12; the two remaining zero-building factions score nothing.
    assert compute_network_scoring(state) == {"engineers": 18, "darklings": 12}


def test_clusters_indirect_matches_network_size_largest_component() -> None:
    a, _r, b = _river_gap()
    s = _place(_place(_finished(), "engineers", a), "engineers", b)
    s = _with_faction_fields(s, "engineers", shipping=1)
    components = clusters(s, "engineers", indirect=True)
    assert max(len(c) for c in components) == network_size(s, "engineers")


# --------------------------------------------------------------------------
# score_resources conversion
# --------------------------------------------------------------------------


def test_apply_score_resources_burns_bowl2_converts_everything_then_c_to_vp() -> None:
    fs = replace(
        FactionState.initial(FACTIONS["nomads"]),
        power=Power(bowl1=2, bowl2=7, bowl3=0),
        coins=1,
        workers=2,
        priests=1,
        vp=20,
    )
    # burn floor(7/2)=3: bowl2 -= 6 -> 1, bowl3 += 3 -> 3.
    # convert 3 pw -> 3c; convert 1 priest -> 1c; convert 2 workers -> 2c.
    # coins = 1 + 3 + 1 + 2 = 7; rate 3 -> vp = 2, leftover coins = 1.
    result = _apply_score_resources(fs, "nomads")
    assert result.power == Power(bowl1=5, bowl2=1, bowl3=0)
    assert result.coins == 1
    assert result.priests == 0
    assert result.workers == 0
    assert result.vp == 22


_ZERO_POWER = Power(bowl1=0, bowl2=0, bowl3=0)


def test_apply_score_resources_leaves_leftover_coins_below_one_vp() -> None:
    fs = replace(
        FactionState.initial(FACTIONS["nomads"]),
        power=_ZERO_POWER,
        workers=0,
        priests=0,
        coins=2,
        vp=20,
    )
    result = _apply_score_resources(fs, "nomads")
    assert result.coins == 2
    assert result.vp == 20


def test_apply_score_resources_uses_alchemists_cheaper_c_to_vp_rate() -> None:
    fs = replace(
        FactionState.initial(FACTIONS["alchemists"]),
        power=_ZERO_POWER,
        workers=0,
        priests=0,
        coins=4,
        vp=20,
    )
    result = _apply_score_resources(fs, "alchemists")
    # Alchemists override C->VP to 2 (factions_data.py exchange_rate_overrides).
    assert result.vp == 22  # 4 // 2 == 2 VP
    assert result.coins == 0


def test_apply_score_resources_reproduces_reference_game_nomads_row() -> None:
    """Corpus citation (scoring.py module docstring): game row 390.
    Power "2/3/0" -> "3/1/0", c_delta +2, w_delta -1, vp_delta 0.
    """
    fs = replace(
        FactionState.initial(FACTIONS["nomads"]),
        power=Power(bowl1=2, bowl2=3, bowl3=0),
        coins=0,
        workers=1,
        priests=0,
        vp=112,
    )
    result = _apply_score_resources(fs, "nomads")
    assert result.power == Power(bowl1=3, bowl2=1, bowl3=0)
    assert result.coins == 2
    assert result.workers == 0
    assert result.vp == 112


def test_apply_score_resources_skips_burn_when_bowl2_below_two() -> None:
    """Corpus citation: game row 392, engineers -- P2=1 -> floor(1/2)=0, no
    burn at all; power snapshot unchanged aside from the later pw loop
    (which has nothing to convert since bowl3 stayed 0).
    """
    fs = replace(
        FactionState.initial(FACTIONS["engineers"]),
        power=Power(bowl1=4, bowl2=1, bowl3=0),
        coins=0,
        workers=1,
        priests=0,
        vp=147,
    )
    result = _apply_score_resources(fs, "engineers")
    assert result.power == Power(bowl1=4, bowl2=1, bowl3=0)
    assert result.coins == 1
    assert result.workers == 0


# --------------------------------------------------------------------------
# handle_score_vp / handle_score_resources
# --------------------------------------------------------------------------


def test_handle_score_vp_applies_matching_row() -> None:
    state = _with_cults(_finished(), {"nomads": {"FIRE": 7, "WATER": 0, "EARTH": 0, "AIR": 0}})
    state = _with_cults(
        state, {"engineers": {"FIRE": 0}, "mermaids": {"FIRE": 0}, "darklings": {"FIRE": 0}}
    )
    cmd = _cmd(n1=8, reason="FIRE")
    new_state = handle_score_vp(state, "nomads", cmd)
    assert new_state.factions["nomads"].vp == state.factions["nomads"].vp + 8


def test_handle_score_vp_rejects_a_mismatched_grant() -> None:
    state = _with_cults(_finished(), {"nomads": {"FIRE": 7, "WATER": 0, "EARTH": 0, "AIR": 0}})
    state = _with_cults(
        state, {"engineers": {"FIRE": 0}, "mermaids": {"FIRE": 0}, "darklings": {"FIRE": 0}}
    )
    cmd = _cmd(n1=99, reason="FIRE")
    with pytest.raises(EngineError):
        handle_score_vp(state, "nomads", cmd)


def test_handle_score_vp_rejects_an_unrecognized_reason() -> None:
    state = _finished()
    cmd = _cmd(n1=3, reason="BOGUS")
    with pytest.raises(EngineError):
        handle_score_vp(state, "nomads", cmd)


def test_handle_score_vp_validates_network() -> None:
    a, _r, b = _river_gap()
    state = _place(_place(_finished(), "engineers", a), "engineers", b)
    state = _with_faction_fields(state, "engineers", shipping=1)
    # Only engineers has buildings, so it alone scores network's top slot.
    cmd = _cmd(n1=18, reason="NETWORK")
    new_state = handle_score_vp(state, "engineers", cmd)
    assert new_state.factions["engineers"].vp == state.factions["engineers"].vp + 18


def test_handle_score_resources_applies_the_conversion() -> None:
    state = _with_faction_fields(
        _finished(), "nomads", power=_ZERO_POWER, priests=0, coins=1, workers=1, vp=112
    )
    cmd = ParsedCommand(verb="score_resources", kind=Kind.SCORING, raw="score_resources")
    new_state = handle_score_resources(state, "nomads", cmd)
    assert new_state.factions["nomads"].coins == 2  # 1 existing + 1 from the worker
    assert new_state.factions["nomads"].vp == 112  # 2 coins short of a VP at rate 3


def test_score_vp_and_score_resources_are_exempt_from_the_turn_gate() -> None:
    """Reference game rows 369-393: score_vp/score_resources rows arrive
    grouped by ranking, not in turn_order rotation -- apply.py's gate must
    not reject them (scoring.py module docstring / apply.py comment).
    """
    state = _with_cults(_finished(), {"nomads": {"FIRE": 7, "WATER": 0, "EARTH": 0, "AIR": 0}})
    state = _with_cults(
        state, {"engineers": {"FIRE": 0}, "mermaids": {"FIRE": 0}, "darklings": {"FIRE": 0}}
    )
    # active_index/turn_order point at whoever GameState.initial left it as
    # (not necessarily "nomads") -- apply() must still accept this row.
    new_state = apply(state, "nomads", _cmd(n1=8, reason="FIRE"))
    assert new_state.factions["nomads"].vp == state.factions["nomads"].vp + 8

    cmd = ParsedCommand(verb="score_resources", kind=Kind.SCORING, raw="score_resources")
    apply(state, "darklings", cmd)  # must not raise for the "wrong" active faction


# --------------------------------------------------------------------------
# final_scoring (simulation mode)
# --------------------------------------------------------------------------


def test_final_scoring_requires_phase_finished() -> None:
    state = replace(_fresh(), round=6, phase=Phase.CLEANUP)
    with pytest.raises(ValueError):
        final_scoring(state)


def test_final_scoring_applies_cult_network_and_resource_vp_and_stays_finished() -> None:
    a, _r, b = _river_gap()
    state = _with_cults(
        _finished(),
        {
            "nomads": {"FIRE": 5, "WATER": 0, "EARTH": 0, "AIR": 0},
            "engineers": {"FIRE": 0, "WATER": 0, "EARTH": 0, "AIR": 0},
            "mermaids": {"FIRE": 0, "WATER": 0, "EARTH": 0, "AIR": 0},
            "darklings": {"FIRE": 0, "WATER": 0, "EARTH": 0, "AIR": 0},
        },
    )
    state = _place(_place(state, "engineers", a), "engineers", b)
    # Zero every faction's starting resources so only the deliberate
    # overrides below drive the resource-conversion half of the assertion.
    for faction in state.factions:
        state = _with_faction_fields(
            state, faction, power=Power(bowl1=0, bowl2=0, bowl3=0), coins=0, workers=0, priests=0
        )
    state = _with_faction_fields(state, "engineers", shipping=1, coins=1, workers=1)
    before_nomads_vp = state.factions["nomads"].vp
    before_engineers_vp = state.factions["engineers"].vp

    result = final_scoring(state)

    assert result.phase == Phase.FINISHED
    # nomads: sole nonzero FIRE position -> full 8 VP; alone at network 0
    # (no buildings) so network never gets a nonzero bucket key for it,
    # and no other faction has any buildings either besides engineers.
    assert result.factions["nomads"].vp == before_nomads_vp + 8
    # engineers: sole network occupant (size 2) -> full 18 VP, plus its
    # resource conversion (1 coin + 1 worker -> 2 coins, below the 3:1 rate).
    assert result.factions["engineers"].vp == before_engineers_vp + 18
    assert result.factions["engineers"].coins == 2
    assert result.factions["engineers"].workers == 0
