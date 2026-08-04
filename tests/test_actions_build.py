"""tests/test_actions_build.py

Scenarios validated against jsnell/terra-mystica ``src/commands.pm``
(``command_build``, ``command_upgrade``, ``command_bridge``) and
``src/resources.pm`` (``adjust_resource``'s ``FAV``/``TW`` branches) --
see ``actions_build.py``'s module docstring for the exact citation trail.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.actions_build import (
    _bridgable_pairs,
    _maybe_queue_town,
    handle_bridge,
    handle_build,
    handle_gain_favor,
    handle_gain_town,
    handle_upgrade,
)
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import BRIDGE_COUNT, FACTIONS
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


def _far_hex() -> str:
    """A land hex not adjacent to ``ANCHOR`` -- used to prove a marker's
    direct-adjacency requirement actually rejects a non-adjacent hex
    (``test_actions_power.py``'s identically-named helper)."""
    land = set(BOARD.land_hexes())
    for h in sorted(land):
        if h != ANCHOR and h not in BOARD.adjacent.get(ANCHOR, frozenset()):
            return h
    raise AssertionError("no suitable far hex found")


def _place(
    state: GameState, faction: str, hex_key: str, building: str, color: str | None = None
) -> GameState:
    hexes = dict(state.hexes)
    color = color if color is not None else FACTIONS[faction].color
    hexes[hex_key] = replace(hexes[hex_key], color=color, building=building, owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings, building: fs.buildings[building] | {hex_key}})
    return replace(with_faction(state, faction, fs), hexes=hexes)


def _clear(state: GameState, faction: str, hex_key: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(
        hexes[hex_key], color=FACTIONS[faction].color, building=None, owner=None
    )
    return replace(state, hexes=hexes)


def _cmd(verb: str, **fields: object) -> ParsedCommand:
    return ParsedCommand(verb=verb, kind=Kind.DECISION, raw=verb, **fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def test_build_pays_cost_and_places_dwelling() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = _clear(s, "engineers", TARGET)
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    fs = s2.factions["engineers"]
    d_cost = FACTIONS["engineers"].buildings["D"].cost
    assert fs.workers == 2 - d_cost["W"]
    assert fs.coins == 10 - d_cost["C"]
    assert s2.hexes[TARGET].building == "D"
    assert s2.hexes[TARGET].owner == "engineers"
    assert TARGET in fs.buildings["D"]


def test_build_wrong_color_rejected() -> None:
    """0 spades banked (fresh ``FactionState``): the implicit auto-transform
    (module docstring) can't afford the recolor, so this is still a hard
    error -- contrast ``test_build_wrong_color_with_enough_spades_auto_
    transforms`` below."""
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color="red", building=None, owner=None)
    s = replace(s, hexes=hexes)
    with pytest.raises(EngineError):
        handle_build(s, "engineers", _cmd("build", loc=TARGET))


def test_build_wrong_color_with_enough_spades_auto_transforms() -> None:
    """Reference-game row 58: an ordinary build on a wrong-colored hex
    implicitly pays ``spades_available`` to recolor it to home color first
    (``command_build``'s internal ``transform $where to $color`` dispatch,
    module docstring) -- no separate ``transform`` command needed, unlike
    ``test_build_wrong_color_rejected`` above (0 spades banked there)."""
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color="red", building=None, owner=None)  # gray<->red: 1
    s = replace(s, hexes=hexes)
    fs = replace(s.factions["engineers"], spades_available=2)
    s = with_faction(s, "engineers", fs)
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    after = s2.factions["engineers"]
    assert s2.hexes[TARGET].color == FACTIONS["engineers"].color
    assert s2.hexes[TARGET].building == "D"
    assert after.spades_available == 1  # 2 banked - 1 spent


def test_build_with_free_tf_marker_is_free_and_needs_direct_adjacency() -> None:
    """Nomads' ACTN (Sandstorm) queues a ``free_tf`` marker; a ``build``'s
    own implicit transform (module docstring) consumes it too, not just a
    bare ``transform`` row (``actions_power.py``'s own ACTN tests cover
    that sibling path) -- task-13 report, reference-game row 174: "action
    ACTN. build F2"."""
    s = _state()
    s = _place(s, "nomads", ANCHOR, "D")
    off_color = "red" if FACTIONS["nomads"].color != "red" else "blue"
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color=off_color, building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = replace(s, pending=(PendingDecision(faction="nomads", kind="free_tf", amount=1),))
    before = s.factions["nomads"]
    s2 = handle_build(s, "nomads", _cmd("build", loc=TARGET))
    assert s2.hexes[TARGET].color == FACTIONS["nomads"].color
    assert s2.hexes[TARGET].building == "D"
    assert s2.factions["nomads"].spades_available == before.spades_available  # free, untouched
    assert not any(p.kind == "free_tf" for p in s2.pending)


def test_build_with_free_tf_marker_rejects_non_adjacent_hex() -> None:
    s = _state()
    s = _place(s, "nomads", ANCHOR, "D")
    far = _far_hex()
    off_color = "red" if FACTIONS["nomads"].color != "red" else "blue"
    hexes = dict(s.hexes)
    hexes[far] = replace(hexes[far], color=off_color, building=None, owner=None)
    s = replace(s, hexes=hexes)
    s = replace(s, pending=(PendingDecision(faction="nomads", kind="free_tf", amount=1),))
    with pytest.raises(EngineError):
        handle_build(s, "nomads", _cmd("build", loc=far))


def _with_dwarves(state: GameState) -> GameState:
    return with_faction(state, "dwarves", FactionState.initial(FACTIONS["dwarves"]))


def _skip_chain() -> tuple[str, str]:
    """(a, b): two land hexes at ``hex_distance`` 2 via some middle hex,
    not directly adjacent -- Dwarves' tunnel range
    (``test_connectivity.py``'s identically-shaped helper)."""
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


def test_build_across_tunnel_pays_teleport_cost_and_gains_vp() -> None:
    """Task-13 report, ``4pLeague_S10_D1L1_G4`` row 71: Dwarves' tunnel
    (``factions_data.py``'s ``TeleportTrack``) was entirely unwired from
    ``handle_build`` -- a build on a hex reachable only by skipping one
    land hex silently dropped the tunnel's own W cost and VP gain,
    leaving the reference deltas 4 VP low and 2 W high.
    ``connectivity.py``'s ``teleport_crossing`` (module docstring) now
    supplies that cost/gain; this pins the fix at the ``handle_build``
    integration level (``test_connectivity.py`` pins the geometry/cost
    primitive itself)."""
    a, b = _skip_chain()
    s = _with_dwarves(_state())
    s = _place(s, "dwarves", a, "D")
    s = _clear(s, "dwarves", b)
    before = s.factions["dwarves"]
    s2 = handle_build(s, "dwarves", _cmd("build", loc=b))
    after = s2.factions["dwarves"]
    d_cost = FACTIONS["dwarves"].buildings["D"].cost
    assert after.workers == before.workers - d_cost["W"] - 2  # + tunnel's 2W (teleport_level 0)
    assert after.coins == before.coins - d_cost["C"]
    assert after.vp == before.vp + 4  # tunnel's flat VP gain
    assert s2.hexes[b].building == "D"
    assert s2.hexes[b].owner == "dwarves"


def test_build_skips_teleport_fee_for_a_hex_already_teleported_to_this_turn() -> None:
    """``FactionState.teleported_hex`` docstring: a build on a hex this
    faction already paid the tunnel fee for *this same turn* (e.g. an
    explicit ``transform`` earlier in the same ledger row) is free --
    Perl's own ``check_reachable`` returns ``({}, {})`` once
    ``$faction->{TELEPORT_TO} eq $where``. Contrast
    ``test_build_pays_teleport_fee_again_in_a_later_turn_for_the_same_hex``
    below: this exemption is strictly same-turn, not permanent (corpus
    ``4pLeague_S10_D3L2_G6`` row 350 pays the *same* hex's fee again 3
    turns later, after ``round_flow.py``'s ``_advance_actions``/
    ``begin_actions`` have cleared ``teleported_hex`` in between)."""
    a, b = _skip_chain()
    s = _with_dwarves(_state())
    s = _place(s, "dwarves", a, "D")
    s = _clear(s, "dwarves", b)  # already dwarves' color -- tf_needed=False
    s = replace(s, factions={**s.factions, "dwarves": replace(s.factions["dwarves"], teleported_hex=b)})
    before = s.factions["dwarves"]
    s2 = handle_build(s, "dwarves", _cmd("build", loc=b))
    after = s2.factions["dwarves"]
    d_cost = FACTIONS["dwarves"].buildings["D"].cost
    assert after.workers == before.workers - d_cost["W"]  # no extra tunnel W charge
    assert after.vp == before.vp  # no tunnel VP gain, already paid


def test_build_pays_teleport_fee_again_in_a_later_turn_for_the_same_hex() -> None:
    """Task-14 fix, corpus ``4pLeague_S10_D3L2_G6`` row 350: dwarves
    ``dig 1. transform A12`` at row 344 pays the tunnel fee for A12;
    ``build A12`` on that same, by-then-already-recolored hex 3 turns
    later (darklings/cultists/chaosmagicians all acted in between,
    clearing ``teleported_hex`` at least twice) pays the *same* fee
    again -- there is no cross-turn memory, only same-turn (contrast the
    sibling test above)."""
    a, b = _skip_chain()
    s = _with_dwarves(_state())
    s = _place(s, "dwarves", a, "D")
    s = _clear(s, "dwarves", b)  # already dwarves' color -- tf_needed=False
    before = s.factions["dwarves"]  # teleported_hex is None (fresh turn)
    s2 = handle_build(s, "dwarves", _cmd("build", loc=b))
    after = s2.factions["dwarves"]
    d_cost = FACTIONS["dwarves"].buildings["D"].cost
    assert after.workers == before.workers - d_cost["W"] - 2  # + tunnel's 2W (teleport_level 0)
    assert after.vp == before.vp + 4  # tunnel's flat VP gain, paid again
    assert s2.hexes[b].building == "D"


def test_build_occupied_hex_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = _place(s, "darklings", TARGET, "D", color=FACTIONS["engineers"].color)
    with pytest.raises(EngineError):
        handle_build(s, "engineers", _cmd("build", loc=TARGET))


def test_build_unreachable_hex_rejected() -> None:
    s = _state()  # engineers has no buildings at all yet -> reachable() is empty
    s = _clear(s, "engineers", TARGET)
    with pytest.raises(EngineError):
        handle_build(s, "engineers", _cmd("build", loc=TARGET))


def test_build_dwelling_max_count_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = _clear(s, "engineers", TARGET)
    max_count = FACTIONS["engineers"].buildings["D"].max_count
    fs = s.factions["engineers"]
    full = frozenset(f"stub{i}" for i in range(max_count))
    fs = replace(fs, buildings={**fs.buildings, "D": full})
    s = with_faction(s, "engineers", fs)
    with pytest.raises(EngineError):
        handle_build(s, "engineers", _cmd("build", loc=TARGET))


def test_build_setup_phase_is_free_and_ignores_reachability() -> None:
    s = _state()
    s = replace(s, round=0, phase=Phase.SETUP_DWELLINGS)
    s = _clear(s, "engineers", TARGET)  # no prior buildings anywhere
    before = s.factions["engineers"]
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    fs = s2.factions["engineers"]
    assert fs.workers == before.workers
    assert fs.coins == before.coins
    assert s2.hexes[TARGET].building == "D"
    assert s2.pending == ()  # no leech during setup


def test_build_enqueues_leech_offers_for_adjacent_opponent() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = _place(s, "darklings", NEIGHBOR, "TP")
    s = _clear(s, "engineers", TARGET)
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    assert s2.pending == (
        PendingDecision(faction="darklings", kind="leech", amount=2, source="engineers"),
    )


def test_build_founds_town_when_cluster_qualifies() -> None:
    s = _state()
    # 3 existing TPs (power 2 each = 6) chained + a new D (power 1) = 7,
    # count 4 -- exactly TOWN_SIZE (7) / count>=4.
    land = set(BOARD.land_hexes())
    chain = [ANCHOR]
    seen = {ANCHOR}
    stack = [ANCHOR]
    while stack and len(chain) < 3:
        node = stack.pop()
        for n in sorted(BOARD.adjacent[node]):
            if n in land and n not in seen and n != TARGET:
                seen.add(n)
                chain.append(n)
                stack.append(n)
                if len(chain) >= 3:
                    break
    assert len(chain) == 3
    s = _place(s, "engineers", chain[0], "TP")
    s = _place(s, "engineers", chain[1], "TP")
    s = _place(s, "engineers", chain[2], "TP")
    s = _clear(s, "engineers", TARGET)
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color=FACTIONS["engineers"].color)
    s = replace(s, hexes=hexes)

    # TARGET must be adjacent to the chain for the cluster to connect --
    # skip gracefully if this particular board layout doesn't cooperate.
    if not (set(BOARD.adjacent[TARGET]) & set(chain)):
        pytest.skip("test hex layout not adjacent to the built-up chain")

    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    town_pendings = [p for p in s2.pending if p.kind == "gain_town"]
    assert len(town_pendings) == 1
    pending = town_pendings[0]
    assert pending.faction == "engineers" and pending.amount == 1
    expected_cluster = frozenset(chain) | {TARGET}
    assert pending.source is not None
    assert frozenset(pending.source.split(",")) == expected_cluster
    # Perl (towns.pm detect_towns_from) marks the town at *detection* time,
    # not at tile-choice time -- see Important-3 fix.
    assert s2.founded_towns["engineers"] == (expected_cluster,)

    # Re-running detection (as would happen on a second board-changing
    # action for this faction before the pending is resolved) must not
    # re-enqueue a pending for the same already-recorded cluster --
    # regression test for the duplicate-gain_town-pending bug.
    s3 = _maybe_queue_town(s2, "engineers")
    assert [p for p in s3.pending if p.kind == "gain_town"] == town_pendings


# --------------------------------------------------------------------------
# upgrade
# --------------------------------------------------------------------------


def test_upgrade_d_to_tp_isolated_pays_double_coins() -> None:
    s = _state()
    s = _place(s, "engineers", TARGET, "D")
    tp_cost = FACTIONS["engineers"].buildings["TP"].cost
    before = s.factions["engineers"]
    s2 = handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TP"))
    fs = s2.factions["engineers"]
    assert fs.workers == before.workers - tp_cost["W"]
    assert fs.coins == before.coins - tp_cost["C"] * 2
    assert s2.hexes[TARGET].building == "TP"
    assert TARGET in fs.buildings["TP"]
    assert TARGET not in fs.buildings["D"]


def test_upgrade_d_to_tp_adjacent_opponent_pays_normal_cost() -> None:
    s = _state()
    s = _place(s, "engineers", TARGET, "D")
    s = _place(s, "darklings", NEIGHBOR, "TP")
    tp_cost = FACTIONS["engineers"].buildings["TP"].cost
    before = s.factions["engineers"]
    s2 = handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TP"))
    fs = s2.factions["engineers"]
    assert fs.coins == before.coins - tp_cost["C"]


def test_upgrade_to_tp_scores_current_round_tiles_build_vp() -> None:
    """Reference-game replay row 47: engineers ``upgrade F6 to TP`` under
    round 1's ``TP >> 3`` tile jumps VP by exactly 3, with no companion
    ledger row (task-13 report; ``tiles.scored_vp``/``commands.pm`` 304)."""
    s = _state()  # round=1; GAME_ID's own score_tiles[0] is `TP >> 3`, build mode.
    s = _place(s, "engineers", TARGET, "D")
    before_vp = s.factions["engineers"].vp
    s2 = handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TP"))
    assert s2.factions["engineers"].vp == before_vp + 3


def test_build_dwelling_scores_current_round_tiles_build_vp_when_keyed() -> None:
    """Round 6's tile (``4pLeague_S10_D1L1_G1``'s ``score_tiles[5]``) is
    ``D >> 2``, build mode -- a fresh dwelling should gain 2 VP; round 1's
    tile (``TP >> 3``) does not key ``D`` at all, so an equivalent build
    earlier in the game gains none."""
    s = replace(_state(), round=6)
    s = _place(s, "engineers", ANCHOR, "D")
    s = _clear(s, "engineers", TARGET)
    before_vp = s.factions["engineers"].vp
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    assert s2.factions["engineers"].vp == before_vp + 2

    s_round1 = _state()
    s_round1 = _place(s_round1, "engineers", ANCHOR, "D")
    s_round1 = _clear(s_round1, "engineers", TARGET)
    before_vp_round1 = s_round1.factions["engineers"].vp
    s2_round1 = handle_build(s_round1, "engineers", _cmd("build", loc=TARGET))
    assert s2_round1.factions["engineers"].vp == before_vp_round1


def test_build_dwelling_during_setup_never_scores_tile_vp() -> None:
    """``command_build``'s ``if ($game{round})`` guard (``commands.pm``
    244-245): setup dwellings (``round == 0``) never trigger the score-tile
    build bonus, whatever the tile's own keys might otherwise say."""
    s = GameState.initial(load_setup(GAME_ID))  # round=0, Phase.SETUP_DWELLINGS
    before_vp = s.factions["engineers"].vp
    s2 = handle_build(s, "engineers", _cmd("build", loc="E7"))
    assert s2.factions["engineers"].vp == before_vp


def test_upgrade_wrong_source_building_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", TARGET, "D")
    with pytest.raises(EngineError):
        handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TE"))


def test_upgrade_te_enqueues_gain_favor_pending() -> None:
    s = _state()
    s = _place(s, "engineers", TARGET, "TP")
    s2 = handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TE"))
    assert PendingDecision(faction="engineers", kind="gain_favor", amount=1) in s2.pending
    assert s2.hexes[TARGET].building == "TE"


def test_upgrade_max_count_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", TARGET, "TP")
    fs = s.factions["engineers"]
    max_count = FACTIONS["engineers"].buildings["TE"].max_count
    full = frozenset(f"stub{i}" for i in range(max_count))
    fs = replace(fs, buildings={**fs.buildings, "TE": full})
    s = with_faction(s, "engineers", fs)
    with pytest.raises(EngineError):
        handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TE"))


def _sh_state(faction: str) -> GameState:
    s = _state()
    fs = replace(FactionState.initial(FACTIONS[faction]), coins=50, workers=20)
    factions = dict(s.factions)
    factions[faction] = fs
    s = replace(s, factions=factions, turn_order=(faction,))
    return _place(s, faction, TARGET, "TP")


def test_upgrade_sh_alchemists_gains_12_pw_directly() -> None:
    s = _sh_state("alchemists")
    before = s.factions["alchemists"].power
    s2 = handle_upgrade(s, "alchemists", _cmd("upgrade", loc=TARGET, building="SH"))
    assert s2.factions["alchemists"].power == before.gain(12)


def test_upgrade_sh_cultists_gains_7_vp_directly() -> None:
    s = _sh_state("cultists")
    before = s.factions["cultists"].vp
    s2 = handle_upgrade(s, "cultists", _cmd("upgrade", loc=TARGET, building="SH"))
    assert s2.factions["cultists"].vp == before + 7


def test_upgrade_sh_halflings_grants_spades_and_vp_immediately() -> None:
    """Fix-2 (task-9 review): the SH's 3-spade grant must land on
    ``spades_available`` -- and its +1 VP/spade passive -- the instant the
    SH is built (``resources.pm`` ``adjust_resource`` gain-mode timing),
    not deferred behind a pending that a player could skip past entirely.
    """
    s = _sh_state("halflings")
    before_vp = s.factions["halflings"].vp
    s2 = handle_upgrade(s, "halflings", _cmd("upgrade", loc=TARGET, building="SH"))
    fs = s2.factions["halflings"]
    assert fs.spades_available == 3
    assert fs.vp == before_vp + 3
    assert [p for p in s2.pending if p.kind == "halflings_spades"] == []


def test_upgrade_sh_darklings_pushes_convert_w_to_p_pending() -> None:
    s = _sh_state("darklings")
    s2 = handle_upgrade(s, "darklings", _cmd("upgrade", loc=TARGET, building="SH"))
    assert PendingDecision(faction="darklings", kind="convert_w_to_p", amount=3) in s2.pending


def test_upgrade_sh_mermaids_advances_shipping_and_scores_advance_vp() -> None:
    """Corrected (task-13 fix, cites ``resources.pm``'s ``adjust_resource``
    ``GAIN_(TELEPORT|SHIP)`` branch, 276-286): a ``GAIN_SHIP`` grant --
    from a build_gain (here, Mermaids' SH) or a town tile alike -- pays
    ``ShippingTrack.advance_vp[level]`` the same as a player-invoked
    ``advance ship`` row (``actions_pass.py``'s ``handle_advance``); it is
    not a free level bump. An earlier revision of this test asserted the
    opposite ("with_no_vp") with no Perl citation -- disproved by
    reference-game row 210 (nomads' TW7 grant), which needs the exact
    same code path to score correctly (``_advance_shipping``'s docstring)."""
    s = _sh_state("mermaids")
    before = s.factions["mermaids"]
    s2 = handle_upgrade(s, "mermaids", _cmd("upgrade", loc=TARGET, building="SH"))
    fs = s2.factions["mermaids"]
    assert fs.shipping == before.shipping + 1
    assert fs.vp == before.vp + FACTIONS["mermaids"].shipping.advance_vp[before.shipping]


# --------------------------------------------------------------------------
# bridge
# --------------------------------------------------------------------------


def _bridge_pair() -> tuple[str, str]:
    return tuple(sorted(next(iter(_bridgable_pairs()))))  # type: ignore[return-value]


# 29 distinct `bridge X:Y` command pairs extracted from every game in
# data/raw/games/*.json.gz (script in task-8-report.md). Committed here as
# a real regression test rather than a one-off manual check: every real
# bridge ever built in the crawled corpus must be a legal span per
# `_bridgable_pairs` (map.pm `setup_valid_bridges`, 164-203). 29 is also
# the map's well-known total legal-bridge count.
_CORPUS_BRIDGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("A11", "C5"),
    ("A3", "C1"),
    ("A7", "C3"),
    ("B1", "C1"),
    ("B1", "D1"),
    ("B2", "C1"),
    ("B3", "C3"),
    ("B4", "C3"),
    ("B5", "C5"),
    ("B6", "C5"),
    ("B6", "D8"),
    ("C2", "D3"),
    ("C2", "D4"),
    ("C2", "E5"),
    ("C4", "D5"),
    ("C5", "D6"),
    ("D6", "E8"),
    ("D6", "E9"),
    ("E4", "G1"),
    ("E8", "G3"),
    ("F1", "H1"),
    ("F2", "G1"),
    ("F2", "H2"),
    ("F3", "G1"),
    ("F4", "G3"),
    ("G2", "H4"),
    ("G2", "I6"),
    ("G4", "H5"),
    ("H6", "I9"),
)


def test_bridgable_pairs_matches_every_real_corpus_bridge() -> None:
    pairs = _bridgable_pairs()
    missing = [pair for pair in _CORPUS_BRIDGE_PAIRS if frozenset(pair) not in pairs]
    assert missing == []
    assert len(pairs) == len(_CORPUS_BRIDGE_PAIRS) == 29


def test_bridge_consumes_pending_and_places_it() -> None:
    a, b = _bridge_pair()
    s = _state()
    s = _place(s, "engineers", a, "D")
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="bridge", amount=1),))
    s2 = handle_bridge(s, "engineers", _cmd("bridge", loc=a, loc2=b))
    assert s2.pending == ()
    assert frozenset({a, b}) in s2.bridges
    assert s2.factions["engineers"].bridges_built == 1


def test_bridge_without_pending_rejected() -> None:
    a, b = _bridge_pair()
    s = _state()
    s = _place(s, "engineers", a, "D")
    with pytest.raises(EngineError):
        handle_bridge(s, "engineers", _cmd("bridge", loc=a, loc2=b))


def test_bridge_illegal_geometry_rejected() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="bridge", amount=1),))
    with pytest.raises(EngineError):
        handle_bridge(s, "engineers", _cmd("bridge", loc=ANCHOR, loc2=NEIGHBOR))


def test_bridge_requires_faction_building_at_an_endpoint() -> None:
    a, b = _bridge_pair()
    s = _state()
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="bridge", amount=1),))
    with pytest.raises(EngineError):
        handle_bridge(s, "engineers", _cmd("bridge", loc=a, loc2=b))


def test_bridge_count_cap_enforced() -> None:
    a, b = _bridge_pair()
    s = _state()
    s = _place(s, "engineers", a, "D")
    fs = replace(s.factions["engineers"], bridges_built=BRIDGE_COUNT)
    s = with_faction(s, "engineers", fs)
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="bridge", amount=1),))
    with pytest.raises(EngineError):
        handle_bridge(s, "engineers", _cmd("bridge", loc=a, loc2=b))


# --------------------------------------------------------------------------
# gain_favor
# --------------------------------------------------------------------------


def test_gain_favor_pops_pending_pool_and_advances_cult() -> None:
    s = _state()
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),))
    before_pool = s.favors_pool["FAV1"]
    s2 = handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV1"))
    fs = s2.factions["engineers"]
    assert "FAV1" in fs.favors
    assert s2.favors_pool["FAV1"] == before_pool - 1
    assert s2.cults["engineers"]["FIRE"] == 3  # FAV1: FIRE, 3 steps
    assert s2.pending == ()


def test_gain_favor_meters_amount_for_multi_favor_grants() -> None:
    s = _state()
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=2),))
    s2 = handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV1"))
    assert s2.pending == (PendingDecision(faction="engineers", kind="gain_favor", amount=1),)
    s3 = handle_gain_favor(s2, "engineers", _cmd("gain_favor", tile="FAV2"))
    assert s3.pending == ()
    assert set(s3.factions["engineers"].favors) == {"FAV1", "FAV2"}


def test_gain_favor_already_held_rejected() -> None:
    s = _state()
    fs = replace(s.factions["engineers"], favors=("FAV1",))
    s = with_faction(s, "engineers", fs)
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),))
    with pytest.raises(EngineError):
        handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV1"))


def test_gain_favor_empty_pool_rejected() -> None:
    s = _state()
    pool = dict(s.favors_pool)
    pool["FAV1"] = 0
    s = replace(
        s,
        favors_pool=pool,
        pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),),
    )
    with pytest.raises(EngineError):
        handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV1"))


def test_gain_favor_never_scores_vp_immediately_even_for_fav10_fav11() -> None:
    """Reference-game row 59: mermaids gain FAV11 (``vp={"D": 2}``) with 1
    dwelling already on the board and VP is unchanged (deltas.parquet
    delta 0) -- ``FavorTile.vp`` is a passive per-build bonus (module
    docstring's ``maybe_score_favor_tile`` citation), never a one-time
    snapshot taken at grant time, so this holds even with existing
    matching buildings already on the board."""
    s = _state()
    s = _place(s, "engineers", ANCHOR, "TP")
    s = _place(s, "engineers", TARGET, "TP")
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),))
    before_vp = s.factions["engineers"].vp
    s2 = handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV10"))
    assert s2.factions["engineers"].vp == before_vp


def test_build_dwelling_scores_held_fav11_vp_passively() -> None:
    """FAV11 (``vp={"D": 2}``) held at build time scores 2 VP for a fresh
    dwelling, same round-1 fixture as ``test_build_dwelling_scores_
    current_round_tiles_build_vp_when_keyed`` (whose ``TP >> 3`` tile does
    not key ``D``, so this isolates the favor-only contribution)."""
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    s = _clear(s, "engineers", TARGET)
    fs = replace(s.factions["engineers"], favors=("FAV11",))
    s = with_faction(s, "engineers", fs)
    before_vp = s.factions["engineers"].vp
    s2 = handle_build(s, "engineers", _cmd("build", loc=TARGET))
    assert s2.factions["engineers"].vp == before_vp + 2


def test_upgrade_scores_held_fav10_vp_passively_alongside_score_tile() -> None:
    """FAV10 (``vp={"TP": 3}``) held at upgrade time scores 3 VP on top of
    round 1's own ``TP >> 3`` score-tile bonus (``test_upgrade_to_tp_
    scores_current_round_tiles_build_vp``) -- both fire from the same
    upgrade, additively (module docstring: ``maybe_score_favor_tile``
    then ``maybe_score_current_score_tile``, independent sources)."""
    s = _state()
    s = _place(s, "engineers", TARGET, "D")
    fs = replace(s.factions["engineers"], favors=("FAV10",))
    s = with_faction(s, "engineers", fs)
    before_vp = s.factions["engineers"].vp
    s2 = handle_upgrade(s, "engineers", _cmd("upgrade", loc=TARGET, building="TP"))
    assert s2.factions["engineers"].vp == before_vp + 3 + 3  # FAV10 + score tile


def test_gain_favor_fav5_immediately_rescans_for_a_newly_qualifying_town() -> None:
    """Corpus game ``4pLeague_S10_D1L1_G3`` row 321: swarmlings'
    "upgrade H2 to TE. gain_favor FAV5. gain_town TW8" -- the TE upgrade
    alone leaves a 4-hex cluster at power 6, short of the default
    ``TOWN_SIZE`` (7), but FAV5's ``TOWN_SIZE => -1`` passive (applied
    the instant it's taken, ``resources.pm``'s ``adjust_resource`` FAV
    branch -- ``handle_gain_favor``'s own docstring) drops the threshold
    to 6, so the *same row*'s ``gain_town TW8`` answer needs a pending
    that only exists because granting FAV5 re-scanned for it -- not
    deferred to this cluster's next build/upgrade."""
    s = _state()
    land = set(BOARD.land_hexes())
    chain = [ANCHOR]
    seen = {ANCHOR}
    stack = [ANCHOR]
    while stack and len(chain) < 4:
        node = stack.pop()
        for n in sorted(BOARD.adjacent[node]):
            if n in land and n not in seen:
                seen.add(n)
                chain.append(n)
                stack.append(n)
                if len(chain) >= 4:
                    break
    assert len(chain) == 4
    for h in chain[:2]:
        s = _place(s, "engineers", h, "TP")  # power 2 each
    for h in chain[2:]:
        s = _place(s, "engineers", h, "D")  # power 1 each -- total 6, below TOWN_SIZE(7)
    assert _maybe_queue_town(s, "engineers").pending == ()  # doesn't qualify yet

    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),))
    s2 = handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV5"))
    assert [p.kind for p in s2.pending] == ["gain_town"]
    assert s2.founded_towns["engineers"] == (frozenset(chain),)


# --------------------------------------------------------------------------
# gain_town
# --------------------------------------------------------------------------


def test_gain_town_applies_tile_and_records_founded_cluster() -> None:
    s = _state()
    land = set(BOARD.land_hexes())
    chain = [ANCHOR]
    seen = {ANCHOR}
    stack = [ANCHOR]
    while stack and len(chain) < 4:
        node = stack.pop()
        for n in sorted(BOARD.adjacent[node]):
            if n in land and n not in seen:
                seen.add(n)
                chain.append(n)
                stack.append(n)
                if len(chain) >= 4:
                    break
    assert len(chain) == 4
    for h in chain:
        s = _place(s, "engineers", h, "TP")  # 4 * power 2 = 8 >= TOWN_SIZE(7), count 4

    s = _maybe_queue_town(s, "engineers")  # realistic detection: records + queues
    assert [p.kind for p in s.pending] == ["gain_town"]
    assert s.founded_towns["engineers"] == (frozenset(chain),)

    before_pool = s.towns_pool["TW1"]
    s2 = handle_gain_town(s, "engineers", _cmd("gain_town", tile="TW1"))
    assert s2.pending == ()
    assert "TW1" in s2.factions["engineers"].towns
    assert s2.founded_towns["engineers"] == (frozenset(chain),)  # unchanged, already recorded
    assert s2.towns_pool["TW1"] == before_pool - 1
    assert any(set(chain) <= set(c) for c in s2.founded_towns["engineers"])


def test_gain_town_n1_resolves_that_many_pendings_with_the_same_tile() -> None:
    """Task-13 report, ``4pLeague_S10_D1L1_G4`` row 257 (raw ``+2TW1``):
    ``cmd.n1`` is a *count* of separate ``gain_town`` pendings being
    resolved with the same tile type in one row, not a scaled reward
    (``handle_gain_town``'s own docstring cites ``resources.pm``
    ``adjust_resource``'s ``TW`` branch, 355-362: ``for (1..$delta) {
    gain $faction, $tiles{$type}{gain}, 'TW' }`` -- the *entire* per-tile
    gain applied twice, not once at double size). Two independently
    queued clusters (distinct ``source``s) both resolving to TW1 must
    grant TW1's VP/resources twice and consume both pendings, leaving the
    pool down by 2."""
    s = _state()
    s = replace(
        s,
        pending=(
            PendingDecision(faction="engineers", kind="gain_town", source="cluster-a"),
            PendingDecision(faction="engineers", kind="gain_town", source="cluster-b"),
        ),
    )
    before = s.factions["engineers"]
    before_pool = s.towns_pool["TW1"]

    # A single grant, for comparison -- proves n1=2 isn't just "apply once
    # at double size" but genuinely two independent applications.
    s_single = replace(
        s, pending=(PendingDecision(faction="engineers", kind="gain_town", source="cluster-a"),)
    )
    single_vp_delta = (
        handle_gain_town(s_single, "engineers", _cmd("gain_town", tile="TW1")).factions[
            "engineers"
        ].vp
        - before.vp
    )

    s2 = handle_gain_town(s, "engineers", _cmd("gain_town", tile="TW1", n1=2))
    after = s2.factions["engineers"]
    assert s2.pending == ()
    assert after.towns == ("TW1", "TW1")
    assert s2.towns_pool["TW1"] == before_pool - 2
    assert after.vp == before.vp + 2 * single_vp_delta


def test_gain_town_without_pending_rejected() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_gain_town(s, "engineers", _cmd("gain_town", tile="TW1"))


def test_gain_town_tw5_drives_cult_advance_on_all_four_tracks() -> None:
    """Reference-game row 203: darklings' TW5 grant (``gain={"KEY": 1,
    "FIRE": 1, "WATER": 1, "EARTH": 1, "AIR": 1}``) must step all four
    cult tracks and apply any threshold power -- ``apply_town_tile``
    itself deliberately leaves those four keys unapplied (``towns.py``'s
    own docstring); ``handle_gain_town`` is the "caller" that docstring
    says must drive ``cults.advance`` (task-13 report)."""
    s = _state()
    fs = replace(s.factions["engineers"], keys=1)
    s = with_faction(s, "engineers", fs)
    s = replace(
        s,
        cults={**s.cults, "engineers": {"FIRE": 2, "WATER": 4, "EARTH": 9, "AIR": 0}},
        pending=(PendingDecision(faction="engineers", kind="gain_town", source="cluster"),),
    )
    s2 = handle_gain_town(s, "engineers", _cmd("gain_town", tile="TW5"))
    assert s2.cults["engineers"] == {"FIRE": 3, "WATER": 5, "EARTH": 10, "AIR": 1}
    # FIRE crosses 3 (+1), WATER crosses 5 (+2), EARTH's 10 costs the key (+3): 6 total.
    assert s2.factions["engineers"].power.as_str() == "0/9/3"  # started 3/9/0, gain(6)
    # started with 1 key; TW5 itself grants +1 (apply_town_tile's own KEY
    # gain), then crossing EARTH's 10 spends 1 -- net back to 1.
    assert s2.factions["engineers"].keys == 1
    assert s2.cult_10["EARTH"] == "engineers"


def test_gain_town_retries_a_cult_blocked_at_9_once_its_own_key_covers_it() -> None:
    """Task-13 report follow-up, ``4pLeague_S10_D1L1_G6`` row 315: nomads'
    ``gain_favor FAV5`` (2 FIRE steps, 8->10) earlier in the same ledger
    row blocked at 9 for lack of a key (``fs.cult_blocked == {"FIRE"}``,
    ``fs.keys == 0``); the very next command, ``gain_town TW7``, grants
    exactly the 1 key needed and must retroactively bump FIRE to 10 --
    ``resources.pm`` ``adjust_resource``'s ``KEY`` branch (355-362, cited
    in ``cults.py`` ``advance``'s docstring), ported as
    ``_retry_blocked_cults``."""
    s = _state()
    fs = replace(s.factions["nomads"], keys=0, cult_blocked=frozenset({"FIRE"}))
    s = with_faction(s, "nomads", fs)
    s = replace(
        s,
        cults={**s.cults, "nomads": {"FIRE": 9, "WATER": 0, "EARTH": 7, "AIR": 0}},
        cult_10={**s.cult_10, "FIRE": None},
        pending=(PendingDecision(faction="nomads", kind="gain_town", source="cluster"),),
    )
    before_power = s.factions["nomads"].power
    s2 = handle_gain_town(s, "nomads", _cmd("gain_town", tile="TW7"))
    assert s2.cults["nomads"]["FIRE"] == 10
    assert s2.cult_10["FIRE"] == "nomads"
    assert s2.factions["nomads"].cult_blocked == frozenset()
    # TW7 grants 1 key; retrying FIRE spends it (net back to 0) and scores
    # the +3 power for crossing into the 10-slot, on top of TW7's own gain.
    assert s2.factions["nomads"].keys == 0
    assert s2.factions["nomads"].power == before_power.gain(3)


def test_gain_town_leaves_multiple_blocked_cults_alone_until_keys_cover_all_of_them() -> None:
    """A single key only retries a batch of blocked cults once it covers
    *every* one of them at once (Perl's own ``>=`` check gates the whole
    batch, not cult-by-cult) -- with two blocked cults and only 1 key,
    neither retries yet."""
    s = _state()
    fs = replace(s.factions["nomads"], keys=0, cult_blocked=frozenset({"FIRE", "WATER"}))
    s = with_faction(s, "nomads", fs)
    s = replace(
        s,
        cults={**s.cults, "nomads": {"FIRE": 9, "WATER": 9, "EARTH": 7, "AIR": 0}},
        cult_10={**s.cult_10, "FIRE": None, "WATER": None},
        pending=(PendingDecision(faction="nomads", kind="gain_town", source="cluster"),),
    )
    s2 = handle_gain_town(s, "nomads", _cmd("gain_town", tile="TW7"))  # grants 1 key
    assert s2.cults["nomads"]["FIRE"] == 9
    assert s2.cults["nomads"]["WATER"] == 9
    assert s2.factions["nomads"].cult_blocked == frozenset({"FIRE", "WATER"})
    assert s2.factions["nomads"].keys == 1  # banked, not spent -- not enough to cover both yet


def test_gain_town_tw7_drives_shipping_advance_vp() -> None:
    """Reference-game row 210: nomads' TW7 grant (``gain={"KEY": 1,
    "VP": 4, "GAIN_SHIP": 1, "carpet_range": 1}``) must also bump
    shipping and score ``advance_vp`` for that step, on top of TW7's own
    flat VP -- ``_apply_town_ship_gain``'s docstring (task-13 report:
    nomads' shipping was already at level 1 by row 210, from an earlier
    ``advance ship`` row, so the missing VP was ``advance_vp[1]``)."""
    s = _state()
    fs = replace(s.factions["nomads"], shipping=1)
    s = with_faction(s, "nomads", fs)
    s = replace(s, pending=(PendingDecision(faction="nomads", kind="gain_town", source="cluster"),))
    before_vp = s.factions["nomads"].vp
    s2 = handle_gain_town(s, "nomads", _cmd("gain_town", tile="TW7"))
    assert s2.factions["nomads"].shipping == 2
    expected_vp = 4 + FACTIONS["nomads"].shipping.advance_vp[1]  # TW7's own VP + the ship advance
    assert s2.factions["nomads"].vp == before_vp + expected_vp
