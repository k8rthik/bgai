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
    s = _state()
    s = _place(s, "engineers", ANCHOR, "D")
    hexes = dict(s.hexes)
    hexes[TARGET] = replace(hexes[TARGET], color="red", building=None, owner=None)
    s = replace(s, hexes=hexes)
    with pytest.raises(EngineError):
        handle_build(s, "engineers", _cmd("build", loc=TARGET))


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


def test_upgrade_sh_mermaids_advances_shipping_with_no_vp() -> None:
    s = _sh_state("mermaids")
    before = s.factions["mermaids"]
    s2 = handle_upgrade(s, "mermaids", _cmd("upgrade", loc=TARGET, building="SH"))
    fs = s2.factions["mermaids"]
    assert fs.shipping == before.shipping + 1
    assert fs.vp == before.vp


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


def test_gain_favor_fav10_grants_vp_scaled_by_tp_count() -> None:
    s = _state()
    s = _place(s, "engineers", ANCHOR, "TP")
    s = _place(s, "engineers", TARGET, "TP")
    s = replace(s, pending=(PendingDecision(faction="engineers", kind="gain_favor", amount=1),))
    before_vp = s.factions["engineers"].vp
    s2 = handle_gain_favor(s, "engineers", _cmd("gain_favor", tile="FAV10"))
    assert s2.factions["engineers"].vp == before_vp + 3 * 2  # FAV10: 3 VP per TP, 2 TPs


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


def test_gain_town_without_pending_rejected() -> None:
    s = _state()
    with pytest.raises(EngineError):
        handle_gain_town(s, "engineers", _cmd("gain_town", tile="TW1"))
