"""Command/state rendering with parser round-trip (LLM-harness plan, task 7)."""

import dataclasses

import pytest
from driver_helpers import fast_forward_setup

from bgai.arena.live_driver import new_game, next_actor, offered_moves
from bgai.arena.setup_factory import fresh_setup
from bgai.data.ledger_parser import parse_command
from bgai.engine.tm.state import PendingDecision
from bgai.mcp.render import diff_factions, render_command, render_moves, render_state

_FIELDS = (
    "loc", "loc2", "building", "tile", "cult", "color", "target", "reason",
    "res1", "res2", "n1", "n2",
)

EXAMPLES = [
    "build A1",
    "upgrade A1 to TP",
    "upgrade F6 to SA",
    "transform A1 to gray",
    "transform A1",
    "dig 2",
    "send p to FIRE",
    "action ACT5",
    "pass BON3",
    "pass",
    "leech 2 from nomads",
    "decline 2 from nomads",
    "convert 1PW to 1C",
    "burn 2",
    "advance ship",
    "advance dig",
    "bridge A1:B1",
    "connect r1",
    "+FAV11",
    "+TW3",
    "+1FIRE",
    "-1FIRE",
    "wait",
    "done",
]


def _fields(cmd):
    return {f: getattr(cmd, f) for f in _FIELDS} | {"verb": cmd.verb}


@pytest.mark.parametrize("text", EXAMPLES)
def test_render_round_trips_through_parser(text):
    parsed = parse_command(text)
    assert parsed is not None, text
    rendered = render_command(parsed)
    reparsed = parse_command(rendered)
    assert reparsed is not None, rendered
    assert _fields(reparsed) == _fields(parsed)


def test_render_unknown_verb_raises():
    from bgai.engine.tm.legal_shared import cmd

    with pytest.raises(ValueError, match="unrenderable"):
        render_command(cmd("flarb"))


def _positions():
    fresh = new_game(fresh_setup(seed=5))
    mid = fast_forward_setup(seed=5)
    leechy = dataclasses.replace(
        fresh,
        pending=(
            PendingDecision(
                faction=fresh.setup.factions[1], kind="leech", amount=2,
                source=fresh.setup.factions[0],
            ),
        ),
    )
    return [fresh, mid, leechy]


def test_every_offered_move_round_trips():
    for state in _positions():
        for faction in state.setup.factions:
            for move in offered_moves(state, faction, turn_open=False):
                rendered = render_command(move)
                reparsed = parse_command(rendered)
                assert reparsed is not None, rendered
                assert _fields(reparsed) == _fields(move), rendered


def test_render_state_smoke():
    state = new_game(fresh_setup(seed=5))
    text = render_state(state)
    for faction in state.setup.factions:
        assert faction in text
    assert "SETUP_DWELLINGS" in text
    assert "round 0/6" in text
    assert "FIRE" in text and "WATER" in text


def test_render_state_viewer_first():
    state = new_game(fresh_setup(seed=5))
    viewer = state.setup.factions[2]
    text = render_state(state, viewer=viewer)
    block_positions = {f: text.index(f"\n{f}") for f in state.setup.factions}
    assert min(block_positions, key=block_positions.get) == viewer


def test_render_moves_numbered():
    state = new_game(fresh_setup(seed=5))
    actor = next_actor(state)
    text = render_moves(offered_moves(state, actor, turn_open=False), actor)
    assert "1." in text
    assert actor in text


def test_diff_factions_reports_resource_change():
    state = fast_forward_setup(seed=5)
    faction = state.setup.factions[0]
    fs = state.factions[faction]
    factions = dict(state.factions)
    factions[faction] = dataclasses.replace(fs, coins=fs.coins + 3, vp=fs.vp + 2)
    after = dataclasses.replace(state, factions=factions)
    text = diff_factions(state, after)
    assert faction in text
    assert "C +3" in text
    assert "VP +2" in text
