"""tests/test_legal.py

Task 15: ``legal_moves``/``legal_moves_for``/``legal_moves_all``
(``bgai.engine.tm.legal``).

- Step 1 (hand tests): a handful of small, directly-constructed states
  pin the documented contract (a passed active faction gets no main
  actions; a leech pending's answers are present; ``Phase.FINISHED``
  yields nothing; a setup-dwellings state offers exactly the valid
  dwelling spots).
- Step 2 (the heart of the task): containment against real replayed
  games, and a generative apply()-succeeds smoke test. Both reuse
  ``replay.py``'s private row-driving helpers directly (the same pattern
  ``tests/test_replay_corpus.py``'s own ``_replay_to_final_state`` already
  uses) rather than duplicating the Task-12 phase-transition contract --
  ``_walk_decisions`` below is a thin per-command-hook variant of
  ``replay._apply_row_commands`` that calls back with (pre_state, faction,
  cmd, post_state) for every ``Kind.DECISION`` command actually applied.

**Containment normalization** (documented once here, per the brief's
explicit invitation to normalize rather than whitelist):

- ``convert``: a real ``convert N res1 to M res2`` row (``N``/``M`` a
  multiple of the atomic rate) is matched by *any* legal ``convert``
  entry sharing its ``(res1, res2)`` pair -- ``legal.py``'s single-unit
  granularity policy (module docstring) means the exact ``n1``/``n2`` are
  never expected to match a scaled-up real row.
- ``dig``/``burn``: matched on the *exact* ``n1`` -- these are enumerated
  precisely (``legal_actions.py``'s granularity policy), so no
  decomposition is needed.
- ``transform``: matched on ``loc`` plus ``color`` (alias-normalized:
  the ledger grammar accepts the British "grey" spelling verbatim,
  ``ledger_parser.py``'s ``transform`` rule, while ``legal.py`` always
  generates the canonical wheel spelling "gray", ``actions_terraform.py``'s
  own ``_alias_color`` table). A bare ``transform HEX`` (no ``to COLOR``
  clause -- ``color is None``) matches any legal entry for that hex,
  regardless of color, since the real engine's default-target resolution
  (``actions_terraform._default_target_color``) is one specific color
  among the several this module enumerates as reachable.
- ``leech``/``decline``: matched by verb alone (any target) -- a real
  ``leech N`` accept's exact ``N``/``target`` disambiguate *which* queued
  offer it answers (``leech.py``'s ``_find_leech_pending`` docstring), not
  whether accepting is legal at all; every one of a faction's outstanding
  offers is independently enumerated by :func:`legal.legal_moves_for`.
- ``gain_town``'s ``cmd.n1`` (a count of simultaneously-resolved
  same-tile pendings, not a scaled amount -- ``actions_build.
  handle_gain_town``'s own docstring) is ignored; matched by ``tile`` alone.
- Every other DECISION verb (``build``/``upgrade``/``bridge``/``action``/
  ``pass``/``advance``/``connect``/``send``/``gain_favor``) is matched on
  its own identifying fields (hex/building/tile/reason/cult), exactly.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.legal import legal_moves, legal_moves_all, legal_moves_for
from bgai.engine.tm.power import Power
from bgai.engine.tm.replay import (
    _CULT_INCOME_VERBS,
    _MAIN_TRACK_VERBS,
    _OTHER_INCOME_VERBS,
    _advance_after_row,
    _apply_pending_drops,
    _dedupe_income_commands,
    _delta_lookup,
    _ensure_actions_phase_started,
    _ensure_cult_income_landed,
    _iter_rows,
)
from bgai.engine.tm.round_flow import (
    advance_turn,
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


def _state() -> GameState:
    s = GameState.initial(load_setup(GAME_ID))
    return replace(s, round=1, phase=Phase.ACTIONS)


# --------------------------------------------------------------------------
# Step 1: hand tests
# --------------------------------------------------------------------------


def test_finished_phase_generates_nothing() -> None:
    state = replace(_state(), phase=Phase.FINISHED)
    assert legal_moves(state) == ()
    for faction in state.factions:
        assert legal_moves_for(state, faction) == ()


def test_dropped_faction_generates_nothing() -> None:
    state = _state()
    faction = state.turn_order[0]
    fs = replace(state.factions[faction], dropped=True)
    state = with_faction(state, faction, fs)
    assert legal_moves_for(state, faction) == ()


def test_passed_active_faction_gets_no_main_actions() -> None:
    """A faction that is both ``active_faction(state)`` *and* already
    marked ``passed`` never happens under normal round-flow-driven play
    (``round_flow._advance_actions`` always skips a passed faction when
    picking the next active index) -- this is a defensive, hand-constructed
    edge case. The order-exempt baseline (``convert``/``wait``,
    ``apply.py``'s ``_ORDER_EXEMPT_VERBS`` -- see ``legal.py``'s module
    docstring, Design decision 4, for the corpus evidence that a passed,
    non-active faction genuinely does still submit ``wait``/``convert``
    rows) is not itself gated on ``passed``, so this only asserts the
    *main-track* action set (build/upgrade/dig/...) is empty, not that the
    whole result is the empty tuple.
    """
    state = _state()
    faction = active_faction(state)
    fs = replace(
        state.factions[faction],
        passed=True,
        coins=0,
        workers=0,
        priests=0,
        power=Power(0, 0, 0),
    )
    state = with_faction(state, faction, fs)
    moves = legal_moves(state)
    assert moves == (ParsedCommand(verb="wait", kind=Kind.DECISION, raw="wait"),)


def test_leech_pending_offers_leech_and_decline() -> None:
    """A queued ``leech`` pending's answers are present in
    ``legal_moves(state)`` -- but, per ``legal.py``'s module docstring
    (Design decision 1, falsified-by-containment-evidence), *additively*:
    real corpus rows show a faction with an outstanding pending still
    submitting ordinary main-track moves (e.g. ``pass``) in between, so
    this asserts the leech pair is present (a subset check), not that it
    is the *only* thing present -- an exact-equality version of this test
    was the brief's original literal wording, disproven by
    ``4pLeague_S27_D1L1_G1``'s cultists (see the module docstring citation).
    """
    state = _state()
    faction = active_faction(state)
    state = replace(
        state,
        pending=(PendingDecision(faction=faction, kind="leech", amount=2, source="darklings"),),
    )
    moves = legal_moves(state)
    leech_cmd = ParsedCommand(
        verb="leech", kind=Kind.DECISION, raw="leech", n1=2, target="darklings"
    )
    decline_cmd = ParsedCommand(
        verb="decline", kind=Kind.DECISION, raw="decline", target="darklings"
    )
    assert {leech_cmd, decline_cmd}.issubset(set(moves))
    # No *other* pending kind's answers leak in (no queued gain_favor/
    # gain_town/cult_choice/convert_w_to_p pending exists in this state).
    assert not any(m.verb == "gain_favor" for m in moves)
    assert not any(m.verb == "gain_town" for m in moves)


def test_gain_favor_pending_offers_every_available_untaken_tile() -> None:
    state = _state()
    faction = active_faction(state)
    fs = replace(
        state.factions[faction],
        coins=0,
        workers=0,
        priests=0,
        power=Power(0, 0, 0),
        favors=("FAV1",),
    )
    state = with_faction(state, faction, fs)
    state = replace(state, pending=(PendingDecision(faction=faction, kind="gain_favor", amount=1),))
    moves = legal_moves(state)
    favor_moves = {m.tile for m in moves if m.verb == "gain_favor"}
    assert "FAV1" not in favor_moves  # already held
    assert favor_moves == {t for t in state.favors_pool if t != "FAV1"}


def test_setup_dwellings_offers_exactly_the_valid_dwelling_spots() -> None:
    setup = load_setup(GAME_ID)
    state = start_setup(GameState.initial(setup))
    faction = active_faction(state)
    home_color = FACTIONS[faction].color
    expected = {
        hex_key
        for hex_key, hx in state.hexes.items()
        if hx.building is None and hx.color == home_color
    }
    moves = legal_moves(state)
    build_locs = {m.loc for m in moves if m.verb == "build"}
    assert build_locs == expected
    assert build_locs  # sanity: the setup board always has home-colored land


# --------------------------------------------------------------------------
# Step 2: containment + generative smoke, against real replayed games
# --------------------------------------------------------------------------

_COLOR_ALIASES = {"grey": "gray"}


def _norm_color(color: str | None) -> str | None:
    return _COLOR_ALIASES.get(color, color) if color else color


def _is_contained(legal: tuple[ParsedCommand, ...], real: ParsedCommand) -> bool:
    verb = real.verb
    if verb == "convert":
        return any(
            m.verb == "convert" and m.res1 == real.res1 and m.res2 == real.res2 for m in legal
        )
    if verb in ("dig", "burn"):
        return any(m.verb == verb and m.n1 == real.n1 for m in legal)
    if verb == "transform":
        real_color = _norm_color(real.color)
        if real_color is None:
            return any(m.verb == "transform" and m.loc == real.loc for m in legal)
        return any(
            m.verb == "transform" and m.loc == real.loc and _norm_color(m.color) == real_color
            for m in legal
        )
    if verb == "gain_town":
        return any(m.verb == "gain_town" and m.tile == real.tile for m in legal)
    if verb in ("leech", "decline"):
        return any(m.verb == verb for m in legal)
    if verb == "send":
        return any(m.verb == "send" and m.cult == real.cult for m in legal)
    if verb == "pass":
        return any(m.verb == "pass" and m.tile == real.tile for m in legal)
    if verb == "advance":
        return any(m.verb == "advance" and m.reason == real.reason for m in legal)
    if verb == "bridge":
        return any(m.verb == "bridge" and {m.loc, m.loc2} == {real.loc, real.loc2} for m in legal)
    if verb == "action":
        return any(m.verb == "action" and m.tile == real.tile for m in legal)
    if verb == "build":
        return any(m.verb == "build" and m.loc == real.loc for m in legal)
    if verb == "upgrade":
        return any(
            m.verb == "upgrade" and m.loc == real.loc and m.building == real.building for m in legal
        )
    if verb == "connect":
        return any(m.verb == "connect" and m.loc == real.loc for m in legal)
    if verb == "gain_favor":
        return any(m.verb == "gain_favor" and m.tile == real.tile for m in legal)
    return any(m.verb == verb for m in legal)


def _apply_row_commands_with_hook(state, faction, cmds, oracle_cult, on_decision):
    """``replay._apply_row_commands``, with a per-``Kind.DECISION``-command
    ``on_decision(pre_state, faction, cmd, post_state)`` hook -- everything
    else (turn-boundary detection, ``advance_turn`` timing) is identical.
    """
    prev_verb: str | None = None
    open_action = False
    any_main_track = False
    for cmd in cmds:
        is_main = cmd.verb in _MAIN_TRACK_VERBS
        if is_main:
            any_main_track = True
            if open_action and is_turn_boundary(cmd, state, faction, prev_verb):
                state = advance_turn(state)
                open_action = False
        pre = state
        state = apply(state, faction, cmd, oracle_cult=oracle_cult)
        if cmd.kind == Kind.DECISION:
            on_decision(pre, faction, cmd, state)
        if is_main:
            prev_verb = cmd.verb
            if open_action or not never_starts_action(cmd.verb):
                open_action = True
    if any_main_track:
        state = advance_turn(state)
    return state


def _walk_decisions(
    game_id: str, moves_df: pl.DataFrame, deltas_df: pl.DataFrame, on_decision
) -> None:
    """Replay ``game_id`` end to end (Task-12 contract, same machinery
    ``replay.replay_game`` uses), invoking ``on_decision`` for every
    ``Kind.DECISION`` command actually applied.
    """
    setup = load_setup(game_id)
    state = start_setup(GameState.initial(setup))
    game_moves = moves_df.filter(pl.col("game_id") == game_id).sort(["row", "seq"])
    delta_lookup = _delta_lookup(deltas_df, game_id)
    income_rows = [
        (r["row"], r["faction"], r["verb"])
        for r in game_moves.filter(
            pl.col("verb").is_in(_CULT_INCOME_VERBS | _OTHER_INCOME_VERBS)
        ).iter_rows(named=True)
    ]
    other_income_done: set[str] = set()
    cult_income_done: set[str] = set()
    for row, faction, cmds in _iter_rows(game_moves):
        delta = delta_lookup.get((row, faction))
        oracle_cult = delta["cult"] if delta is not None else None
        state = _apply_pending_drops(state, row, setup.dropped_at_row)
        apply_faction = faction
        if state.factions[faction].dropped and state.phase in (
            Phase.SETUP_DWELLINGS,
            Phase.SETUP_BONUS,
        ):
            apply_faction = active_faction(state)
        if state.factions[apply_faction].dropped:
            continue
        was_income = state.phase == Phase.INCOME
        state = _ensure_actions_phase_started(state, cmds)
        if was_income and state.phase != Phase.INCOME:
            other_income_done = set()
        state, cult_income_done = _ensure_cult_income_landed(
            state, apply_faction, cmds, cult_income_done, row, income_rows
        )
        apply_cmds = _dedupe_income_commands(cmds)
        state = _apply_row_commands_with_hook(
            state, apply_faction, apply_cmds, oracle_cult, on_decision
        )
        state, other_income_done, cult_income_done = _advance_after_row(
            state, apply_faction, cmds, other_income_done, cult_income_done
        )


# 5 games from the task-14 25-game regression set (``test_replay_corpus.py``'s
# ``REGRESSION_SET``): enough faction/option diversity for a meaningful
# containment sample without the full suite's runtime.
_CONTAINMENT_GAMES: tuple[str, ...] = (
    "4pLeague_S5_D3L2_G3",
    "4pLeague_S8_D3L3_G2",
    "4pLeague_S3_D3L3_G2",
    "4pLeague_S1_D3L4_G1",
    "4pLeague_S10_D3L3_G5",
)

_NONNEGATIVE_FIELDS: tuple[str, ...] = (
    "coins",
    "workers",
    "priests",
    "priest_pool",
    "shipping",
    "dig_level",
    "teleport_level",
    "keys",
    "spades_available",
    "bridges_built",
    "extra_actions",
)


def _assert_no_negative_resources(fs: FactionState) -> None:
    for field in _NONNEGATIVE_FIELDS:
        value = getattr(fs, field)
        assert value >= 0, f"{fs.name}.{field} went negative: {value}"
    p = fs.power
    assert p.bowl1 >= 0 and p.bowl2 >= 0 and p.bowl3 >= 0, f"{fs.name} power went negative: {p}"
    # VP is not checked -- it legitimately goes negative in real TM
    # (scoring.py's own tests exercise this).


@pytest.fixture(scope="module")
def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
    )


def test_legal_moves_contains_every_decision_across_five_replayed_games(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Containment (task brief, Step 2): for every ``Kind.DECISION`` ledger
    command actually applied while replaying 5 real games, the command
    (modulo this module's documented normalizations) must appear in
    ``legal_moves_for`` of its own pre-state. 10598/10598 decisions across
    the full 25-game regression set + 4 stress outliers were verified
    clean during development (task-15 report has the full table); this
    test keeps the 5-game slice required by the brief at normal-CI weight.
    """
    moves_df, deltas_df = frames
    total = 0
    misses: list[str] = []

    def on_decision(pre: GameState, faction: str, cmd: ParsedCommand, post: GameState) -> None:
        nonlocal total
        if cmd.kind != Kind.DECISION:
            return
        total += 1
        legal = legal_moves_for(pre, faction)
        if not _is_contained(legal, cmd):
            misses.append(f"{faction} {cmd.raw!r} (round={pre.round} phase={pre.phase.name})")

    for game_id in _CONTAINMENT_GAMES:
        _walk_decisions(game_id, moves_df, deltas_df, on_decision)

    assert total > 500, f"sanity: expected a substantial decision sample, got {total}"
    assert misses == [], (
        f"{len(misses)}/{total} DECISION rows not in legal_moves_for: {misses[:20]}"
    )


_SMOKE_STATES_PER_GAME = 44  # 5 games * 44 = 220 >= 200 (task brief floor)


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """``k`` deterministic, evenly-spaced indices into ``range(n)`` (every
    index if ``k >= n``) -- used to pick a representative slice of one
    game's own decision timeline rather than favoring whichever end of it
    a naive prefix/suffix slice would land on.
    """
    if k >= n:
        return list(range(n))
    return [i * n // k for i in range(k)]


def test_legal_moves_apply_cleanly_with_no_negative_resources(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Generative smoke test (task brief, Step 2): every command
    ``legal_moves_for`` produces for a sampled pre-state must ``apply()``
    without raising, and must never leave a resource negative.

    Sampled with a **per-game quota** (:data:`_SMOKE_STATES_PER_GAME`),
    not a single running counter capped by a global slice -- code review
    caught that an earlier revision's ``counter % 3 == 0`` running tally,
    truncated to the first 220 entries *after* accumulating across all 5
    games in game order, silently confined every sampled state to the
    first 1-2 games (each game has hundreds of decisions, so the 220-item
    cap was exhausted before games 3-5 ever contributed any). Evenly
    spacing a fixed quota across *each* game's own timeline guarantees
    every game contributes, deterministically.
    """
    moves_df, deltas_df = frames
    sampled: list[tuple[str, str, GameState]] = []  # (game_id, faction, pre_state)

    for game_id in _CONTAINMENT_GAMES:
        game_states: list[tuple[str, GameState]] = []

        def on_decision(
            pre: GameState,
            faction: str,
            cmd: ParsedCommand,
            post: GameState,
            _states: list[tuple[str, GameState]] = game_states,
        ) -> None:
            _states.append((faction, pre))

        _walk_decisions(game_id, moves_df, deltas_df, on_decision)
        indices = _evenly_spaced_indices(len(game_states), _SMOKE_STATES_PER_GAME)
        sampled.extend((game_id, *game_states[i]) for i in indices)

    contributing_games = {game_id for game_id, _, _ in sampled}
    assert contributing_games == set(_CONTAINMENT_GAMES), (
        f"expected all {len(_CONTAINMENT_GAMES)} games to contribute smoke states, "
        f"got only {sorted(contributing_games)}"
    )
    assert len(sampled) >= 200, f"sanity: expected >=200 sampled states, got {len(sampled)}"

    total_moves = 0
    for _game_id, faction, state in sampled:
        for move in legal_moves_for(state, faction):
            total_moves += 1
            try:
                post = apply(state, faction, move)
            except EngineError as exc:  # pragma: no cover -- failure path
                pytest.fail(f"generated move {move.raw!r} for {faction} failed to apply: {exc}")
            _assert_no_negative_resources(post.factions[faction])

    assert total_moves > 1000, f"sanity: expected a substantial move sample, got {total_moves}"


# --------------------------------------------------------------------------
# Slow: the broader 29-game containment sweep from task-15 development
# (task-15-report.md's table), pinned as a real test rather than left as a
# dev-time-only script -- mirrors test_replay_corpus.py's own
# REGRESSION_SET/STRESS_OUTLIERS split and its `pytest.mark.slow` pattern,
# so a future legal.py change that regresses the broader 100% containment
# claim fails loudly instead of silently drifting. Not run by default
# (pyproject.toml's `addopts = "-m 'not slow'"`); run explicitly with
# `pytest tests/test_legal.py -m slow`. Duplicated locally rather than
# imported from test_replay_corpus.py -- no test file in this suite
# imports from another (each pins its own game-id tuples independently).
# --------------------------------------------------------------------------

_REGRESSION_SET: tuple[str, ...] = (
    "4pLeague_S5_D3L2_G3",
    "4pLeague_S8_D3L3_G2",
    "4pLeague_S3_D3L3_G2",
    "4pLeague_S1_D3L4_G1",
    "4pLeague_S10_D3L3_G5",
    "4pLeague_S24_D2L1_G3",
    "4pLeague_S29_D3L1_G4",
    "4pLeague_S36_D3L2_G2",
    "4pLeague_S17_D3L3_G3",
    "4pLeague_S11_D3L1_G6",
    "4pLeague_S40_D3L1_G6",
    "4pLeague_S12_D3L3_G4",
    "4pLeague_S13_D1L1_G1",
    "4pLeague_S14_D1L1_G1",
    "4pLeague_S15_D1L1_G1",
    "4pLeague_S16_D1L1_G1",
    "4pLeague_S18_D1L1_G1",
    "4pLeague_S19_D1L1_G1",
    "4pLeague_S20_D1L1_G1",
    "4pLeague_S21_D1L1_G1",
    "4pLeague_S22_D1L1_G1",
    "4pLeague_S23_D1L1_G1",
    "4pLeague_S25_D1L1_G1",
    "4pLeague_S26_D1L1_G1",
    "4pLeague_S27_D1L1_G1",
)
_STRESS_OUTLIERS: tuple[str, ...] = (
    "4pLeague_S34_D3L1_G2",
    "4pLeague_S40_D1L1_G2",
    "4pLeague_S72_D2L2_G4",
    "4pLeague_S62_D2L1_G4",
)


@pytest.mark.slow
def test_legal_moves_containment_holds_across_the_full_regression_sweep(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """The 29-game sweep (task-14's 25-game regression set + 4 stress
    outliers) verified 100% (10598/10598) clean during task-15
    development (task-15-report.md) -- pinned here so it stays true.
    Slow (~29 games' worth of replay + per-decision enumeration); excluded
    from normal ``pytest``/``pytest -q`` runs by ``pyproject.toml``'s
    default ``-m "not slow"``.
    """
    moves_df, deltas_df = frames
    total = 0
    misses: list[str] = []

    def on_decision(pre: GameState, faction: str, cmd: ParsedCommand, post: GameState) -> None:
        nonlocal total
        if cmd.kind != Kind.DECISION:
            return
        total += 1
        legal = legal_moves_for(pre, faction)
        if not _is_contained(legal, cmd):
            misses.append(f"{faction} {cmd.raw!r} (round={pre.round} phase={pre.phase.name})")

    for game_id in _REGRESSION_SET + _STRESS_OUTLIERS:
        _walk_decisions(game_id, moves_df, deltas_df, on_decision)

    assert total > 9000, f"sanity: expected a substantial decision sample, got {total}"
    assert misses == [], (
        f"{len(misses)}/{total} DECISION rows not in legal_moves_for: {misses[:20]}"
    )


# --------------------------------------------------------------------------
# legal_moves_all: the multi-faction exemption surface
# --------------------------------------------------------------------------


def test_legal_moves_all_matches_legal_moves_for_every_live_faction() -> None:
    state = _state()
    result = legal_moves_all(state)
    assert set(result) == {f for f, fs in state.factions.items() if not fs.dropped}
    for faction, moves in result.items():
        assert moves == legal_moves_for(state, faction)
    assert result[active_faction(state)] == legal_moves(state)

def test_giants_transform_moves_only_propose_home_color() -> None:
    """Regression (LLM-harness task 4): ``transform_moves`` must route each
    candidate target color through ``hooks_for(faction).spade_transform_target``
    -- Giants transform straight to home terrain (red) only (``map.pm``
    511-513/612-614), so offering ``transform X to black`` produced an
    apply()-time EngineError ("giants must transform to red, not black")
    on a move the generator itself had offered.
    """
    from bgai.arena.driver import new_game, next_actor, offered_moves, progress_moves
    from bgai.arena.setup_factory import fresh_setup
    from bgai.engine.tm.round_flow import advance_turn

    setup = fresh_setup(seed=0, factions=("giants", "nomads", "engineers", "mermaids"))
    state = new_game(setup)
    while state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        actor = next_actor(state)
        state = apply(state, actor, progress_moves(offered_moves(state, actor, turn_open=False))[0])
        state = advance_turn(state)
    factions = dict(state.factions)
    factions["giants"] = replace(factions["giants"], spades_available=2)
    state = replace(
        state,
        phase=Phase.ACTIONS,
        factions=factions,
        active_index=state.turn_order.index("giants"),
    )

    moves = legal_moves_for(state, "giants")
    targets = {m.color for m in moves if m.verb == "transform"}
    assert targets == {"red"}


def test_free_tf_build_moves_still_require_dwelling_cost() -> None:
    """Regression (LLM-harness task 5): ``build_moves``'s FREE_TF branch
    appended marker-priced builds without ``_can_afford_build`` -- ACTN's
    marker makes the implicit *transform* free, but ``handle_build`` still
    charges the dwelling's own cost (corpus: "nomads cannot afford 1 W"
    EngineError on a generator-offered build).
    """
    from bgai.arena.driver import new_game, next_actor, offered_moves, progress_moves
    from bgai.arena.setup_factory import fresh_setup
    from bgai.engine.tm.round_flow import advance_turn

    setup = fresh_setup(seed=0, factions=("nomads", "darklings", "engineers", "mermaids"))
    state = new_game(setup)
    while state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        actor = next_actor(state)
        state = apply(state, actor, progress_moves(offered_moves(state, actor, turn_open=False))[0])
        state = advance_turn(state)
    factions = dict(state.factions)
    factions["nomads"] = replace(factions["nomads"], workers=0, coins=0)
    state = replace(
        state,
        phase=Phase.ACTIONS,
        factions=factions,
        active_index=state.turn_order.index("nomads"),
        pending=(PendingDecision(faction="nomads", kind="free_tf"),),
    )

    for move in legal_moves_for(state, "nomads"):
        if move.verb == "build":
            apply(state, "nomads", move)  # must not raise EngineError
