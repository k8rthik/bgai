# LLM Agent Phase 1: MCP Harness + Tool Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a frozen Claude pilot Terra Mystica seats through an MCP server exposing fact-only engine tools, measured by win rate against baseline bots in a reproducible arena.

**Architecture:** A live-game **driver** (`arena/driver.py`) adapts the replay harness's Task-12 contract to games with no ledger: it auto-runs bookkeeping phases (income, cleanup, final scoring) and routes decisions to per-faction agents. Baseline bot agents plug into the driver directly; the LLM seat plugs in through an **MCP server** whose tools are gated into rungs (state/act → factual analysis → sandbox branching). A headless runner spawns `claude -p` per game. A knowledge pipeline mines the corpus into win-conditioned statistics for the rung-4 compendium.

**Tech Stack:** Python 3.12, uv, polars, `mcp` (FastMCP, stdio), `trueskill`, pytest, Claude Code CLI (`claude -p`).

**Spec:** `docs/superpowers/specs/2026-08-04-llm-tm-mcp-harness-design.md`. One documented deviation: the spec's "replay a corpus game's recorded moves through `play_move`" integration test is replaced by (a) the existing corpus-containment suite (already proves legal-move coverage), (b) a scripted-agent exact-state opening test, and (c) multi-seed full-random-game invariant tests. Rationale: ledger rows interleave bookkeeping rows (income, `cultist_leech_bonus`, `score_*`) that the live driver generates itself, so feeding raw ledger rows through `play_move` double-applies them; reconciling that inside the test would re-implement `replay.py` badly.

## Global Constraints

- Python ≥3.12, run everything via `uv run`; ruff line-length 100, target py312.
- Immutability: `GameState`/`FactionState` are frozen; never mutate — always `dataclasses.replace` / new dicts. The only mutable object allowed is the MCP session holder (one per process, documented).
- The facts/judgment boundary (spec): no MCP tool may rank moves, score desirability, or expose any evaluation signal. Deterministic arithmetic and mechanically guaranteed projections only.
- 4-player games only (engine is oracle-validated at 4p).
- New engine-package files must not import from `bgai.data` beyond `ledger_parser` (matches existing `tests/test_engine_import_isolation.py` policy — read it before adding imports).
- Tests: pytest; corpus-dependent or claude-spawning tests get `@pytest.mark.slow` (default addopts exclude slow). No network in tests.
- Fresh games use default `GameOptions()` (all flags False).
- Commit after every task (conventional commits; no attribution trailer — disabled globally).
- Baseline bots (`HeuristicAgent`) are arena opponents only; nothing may surface their logic through MCP tools.

---

### Task 1: Dependencies, Agent protocol, RandomAgent

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bgai/agents/base.py`
- Create: `src/bgai/agents/random_agent.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `Agent` protocol with `name: str` and `choose(state: GameState, faction: str, moves: tuple[ParsedCommand, ...]) -> ParsedCommand`; `RandomAgent(seed: int)`; `progress_moves(moves) -> tuple[ParsedCommand, ...]`; `AUX_VERBS: frozenset[str]`.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `[project] dependencies`, add `"mcp>=1.2"` and `"trueskill>=0.4.5"`. Run `uv sync` and verify `uv run python -c "import mcp, trueskill"` succeeds.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_agents.py
from bgai.agents.base import AUX_VERBS, progress_moves
from bgai.agents.random_agent import RandomAgent
from bgai.engine.tm.legal_shared import cmd


def test_progress_moves_filters_aux_verbs():
    moves = (cmd("convert", res1="PW", n1=1, res2="C", n2=1), cmd("build", loc="A1"),
             cmd("burn", n1=1), cmd("wait"))
    assert tuple(m.verb for m in progress_moves(moves)) == ("build",)


def test_progress_moves_falls_back_when_only_aux():
    moves = (cmd("convert", res1="PW", n1=1, res2="C", n2=1), cmd("wait"))
    assert progress_moves(moves) == moves


def test_random_agent_is_deterministic_per_seed():
    moves = tuple(cmd("build", loc=f"A{i}") for i in range(1, 9))
    picks_a = [RandomAgent(seed=7).choose(None, "nomads", moves) for _ in range(5)]
    picks_b = [RandomAgent(seed=7).choose(None, "nomads", moves) for _ in range(5)]
    assert picks_a == picks_b


def test_random_agent_never_picks_aux_when_progress_exists():
    moves = (cmd("convert", res1="PW", n1=1, res2="C", n2=1), cmd("pass", tile="BON1"))
    agent = RandomAgent(seed=1)
    assert all(agent.choose(None, "nomads", moves).verb == "pass" for _ in range(20))
```

- [ ] **Step 3: Run tests, verify failure** — `uv run pytest tests/test_agents.py -v` → ImportError.

- [ ] **Step 4: Implement**

```python
# src/bgai/agents/base.py
"""Agent protocol shared by baseline bots and the arena driver."""
from __future__ import annotations
from typing import Protocol
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState

AUX_VERBS: frozenset[str] = frozenset({"convert", "burn", "wait"})
"""Order-exempt resource moves, always legal — an unbiased random pick over
the full legal set would convert forever and never end its turn."""


class Agent(Protocol):
    name: str

    def choose(
        self, state: GameState, faction: str, moves: tuple[ParsedCommand, ...]
    ) -> ParsedCommand: ...


def progress_moves(moves: tuple[ParsedCommand, ...]) -> tuple[ParsedCommand, ...]:
    """Moves that advance the game (non-AUX); falls back to `moves` if empty."""
    filtered = tuple(m for m in moves if m.verb not in AUX_VERBS)
    return filtered or moves
```

```python
# src/bgai/agents/random_agent.py
"""Uniform-random baseline over progress moves. Floor of the arena ladder."""
from __future__ import annotations
import random
from bgai.data.ledger_parser import ParsedCommand
from bgai.agents.base import progress_moves
from bgai.engine.tm.state import GameState


class RandomAgent:
    def __init__(self, seed: int, name: str = "random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose(
        self, state: GameState | None, faction: str, moves: tuple[ParsedCommand, ...]
    ) -> ParsedCommand:
        if not moves:
            raise ValueError(f"{faction}: no legal moves offered")
        return self._rng.choice(progress_moves(moves))
```

`state` is unused by RandomAgent but part of the protocol (`GameState | None` in the annotation only here, so tests can pass None).

- [ ] **Step 5: Run tests, verify pass** — `uv run pytest tests/test_agents.py -v`
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: agent protocol + RandomAgent baseline"`

---

### Task 2: Fresh-setup factory and public legal-move constants

**Files:**
- Modify: `src/bgai/engine/tm/legal.py` (public alias only)
- Create: `src/bgai/arena/setup_factory.py`
- Test: `tests/test_setup_factory.py`

**Interfaces:**
- Consumes: `GameSetup`, `GameOptions` (`bgai.engine.tm.setup`), `SCORE_TILES`, `BONUS_TILES`, `TILE_OPTIONS` (`bgai.engine.tm.tiles`), `FACTIONS` (`bgai.engine.tm.factions_data`).
- Produces: `BLOCKING_PENDING_KINDS` public in `legal.py` (alias of `_BLOCKING_PENDING_KINDS`); `fresh_setup(seed: int, factions: tuple[str, str, str, str] | None = None) -> GameSetup`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_setup_factory.py
from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.legal import BLOCKING_PENDING_KINDS


def test_blocking_pending_kinds_is_public():
    assert "leech" in BLOCKING_PENDING_KINDS


def test_fresh_setup_shape():
    s = fresh_setup(seed=42)
    assert s.player_count == 4 and len(s.factions) == 4
    assert len(s.score_tiles) == 6
    assert len(s.bonus_tiles) == 7  # player_count + 3
    assert s.game_id == "arena_42"
    colors = [FACTIONS[f].color for f in s.factions]
    assert len(set(colors)) == 4  # one faction per color


def test_fresh_setup_deterministic_and_seed_sensitive():
    assert fresh_setup(seed=1) == fresh_setup(seed=1)
    assert fresh_setup(seed=1) != fresh_setup(seed=2)


def test_fresh_setup_respects_explicit_factions():
    factions = ("nomads", "darklings", "engineers", "mermaids")
    assert fresh_setup(seed=3, factions=factions).factions == factions


def test_fresh_setup_excludes_option_gated_tiles():
    for seed in range(30):
        s = fresh_setup(seed=seed)
        assert "BON10" not in s.bonus_tiles      # shipping-bonus option
        assert all(t is not None for t in s.score_tiles)
```

- [ ] **Step 2: Run tests, verify failure** — `uv run pytest tests/test_setup_factory.py -v`

- [ ] **Step 3: Implement**

In `legal.py`, directly below `_BLOCKING_PENDING_KINDS`, add:

```python
BLOCKING_PENDING_KINDS = _BLOCKING_PENDING_KINDS
"""Public alias for arena/MCP decision routing (Design decision 1)."""
```

```python
# src/bgai/arena/setup_factory.py
"""Seeded fresh-game GameSetup factory for arena/MCP play.

Simplifications vs. snellman lobby rules (documented, deliberate):
uniform 6-of-SCORE1..8 sampling (no "no-spade-tile in rounds 5/6" rule),
uniform 7-of-BON1..9 pool, default GameOptions (all flags off). SCORE9
(temple-scoring-tile) and BON10 (shipping-bonus) are option-gated per
tiles.TILE_OPTIONS and excluded under default options.
"""
from __future__ import annotations
import random
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import GameOptions, GameSetup
from bgai.engine.tm.tiles import BONUS_TILES, SCORE_TILES, TILE_OPTIONS


def fresh_setup(
    seed: int, factions: tuple[str, str, str, str] | None = None
) -> GameSetup:
    rng = random.Random(seed)
    if factions is None:
        by_color: dict[str, list[str]] = {}
        for name, data in sorted(FACTIONS.items()):
            by_color.setdefault(data.color, []).append(name)
        colors = rng.sample(sorted(by_color), 4)
        factions = tuple(rng.choice(by_color[c]) for c in colors)
    score_pool = sorted(t for t in SCORE_TILES if TILE_OPTIONS.get(t) is None)
    bonus_pool = sorted(t for t in BONUS_TILES if TILE_OPTIONS.get(t) is None)
    score_ids = rng.sample(score_pool, 6)
    bonus_ids = tuple(sorted(rng.sample(bonus_pool, 7)))
    return GameSetup(
        game_id=f"arena_{seed}",
        options=GameOptions(),
        factions=factions,
        score_tiles=tuple(SCORE_TILES[t] for t in score_ids),
        bonus_tiles=bonus_ids,
        player_count=4,
    )
```

Check `GameSetup`'s dataclass field order/names against `setup.py` before writing; `dropped_at_row` defaults to `{}` and is correct for fresh games. If a distinct-color 4-sample is impossible (it isn't — 7 colors exist), `rng.sample` raising is acceptable.

- [ ] **Step 4: Run tests, verify pass**; also run `uv run pytest tests/test_legal.py -q` to confirm the alias broke nothing.
- [ ] **Step 5: Commit** — `git commit -am "feat: seeded fresh-game setup factory + public blocking-pending kinds"`

---

### Task 3: Driver — bookkeeping auto-advance and decision routing

**Files:**
- Create: `src/bgai/arena/driver.py`
- Test: `tests/test_driver.py`

**Interfaces:**
- Consumes: `apply`, `EngineError` (`apply.py`); `start_setup`, `begin_actions`, `end_of_round`, `advance_turn` (`round_flow.py`); `final_scoring` (`scoring.py`); `legal_moves_for`, `BLOCKING_PENDING_KINDS` (`legal.py`); `cmd` (`legal_shared.py`); `Kind` (`ledger_parser.py`); `fresh_setup` (Task 2).
- Produces: `new_game(setup) -> GameState`; `live_factions(state) -> tuple[str, ...]`; `advance_bookkeeping(state) -> GameState`; `next_actor(state) -> str | None`; `DriverError(Exception)`; `MAIN_TRACK_VERBS`, `ALWAYS_END_VERBS`, `CONTINUATION_MARKER_KINDS` constants. (Turn protocol and game loop come in Task 4 — same file.)

The driver is the live-play analogue of `replay.py`'s row grouping. Read `round_flow.py`'s "Task 12 contract" docstring (lines ~209–260) before implementing; the semantics below are that contract restated for a caller that *generates* rows instead of parsing them:

- `new_game` = `GameState.initial(setup)` then `start_setup(state)`.
- `Phase.INCOME`: apply `cmd("other_income_for_faction", kind=Kind.BOOKKEEPING)` once per live faction (seat order; order is provably irrelevant), then `begin_actions`.
- `Phase.CLEANUP`: only when no blocking pendings remain — apply `cmd("cult_income_for_faction", kind=Kind.BOOKKEEPING)` per live faction, then `end_of_round`; if that lands in `Phase.FINISHED`, apply `final_scoring` exactly once (it is not idempotent) and return.
- `next_actor`: the oldest blocking pending's faction (kinds in `BLOCKING_PENDING_KINDS`) wins; else `active_faction(state)` during `SETUP_DWELLINGS`/`SETUP_BONUS`/`ACTIONS` if not passed/dropped; else `None` (bookkeeping owes an advance).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_driver.py
import dataclasses
import pytest
from bgai.arena.driver import advance_bookkeeping, live_factions, new_game, next_actor
from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.state import GameState, PendingDecision, Phase


def test_new_game_starts_setup_dwellings_snake_order():
    state = new_game(fresh_setup(seed=5))
    assert state.phase is Phase.SETUP_DWELLINGS
    assert next_actor(state) == state.turn_order[0]


def test_next_actor_prefers_oldest_blocking_pending():
    state = new_game(fresh_setup(seed=5))
    offer = PendingDecision(faction=state.setup.factions[2], kind="leech",
                            amount=1, source=state.setup.factions[0])
    state = dataclasses.replace(state, pending=(offer,))
    assert next_actor(state) == state.setup.factions[2]


def test_next_actor_ignores_marker_pendings():
    state = new_game(fresh_setup(seed=5))
    marker = PendingDecision(faction=state.setup.factions[1], kind="free_d")
    state = dataclasses.replace(state, pending=(marker,))
    assert next_actor(state) == state.turn_order[0]


def test_advance_bookkeeping_grants_income_and_enters_actions():
    # Build a state parked at round-1 INCOME by fast-forwarding setup with
    # scripted legal moves (helper below), then check income lands.
    from tests.driver_helpers import fast_forward_setup
    state = fast_forward_setup(seed=5)
    assert state.phase is Phase.INCOME
    before = {f: state.factions[f].workers for f in live_factions(state)}
    state = advance_bookkeeping(state)
    assert state.phase is Phase.ACTIONS
    assert any(state.factions[f].workers > before[f] for f in live_factions(state))


def test_advance_bookkeeping_is_noop_during_actions():
    from tests.driver_helpers import fast_forward_setup
    state = advance_bookkeeping(fast_forward_setup(seed=5))
    assert advance_bookkeeping(state) is state
```

Also create `tests/driver_helpers.py` with `fast_forward_setup(seed) -> GameState`: loop — while `state.phase` in (SETUP_DWELLINGS, SETUP_BONUS): take `next_actor`, pick the *first* move of `progress_moves(legal_moves_for(state, actor))`, `apply`, then `advance_turn` (per Task-12 contract, one call per setup build/pass). Keep it ~15 lines; Task 4's real loop supersedes it for gameplay but this helper stays for phase-boundary tests.

- [ ] **Step 2: Run tests, verify failure** — `uv run pytest tests/test_driver.py -v`

- [ ] **Step 3: Implement**

```python
# src/bgai/arena/driver.py (Task-3 portion)
"""Live-game driver: the Task-12 contract for games with no ledger.

`replay.py` mirrors corpus rows; this module *generates* the bookkeeping
those rows record (income, cult income, end-of-round, final scoring) and
routes genuine decisions to agents. Engine hooks handle reactive effects
(e.g. Cultists' leech bonus fires inside handle_leech/handle_decline via
_resolve_cultist_watch), so the driver never emits those verbs.
"""
from __future__ import annotations
from bgai.data.ledger_parser import Kind
from bgai.engine.tm.apply import apply
from bgai.engine.tm.legal import BLOCKING_PENDING_KINDS
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.round_flow import begin_actions, end_of_round, start_setup
from bgai.engine.tm.scoring import final_scoring
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState, Phase, active_faction

MAIN_TRACK_VERBS = frozenset(
    {"build", "upgrade", "action", "advance", "pass", "send", "dig",
     "transform", "bridge", "connect"}
)
ALWAYS_END_VERBS = frozenset({"pass", "send", "advance"})
CONTINUATION_MARKER_KINDS = frozenset({"free_d", "free_tp", "free_tf", "bridge"})


class DriverError(Exception):
    """Driver-level failure (budget exhausted, agent returned illegal move)."""


def new_game(setup: GameSetup) -> GameState:
    return start_setup(GameState.initial(setup))


def live_factions(state: GameState) -> tuple[str, ...]:
    return tuple(f for f in state.setup.factions if not state.factions[f].dropped)


def _has_blocking_pending(state: GameState) -> bool:
    return any(p.kind in BLOCKING_PENDING_KINDS for p in state.pending)


def next_actor(state: GameState) -> str | None:
    for p in state.pending:
        if p.kind in BLOCKING_PENDING_KINDS:
            return p.faction
    if state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS, Phase.ACTIONS):
        actor = active_faction(state)
        fs = state.factions[actor]
        if not fs.passed and not fs.dropped:
            return actor
    return None


def advance_bookkeeping(state: GameState) -> GameState:
    """Run INCOME/CLEANUP grants + phase transitions; stop at any decision
    point, at FINISHED (after one-shot final_scoring), or when nothing is
    owed. Returns `state` unchanged (identity) when there is nothing to do.
    """
    while True:
        if state.phase is Phase.INCOME:
            for faction in live_factions(state):
                state = apply(
                    state, faction,
                    cmd("other_income_for_faction", kind=Kind.BOOKKEEPING),
                )
            state = begin_actions(state)
            continue
        if state.phase is Phase.CLEANUP and not _has_blocking_pending(state):
            for faction in live_factions(state):
                state = apply(
                    state, faction,
                    cmd("cult_income_for_faction", kind=Kind.BOOKKEEPING),
                )
            state = end_of_round(state)
            if state.phase is Phase.FINISHED:
                return final_scoring(state)
            continue
        return state
```

Verify against the actual `handle_income_row` dispatch in `round_flow.py`: it keys on `cmd.verb` — if it needs any other field populated, mirror what `replay.py`'s `_parsed_command_from_row` builds for these verbs. If `cmd(...)`'s `raw` default doesn't satisfy `EngineError` formatting, pass `raw` explicitly. A cult-income SPADE grant can force a `transform` during CLEANUP (`legal.py` Design decision 4) — that surfaces as `spades_available > 0`; handle it in Task 4's loop (bot must transform before `end_of_round`), and in this task guard `advance_bookkeeping`'s CLEANUP branch with `all(state.factions[f].spades_available == 0 for f in live_factions(state))` as well.

- [ ] **Step 4: Run tests, verify pass** — `uv run pytest tests/test_driver.py -v`
- [ ] **Step 5: Commit** — `git commit -am "feat: live-game driver bookkeeping + decision routing"`

---

### Task 4: Driver — turn protocol, game loop, full random games

**Files:**
- Modify: `src/bgai/arena/driver.py`
- Test: `tests/test_driver_loop.py`

**Interfaces:**
- Consumes: everything from Task 3; `Agent`/`progress_moves` (Task 1); `legal_moves_for` (`legal.py`); `advance_turn`, `is_turn_boundary` docstring semantics (`round_flow.py`).
- Produces: `DONE = cmd("done")` sentinel; `offered_moves(state, faction, turn_open: bool) -> tuple[ParsedCommand, ...]`; `play_until_decision(state, bots: Mapping[str, Agent], external: frozenset[str], events: list[str], max_commands: int = 10000) -> GameState`; `GameResult` frozen dataclass (`game_id, seed_note, vp: Mapping[str, int], winners: tuple[str, ...], commands_applied: int`); `run_game(setup, agents: Mapping[str, Agent], max_commands=10000) -> GameResult`. `events` receives human-readable strings (uses each cmd's fields; Task 7 upgrades rendering).

**Turn protocol** (the load-bearing design; encode exactly):

A full action for faction F, driven one command at a time:
1. `moves = legal_moves_for(state, F)`; if a main-track verb has already been applied this action AND F has no mandatory affordance, append `DONE`.
2. Mandatory affordance = `fs.spades_available > 0` or F holds a pending with kind in `CONTINUATION_MARKER_KINDS` (markers are declinable — `marker_decline_moves` are already in the legal set — but the turn cannot end while one is queued).
3. Apply the chosen command via `apply(state, F, move)` unless it is `DONE`.
4. The action ends (driver calls `advance_turn`) when: the agent chose `DONE`; or the applied verb ∈ `ALWAYS_END_VERBS`; or the applied verb ∈ {`build`, `upgrade`, `connect`, `bridge`, `action`} and no mandatory affordance remains. After `transform`/`dig` with no affordance the action stays open (optional build continuation per `is_turn_boundary`'s `_CONTINUATION_PREV_VERBS`) — the agent must play a continuation or `DONE`.
5. `advance_turn` itself honors Chaos Magicians' `extra_actions` (same faction goes again — the outer loop re-routes to them naturally).
6. Blocking-pending answers (leech/decline/gain_favor/gain_town/cult_choice/convert_w_to_p) are applied WITHOUT `advance_turn` and outside any open action.

Outer loop (`play_until_decision`): repeat — if FINISHED return; `actor = next_actor(state)`; if None → `advance_bookkeeping` (if it returns identity AND no actor exists, raise `DriverError` — stuck state, a bug); if actor ∈ external → return (caller owns that seat); if actor owes a blocking pending → single answer step with `moves = legal_moves_for(state, actor)` restricted to that pending's answers plus AUX (restrict by generating: answers = those moves not in the would-be set without the pending is overkill — instead just pass the full `legal_moves_for` output through the agent; RandomAgent's progress filter and the engine's own gating make any legal choice acceptable, including jumping straight to a main-track move if actor is also active); else run one full-action step as above. Decrement a shared command budget on every `apply`; raise `DriverError` at 0.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_driver_loop.py
import pytest
from bgai.agents.random_agent import RandomAgent
from bgai.arena.driver import GameResult, run_game
from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.state import Phase


def _agents(setup, base_seed):
    return {f: RandomAgent(seed=base_seed + i, name=f"random{i}")
            for i, f in enumerate(setup.factions)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_random_game_finishes(seed):
    setup = fresh_setup(seed=seed)
    result = run_game(setup, _agents(setup, seed))
    assert isinstance(result, GameResult)
    assert set(result.vp) == set(setup.factions)
    assert all(v >= 0 for v in result.vp.values())
    assert result.winners and set(result.winners) <= set(setup.factions)
    assert max(result.vp.values()) == result.vp[result.winners[0]]


def test_full_random_game_deterministic():
    setup = fresh_setup(seed=9)
    r1 = run_game(setup, _agents(setup, 9))
    r2 = run_game(setup, _agents(setup, 9))
    assert r1 == r2


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(20))
def test_random_game_sweep(seed):
    setup = fresh_setup(seed=100 + seed)
    result = run_game(setup, _agents(setup, seed))
    assert sum(result.vp.values()) > 4 * 20  # everyone started at 20 VP
```

Plus a scripted exact-state test: a `ScriptedAgent` (in `tests/driver_helpers.py`) that pops moves from a list, selecting each by `(verb, loc, tile)` match against the offered set; script the full SETUP_DWELLINGS + SETUP_BONUS phases for `fresh_setup(seed=5, factions=("nomads","darklings","engineers","mermaids"))` — choose specific home hexes off the rendered legal sets (write the test after printing them once; hard-code the chosen hexes and assert the resulting `state.factions[...].buildings["D"]` contents and round-1 `Phase.ACTIONS` entry exactly).

- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement** per the protocol above. Keep `driver.py` under 400 lines; extract `_bot_full_action(state, agent, faction, events, budget) -> GameState` and `_bot_answer_pending(...)` helpers. On an agent returning a move not in the offered tuple (field-wise equality), raise `DriverError` naming agent, faction, and move. Winners: all factions tied at max VP (snellman shares wins).
- [ ] **Step 4: Run** `uv run pytest tests/test_driver_loop.py -v` then the slow sweep once: `uv run pytest tests/test_driver_loop.py -m slow -v`. Expect real engine friction here (this is the first non-ledger consumer); debug with the corpus-validated invariant in mind: any `EngineError` on a move `legal_moves_for` offered is an engine/driver contract bug — investigate against `round_flow` docstrings, don't paper over.
- [ ] **Step 5: Run full suite** — `uv run pytest -q` (fast set) to confirm no regression.
- [ ] **Step 6: Commit** — `git commit -am "feat: driver turn protocol + full random-game loop"`

---

### Task 5: HeuristicAgent

**Files:**
- Create: `src/bgai/agents/heuristic.py`
- Test: `tests/test_heuristic_agent.py`

**Interfaces:**
- Consumes: `Agent` protocol, `progress_moves`; `GameState` fields.
- Produces: `HeuristicAgent(seed: int, name: str = "heuristic")` satisfying `Agent`.

A deliberately simple scripted opponent — a yardstick above random, NOT a strategy oracle (never exposed via MCP). Policy, in priority order over offered moves (ties broken by seeded rng):
1. Answer any leech offer: accept if `amount <= 2` else decline (rough VP-cost rule of thumb).
2. `gain_favor`: prefer FAV11 > FAV10 > highest-numbered remaining; `gain_town`: prefer TW3 > TW1 > first offered.
3. During SETUP_DWELLINGS: the build whose hex has the most same-color neighbors is approximated by preferring central rows D/E/F — pick the first offered build with `loc[0] in "DEF"`, else first build.
4. During ACTIONS: first match wins — `upgrade` to SH/SA if offered; `upgrade` to TP/TE; `build`; `send` (priest to cult); `action` (any power/special action); `dig`/`transform`+`build` continuation; `pass` (prefer the bonus tile with the highest printed coin count via `BONUS_TILES[t].income.get("C", 0)`).
5. `DONE` if offered and nothing above matched; else random progress move.

- [ ] **Step 1: Write failing tests** — unit-test the priority function directly on hand-built move tuples (no full game needed): leech accept/decline threshold; SH-upgrade preferred over build; pass-tile coin preference. Plus one integration test: `run_game(fresh_setup(seed=11), {**3 random seats, 1 heuristic})` finishes, and a `@pytest.mark.slow` 20-game sweep asserting HeuristicAgent's mean VP exceeds the random seats' mean (it should, comfortably; if not, the policy or driver has a bug worth finding).
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement** (~120 lines; pure function `_pick(state, faction, moves, rng)` + thin class).
- [ ] **Step 4: Run tests incl. slow sweep, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: heuristic baseline agent"`

---

### Task 6: Arena match runner, rotation, reports, CLI

**Files:**
- Create: `src/bgai/arena/match.py`
- Create: `src/bgai/arena/report.py`
- Create: `src/bgai/arena/__main__.py`
- Test: `tests/test_arena.py`

**Interfaces:**
- Consumes: `run_game`, `GameResult`, `fresh_setup`, `RandomAgent`, `HeuristicAgent`.
- Produces:
  - `match.py`: `MatchSpec` frozen dataclass (`seed: int, agent_names: tuple[str, str, str, str]` — seat i gets agent i); `build_agent(name: str, seed: int) -> Agent` (registry: `"random"`, `"heuristic"`; raises `ValueError` on unknown); `run_match(spec) -> MatchOutcome` (frozen: `spec`, `result: GameResult`, `agent_by_faction: Mapping[str, str]`); `rotation(agent_names: tuple[str, ...], games: int, base_seed: int) -> tuple[MatchSpec, ...]` — game g uses seed `base_seed + g` and seat assignment rotated by `g % 4`, so each agent name occupies each seat equally over multiples of 4.
  - `report.py`: `summarize(outcomes: Sequence[MatchOutcome]) -> ArenaReport` (frozen: per-agent-name `games, wins, win_rate, mean_vp, trueskill_mu, trueskill_sigma`) using `trueskill.rate` with one rating group per seat, `ranks=` from VP ordering (ties share rank); `to_json(report) -> str`; `format_table(report) -> str`.
  - `__main__.py`: `uv run python -m bgai.arena --agents random,random,random,heuristic --games 8 --seed 7 [--out report.json]` → prints table, optionally writes JSON.

- [ ] **Step 1: Write failing tests** — `rotation` seat-fairness (over 8 games each name sits each seat twice); `summarize` on hand-built `MatchOutcome`s (fabricate `GameResult`s directly — no games needed) checking win_rate/mean_vp arithmetic and that a dominant agent's trueskill_mu exceeds a dominated one's; CLI smoke via `run_module` with `--games 2` monkeypatching `run_match` to return canned outcomes.
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement.** trueskill: create one `trueskill.TrueSkill(draw_probability=0.02)` env in `report.py`; ratings keyed by agent *name* persist across the outcome sequence. Validate `--agents` has exactly 4 entries.
- [ ] **Step 4: Run tests; then a real smoke:** `uv run python -m bgai.arena --agents random,random,random,heuristic --games 4 --seed 1`. Confirm the table prints and heuristic leads.
- [ ] **Step 5: Commit** — `git commit -am "feat: arena match runner, rotation, trueskill reports, CLI"`

---

### Task 7: Rendering — commands, state, diffs

**Files:**
- Create: `src/bgai/mcp/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `ParsedCommand`, `parse_command` (`ledger_parser.py`); `GameState`/`FactionState`; `cult_string` (`state.py`); `BONUS_TILES`/`FAVOR_TILES`/`TOWN_TILES`/`SCORE_TILES` (`tiles.py`).
- Produces: `render_command(cmd: ParsedCommand) -> str` (ledger-grammar text that `parse_command` round-trips); `render_state(state: GameState, viewer: str | None = None) -> str`; `render_moves(moves, viewer_faction) -> str` (numbered list); `diff_factions(before: GameState, after: GameState) -> str` (per-faction resource/VP/power/cult deltas, new pendings — factual only).

`render_command` per verb, derived from `ledger_parser._RULES` patterns (read them first; exact grammar is the source of truth). Examples to encode as tests: `build A1`; `upgrade A1 to TP`; `transform A1 to gray`; `dig 2`; `send p to FIRE`; `action ACT5`; `pass BON3`; `leech 2 from nomads`; `decline 2 from nomads`; `convert 1PW to 1C`; `burn 2`; `advance ship`; `advance dig`; `bridge A1:B1`; `connect r1`; `+1FIRE` (cult_choice answer / gain_cult); `wait`; `done`.

`render_state` sections (token-lean, fixed order): header (`round/phase/turn_order/active/pends`); score tiles (all 6, current marked); per-faction block (VP, C/W/P, power bowls `x/y/z`, shipping/dig level, building counts + hex lists, favors, bonus + its income/action, towns, passed flag); cult tracks table with cult_10 occupants and priest-slot occupancy; board as one line per occupied hex (`E5 brown TP darklings`) plus a terraformed-but-empty list (hexes whose color ≠ `base_board()` color); pools (favors remaining, towns remaining, bonus-coin accumulation, power actions taken this round). Viewer arg only reorders (viewer's faction first) — no hidden information exists in TM.

- [ ] **Step 1: Write failing tests** — round-trip property: for each example string s above, `render_command(parse_command(s))` == canonical form and `parse_command(render_command(c))` reproduces `c`'s fields (compare verb + all non-`raw`, non-`kind` fields); a broader round-trip over every move `legal_moves_for` yields in 3 driver-generated positions (setup start, round-1 actions, a leech-pending state built in the test); `render_state` smoke on a fresh game asserting the four faction names, "SETUP_DWELLINGS", and a known home hex line appear.
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement** (~250 lines). Unknown verb in `render_command` → raise `ValueError` (never emit text that won't parse back).
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: command/state rendering with parser round-trip"`

---

### Task 8: MCP server — session config + rung 1 (state & act)

**Files:**
- Create: `src/bgai/mcp/config.py`
- Create: `src/bgai/mcp/session.py`
- Create: `src/bgai/mcp/server.py`
- Create: `.mcp.json` (repo root)
- Test: `tests/test_mcp_session.py`

**Interfaces:**
- Consumes: driver + setup_factory + agents + render (Tasks 1–7).
- Produces:
  - `config.py`: `SessionConfig` frozen dataclass — `seed: int`, `llm_faction_index: int` (seat, 0–3; `-1` = ALL seats external, for interactive analysis), `opponents: str` (`"random" | "heuristic"`), `rungs: tuple[int, ...]` (subset of `(1, 2, 3)`; rung 1 always required), `result_path: str | None`, `max_commands: int = 10000`, `factions: tuple[str, str, str, str] | None`. `load_config() -> SessionConfig`: reads JSON at `$BGAI_TM_SESSION` if set (schema-validate: unknown keys rejected, types checked, fail fast with the offending key in the message), else interactive defaults (`seed=0, llm_faction_index=0, opponents="heuristic", rungs=(1, 2, 3), result_path=None`).
  - `session.py`: `class Session` — the one sanctioned mutable object. Fields: `config`, `state: GameState | None`, `events: list[str]`, `turn_open_verb: str | None`, `commands_used: int`, `branches: dict[str, GameState]` (rung 3, Task 10). Methods: `start() -> str` (fresh_setup + new_game + `_advance()` + render); `llm_faction -> str`; `external: frozenset[str]`; `_advance()` — `play_until_decision(state, bots, external, events)`; on FINISHED: append final VP table to events, write result JSON to `config.result_path` (`{game_id, seed, llm_faction, vp, winners, commands_used, rungs}`); `offered() -> tuple[ParsedCommand, ...]` (reuses driver `offered_moves` incl. DONE logic for the open turn); `submit(text: str) -> str` — parse (`parse_command`; `None` → error string listing `render_moves(offered())`), membership-check against `offered()` field-wise, apply via the driver turn-protocol step (updating `turn_open_verb`), then `_advance()`; every failure path returns a structured message (`ERROR: <reason>` + current legal moves), never raises out.
  - `server.py`: FastMCP wiring only —

```python
from mcp.server.fastmcp import FastMCP
from bgai.mcp.config import load_config
from bgai.mcp.session import Session

mcp = FastMCP("terra-mystica")
SESSION = Session(load_config())


@mcp.tool()
def new_game() -> str:
    """Start (or restart) the configured game; returns the opening state
    once it is your seat's turn."""
    return SESSION.start()


@mcp.tool()
def get_state() -> str:
    """Current full game state (factual render)."""
    ...  # delegate to session; ERROR string if no game started

# legal_moves() -> str, play_move(move: str) -> str similarly delegate.

if __name__ == "__main__":
    mcp.run()
```

  Rung gating: `server.py` registers rung-2/3 tools only if their rung ∈ `config.rungs` (plain `if` around `@mcp.tool()` registrations — build tools via `mcp.tool()(fn)` calls inside `_register(config)`).
  - `.mcp.json`: `{"mcpServers": {"tm": {"command": "uv", "args": ["run", "python", "-m", "bgai.mcp.server"]}}}` — makes "hey Claude, pilot this game" work in this repo interactively.

- [ ] **Step 1: Write failing tests** (all against `Session` directly — no MCP client needed):

```python
# tests/test_mcp_session.py — representative cases
def test_start_reaches_llm_decision(): ...      # seat 0: state phase SETUP_DWELLINGS, offered() non-empty
def test_submit_legal_move_advances(): ...      # submit first offered build; bots play; it's our turn again or new phase
def test_submit_illegal_move_returns_error():   # "build Z9" -> startswith "ERROR:", state unchanged, legal list included
def test_submit_unparseable_returns_error():    # "flarb" -> ERROR + legal list
def test_full_llm_random_game_via_session():    # drive session by always submitting render_command(offered()[0]); asserts FINISHED + result file written (tmp_path result_path)
def test_all_external_mode():                   # llm_faction_index=-1: submit moves for every faction in turn (session routes by next_actor)
def test_config_rejects_unknown_keys(): ...
def test_config_env_roundtrip(tmp_path, monkeypatch): ...
```

  The full-game session test may be slow-ish but stays in the fast set if it completes in a few seconds (random games are ~300–600 commands, pure Python — check; if >5 s, mark slow and add a 2-round fast variant).
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement.** `session.py` owns all logic; `server.py` stays <100 lines of wiring. In all-external mode `submit` requires a `faction:` prefix? No — route implicitly: the session knows `next_actor`; the submitted move is applied for that faction (document in the tool docstring).
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Manual smoke (interactive):** `uv run python -m bgai.mcp.server` starts and idles on stdio without traceback (Ctrl-C out). Optionally verify tool listing with `uv run python -c` snippet using `mcp` client over stdio if quick; otherwise the headless test in Task 11 covers it end-to-end.
- [ ] **Step 6: Commit** — `git commit -am "feat: MCP server rung 1 — session, state/act tools, repo .mcp.json"`

---

### Task 9: MCP rung 2 — factual analysis tools

**Files:**
- Create: `src/bgai/mcp/analysis.py`
- Modify: `src/bgai/mcp/server.py` (register rung-2 tools), `src/bgai/mcp/session.py` (delegates)
- Test: `tests/test_mcp_analysis.py`

**Interfaces:**
- Consumes: `apply`, `legal_moves_for`, `diff_factions` (Task 7), `compute_cult_scoring`, `compute_network_scoring`, `network_size` (`scoring.py`), current `ScoringTile` (`state.setup.score_tiles[state.round - 1]`), `tiles` data.
- Produces: `preview_move(state, faction, cmd) -> str` — apply on the immutable state (a branch by construction), return `diff_factions(before, after)` + any new pendings (leech offers with exact amounts and targets) + resulting phase/turn note; never commits, never scores desirability. `score_projection(state) -> str` — sections: (a) this round's scoring-tile cult income each faction would receive at cleanup from current cult positions (exact `floor(pos/req) * income` arithmetic); (b) end-game cult scoring if standings froze now (`compute_cult_scoring` output labeled "IF standings froze now"); (c) network sizes + `compute_network_scoring` under the same framing; (d) per-faction pass-VP arithmetic of held bonus/favor tiles and leftover-resource conversion at current holdings. All numbers are mechanical consequences of frozen state — the facts/judgment boundary in action; the wording in the output must say "if ... ended now", never "good/bad/strong".

- [ ] **Step 1: Write failing tests** — `preview_move` on a driver-built position: previewing a build shows the exact C/W delta of that faction's D cost and lists the leech offer it triggers (build adjacent to a bot's dwelling in a scripted position); previewing does not change the session state (submit-after-preview still legal). `score_projection` on a mid-game scripted state: cult-income arithmetic matches hand-computed `floor(pos/req)*income`; cult 8/4/2 split matches `compute_cult_scoring`. Grep-test the boundary: output contains no words from `("recommend", "best", "should", "strong", "weak", "good", "bad")`.
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement**; register in `server.py` under `2 in config.rungs` with docstrings stating they are factual previews ("no evaluation — arithmetic only").
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: MCP rung 2 factual analysis tools"`

---

### Task 10: MCP rung 3 — sandbox branching

**Files:**
- Create: `src/bgai/mcp/branching.py`
- Modify: `src/bgai/mcp/server.py`, `src/bgai/mcp/session.py`
- Test: `tests/test_mcp_branching.py`

**Interfaces:**
- Produces (tools, gated on `3 in config.rungs`): `branch(label: str) -> str` — fork the *live* state under a label (max 8 concurrent; error listing labels when full; duplicate label → error); `branch_play(label: str, move: str) -> str` — apply a move in the branch **for whichever faction `next_actor` says owes a decision** (the LLM plays ALL seats in branches — that's the point: it models opponents itself), with driver bookkeeping auto-advancing between decisions and the same turn-protocol/DONE handling as the live seat (each branch tracks its own `turn_open_verb`); `branch_state(label) -> str`, `branch_moves(label) -> str`, `discard_branch(label) -> str`, `list_branches() -> str`. Branch games run `final_scoring` on completion like the live driver (a fully played-out branch reports factual final VP — arithmetic, not evaluation). Branches never touch the live game or the bots; `play_move` on the live game leaves branches untouched (they represent hypotheticals from an older position — `branch_state` output must carry a header line `branched from: round R, phase P` so stale branches are self-describing).
- Consumes: `Session.branches: dict[str, BranchState]` where `BranchState` is a small frozen dataclass (`state, turn_open_verb, created_at_round, created_at_phase`) replaced on every play (immutability).

- [ ] **Step 1: Write failing tests** — branch + play a full bot-free round in the branch while live state is unchanged; branch cap and duplicate-label errors; branch played to FINISHED reports VP table; discard removes; `branch_play` routes multi-faction decisions correctly (after our build in-branch, the leech-owed opponent is the next actor and `branch_moves` shows leech/decline for THEM).
- [ ] **Step 2: Run, verify failure**
- [ ] **Step 3: Implement** (`branching.py` pure functions over `BranchState`; session holds the dict).
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: MCP rung 3 sandbox branching"`

---

### Task 11: Headless LLM arena runner + pilot prompt

**Files:**
- Create: `prompts/tm_pilot.md`
- Create: `scripts/arena_llm.py`
- Test: `tests/test_arena_llm.py` (config/aggregation logic only; claude-spawning test is `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `SessionConfig` JSON schema (Task 8), `ArenaReport` machinery (Task 6).
- Produces: `scripts/arena_llm.py` CLI —
  `uv run python scripts/arena_llm.py --games 4 --seed 100 --opponents heuristic --rungs 1,2,3 --out runs/llm_r123/ [--model sonnet] [--claude-bin claude] [--dry-run]`
  Per game g: build `SessionConfig(seed=seed+g, llm_faction_index=g % 4, opponents=..., rungs=..., result_path=<out>/game_<g>/result.json)`; write it to `<out>/game_<g>/session.json`; write `<out>/game_<g>/mcp.json` containing `{"mcpServers": {"tm": {"command": "uv", "args": ["run", "python", "-m", "bgai.mcp.server"], "env": {"BGAI_TM_SESSION": "<abs path to session.json>"}}}}`; spawn `claude -p "$(cat prompts/tm_pilot.md)" --mcp-config <mcp.json> --allowedTools "mcp__tm__*" --max-turns 600 --output-format json` with `subprocess.run(timeout=3600)`, capturing stdout to `<out>/game_<g>/claude.json` (its `total_cost_usd` and `duration_ms` fields feed the report; tolerate their absence — older CLIs — by recording null) and stderr to `<out>/game_<g>/transcript.txt`; afterwards read `result.json` — missing/invalid → the game is recorded `{"status": "invalid"}` and retried up to 2 times with the same seed (spec's error-handling rule) before being excluded from stats. Aggregate valid results: LLM win rate, mean VP, mean VP margin vs. best opponent, commands used, wall-clock and $ per game (from `claude.json`) → printed table + `<out>/summary.json`. `--dry-run` writes all configs and prints the commands without spawning (this is what the fast test exercises).
- `prompts/tm_pilot.md` — the pilot system prompt: you are playing faction X of a 4p Terra Mystica game via `tm` MCP tools; call `new_game` once, then loop `get_state`/`legal_moves`/`play_move`; explain the turn protocol (compound actions, `done`), leech etiquette (answer offers when asked), budget (~30 s of thinking per decision; prefer decisive play over exhaustive analysis); rung-conditional paragraphs for rung 2 (factual preview tools exist — use them for arithmetic, the judgment is yours) and rung 3 (branch tools for lookahead — you play the opponents in branches); end when the game reports FINISHED, then stop calling tools. No strategy content in the prompt itself at rungs 1–3 (that's rung 4's compendium, appended by the runner when configured — leave a `{{COMPENDIUM}}` placeholder line the runner substitutes or strips).
- [ ] **Step 1: Write failing tests** — dry-run produces N config dirs with valid JSON (schema round-trips through `load_config` via monkeypatched env); aggregation over hand-written result.json fixtures (2 valid + 1 invalid) computes win rate over valid only and lists the invalid game; retry logic invoked exactly twice on persistent failure (monkeypatched spawn).
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Implement** (argparse; ~250 lines; pure helpers separated from subprocess calls for testability).
- [ ] **Step 4: Run fast tests; then one real headless game** — `uv run python scripts/arena_llm.py --games 1 --seed 500 --rungs 1 --out /tmp/llm_smoke` (requires `claude` on PATH; report cost/duration in the commit message body). Add that invocation as `@pytest.mark.slow` `test_one_real_headless_game`, skipped when `shutil.which("claude") is None`.
- [ ] **Step 5: Commit** — `git commit -am "feat: headless claude-code arena runner + pilot prompt"`

---

### Task 12: Knowledge pipeline — corpus stats mining + compendium procedure

**Files:**
- Create: `src/bgai/knowledge/__init__.py`, `src/bgai/knowledge/stats.py`, `src/bgai/knowledge/__main__.py`
- Create: `docs/knowledge/tm-compendium/GENERATE.md`
- Test: `tests/test_knowledge_stats.py`

**Interfaces:**
- Consumes: `data/datasets/moves.parquet`, `data/datasets/deltas.parquet` (inspect schemas first: `uv run python -c "import polars as pl; print(pl.read_parquet('data/datasets/moves.parquet').head(3)); print(pl.read_parquet('data/datasets/deltas.parquet').head(3))"` — column names below are to be reconciled against reality in step 3, not guessed).
- Produces: `stats.py` pure polars functions, each `(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame`:
  - `final_standings` — per (game_id, faction): final VP (last delta row per faction), rank, won flag (ties share rank 1).
  - `round_index` — per (game_id, row): round number, derived by counting `other_income_for_faction` batches (round r begins at the r-th batch; rows before the first batch are setup).
  - `faction_win_rates` — games, wins, win rate, mean final VP per faction.
  - `opening_patterns` — per faction, win-conditioned: mean dwellings built during setup+round 1, first-upgrade timing (round of first TP/TE/SH/SA), round-1 pass position distribution.
  - `timing_curves` — per faction × round, win-conditioned means: builds, upgrades, priests sent (`send` rows), power actions taken (`action ACTx` rows), towns founded (`gain_town` rows).
  - `tile_pick_rates` — bonus-tile pick rate by round × faction × won; favor-tile pick rate by faction × won.
  - `write_compendium_tables(out_dir: Path)` — runs all of the above on the real corpus and writes one markdown file per faction (`darklings.md`, ...) of win-vs-loss tables plus `_overview.md` (cross-faction win rates, tile tables). Numbers only — no prose, no advice.
  - `__main__.py`: `uv run python -m bgai.knowledge --out docs/knowledge/tm-compendium/tables/`.
- `GENERATE.md`: the operational procedure for the prose playbooks — the exact prompt to run in Claude Code ("From these win-conditioned statistics tables alone, write a ≤600-word playbook of principles with effect sizes for faction X; cite the numbers; no move prescriptions, no position references"), the review checklist (every claim traceable to a table; no invented facts), and where outputs land (`docs/knowledge/tm-compendium/<faction>-playbook.md`). The runner (Task 11) substitutes the concatenated playbooks for `{{COMPENDIUM}}` when `--rungs` includes 4.
- [ ] **Step 1: Write failing tests** — build a tiny in-memory fixture frame (2 fake games, 2 factions, hand-written rows incl. income batches and final score rows); assert `final_standings` ranks and win flags, `round_index` boundaries, one `timing_curves` cell against hand-computed values. Real-corpus run is `@pytest.mark.slow`: `write_compendium_tables` to `tmp_path` completes and `darklings.md` contains a win-rate line.
- [ ] **Step 2: Run tests, verify failure**
- [ ] **Step 3: Reconcile column names** against the printed schemas, adjust fixture + code, implement.
- [ ] **Step 4: Run tests (fast + slow), verify pass; run the real CLI once** and commit the generated `tables/` output (it's documentation-grade data, deterministic from the corpus).
- [ ] **Step 5: Update README** — add an "LLM agent track" section: MCP server + `.mcp.json` interactive piloting, headless arena runner, rung flags, knowledge pipeline. Keep it under 40 lines, matching the existing README voice.
- [ ] **Step 6: Commit** — `git commit -am "feat: corpus stats mining + compendium generation procedure; README LLM-track section"`

---

## Execution order & dependencies

1 → 2 → 3 → 4 → 5 → 6 (arena track complete) → 7 → 8 → 9 → 10 (MCP track) → 11 (needs 6 + 8) → 12 (independent of 7–11; can run any time after 4).

Task 4 is the risk concentrator (first non-ledger engine consumer); budget debugging time there, not in the MCP layer.

## Verification at the end

- `uv run pytest -q` green; `uv run pytest -m slow -q` green (corpus + sweeps).
- `uv run python -m bgai.arena --agents random,random,random,heuristic --games 8 --seed 1` → heuristic on top.
- One real headless rung-1 game via `scripts/arena_llm.py` producing a valid `result.json`.
- Interactive: open Claude Code in the repo, "pilot a game" → tools appear, a few moves play cleanly.
