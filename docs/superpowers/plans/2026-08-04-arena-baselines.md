# Phase 4: Arena + Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An arena that plays complete headless Terra Mystica games between pluggable agents on corpus-sampled setups with mirrored seat rotation, rates them with TrueSkill, and emits an HTML report — verified by the master-plan gate: greedy ≫ random over ≥200 games.

**Architecture:** A self-driving simulation loop (`arena/sim.py`) composes the existing replay-validated engine primitives (`legal_moves_for`, `apply`, `advance_turn`, `is_turn_boundary`, `begin_actions`, `end_of_round`, `final_scoring`) into a decision scheduler: drain blocking pending decisions first, drive INCOME/CLEANUP windows with synthetic income commands, and run the active faction's turn protocol (one fresh main-track action + its continuations per turn). Agents implement one method — choose a `ParsedCommand` from an offered tuple. Setups are sampled from the crawled corpus via `load_setup` (drops cleared), never generated synthetically (master-plan decision 2026-08-04).

**Tech Stack:** Python 3.12, uv, pytest, polars (existing). New dependency: `trueskill` (PyPI 0.4.5, MIT — the standard Python implementation; battle-tested, pure Python).

## Global Constraints

- Run tests with `uv run pytest <path> -x -q` from repo root `/Users/keerthikmuruganandam/code/bgai`.
- IMMUTABILITY: all new state types are `@dataclass(frozen=True)`; rating updates return new dicts, never mutate.
- Files under ~400 lines; split by responsibility.
- `from __future__ import annotations` + full type annotations in every module.
- Engine modules are modified ONLY in Task 2 (two small public wrappers in `legal.py`); everything else is additive under `src/bgai/agents/` and `src/bgai/arena/`.
- Commit after every green test cycle, conventional commits (`feat:`/`test:`/`fix:`), no attribution footers.
- Existing engine API consumed (do not reimplement): `bgai.engine.tm.setup.load_setup(game_id, raw_dir=Path("data/raw/games")) -> GameSetup` (fields: `game_id, options, factions, score_tiles, bonus_tiles, player_count, dropped_at_row`); `bgai.engine.tm.state` (`GameState.initial(setup)`, `active_faction(state)`, `Phase`); `bgai.engine.tm.round_flow` (`start_setup`, `begin_actions`, `end_of_round`, `advance_turn`, `is_turn_boundary(cmd, state_before, faction, prev_verb)`); `bgai.engine.tm.apply.apply(state, faction, cmd)` (+ `EngineError`); `bgai.engine.tm.legal` (`legal_moves`, `legal_moves_for`); `bgai.engine.tm.scoring.final_scoring(state)` (requires `Phase.FINISHED`); `bgai.engine.tm.legal_shared.cmd(verb, **fields)`; `bgai.data.ledger_parser.ParsedCommand`.
- Simulation-mode invariants: never pass `oracle_cult` to `apply` (default `None` = fixed FIRE>WATER>EARTH>AIR tiebreak); call `final_scoring` exactly once at `Phase.FINISHED`.
- Data: `data/raw/games/*.json.gz` (3,564 files), `data/datasets/games_meta.parquet`. 10 `nofaction*` games raise `ValueError` in `load_setup`; 179 games have non-empty `dropped_at_row`. The arena excludes both classes.
- Slow tests follow the existing convention in `tests/test_replay_corpus.py` (check its marker/skip mechanism at execution time and mirror it).

## File Map

| File | Responsibility |
|---|---|
| `src/bgai/agents/__init__.py` | Re-export `Agent`, `RandomAgent`, `GreedyAgent` |
| `src/bgai/agents/base.py` | `Agent` protocol |
| `src/bgai/agents/random_agent.py` | `RandomAgent` |
| `src/bgai/agents/greedy.py` | `GreedyAgent` + `evaluate(state, faction)` heuristic |
| `src/bgai/arena/__init__.py` | empty package init |
| `src/bgai/arena/setups.py` | corpus setup sampler (clean 4p setups, drops cleared) |
| `src/bgai/arena/sim.py` | self-driving game loop: `run_game(setup, seats, rng)` → `GameResult` |
| `src/bgai/arena/rotation.py` | mirrored seat rotations |
| `src/bgai/arena/ratings.py` | TrueSkill wrapper (immutable updates), placement stats |
| `src/bgai/arena/series.py` | multi-table orchestration: `run_series(...)` → `SeriesResult` |
| `src/bgai/arena/report.py` | self-contained HTML report |
| `src/bgai/arena/run.py` | CLI: `python -m bgai.arena.run` |
| Modify: `src/bgai/engine/tm/legal.py` | add `pending_answer_moves`, `has_blocking_pending` (public wrappers) |

Tests mirror one-to-one: `tests/test_agents.py`, `tests/test_arena_setups.py`, `tests/test_arena_sim.py`, `tests/test_arena_rotation.py`, `tests/test_arena_ratings.py`, `tests/test_arena_series.py`, `tests/test_arena_report.py`, `tests/test_legal_pending_api.py`, plus the slow gate `tests/test_arena_verify.py`.

---

### Task 1: Agent protocol + RandomAgent (`agents/base.py`, `agents/random_agent.py`)

**Files:**
- Create: `src/bgai/agents/__init__.py`, `src/bgai/agents/base.py`, `src/bgai/agents/random_agent.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `Agent` protocol — `name: str` attribute; `choose(self, state: GameState, faction: str, offer: tuple[ParsedCommand, ...], rng: random.Random) -> ParsedCommand`. `RandomAgent(name: str = "random")`. The arena guarantees `offer` is non-empty; agents must return an element of `offer`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_agents.py"""
from __future__ import annotations

import random

from bgai.agents import RandomAgent
from bgai.engine.tm.legal_shared import cmd


def test_random_agent_picks_from_offer_deterministically() -> None:
    offer = (cmd("pass", tile="BON1"), cmd("pass", tile="BON2"), cmd("pass", tile="BON3"))
    agent = RandomAgent()
    picks_a = [agent.choose(None, "witches", offer, random.Random(7)) for _ in range(5)]
    picks_b = [agent.choose(None, "witches", offer, random.Random(7)) for _ in range(5)]
    assert picks_a == picks_b
    assert all(p in offer for p in picks_a)


def test_random_agent_has_name() -> None:
    assert RandomAgent().name == "random"
    assert RandomAgent(name="rnd2").name == "rnd2"
```

(`state` is unused by `RandomAgent`, so `None` is fine in the unit test; the protocol types it as `GameState`.)

- [ ] **Step 2: Run test, verify it fails** — `uv run pytest tests/test_agents.py -x -q`, expect `ModuleNotFoundError: bgai.agents`.

- [ ] **Step 3: Implement**

```python
"""src/bgai/agents/base.py"""
from __future__ import annotations

import random
from typing import Protocol

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState


class Agent(Protocol):
    """Anything that can play Terra Mystica in the arena.

    The arena calls ``choose`` once per decision — main actions, setup
    dwelling placements, leech answers, favor picks, and income-window
    spade transforms all arrive through this single method (the engine's
    pending-decision queue makes them all ordinary moves). ``offer`` is
    always non-empty and the return value must be one of its elements.
    ``rng`` is the arena's seeded generator: agents must draw randomness
    only from it so games are reproducible from (setup, seats, seed).
    """

    name: str

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand: ...
```

```python
"""src/bgai/agents/random_agent.py"""
from __future__ import annotations

import random

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState


class RandomAgent:
    """Uniform-random baseline: the floor of the rating scale."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        return offer[rng.randrange(len(offer))]
```

```python
"""src/bgai/agents/__init__.py"""
from __future__ import annotations

from bgai.agents.base import Agent
from bgai.agents.random_agent import RandomAgent

__all__ = ["Agent", "RandomAgent"]
```

(`GreedyAgent` is added to `__init__` in Task 6.)

- [ ] **Step 4: Run test, verify pass** — `uv run pytest tests/test_agents.py -x -q`.
- [ ] **Step 5: Commit** — `feat: Agent protocol and RandomAgent baseline`

---

### Task 2: Public pending-answer API in `legal.py`

**Files:**
- Modify: `src/bgai/engine/tm/legal.py` (append two functions at module bottom)
- Test: `tests/test_legal_pending_api.py`

**Interfaces:**
- Produces: `pending_answer_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]` — the faction's outstanding blocking-decision answers (empty when none). `has_blocking_pending(state: GameState, faction: str) -> bool`.
- Rationale: `sim.py` must (a) route non-active factions to their leech/cult-choice answers and (b) forbid ending a turn while the active faction owes a favor/town pick. The logic exists as `_own_pending_answers`/`_BLOCKING_PENDING_KINDS`; the arena must not import underscored names (the refactor queue already flags a private cross-module import as a smell — don't add another).

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_legal_pending_api.py"""
from __future__ import annotations

from bgai.engine.tm.legal import has_blocking_pending, pending_answer_moves
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState


def test_no_pending_at_setup_start() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    for faction in setup.factions:
        assert pending_answer_moves(state, faction) == ()
        assert not has_blocking_pending(state, faction)
```

- [ ] **Step 2: Run test, verify it fails** — `ImportError: cannot import name 'pending_answer_moves'`.

- [ ] **Step 3: Implement** (append to `legal.py`)

```python
def pending_answer_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """Public wrapper over :func:`_own_pending_answers` for simulation
    drivers (``bgai.arena.sim``): every answer ``faction`` can give right
    now to its own outstanding blocking pending decisions, and nothing
    else — no baseline converts, no main-track moves. Empty when the
    faction owes no blocking answer.
    """
    return _own_pending_answers(state, faction, set())


def has_blocking_pending(state: GameState, faction: str) -> bool:
    """Whether ``faction`` currently owes an answer to a blocking pending
    decision (:data:`_BLOCKING_PENDING_KINDS`) — the condition a
    simulation driver must clear before letting that faction's turn end.
    """
    return any(
        p.faction == faction and p.kind in _BLOCKING_PENDING_KINDS for p in state.pending
    )
```

- [ ] **Step 4: Run new test + full engine suite** — `uv run pytest tests/test_legal_pending_api.py tests/test_legal.py -x -q`.
- [ ] **Step 5: Commit** — `feat: public pending-answer API on legal.py for simulation drivers`

---

### Task 3: Corpus setup sampler (`arena/setups.py`)

**Files:**
- Create: `src/bgai/arena/__init__.py` (empty), `src/bgai/arena/setups.py`
- Test: `tests/test_arena_setups.py`

**Interfaces:**
- Produces: `clean_game_ids(raw_dir: Path = Path("data/raw/games")) -> tuple[str, ...]` — sorted game ids that load and have no dropped faction; cached with `functools.lru_cache`. `sample_setup(rng: random.Random, raw_dir: Path = Path("data/raw/games")) -> GameSetup` — a uniformly sampled clean setup with `dropped_at_row` cleared.
- Master-plan decision 2026-08-04: corpus-sampled setups only, no synthetic generator.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_arena_setups.py"""
from __future__ import annotations

import random

from bgai.arena.setups import clean_game_ids, sample_setup


def test_clean_ids_exclude_known_bad_classes() -> None:
    ids = clean_game_ids()
    # 3,564 raw files minus 10 nofaction load failures minus 179 dropped-faction games.
    assert len(ids) == 3564 - 10 - 179
    assert ids == tuple(sorted(ids))


def test_sample_setup_is_clean_and_deterministic() -> None:
    a = sample_setup(random.Random(3))
    b = sample_setup(random.Random(3))
    assert a.game_id == b.game_id
    assert dict(a.dropped_at_row) == {}
    assert len(a.factions) == 4
    assert len(a.score_tiles) == 6
```

- [ ] **Step 2: Run test, verify it fails** — `ModuleNotFoundError: bgai.arena`.

- [ ] **Step 3: Implement**

```python
"""src/bgai/arena/setups.py

Corpus-sampled arena setups (master plan, Phase 4 decision 2026-08-04):
every arena table reuses a real Div 1-3 game's fixed configuration —
score tiles, bonus-tile pool, faction lineup, seat order, options — via
the replay-validated ``load_setup``, with the drop history cleared
(arena games have four live seats). No synthetic setup generator: human
lineups are pre-vetted for coherence, and evaluation stays on the
Phase 5 training distribution.
"""
from __future__ import annotations

import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from bgai.engine.tm.setup import GameSetup, load_setup

_DEFAULT_RAW_DIR = Path("data/raw/games")


@lru_cache(maxsize=4)
def clean_game_ids(raw_dir: Path = _DEFAULT_RAW_DIR) -> tuple[str, ...]:
    """Sorted ids of every raw game that loads and never dropped a
    faction. ``load_setup`` raising ``ValueError`` (the 10 ``nofaction*``
    incomplete crawls) and non-empty ``dropped_at_row`` (179 AFK-drop
    games) are the two exclusion classes.
    """
    ids: list[str] = []
    for path in sorted(raw_dir.glob("*.json.gz")):
        game_id = path.name.removesuffix(".json.gz")
        try:
            setup = load_setup(game_id, raw_dir)
        except ValueError:
            continue
        if setup.dropped_at_row:
            continue
        ids.append(game_id)
    return tuple(ids)


def sample_setup(rng: random.Random, raw_dir: Path = _DEFAULT_RAW_DIR) -> GameSetup:
    """A uniformly drawn clean setup, drop history cleared."""
    ids = clean_game_ids(raw_dir)
    game_id = ids[rng.randrange(len(ids))]
    setup = load_setup(game_id, raw_dir)
    return replace(setup, dropped_at_row={})
```

- [ ] **Step 4: Run test, verify pass.** The first `clean_game_ids()` call loads all 3,564 raw files — if the test exceeds ~60s, add a parquet-metadata fast path (read `games_meta.parquet`, pre-filter, verify count unchanged) before proceeding; otherwise keep the simple version and note the one-time cost in the docstring.
- [ ] **Step 5: Commit** — `feat: corpus-sampled arena setups with drops cleared`

---

### Task 4: Simulation driver (`arena/sim.py`) — the core

**Files:**
- Create: `src/bgai/arena/sim.py`
- Test: `tests/test_arena_sim.py`

**Interfaces:**
- Consumes: everything in Global Constraints, plus Task 2's `pending_answer_moves`/`has_blocking_pending`, Task 1's `Agent`.
- Produces:
  - `GameResult` frozen dataclass: `setup_game_id: str`, `seats: Mapping[str, str]` (faction → agent name), `vps: Mapping[str, int]`, `ranks: Mapping[str, int]` (0-based competition ranking, ties share), `decisions: int`, `error: str | None`, `anomalies: tuple[str, ...]`.
  - `run_game(setup: GameSetup, seats: Mapping[str, Agent], rng: random.Random, max_decisions: int = 5000) -> GameResult`.
- Scheduler design (locked during planning against the engine source):
  1. **Init:** `state = start_setup(GameState.initial(setup))`.
  2. **Setup phases** (`SETUP_DWELLINGS`, `SETUP_BONUS`): active faction picks one move from the phase moves only (filter `legal_moves(state)` to `verb == "build"` / `verb == "pass"` respectively — no converts/noops during setup), `apply`, drain, `advance_turn`. `_advance_setup_bonus` flips to `Phase.INCOME` round 1 by itself.
  3. **INCOME window:** for each faction in `state.turn_order`: `apply(state, f, cmd("other_income_for_faction", kind=Kind.INCOME))` (registered handler; documented synthetic-command pattern in `round_flow.grant_missing_cult_income`'s docstring). Then the **spade window** (below), then `begin_actions(state)`.
  4. **ACTIONS:** loop: drain; if `state.phase` left `ACTIONS` (all passed → `CLEANUP`), break; else run one **turn protocol** for `active_faction(state)`.
  5. **CLEANUP window:** for each faction in turn order: `apply(state, f, cmd("cult_income_for_faction", kind=Kind.INCOME))` (uses `score_tiles[state.round - 1]` — round is still r here, correct tile). Then the spade window, then `end_of_round(state)` → next `INCOME` or `FINISHED`.
  6. **FINISHED:** `state = final_scoring(state)`; read `vps` from `state.factions[f].vp`; competition ranks from VP descending, ties share rank.
  - **Drain** (`_drain(state, seats, rng, skip: str | None)`): while any `p` in `state.pending` has `p.kind` blocking and `p.faction != skip` and `pending_answer_moves(state, p.faction)` non-empty: that faction's agent chooses among exactly those answers; `apply`; repeat. Queue order (leech offers enqueue clockwise — head-first matches snellman's default answer order).
  - **Spade window:** while any faction (turn order) has `spades_available > 0`: offer = its `transform` moves from `legal_moves_for(state, f)` (the INCOME/CLEANUP branch serves them) plus a synthetic forfeit `cmd("lose_resource", res1="SPADE", n1=<balance>)`; if no transform moves exist, auto-forfeit without consulting the agent. Forfeiting prevents a leftover income spade from leaking into ACTIONS as a free build-continuation (`is_turn_boundary` returns False whenever `spades_available > 0`).
  - **Turn protocol** (ACTIONS, faction f): `fresh_taken = False`, `prev_verb = None`; loop:
    - `answers = pending_answer_moves(state, f)`; if non-empty → offer = answers (a favor/town pick mid-turn is forced before anything else).
    - elif not `fresh_taken` → offer = `legal_moves(state)` minus verbs `{"wait", "done"}` (pass is always present, so never empty).
    - else → offer = continuations + free actions + `cmd("done")`, where continuations = `[m for m in legal_moves(state) if m.verb in MAIN_TRACK_VERBS and not is_turn_boundary(m, state, f, prev_verb)]` and free actions = `[m for m in legal_moves(state) if m.verb in {"convert", "burn"}]`.
    - agent chooses; `done` → break; else `apply`, then `_drain(..., skip=f)`.
    - if applied verb in `MAIN_TRACK_VERBS`: set `prev_verb`; if it was classified fresh (`is_turn_boundary` against the pre-apply state) → `fresh_taken = True`.
    - `pass` → break immediately after apply+drain.
    - Loop guard: turn may not end (`done` rejected from offer composition) while `has_blocking_pending(state, f)` — guaranteed by the answers-first branch.
    - After loop: `advance_turn(state)` (handles ACTC `extra_actions` internally by re-serving the same faction).
  - `MAIN_TRACK_VERBS = frozenset({"build", "upgrade", "dig", "bridge", "transform", "connect", "action", "pass", "advance", "send"})` — module constant, mirroring `round_flow`'s classification universe (convert/burn/leech answers are never turn-relevant, per `is_turn_boundary`'s docstring).
  - **Error handling:** wrap the whole loop; `EngineError` → `GameResult` with `error` = message + repr of offending cmd + decision index (the arena doubles as a legal_moves-soundness fuzzer; a crash is a recorded finding, never silent). Decision count exceeding `max_decisions` → error `"decision cap exceeded"`.
  - `Kind` import: `from bgai.data.ledger_parser import Kind`; if `Kind.INCOME` does not exist, use the `Kind` member the parser assigns income rows (check `ledger_parser.py` at execution; `cmd()`'s default `Kind.DECISION` is acceptable — `handle_income_row` never reads `kind`).

- [ ] **Step 1: Write the failing smoke test**

```python
"""tests/test_arena_sim.py"""
from __future__ import annotations

import random

from bgai.agents import RandomAgent
from bgai.arena.sim import GameResult, run_game
from bgai.engine.tm.setup import load_setup


def _seats(setup, agents):
    return {faction: agents[i % len(agents)] for i, faction in enumerate(setup.factions)}


def test_random_game_runs_to_completion() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    seats = _seats(setup, [RandomAgent("r1"), RandomAgent("r2")])
    result = run_game(setup, seats, random.Random(11))
    assert isinstance(result, GameResult)
    assert result.error is None, result.error
    assert set(result.vps) == set(setup.factions)
    assert sorted(result.ranks.values())[0] == 0
    assert result.decisions < 5000


def test_same_seed_same_result() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    seats = _seats(setup, [RandomAgent("r1"), RandomAgent("r2")])
    a = run_game(setup, seats, random.Random(5))
    b = run_game(setup, seats, random.Random(5))
    assert a.vps == b.vps and a.decisions == b.decisions


def test_five_seeds_three_setups_all_complete() -> None:
    for game_id in ("4pLeague_S10_D1L1_G1", "4pLeague_S1_D1L1_G1", "4pLeague_S1_D1L1_G2"):
        setup = load_setup(game_id)
        seats = _seats(setup, [RandomAgent()])
        for seed in range(5):
            result = run_game(setup, seats, random.Random(seed))
            assert result.error is None, f"{game_id} seed {seed}: {result.error}"
```

- [ ] **Step 2: Run test, verify it fails** — `ModuleNotFoundError` / `ImportError`.
- [ ] **Step 3: Implement `sim.py` per the scheduler design above.** Structure: `GameResult`; `MAIN_TRACK_VERBS`; private helpers `_drain`, `_spade_window`, `_income_window`, `_cleanup_window`, `_setup_phase_step`, `_actions_turn`; public `run_game` composing them with the decision counter and error wrapper. Keep under 400 lines; every helper's docstring cites the engine function it composes.
- [ ] **Step 4: Iterate to green.** Expected debugging surface (budget for it): turn-boundary misclassification (symptom: `EngineError: faction acted out of turn`), spade-window leaks, pending kinds that aren't blocking (marker kinds `free_d`/`free_tp`/`free_tf`/`bridge` must NOT be drained — `pending_answer_moves` already excludes them). Use `superpowers:systematic-debugging` if a failure isn't understood within two hypotheses. If a genuine `legal_moves` soundness gap surfaces (legal move that `apply` rejects), pin it as a test in `tests/test_arena_sim.py` with a comment and route around it in the driver only if the fix belongs in Phase 4's scope; otherwise fix the engine with its own test.
- [ ] **Step 5: Run the full suite** — `uv run pytest tests/ -x -q` (fast tests only, whatever marker convention `test_replay_corpus.py` uses to exclude slow).
- [ ] **Step 6: Commit** — `feat: self-driving arena simulation loop over the replay-validated engine`

---

### Task 5: Timing benchmark note (folded into Task 4's commit if trivial)

After Task 4 is green, measure: `uv run python - <<'EOF'` … time 10 random-vs-random games on 10 sampled setups … `EOF`. Record games/sec in `sim.py`'s module docstring. The master-plan gate is <1s per headless 4p game: if random-vs-random exceeds it materially, do NOT optimize now — record the number and flag it in the final report (plan: speedup work only when MCTS demands it).

---

### Task 6: GreedyAgent (`agents/greedy.py`)

**Files:**
- Create: `src/bgai/agents/greedy.py`; Modify: `src/bgai/agents/__init__.py` (add export)
- Test: extend `tests/test_agents.py`

**Interfaces:**
- Produces: `evaluate(state: GameState, faction: str) -> float`; `GreedyAgent(name: str = "greedy")` — one-ply: for each offered move, `apply` it to a copy of the decision state, score with `evaluate`, pick argmax (stable: first max wins; `EngineError` during what-if → candidate scored `-inf` and recorded — soundness finding, not a crash).
- `evaluate` weights (module constant `WEIGHTS`, one place): `vp: 1.0`, `c: 0.10`, `w: 0.30`, `p: 0.40`, bowl2 power `0.03`, bowl3 power `0.08`, each cult track position `0.15`, each placed building `0.5`, shipping/dig level `0.4` each. Deliberately crude — its only job is to be unambiguously stronger than random.

- [ ] **Step 1: Write the failing tests**

```python
def test_greedy_prefers_vp_over_noop() -> None:
    """On a real mid-setup state, greedy must pick a dwelling placement
    (adds a building to evaluate) over conceptually available junk."""
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    faction = active_faction(state)
    offer = legal_moves(state)
    build_offers = tuple(m for m in offer if m.verb == "build")
    pick = GreedyAgent().choose(state, faction, offer, random.Random(0))
    assert pick in build_offers


def test_greedy_is_deterministic() -> None:
    setup = load_setup("4pLeague_S10_D1L1_G1")
    state = start_setup(GameState.initial(setup))
    faction = active_faction(state)
    offer = legal_moves(state)
    picks = {GreedyAgent().choose(state, faction, offer, random.Random(i)).raw for i in range(3)}
    assert len(picks) == 1
```

(add imports to the existing test module: `load_setup`, `GameState`, `start_setup`, `active_faction`, `legal_moves`, `GreedyAgent`)

- [ ] **Step 2: Run, verify fail** → **Step 3: Implement** →

```python
"""src/bgai/agents/greedy.py"""
from __future__ import annotations

import random

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.factions_data import CULTS
from bgai.engine.tm.state import GameState

WEIGHTS = {
    "vp": 1.0, "c": 0.10, "w": 0.30, "p": 0.40,
    "bowl2": 0.03, "bowl3": 0.08, "cult": 0.15,
    "building": 0.5, "track": 0.4,
}


def evaluate(state: GameState, faction: str) -> float:
    """Crude static evaluation — exists only to beat random decisively."""
    fs = state.factions[faction]
    score = (
        WEIGHTS["vp"] * fs.vp
        + WEIGHTS["c"] * fs.c
        + WEIGHTS["w"] * fs.w
        + WEIGHTS["p"] * fs.p
        + WEIGHTS["bowl2"] * fs.power.bowl2
        + WEIGHTS["bowl3"] * fs.power.bowl3
        + WEIGHTS["cult"] * sum(state.cults[faction][c] for c in CULTS)
        + WEIGHTS["building"] * sum(len(hexes) for hexes in fs.buildings.values())
        + WEIGHTS["track"] * (fs.shipping_level + fs.dig_level)
    )
    return score


class GreedyAgent:
    """One-ply lookahead over the offered moves under ``evaluate``."""

    def __init__(self, name: str = "greedy") -> None:
        self.name = name

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        best, best_score = offer[0], float("-inf")
        for move in offer:
            try:
                child = apply(state, faction, move)
            except EngineError:
                continue
            score = evaluate(child, faction)
            if score > best_score:
                best, best_score = move, score
        return best
```

**Field-name check at execution:** `FactionState` attribute names for resources/power/tracks/buildings must be read from `state.py:68-163` before writing this file (the plan's names — `fs.c/w/p`, `fs.power.bowl2/bowl3`, `fs.shipping_level/dig_level`, `fs.buildings` — were taken from the state-model docs; verify exact spellings, fix here if they differ).

- [ ] **Step 4: Run tests** → **Step 5: Commit** — `feat: one-ply greedy heuristic baseline`

---

### Task 7: Mirrored rotation (`arena/rotation.py`)

**Files:**
- Create: `src/bgai/arena/rotation.py`
- Test: `tests/test_arena_rotation.py`

**Interfaces:**
- Produces: `seat_rotations(agent_names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]` — input length 4 (agent name per seat, duplicates allowed, e.g. `("greedy", "random", "greedy", "random")`); output = the 4 cyclic rotations. One sampled setup is played once per rotation, so every agent name occupies every seat (and thus every faction) exactly as often as it appears in the base assignment — faction/seat strength cancels out of pairwise comparisons.

- [ ] **Step 1: Failing test**

```python
"""tests/test_arena_rotation.py"""
from __future__ import annotations

from bgai.arena.rotation import seat_rotations


def test_four_cyclic_rotations() -> None:
    rots = seat_rotations(("a", "b", "c", "d"))
    assert rots == (
        ("a", "b", "c", "d"),
        ("b", "c", "d", "a"),
        ("c", "d", "a", "b"),
        ("d", "a", "b", "c"),
    )


def test_each_agent_visits_each_seat_once() -> None:
    rots = seat_rotations(("g", "r", "g", "r"))
    for seat in range(4):
        assert sorted(rot[seat] for rot in rots) == ["g", "g", "r", "r"]
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement**

```python
"""src/bgai/arena/rotation.py"""
from __future__ import annotations


def seat_rotations(agent_names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """The 4 cyclic rotations of a base seat assignment (master plan:
    mirrored seat/faction rotation — same setup replayed once per
    rotation so seat and faction effects cancel out of agent
    comparisons).
    """
    if len(agent_names) != 4:
        raise ValueError(f"expected 4 seats, got {len(agent_names)}")
    return tuple(
        tuple(agent_names[(seat + shift) % 4] for seat in range(4)) for shift in range(4)
    )
```

- [ ] **Step 4: Run** → **Step 5: Commit** — `feat: mirrored 4-seat cyclic rotation`

---

### Task 8: TrueSkill ratings (`arena/ratings.py`)

**Files:**
- Create: `src/bgai/arena/ratings.py`
- Test: `tests/test_arena_ratings.py`
- Dependency: `uv add trueskill`

**Interfaces:**
- Consumes: Task 4's `GameResult`.
- Produces: `Ratings = Mapping[str, trueskill.Rating]` (agent name → rating); `new_ratings(agent_names: Iterable[str]) -> Ratings`; `update(ratings: Ratings, result: GameResult) -> Ratings` — IMMUTABLE: returns a new dict; games with `result.error` are skipped unchanged. `conservative(rating) -> float` (μ − 3σ, the standard sort key). `placement_table(results) -> dict[tuple[str, str], tuple[int, float]]` keyed by `(agent, faction)` and a seat variant keyed by `(agent, seat_index)` → `(n_games, mean_rank)` — the per-faction/per-seat covariate reporting the master plan requires.
- TrueSkill mapping: one rating per agent *name*; a 4-seat game is 4 single-player "teams" `rate([{name_i: r_i} …], ranks=[rank_i …])`. When one agent name occupies several seats, its seats are still separate teams — merge the resulting ratings by taking the team with the *lowest* σ (conservative merge; document in docstring). Default `trueskill.TrueSkill()` env; ties pass equal ranks (trueskill handles draws natively).

- [ ] **Step 1: Failing tests** — winner's μ rises, loser's falls; update returns a NEW mapping (input unchanged); errored `GameResult` is a no-op; `placement_table` on two hand-built `GameResult`s gives exact `(n, mean_rank)` values. Write all four with hand-constructed `GameResult` objects (no simulation needed).

```python
"""tests/test_arena_ratings.py"""
from __future__ import annotations

from bgai.arena.ratings import conservative, new_ratings, placement_table, update
from bgai.arena.sim import GameResult


def _result(ranks: dict[str, int], seats: dict[str, str], error: str | None = None) -> GameResult:
    return GameResult(
        setup_game_id="g", seats=seats, vps={f: 100 - r for f, r in ranks.items()},
        ranks=ranks, decisions=100, error=error, anomalies=(),
    )


SEATS = {"witches": "greedy", "nomads": "random", "engineers": "greedy", "darklings": "random"}
RANKS = {"witches": 0, "engineers": 1, "nomads": 2, "darklings": 3}


def test_update_moves_winner_up_loser_down_immutably() -> None:
    before = new_ratings(["greedy", "random"])
    after = update(before, _result(RANKS, SEATS))
    assert after["greedy"].mu > before["greedy"].mu
    assert after["random"].mu < before["random"].mu
    assert before["greedy"].mu == new_ratings(["greedy"])["greedy"].mu  # input untouched


def test_errored_game_is_noop() -> None:
    before = new_ratings(["greedy", "random"])
    after = update(before, _result(RANKS, SEATS, error="boom"))
    assert after == before


def test_conservative_orders_by_mu_minus_3sigma() -> None:
    r = new_ratings(["a"])["a"]
    assert conservative(r) == r.mu - 3 * r.sigma
```

- [ ] **Step 2: Verify fail** → **Step 3: Implement per the interface spec** (~80 lines; seat order for the seat-index table comes from `setup.factions` ordering carried in `GameResult.seats` iteration order — `dict` preserves insertion; `run_game` must build `seats`/`vps`/`ranks` dicts in seat order, note this cross-task contract in `sim.py`'s `GameResult` docstring).
- [ ] **Step 4: Run** → **Step 5: Commit** — `feat: immutable TrueSkill ratings with per-faction/per-seat placement tables`

---

### Task 9: Series orchestration + CLI (`arena/series.py`, `arena/run.py`)

**Files:**
- Create: `src/bgai/arena/series.py`, `src/bgai/arena/run.py`
- Test: `tests/test_arena_series.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 7, 8.
- Produces: `SeriesResult` frozen dataclass: `results: tuple[GameResult, ...]`, `ratings: Ratings`, `n_errors: int`. `run_series(agents: Mapping[str, Agent], base_seats: tuple[str, ...], n_tables: int, seed: int, raw_dir: Path = ...) -> SeriesResult` — per table: `sample_setup`, play all 4 `seat_rotations(base_seats)` on that same setup (fresh `random.Random(hash((seed, table, rotation)) & 0xFFFFFFFF)` per game — derive deterministically, document), update ratings after each game. CLI (`run.py`): `--tables N --seed S --report PATH --agents random,greedy` → builds `{random: RandomAgent(), greedy: GreedyAgent()}`, `base_seats = ("greedy", "random", "greedy", "random")`, runs, writes the Task 10 HTML report, prints a one-line summary (`games, errors, ratings sorted by conservative()`).

- [ ] **Step 1: Failing test** — `run_series` with 2 tables × 4 rotations returns 8 results, deterministic across two identical calls (compare `[r.vps for r in results]`), `ratings` contains both agent names, and every `GameResult.error is None` (uses `RandomAgent` vs `RandomAgent` for speed).
- [ ] **Step 2: Verify fail** → **Step 3: Implement** (~90 lines + ~60-line CLI with `argparse`).
- [ ] **Step 4: Run** → **Step 5: Commit** — `feat: arena series runner with deterministic per-game seeding and CLI`

---

### Task 10: HTML report (`arena/report.py`)

**Files:**
- Create: `src/bgai/arena/report.py`
- Test: `tests/test_arena_report.py`

**Interfaces:**
- Produces: `render_report(series: SeriesResult) -> str` (self-contained HTML, no external assets) and `write_report(series, path: Path) -> None`. Sections: (1) header with game/error counts; (2) ratings table sorted by `conservative()` (μ, σ, conservative, n games); (3) per-faction mean-placement table (rows = factions, columns = agents, cells = `mean_rank (n)`); (4) per-seat table, same shape; (5) mean VP per agent; (6) anomaly/error list (every non-None `error` and every `anomalies` entry verbatim — the fuzzer findings must be visible, never swallowed). Plain `<table>` markup, one `<style>` block, f-string template; escape all dynamic strings with `html.escape`.

- [ ] **Step 1: Failing test** — build a tiny `SeriesResult` from two hand-made `GameResult`s (reuse Task 8's `_result` pattern), assert the HTML contains both agent names, a faction row label, the string `μ`, and the escaped error text of an errored game.
- [ ] **Step 2: Verify fail** → **Step 3: Implement** (~120 lines) → **Step 4: Run** → **Step 5: Commit** — `feat: self-contained HTML arena report`

---

### Task 11: The Phase 4 gate — greedy ≫ random over ≥200 games

**Files:**
- Create: `tests/test_arena_verify.py` (slow-marked, mirroring `test_replay_corpus.py`'s convention)
- Modify: `README.md` (arena section), master plan Phase 4 checklist

**Interfaces:** consumes everything above.

- [ ] **Step 1: Write the gate test**

```python
"""tests/test_arena_verify.py — Phase 4 master-plan gate (slow).

greedy >> random over >=200 mirrored games: 50 corpus-sampled tables x 4
cyclic rotations, base seats (greedy, random, greedy, random). Gate
criteria: (1) zero errored games; (2) greedy's mean placement beats
random's by >= 0.4 ranks; (3) TrueSkill conservative(greedy) >
conservative(random).
"""
from __future__ import annotations

import random  # noqa: F401  (seed derivation lives in run_series)

from bgai.agents import GreedyAgent, RandomAgent
from bgai.arena.ratings import conservative
from bgai.arena.series import run_series

# apply the same slow/skip marker used by tests/test_replay_corpus.py


def test_greedy_beats_random_over_200_games() -> None:
    series = run_series(
        agents={"greedy": GreedyAgent(), "random": RandomAgent()},
        base_seats=("greedy", "random", "greedy", "random"),
        n_tables=50,
        seed=20260804,
    )
    assert series.n_errors == 0, [r.error for r in series.results if r.error]
    mean_rank = {}
    for name in ("greedy", "random"):
        ranks = [
            r.ranks[f] for r in series.results for f, a in r.seats.items() if a == name
        ]
        mean_rank[name] = sum(ranks) / len(ranks)
    assert mean_rank["greedy"] + 0.4 <= mean_rank["random"], mean_rank
    assert conservative(series.ratings["greedy"]) > conservative(series.ratings["random"])
```

- [ ] **Step 2: Run it** — `uv run pytest tests/test_arena_verify.py -x -q` (expect minutes; run in background). If it fails on the margin (greedy not clearly better), strengthen `evaluate`'s building/track weights before touching anything else — do NOT weaken the gate. If it fails on errors, each error is a Task 4/engine bug: fix with a pinned test, rerun.
- [ ] **Step 3: Generate the demonstration report** — `uv run python -m bgai.arena.run --tables 50 --seed 20260804 --report /tmp/arena_phase4.html --agents random,greedy`; sanity-check the HTML per-faction table has no starved rows worth flagging (decision 2026-08-04: stratify only if a report shows starvation).
- [ ] **Step 4: Timing measurement** — from the series wall-clock, compute seconds/game; record honestly in README (<1s target: state pass/fail; no optimization either way this phase).
- [ ] **Step 5: Docs** — README gains an "Arena" section (how to run the CLI, what the gate asserts, current timing); master plan Phase 4 marked complete with the gate numbers.
- [ ] **Step 6: Commit** — `test: pin Phase 4 gate — greedy beats random over 200 mirrored arena games` then `docs: README arena section + master plan Phase 4 completion`

---

## Self-Review (done at planning time)

- **Spec coverage:** Agent interface (T1), baselines (T1, T6), match runner (T4, T9), corpus-sampled setups (T3), mirrored rotation (T7), TrueSkill + per-faction/per-seat covariate reporting (T8), HTML reports (T10), greedy≫random ≥200 games gate (T11), <1s timing measurement (T5, T11), tmai `ai_lode` baseline explicitly SKIPPED (optional in master plan; Node subprocess dependency not worth it this phase — noted here so the omission is a decision, not a miss).
- **Type consistency:** `GameResult` fields match between T4 (producer) and T8/T9/T10/T11 (consumers); `Agent.choose` signature identical in T1/T4/T6; `seat_rotations` output feeds `run_series` base-seats contract.
- **Known execution-time verifications (flagged inline):** `FactionState` field spellings (T6), `Kind` member for income rows (T4), slow-test marker convention (T3 timing, T11), `lose_resource` handling of `res1="SPADE"` (T4 spade forfeit).
