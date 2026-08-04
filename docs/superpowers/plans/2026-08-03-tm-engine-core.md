# Terra Mystica Rules Engine Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `apply(state, faction, move) -> GameState` (plus supporting state model, decision queue, and round flow) that replays all 3,563 crawled tournament games with zero per-move delta mismatches against `data/datasets/deltas.parquet`, then expose `legal_moves(state)`.

**Architecture:** Immutable frozen-dataclass `GameState`/`FactionState`; a pending-decision queue models forced follow-ups (leech offers, favor/town picks, stronghold sequences); `apply` dispatches on the parser's `ParsedCommand` verbs (the engine consumes `moves.parquet` rows directly — no second grammar). A replay harness checks the acting faction's VP/C/W/P/PW/CULT against `deltas.parquet` after every ledger row. Every rule is ported from the reference implementation (jsnell/terra-mystica, Perl), never from memory — cite source file + lines in docstrings (existing repo convention).

**Tech Stack:** Python 3.12, uv, pytest, polars (already deps). No new dependencies.

## Global Constraints

- Run tests with `uv run pytest <path> -x -q` from the repo root `/Users/keerthikmuruganandam/code/bgai`.
- IMMUTABILITY: all state types are `@dataclass(frozen=True)`; `apply` returns a new `GameState`, never mutates. Use `dataclasses.replace` and small `with_*` helpers.
- Files stay under ~400 lines; split by mechanic, not by layer.
- Type annotations on all function signatures. `from __future__ import annotations` at top of every module.
- Rules provenance: every nontrivial mechanic's docstring cites the reference Perl file (e.g. "resources.pm lines 190-231"). The reference clone lives at `/Users/keerthikmuruganandam/code/terra-mystica` (Task 1 creates it; never committed to this repo).
- The oracle is exact: `deltas.parquet` columns `vp_value/c_value/w_value/p_value` (absolute), `pw` (e.g. `"5/7/0"`), `cult` (e.g. `"0/1/1/0"` = FIRE/WATER/EARTH/AIR) for the acting faction of each ledger row.
- Commit after every green test cycle, conventional commits (`feat:`/`test:`/`fix:`), no attribution footers.
- Existing building blocks (do not reimplement): `bgai.engine.tm.board` (`base_board()`, `hex_distance`), `bgai.engine.tm.power` (`Power`), `bgai.engine.tm.cults` (`advance`, `CultAdvance`, `PRIEST_SLOT_STEPS`), `bgai.engine.tm.terraform` (`spade_distance`), `bgai.engine.tm.factions_data` (`FACTIONS`, `FactionData`, `FACTION_SPECIAL_ACTIONS`, `BUILDING_MAX_COUNT`, `BASE_EXCHANGE_RATES`, `MAX_PRIESTS`, `TOWN_SIZE`, `BRIDGE_COUNT`, `CULTS`), `bgai.data.ledger_parser` (`ParsedCommand`, `Kind`).
- Datasets: `data/datasets/moves.parquet` (game_id, row, seq, faction, kind, verb, loc, loc2, building, tile, cult, color, target, reason, res1, res2, n1, n2), `data/datasets/deltas.parquet`, `data/datasets/games_meta.parquet` (game_id, options CSV, player_count, factions CSV **alphabetical — not seat order**, final_vp JSON). Raw per-game JSON: `data/raw/games/<game_id>.json.gz` with keys `score_tiles` (list of 6 resolved dicts), `pool` (BON subset = keys matching `BON\d+`), `order`, `players`, `ledger`, `options`.
- Known reference game for exact-value tests: `4pLeague_S10_D1L1_G1` — seat/turn order engineers, darklings, nomads, mermaids; setup ledger rows 24-27 carry initial-state deltas (engineers pw `3/9/0`, darklings cult `0/1/1/0`); first income rows 42-45; round-1 score tile: cult WATER, req 4, income `{SPADE: 1}`, vp_mode build, vp `{TP: 3}`.

## File Map

Create under `src/bgai/engine/tm/` (existing modules unchanged unless noted):

| File | Responsibility |
|---|---|
| `tiles.py` | Static data: BON1-10, FAV1-12, TW1-8, ACT1-6 power actions, `ScoringTile` |
| `setup.py` | `GameOptions`, `GameSetup`, raw-JSON loader |
| `state.py` | `Phase`, `HexState`, `PendingDecision`, `FactionState`, `GameState`, initial-state constructors, `active_faction` |
| `income.py` | Per-faction income computation, split into the ledger's income categories |
| `connectivity.py` | Reachability (adjacency+bridge+shipping, tunnel, carpet), connected clusters |
| `towns.py` | Town formation detection (incl. Mermaids river skip), town founding rewards |
| `leech.py` | Leech offer computation + enqueue ordering, accept/decline, Cultists hooks |
| `apply.py` | `apply()` dispatch, decision-queue mechanics, convert/burn/bookkeeping verbs |
| `actions_build.py` | build, upgrade, bridge |
| `actions_terraform.py` | transform, dig, spade accounting, faction spade behavior |
| `actions_power.py` | `action ACTx` — ACT1-6, ACTA/C/E/G/N/S/W, BON/FAV actions, once-per-round tracking |
| `actions_pass.py` | pass (bonus swap, pass-VP), advance, send priest, connect |
| `round_flow.py` | phase transitions, income application, cleanup, turn order |
| `scoring.py` | Final scoring: cults 8/4/2 with ties, network, score_resources |
| `factions/hooks.py` | `FactionHooks` protocol + registry (per-faction modules only if a hook outgrows a method) |
| `replay.py` | `ReplayResult`, `replay_game()`, delta checking, batch CLI |
| `legal.py` | `legal_moves(state)` (final task) |

Tests mirror one-to-one: `tests/test_tiles.py`, `tests/test_setup.py`, `tests/test_state.py`, `tests/test_income.py`, `tests/test_connectivity.py`, `tests/test_towns.py`, `tests/test_leech.py`, `tests/test_apply.py`, `tests/test_actions_build.py`, `tests/test_actions_terraform.py`, `tests/test_actions_power.py`, `tests/test_actions_pass.py`, `tests/test_round_flow.py`, `tests/test_scoring.py`, `tests/test_replay.py`, `tests/test_legal.py`.

---

### Task 1: Reference clone + tile data (`tiles.py`)

**Files:**
- Create: `src/bgai/engine/tm/tiles.py`
- Test: `tests/test_tiles.py`
- Reference clone (outside repo): `/Users/keerthikmuruganandam/code/terra-mystica`

**Interfaces:**
- Produces: `BONUS_TILES: dict[str, BonusTile]`, `FAVOR_TILES: dict[str, FavorTile]`, `TOWN_TILES: dict[str, TownTile]`, `POWER_ACTIONS: dict[str, PowerAction]` (ACT1-6), `ScoringTile` frozen dataclass with `from_snellman(d: dict) -> ScoringTile`, `FAVOR_POOL_COUNTS: dict[str, int]`, `TOWN_POOL_COUNTS: dict[str, int]` (base + mini-expansion variants).

- [ ] **Step 1: Clone the reference repo (skip if present)**

```bash
test -d /Users/keerthikmuruganandam/code/terra-mystica || \
  git clone --depth 1 https://github.com/jsnell/terra-mystica /Users/keerthikmuruganandam/code/terra-mystica
```

- [ ] **Step 2: Read the source data before writing anything**

Read `/Users/keerthikmuruganandam/code/terra-mystica/src/Game/Constants.pm`: `%bonus_tiles`, `%favors`, `%tiles` (town tiles TWx), `%actions` (ACT1-ACT6 entries), `%score_tiles` / score-tile setup, and which tiles the `mini-expansion-1`, `shipping-bonus`, `temple-scoring-tile` options add (grep those option names across `src/`). Do not write tile values from memory — transcribe.

- [ ] **Step 3: Write the failing test**

```python
"""tests/test_tiles.py"""
import gzip, json
from pathlib import Path
import pytest
from bgai.engine.tm.tiles import (
    BONUS_TILES, FAVOR_TILES, TOWN_TILES, POWER_ACTIONS, ScoringTile,
)

def test_tile_id_coverage() -> None:
    assert set(BONUS_TILES) >= {f"BON{i}" for i in range(1, 11)}
    assert set(FAVOR_TILES) == {f"FAV{i}" for i in range(1, 13)}
    assert set(TOWN_TILES) >= {f"TW{i}" for i in range(1, 9)}
    assert set(POWER_ACTIONS) == {f"ACT{i}" for i in range(1, 7)}

def test_known_tile_values() -> None:
    # Anchor literals: write these AFTER reading Constants.pm in Step 2, by
    # hand-copying three values from the Perl source (not from tiles.py — the
    # point is an independent transcription check). Expected shape:
    assert POWER_ACTIONS["ACT4"].cost_power == 4   # confirm against %actions
    assert FAVOR_TILES["FAV1"].cult == "FIRE" and FAVOR_TILES["FAV1"].steps == 3
    assert BONUS_TILES["BON3"].income == {"C": 6}

def test_scoring_tile_from_snellman() -> None:
    raw = {"vp": {"TP": 3}, "cult": "WATER", "vp_mode": "build",
           "income": {"SPADE": 1}, "req": 4,
           "income_display": "4 WATER -> 1 SPADE", "vp_display": "TP >> 3"}
    t = ScoringTile.from_snellman(raw)
    assert t.cult == "WATER" and t.req == 4
    assert t.vp_mode == "build" and t.vp == (("TP", 3),)
    assert t.cult_income == (("SPADE", 1),)

def test_corpus_pool_coverage() -> None:
    """Every BON/TW/FAV/ACT id appearing in a 50-game sample is defined."""
    games = sorted(Path("data/raw/games").glob("*.json.gz"))[:50]
    known = set(BONUS_TILES) | set(FAVOR_TILES) | set(TOWN_TILES) | set(POWER_ACTIONS)
    for path in games:
        with gzip.open(path) as f:
            pool = json.load(f)["pool"]
        ids = {k for k in pool if k[:3] in ("BON", "FAV", "ACT") or k[:2] == "TW"}
        assert ids <= known, f"{path.name}: unknown tiles {ids - known}"
```

Adjust `test_known_tile_values` anchors to the actual transcribed values in the same commit — the point is that at least three literal values are double-checked by hand against Constants.pm.

- [ ] **Step 4: Run to verify failure** — `uv run pytest tests/test_tiles.py -x -q` → ImportError.

- [ ] **Step 5: Implement `tiles.py`**

Frozen dataclasses; transcribe every field Constants.pm defines that the tournament config uses:

```python
@dataclass(frozen=True)
class BonusTile:
    income: Mapping[str, int]          # granted during income phase while held
    special_action: Mapping[str, int] | None  # e.g. BON1 spade, BON2 cult step
    pass_vp: tuple[tuple[str, int], ...]      # e.g. BON6: (("SH",4),("SA",4))
    passive: Mapping[str, int]         # e.g. BON4 +1 shipping while held

@dataclass(frozen=True)
class FavorTile:
    cult: str; steps: int
    income: Mapping[str, int]
    special_action: Mapping[str, int] | None  # FAV6-class cult action
    passive: Mapping[str, int]         # e.g. TOWN_SIZE->6, TP-build VP, pass-VP
    vp_mode: str | None                # for gain/pass VP favors

@dataclass(frozen=True)
class TownTile:
    vp: int; gain: Mapping[str, int]   # workers/coins/priests/power/cult-all/key count

@dataclass(frozen=True)
class PowerAction:
    cost_power: int
    gain: Mapping[str, int]            # BRIDGE/P/W/C/SPADE counts
```

Include `FAVOR_POOL_COUNTS` (FAV1-4: 1 copy, FAV5-12: 3 copies — verify in cults.pm/Constants.pm) and `TOWN_POOL_COUNTS` keyed by whether `mini-expansion-1` is on. Docstring cites exact Constants.pm line ranges.

- [ ] **Step 6: Run to green** — `uv run pytest tests/test_tiles.py -x -q`.

- [ ] **Step 7: Commit** — `git add src/bgai/engine/tm/tiles.py tests/test_tiles.py && git commit -m "feat: tile data (BON/FAV/TW/ACT/SCORE) ported from reference Constants.pm"`

---

### Task 2: Game options + setup loader (`setup.py`)

**Files:**
- Create: `src/bgai/engine/tm/setup.py`
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `ScoringTile` from Task 1.
- Produces:
  - `GameOptions` frozen dataclass with bools `strict_leech, errata_cultist_power, variable_turn_order, maintain_player_order, strict_darkling_sh, strict_chaosmagician_sh, mini_expansion_1, shipping_bonus, temple_scoring_tile, loose_dig, merge_income_phases`; classmethod `GameOptions.from_csv(text: str) -> GameOptions` (unknown options like `email-notify` ignored).
  - `GameSetup` frozen dataclass: `game_id: str`, `options: GameOptions`, `factions: tuple[str, ...]` (**seat order**, from raw JSON `order`), `score_tiles: tuple[ScoringTile, ...]` (len 6), `bonus_tiles: tuple[str, ...]` (sorted BON ids from `pool`), `player_count: int`.
  - `load_setup(game_id: str, raw_dir: Path = Path("data/raw/games")) -> GameSetup`.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_setup.py"""
from bgai.engine.tm.setup import GameOptions, load_setup

def test_options_from_csv() -> None:
    opts = GameOptions.from_csv(
        "email-notify,errata-cultist-power,strict-leech,mini-expansion-1"
    )
    assert opts.errata_cultist_power and opts.strict_leech and opts.mini_expansion_1
    assert not opts.variable_turn_order

def test_load_reference_game() -> None:
    s = load_setup("4pLeague_S10_D1L1_G1")
    assert s.player_count == 4
    assert s.factions == ("engineers", "darklings", "nomads", "mermaids")
    assert len(s.score_tiles) == 6
    assert s.score_tiles[0].cult == "WATER" and s.score_tiles[0].req == 4
    assert set(s.bonus_tiles) == {"BON1", "BON3", "BON4", "BON5", "BON7", "BON9", "BON10"}
    assert s.options.strict_leech
```

Note: verify the seat-order tuple against the raw JSON `order` field while implementing; the ledger setup rows 24-27 confirm engineers→darklings→nomads→mermaids. If `order` turns out to be keyed differently (dict of faction→index), adapt the loader, not the expected tuple.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_setup.py -x -q`.

- [ ] **Step 3: Implement** — read gzipped JSON, build `GameOptions` from the game's own `options` list (fall back to `games_meta` CSV only if absent), `ScoringTile.from_snellman` per entry, bonus subset = `sorted(k for k in raw["pool"] if re.fullmatch(r"BON\d+", k))`. Validate: exactly 6 score tiles, `player_count + 3` bonus tiles; raise `ValueError` with game_id context otherwise.

- [ ] **Step 4: Run to green.**

- [ ] **Step 5: Commit** — `git commit -m "feat: game options + per-game setup loader from raw snellman JSON"`

---

### Task 3: State model (`state.py`)

**Files:**
- Create: `src/bgai/engine/tm/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `GameSetup` (Task 2), `FACTIONS`/`FactionData`, `Power`, `base_board`, tiles pools (Task 1).
- Produces (all frozen dataclasses):
  - `Phase(Enum)`: `SETUP_DWELLINGS, SETUP_BONUS, INCOME, ACTIONS, CLEANUP, FINISHED`.
  - `HexState(color: str, building: str | None, owner: str | None)`.
  - `PendingDecision(faction: str, kind: str, amount: int = 0, source: str | None = None, options: tuple[str, ...] = ())` — kinds used later: `"leech"`, `"gain_favor"`, `"gain_town"`, `"cult_choice"`, `"convert_w_to_p"`, `"halflings_spades"`, `"bonus_choice"`, `"spade_use"`.
  - `FactionState(name, coins, workers, priests, priest_pool, power: Power, vp, shipping, dig_level, teleport_level, buildings: Mapping[str, frozenset[str]], favors: tuple[str, ...], bonus: str | None, towns: tuple[str, ...], keys: int, passed: bool, actions_used: frozenset[str], cult_blocked: frozenset[str])`; classmethod `FactionState.initial(data: FactionData) -> FactionState` (vp=20, priest_pool=MAX_PRIESTS, buildings all empty).
  - `GameState(setup: GameSetup, round: int, phase: Phase, turn_order: tuple[str, ...], passed_order: tuple[str, ...], active_index: int, pending: tuple[PendingDecision, ...], hexes: Mapping[str, HexState], bridges: frozenset[frozenset[str]], factions: Mapping[str, FactionState], cults: Mapping[str, Mapping[str, int]], cult_10: Mapping[str, str | None], priest_slots: Mapping[str, tuple[str | None, ...]], favors_pool: Mapping[str, int], towns_pool: Mapping[str, int], bonus_coins: Mapping[str, int], power_actions_taken: frozenset[str])`; classmethod `GameState.initial(setup: GameSetup) -> GameState`.
  - Helpers: `active_faction(state) -> str` (head of `pending` if any, else `turn_order[active_index]`); `with_faction(state, name, fs) -> GameState`; `cult_string(state, faction) -> str` (`"F/W/E/A"`); these are what the replay harness compares.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_state.py"""
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, Phase, active_faction, cult_string

def test_initial_state_matches_setup_deltas() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert s.phase is Phase.SETUP_DWELLINGS and s.round == 0
    eng = s.factions["engineers"]
    assert (eng.coins, eng.workers, eng.priests, eng.vp) == (10, 2, 0, 20)
    assert eng.power.as_str() == "3/9/0"          # deltas row 24
    dk = s.factions["darklings"]
    assert (dk.coins, dk.workers, dk.priests) == (15, 1, 1)
    assert cult_string(s, "darklings") == "0/1/1/0"  # deltas row 25
    assert cult_string(s, "nomads") == "1/0/1/0"     # deltas row 26
    assert active_faction(s) == "engineers"

def test_pools_initialized() -> None:
    s = GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))
    assert s.favors_pool["FAV1"] == 1 and s.favors_pool["FAV5"] == 3
    assert sum(s.towns_pool.values()) >= 10
    assert all(h.building is None for h in s.hexes.values())
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — hexes from `base_board()` (`HexState(color=hex.color, building=None, owner=None)` including rivers), cult positions from `FactionData.cults`, pools from Task 1 counts (town pool respects `mini_expansion_1`), `turn_order = setup.factions`. Keep pure construction; no game logic here.

- [ ] **Step 4: Run to green.**

- [ ] **Step 5: Commit** — `git commit -m "feat: immutable GameState/FactionState model with pending-decision queue"`

---

### Task 4: Income computation (`income.py`)

**Files:**
- Create: `src/bgai/engine/tm/income.py`
- Test: `tests/test_income.py`

**Interfaces:**
- Consumes: `FactionState`, `FactionData`, tiles (Task 1).
- Produces: `faction_income(fs: FactionState, data: FactionData) -> dict[str, dict[str, int]]` returning category-keyed income: `{"base": {...}, "buildings": {...}, "bonus": {...}, "favors": {...}}` with resource keys `C/W/P/PW`. (How categories map onto the ledger's `cult_income_for_faction` vs `other_income_for_faction` rows is resolved in Task 10 — this task only computes correct totals per source.)

- [ ] **Step 1: Read the reference** — grep `income` in `/Users/keerthikmuruganandam/code/terra-mystica/src/` (`Game/Income` handling lives around `resources.pm` / `acting.pm`; find where building-track income and `faction->{income}` are summed). Confirm: dwellings use the cumulative track (`income["W"][n]`), Engineers D income starts at 0, base faction income exists for some factions (check each faction's `income` key in the .pm files — e.g. most get 1 W base? transcribe, don't assume).

- [ ] **Step 2: Write the failing test**

```python
"""tests/test_income.py"""
from dataclasses import replace
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.state import FactionState

def _fs(name: str, **kw) -> FactionState:
    fs = FactionState.initial(FACTIONS[name])
    return replace(fs, **kw)

def test_engineers_two_dwellings() -> None:
    fs = _fs("engineers", buildings={"D": frozenset({"E7", "F6"}),
                                     "TP": frozenset(), "TE": frozenset(),
                                     "SH": frozenset(), "SA": frozenset()})
    inc = faction_income(fs, FACTIONS["engineers"])
    assert inc["buildings"].get("W", 0) == 2   # track (0,1,2,...) index 2

def test_bonus_tile_income() -> None:
    fs = _fs("engineers", bonus="BON3",
             buildings={k: frozenset() for k in ("D", "TP", "TE", "SH", "SA")})
    inc = faction_income(fs, FACTIONS["engineers"])
    assert inc["bonus"] == {"C": 6}

def test_favor_income() -> None:
    # Pick, from the Task-1 tiles.py data, one favor tile whose income dict is
    # non-empty (there is at least one coin/power/worker-income favor). Give a
    # faction that single favor and assert inc["favors"] equals that tile's
    # income dict verbatim, e.g.:
    fav_id = next(f for f, t in FAVOR_TILES.items() if t.income)
    fs = _fs("darklings", favors=(fav_id,),
             buildings={k: frozenset() for k in ("D", "TP", "TE", "SH", "SA")})
    inc = faction_income(fs, FACTIONS["darklings"])
    assert inc["favors"] == dict(FAVOR_TILES[fav_id].income)
```

(Import `FAVOR_TILES` from `bgai.engine.tm.tiles` in the test module.)

- [ ] **Step 3: Run to verify failure. Step 4: Implement. Step 5: Run to green.**

- [ ] **Step 6: Data-driven check against real round-1 income** — extend the test:

```python
def test_matches_reference_game_round1_income() -> None:
    import polars as pl
    d = pl.read_parquet("data/datasets/deltas.parquet").filter(
        (pl.col("game_id") == "4pLeague_S10_D1L1_G1") & (pl.col("row").is_in([42, 43, 44, 45]))
    )
    # After setup, every faction has 2 dwellings (nomads 3) and a bonus tile:
    # engineers BON4, darklings BON3, nomads BON5, mermaids BON1 (ledger rows 37-40).
    # Assert summed income (all categories) equals each row's recorded deltas.
```

Fill in the body: build each faction's post-setup `FactionState` by hand (dwelling hexes from ledger rows 28-36: engineers E7+F6, darklings G5+E10, nomads D3+F3+G4, mermaids D2+D5), call `faction_income`, sum categories, and compare to `c_delta/w_delta/p_delta` and the `pw` gain implied by bowl movement (use `Power.gain`). This pins income arithmetic to the oracle before any apply() exists.

- [ ] **Step 7: Commit** — `git commit -m "feat: per-faction income computation validated against reference game deltas"`

---

### Task 5: Connectivity (`connectivity.py`)

**Files:**
- Create: `src/bgai/engine/tm/connectivity.py`
- Test: `tests/test_connectivity.py`

**Interfaces:**
- Consumes: `base_board`, `GameState`, `FactionState`.
- Produces:
  - `directly_adjacent(state, hex_key) -> frozenset[str]` (land adjacency + bridges).
  - `reachable(state, faction) -> frozenset[str]`: hexes the faction may build/transform on considering direct adjacency to own buildings, bridges, shipping range over river hexes, Dwarves tunneling (`teleport_level` range 1 skip), Fakirs carpet range. Port from `map.pm` (`compute_network`/reachability helpers — find the functions used by the build command validation in `commands.pm`).
  - `clusters(state, faction, *, river_skip: bool = False) -> tuple[frozenset[str], ...]`: connected groups of the faction's buildings via direct adjacency + bridges (+ optional single-river-hex joins for Mermaids town checks — flag consumed by Task 6 towns and final network scoring).

- [ ] **Step 1: Write the failing test** — hand-built board scenarios:

Derive scenario hexes programmatically from the real board so no literal is guessed — the tests are concrete but self-locating:

```python
"""tests/test_connectivity.py"""
from dataclasses import replace
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import reachable, clusters
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState, HexState

BOARD = base_board()

def _river_gap() -> tuple[str, str, str]:
    """(land_a, river, land_b): two land hexes joined only via one river hex."""
    for r in (k for k, h in BOARD.hexes.items() if h.color == RIVER):
        lands = [n for n in BOARD.adjacent[r] if BOARD.hexes[n].color != RIVER]
        for a in lands:
            for b in lands:
                if a != b and b not in BOARD.adjacent[a]:
                    return a, r, b
    raise AssertionError("no river gap on base map")

def _place(state: GameState, faction: str, hex_key: str) -> GameState:
    hexes = dict(state.hexes)
    hexes[hex_key] = replace(hexes[hex_key], building="D", owner=faction)
    fs = state.factions[faction]
    fs = replace(fs, buildings={**fs.buildings,
                                "D": fs.buildings["D"] | {hex_key}})
    return replace(state, hexes=hexes,
                   factions={**state.factions, faction: fs})

def _fresh() -> GameState:
    return GameState.initial(load_setup("4pLeague_S10_D1L1_G1"))

def test_shipping_zero_cannot_cross_river() -> None:
    a, _r, b = _river_gap()
    s = _place(_fresh(), "engineers", a)   # engineers ship level 0 initially
    assert b not in reachable(s, "engineers")

def test_shipping_one_reaches_across_single_river_hex() -> None:
    a, _r, b = _river_gap()
    s = _place(_fresh(), "engineers", a)
    s = replace(s, factions={**s.factions, "engineers":
                             replace(s.factions["engineers"], shipping=1)})
    assert b in reachable(s, "engineers")

def test_direct_neighbors_always_reachable() -> None:
    a, _r, _b = _river_gap()
    s = _place(_fresh(), "engineers", a)
    land_neighbors = {n for n in BOARD.adjacent[a] if BOARD.hexes[n].color != RIVER}
    assert land_neighbors <= reachable(s, "engineers")

def test_clusters_merge_via_bridge() -> None:
    # Find two land hexes at hex_distance 2 with a river between (bridge shape
    # rules ported in Task 8; here just assert clusters() honors state.bridges).
    a, r, b = _river_gap()
    s = _place(_place(_fresh(), "engineers", a), "engineers", b)
    assert len(clusters(s, "engineers")) == 2
    s = replace(s, bridges=frozenset({frozenset({a, b})}))
    assert len(clusters(s, "engineers")) == 1

def test_dwarves_tunnel_skips_one_land_hex() -> None:
    # Locate land->land->land chain where the ends are not adjacent.
    for mid in (k for k, h in BOARD.hexes.items() if h.color != RIVER):
        lands = [n for n in BOARD.adjacent[mid] if BOARD.hexes[n].color != RIVER]
        pair = next(((a, b) for a in lands for b in lands
                     if a != b and b not in BOARD.adjacent[a]), None)
        if pair:
            a, b = pair
            s = _place(_fresh(), "darklings", a)  # non-dwarves: not reachable
            assert b not in reachable(s, "darklings")
            # Reload a setup containing dwarves for the positive case, or
            # construct FactionState for dwarves directly; dwarves at teleport
            # range 1 must reach b.
            return
    raise AssertionError("no skip-chain found")
```

Adjust the dwarves positive case while implementing: pick any corpus game containing dwarves (query `games_meta.factions`) or build the dwarves `FactionState` directly — the assertion is `b in reachable(s, "dwarves")`.

- [ ] **Step 2: Run to verify failure. Step 3: Implement** (BFS over own-building hexes; shipping = river-hex path length ≤ level; tunnel/carpet = `hex_distance` + land checks per `map.pm`). **Step 4: Run to green.**

- [ ] **Step 5: Commit** — `git commit -m "feat: board connectivity (bridges, shipping, tunnel, carpet) and building clusters"`

---

### Task 6: Towns (`towns.py`)

**Files:**
- Create: `src/bgai/engine/tm/towns.py`
- Test: `tests/test_towns.py`

**Interfaces:**
- Consumes: `clusters` (Task 5), `TOWN_TILES` (Task 1), `TOWN_SIZE`, favor passives (town size 6 favor).
- Produces:
  - `building_power_value(building: str) -> int` (D=1, TP/TE=2, SH/SA=3 — verify in towns.pm).
  - `new_towns(state, faction) -> tuple[frozenset[str], ...]`: clusters that now qualify as towns (≥4 buildings — SA counting per towns.pm — and total power ≥ TOWN_SIZE, reduced to 6 with the relevant favor; Mermaids `river_skip=True`) minus clusters already counted as towns. Track founded-town hexes on `FactionState.towns`-adjacent bookkeeping — decide representation while implementing: store founded cluster hex-sets in `GameState` (add field `founded_towns: Mapping[str, tuple[frozenset[str], ...]]` in this task via a widening migration of `state.py`).
  - `apply_town_tile(state, faction, tile: str) -> GameState`: VP + gains (incl. 2-cult-steps-on-all tile semantics, key count; Witches +5 VP passive, Swarmlings +3 W passive).

- [ ] **Step 1: Read `towns.pm`** in the reference clone end-to-end (it is short) — town qualification, Mermaids river connection, when a founded town's hexes block re-founding.

- [ ] **Step 2: Write failing tests** — hand-built states: a 4-building power-7 cluster qualifies; 3 buildings power-8 does not; SA in cluster with town-size-6 favor qualifies at power 6; Mermaids cluster joined across one river hex qualifies only for mermaids; founding twice returns nothing new; `apply_town_tile` grants transcribed TW values + faction passives.

- [ ] **Step 3-5: Red → implement → green. Step 6: Commit** — `git commit -m "feat: town formation detection and town tile rewards"`

---

### Task 7: Apply skeleton + decision queue + conversions (`apply.py`, `factions/hooks.py`)

**Files:**
- Create: `src/bgai/engine/tm/apply.py`, `src/bgai/engine/tm/factions/__init__.py`, `src/bgai/engine/tm/factions/hooks.py`
- Test: `tests/test_apply.py`

**Interfaces:**
- Consumes: everything above; `ParsedCommand`.
- Produces:
  - `apply(state: GameState, faction: str, cmd: ParsedCommand) -> GameState` — the single entry point. Asserts `faction == active_faction(state)` unless `cmd.verb` is order-exempt (`wait`, `annotation`, leech/decline answering an outstanding offer for that faction anywhere in the queue). Dispatches on `cmd.verb` to handler functions registered in a `dict[str, Handler]`; unknown verb → `EngineError` (a new exception type carrying state context).
  - Queue mechanics: `push_pending(state, *decisions) -> GameState`, `pop_pending(state) -> GameState`; answering verbs (`gain_favor`, `gain_town`, `leech`, `decline`, faction-specific pendings) must match the head (or, for leech, the first queue entry for that faction — snellman lets later leechers answer early only per strict-leech rules; port exact semantics from `acting.pm`).
  - This task implements the simple verbs fully: `convert` (BASE_EXCHANGE_RATES + faction overrides + Alchemists VP→C; power spends via `Power.spend`), `burn`, `wait`, `done`, `resign` (no-op), `annotation`, `setup` (no-op anchors), `lose_*`/`convert_marker` bookkeeping verbs (apply the literal stated loss), `gain_cult`/`lose_cult` bookkeeping rows (advance/retreat on the named track via `cults.advance`, consuming a matching `cult_choice` pending when one is queued — these rows carry Cultists' leech choice, ACTA/BON2/FAV-action steps, and town-tile cult gains), `send` (priest → cult track: slot choice per `cmd.n1` (steps) — `send p to FIRE for 2` takes a 2-slot; default = biggest open slot, priest leaves `priest_pool` permanently unless track full → 1-step bounce per cults.pm; uses `cults.advance` for power/keys).
  - `factions/hooks.py`: `class FactionHooks` with no-op defaults — methods added as later tasks need them: `spade_transform_target(...)`, `extra_dig_gain(...)`, `on_stronghold_built(...)`, `on_leech_resolved(...)`, `pass_vp_extra(...)`, `reachable_extra(...)`; `HOOKS: dict[str, FactionHooks]` registry (default instance for factions without overrides).

- [ ] **Step 1: Write failing tests** — convert C→VP at 3:1 base and 2:1 for alchemists (plus VP→C only for alchemists), burn moves bowls, send-priest slot accounting (priest_pool decrements; 4th priest to a full track advances 1 and returns? — port the exact rule first, then write the test to it), pending-queue LIFO/FIFO ordering (FIFO), active_faction override by queue head, `apply` rejects out-of-turn non-exempt moves.

- [ ] **Step 2-4: Red → implement → green. Step 5: Commit** — `git commit -m "feat: apply dispatch, decision queue, conversions, priest-to-cult"`

---

### Task 8: Build, upgrade, bridge + leech (`actions_build.py`, `leech.py`)

**Files:**
- Create: `src/bgai/engine/tm/actions_build.py`, `src/bgai/engine/tm/leech.py`
- Test: `tests/test_actions_build.py`, `tests/test_leech.py`

**Interfaces:**
- Consumes: apply registry (Task 7), connectivity, towns, tiles.
- Produces:
  - `handle_build(state, faction, cmd)`: dwelling on reachable + correct-color hex, pay cost, D-count cap, score-tile build VP (`vp_mode == "build"`, `vp` key `D`), FREE_D marker consumption (Witches' ride — set by Task 9), setup-phase builds free & unrestricted-reachability, town check, **enqueue leech offers**.
  - `handle_upgrade(state, faction, cmd)`: D→TP (cost doubles without adjacent opponent — port the neighbour discount test), TP→TE/SH, TE→SA, building-count caps, GAIN_FAVOR pendings (count from `build_gain`, Chaos Magicians 2), on-SH hooks (Halflings `halflings_spades` pending ×3, Darklings `convert_w_to_p` pending (cap 3, `strict_darkling_sh` semantics), faction ACTx grants, Alchemists +12 PW, Cultists +7 VP, Mermaids free ship level...) — all driven by `build_gain` data + `HOOKS[faction].on_stronghold_built`, score-tile VP, town check, leech offers.
  - `handle_bridge(state, faction, cmd)`: place bridge from pending BRIDGE marker (ACT1/ACTE set it), validate river-crossing geometry per `map.pm` bridge rules, BRIDGE_COUNT cap.
  - `handle_gain_favor(state, faction, cmd)`: pop the matching `gain_favor` pending, take `cmd.tile` from `favors_pool` (must be >0; one copy of each FAV per faction max), advance its cult steps via `cults.advance`, record passives (town-size-6, per-TP-build VP etc. are read from `FactionState.favors` by the mechanics that care).
  - `handle_gain_town(state, faction, cmd)`: pop the matching `gain_town` pending, take `cmd.tile` from `towns_pool`, apply via `towns.apply_town_tile`.
  - `leech.py`: `offers_for_build(state, builder, hex) -> tuple[PendingDecision, ...]` — for each other faction with buildings directly adjacent (incl. bridges): power = sum of `building_power_value`, capped by `Power.gainable`; zero-cap → auto-skip (no pending; but check acting.pm: snellman still emits offers a faction *could* partially use); enqueue clockwise in seat order starting after builder. `handle_leech` (accept: `Power.gain(n)`, pay `n-1` VP, VP floor rules per strict-leech), `handle_decline`. Cultists: `HOOKS["cultists"].on_leech_resolved` — after ALL offers from one build resolve: any accepted → cultists get `cult_choice` pending (1 step, choice arrives as a `gain_cult` bookkeeping row — wire in Task 10's row handling); all declined → +1 power (`errata_cultist_power`).

- [ ] **Step 1: Read the reference first** — `acting.pm` leech flow + `strict-leech` grep, `commands.pm` build/upgrade validation, `resources.pm` leech VP payment. Transcribe semantics into module docstrings before coding.

- [ ] **Step 2: Write failing tests** covering: build pays W+C and places building; build on wrong color rejected; upgrade D→TP neighbour discount both ways; TE upgrade enqueues `gain_favor`; leech offers enqueue clockwise with correct amounts (build a 3-faction hand state where two factions adjoin); accept pays VP; decline is free; cultists cult_choice after any accept; cultists +1 PW after all-decline.

- [ ] **Step 3-5: Red → implement → green. Step 6: Commit** — `git commit -m "feat: build/upgrade/bridge with leech offer queue and cultists hooks"`

---

### Task 9: Terraform + dig (`actions_terraform.py`)

**Files:**
- Create: `src/bgai/engine/tm/actions_terraform.py`
- Test: `tests/test_actions_terraform.py`

**Interfaces:**
- Consumes: `spade_distance`, dig tracks, hooks.
- Produces:
  - Spade bookkeeping on a transient `spades_available` field (add to `FactionState`, default 0): `dig N` converts resources → N spades (Darklings: priests, +2 VP each, `dig_gain`; Halflings passive +1 VP/spade; round-tile SPADE gain VP; Alchemists SH +2 PW/spade); `transform HEX [to color]` spends spades = `spade_distance` (Giants: always 2, always to home color — `HOOKS["giants"].spade_transform_target`); leftover-spade `lose_spade` rows zero the balance; `-FREE_TF` markers (ACTN sandstorm) and ACT5/ACT6-granted spades feed the same balance.
  - `transform` without spades but with pending spade source → consume from pending (`spade_use`).
  - Building on a just-transformed hex happens via a separate `build` row — no coupling needed beyond hex color.

- [ ] **Steps: red tests** (standard dig 3W→1 spade at level 0; transform 1 step consumes 1 spade; giants always 2; darklings dig pays P and gains 2VP; halflings VP passive; transform onto occupied hex rejected) **→ implement → green → commit** — `git commit -m "feat: terraform/dig with faction spade behaviors"`

---

### Task 10: Power actions + specials (`actions_power.py`)

**Files:**
- Create: `src/bgai/engine/tm/actions_power.py`
- Test: `tests/test_actions_power.py`

**Interfaces:**
- Consumes: `POWER_ACTIONS`, `FACTION_SPECIAL_ACTIONS`, bonus/favor special actions, hooks.
- Produces: `handle_action(state, faction, cmd)` for `cmd.tile` in:
  - `ACT1-6`: pay `cost_power` from bowl3, mark `power_actions_taken` (blocking, once per round globally), gains: ACT1 bridge marker, ACT2 priest, ACT3 workers, ACT4 coins, ACT5/6 spades (→ spade balance; a follow-up `transform`/`build` row consumes them).
  - `ACTA/ACTC/ACTE/ACTG/ACTN/ACTS/ACTW`: per-faction once-per-round (`FactionState.actions_used`), effects per `FACTION_SPECIAL_ACTIONS` (ACTA: 2 cult steps one track — steps arrive as `gain_cult` rows; ACTC: double turn — set a `double_turn` flag consumed by turn advancement in Task 11, `strict_chaosmagician_sh` semantics from reference; ACTE: 2W → bridge marker; ACTG: 2 free spades to home; ACTN: FREE_TF marker (direct adjacency constraint); ACTS: FREE_TP marker consumed by a following `upgrade` row; ACTW: FREE_D marker consumed by a following `build` row).
  - `BONx`/`FAVx` action ids (BON1 spade, BON2 cult, FAV6-class cult): once per round via `actions_used`.

- [ ] **Steps: red tests** (ACT6 requires 6 in bowl3 and blocks second take; ACT2 priest respects priest_pool; ACTW marker then free build anywhere green; per-round reset covered in Task 11) **→ implement → green → commit** — `git commit -m "feat: power actions and faction/bonus special actions"`

---

### Task 11: Pass, income phase, cleanup, round flow (`actions_pass.py`, `round_flow.py`)

**Files:**
- Create: `src/bgai/engine/tm/actions_pass.py`, `src/bgai/engine/tm/round_flow.py`
- Test: `tests/test_actions_pass.py`, `tests/test_round_flow.py`

**Interfaces:**
- Consumes: income (Task 4), tiles, hooks, everything above.
- Produces:
  - `handle_pass(state, faction, cmd)`: pass-VP from held bonus tile + favors (transcribed pass_vp data; Engineers SH bridge VP via `HOOKS["engineers"].pass_vp_extra`), return old bonus + take `cmd.tile` (with accumulated `bonus_coins`), append to `passed_order`, mark passed; round 6: no tile taken (`pass` bare).
  - `handle_advance(state, faction, cmd)`: ship/dig track advance, costs + advance_vp, Mermaids free-ship path (from SH `GAIN_SHIP`), Fakirs SH carpet range.
  - `handle_connect(state, faction, cmd)`: Mermaids river-hex town designation (`connect r9`) → town check with that river hex bridging.
  - `round_flow.py`:
    - `handle_income_row(state, faction, cmd)` for `other_income_for_faction` / `cult_income_for_faction` / `all_income_for_faction`. **Step 1 of this task is empirical**: for 3 games, join `moves` and `deltas` on income rows and determine exactly which Task-4 categories each row type grants (hypothesis: `cult_income` = cult-track-derived? favors? — settle it against the data + grep `cult_income_for_faction` in the reference `src/`). Document the finding in the module docstring; `merge_income_phases` option = single `all_income` row.
    - Turn advancement after each completed action (skip passed factions; `double_turn` flag; queue interrupts already handled by `active_faction`).
    - `end_of_round(state) -> GameState`: score-tile cult rewards (`req`-based integer division per tile `cult_income`), reset `power_actions_taken`/`actions_used`/spade balances, bonus_coins +1 on untaken tiles, turn_order ← passed_order (`variable_turn_order`; else keep seat order), phase → INCOME of next round or FINISHED-scoring path after round 6.
    - Setup-phase flow: `SETUP_DWELLINGS` (snake order per `acting.pm` setup_order: normal order once, reverse once, Nomads third D, Chaos Magicians single D last) → `SETUP_BONUS` (reverse-order pass rows) → round 1 INCOME.

- [ ] **Steps:** empirical income-split step (documented), then red tests: pass-VP for BON7 with 2 TP (exact transcribed value); bonus swap returns accumulated coins; passed faction skipped in rotation; end_of_round grants `4 WATER -> 1 SPADE` as spade balance...; cult reward integer division (7 EARTH, req 4 → 1 unit); variable turn order applied; setup snake order for the reference game reproduces ledger rows 28-36 exactly (engineers E7, darklings G5, nomads D3, mermaids D2, mermaids D5, nomads F3, darklings E10, engineers F6, nomads G4). **Red → implement → green → commit** — `git commit -m "feat: pass/advance, income phase, cleanup, setup and round flow"`

---

### Task 12: Final scoring (`scoring.py`)

**Files:**
- Create: `src/bgai/engine/tm/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `clusters` (network per towns rules incl. shipping/bridges — port `compute_network` scoring semantics from `scoring.pm` exactly: which connectivity counts for final network, Dwarves/Fakirs do NOT tunnel for network? — transcribe), cult standings.
- Produces: `final_scoring(state) -> GameState` — per cult: 8/4/2 VP for top three positions with equal-split-on-ties (integer floor per `scoring.pm`), position 0 scores nothing; network: 18/12/6 with same tie handling, size = largest connected building group; `score_resources` per faction: leftover C/W/P (+bowl power?) → VP conversions per `scoring.pm` `score_resources` (3 resources → 1 VP class rules; Alchemists 2C→1VP? use exchange rates — transcribe exactly).
- Also produces `handle_score_vp(state, faction, cmd)` registered for `score_vp` rows (`+N vp for X`) — validation anchors: engine computes these itself; the handler asserts the engine's just-computed grant matches `cmd.n1`/`cmd.reason` (this is how scoring bugs surface as loud errors rather than silent drift). Same pattern for `score_resources` rows.

- [ ] **Steps: red tests** (tie-split arithmetic: two factions tied 1st on FIRE → (8+4)//2 each, third gets 2; network tie; resource conversion for a hand-built end state; verify against reference game final VPs from `games_meta.final_vp` once replay exists — deferred assertion noted in test as skip until Task 13) **→ implement → green → commit** — `git commit -m "feat: final scoring (cults, network, resources)"`

---

### Task 13: Replay harness (`replay.py`)

> Deviation from the design spec: the spec suggested starting with the reference repo's `test/testgame*.txt` fixtures. Those are raw ledger text needing a separate loader; since our harness consumes the already-parsed parquet datasets and the deltas oracle, we start directly with crawled games — same coverage, one fewer input format.

**Files:**
- Create: `src/bgai/engine/tm/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: everything; `moves.parquet`, `deltas.parquet`.
- Produces:
  - `@dataclass(frozen=True) class Mismatch: game_id: str; row: int; faction: str; field: str; expected: str; actual: str; raw: str`
  - `@dataclass(frozen=True) class ReplayResult: game_id: str; rows_checked: int; mismatches: tuple[Mismatch, ...]; error: str | None` (error = exception with row context if apply raised).
  - `replay_game(game_id: str, moves_df, deltas_df) -> ReplayResult`: build `GameState.initial(load_setup(game_id))`; iterate move rows grouped by `row` ordered by (`row`, `seq`); `apply` each; after the last seq of a row, if a deltas row exists for (row, acting faction), compare `vp/c/w/p` absolutes, `pw` string, `cult` string; collect mismatches (configurable `stop_after: int = 5` mismatches per game).
  - CLI: `uv run python -m bgai.engine.tm.replay --limit 10 [--game-id X] [--report out.json]` → per-game pass/fail, per-faction row pass rates, mismatch dump sorted by frequency of (verb, field) so the most common bug class is on top.

- [ ] **Step 1: Write the failing test**

```python
"""tests/test_replay.py"""
import polars as pl
import pytest
from bgai.engine.tm.replay import replay_game

@pytest.fixture(scope="session")
def frames():
    return (pl.read_parquet("data/datasets/moves.parquet"),
            pl.read_parquet("data/datasets/deltas.parquet"))

def test_reference_game_replays_clean(frames) -> None:
    r = replay_game("4pLeague_S10_D1L1_G1", *frames)
    assert r.error is None
    assert r.mismatches == ()
    assert r.rows_checked > 200
```

- [ ] **Step 2: Run — it will fail with real engine bugs, not just ImportError.** This is the point: Task 13's loop is *diagnose → fix in the owning module (with a pinned unit test) → rerun* until the reference game is clean. Every fix commits separately (`fix: <mechanic> — <game_id> row <n>`).

- [ ] **Step 3: Extend to 10 games** — add `test_first_ten_games_replay_clean` iterating the first 10 game_ids from `games_meta` sorted; same loop.

- [ ] **Step 4: Commit harness** — `git commit -m "feat: replay harness with per-row delta oracle and mismatch triage CLI"`

---

### Task 14: Corpus rollout to zero mismatches

**Files:**
- Modify: whichever engine modules the mismatches implicate; every fix adds a pinned unit test in that module's test file.
- Create: `tests/test_replay_corpus.py` (marked `@pytest.mark.slow`)

**Process (iterative — this task is a loop, not a step list):**

- [ ] **Batch 100**: `uv run python -m bgai.engine.tm.replay --limit 100 --report /tmp/replay100.json`. Triage by (verb, field) frequency; fix the top class; rerun. Repeat until 100/100 clean. The ≤8-game exotic options (`loose-dig`, `merge-income-phases`, `loose-convert-phase`, `loose-cult-loss`, `loose-lose-cult`) may be excluded here and handled last.
- [ ] **Full corpus**: all 3,563 games (runtime target: minutes, pure Python is fine; parallelize with `concurrent.futures.ProcessPoolExecutor` over games if >10 min). Report per-faction pass rates — all 14 factions must hit 100% of games containing them (asymmetry requirement from the design spec).
- [ ] **Final-VP cross-check**: after each clean replay, also assert final VP per faction equals `games_meta.final_vp` — catches final-scoring bugs the last delta row can't see.
- [ ] **Pin**: `tests/test_replay_corpus.py` replays a fixed 25-game regression set (every faction ≥3 appearances, every exotic option ≥1 game) in normal CI; full corpus stays a manual/slow command documented in README.
- [ ] **Commit** — `git commit -m "feat: full-corpus replay green (3563/3563 games, zero delta mismatches)"` (only when literally true; otherwise commit progress honestly, e.g. `fix: ... (3540/3563 clean)`).

---

### Task 15: `legal_moves` (`legal.py`)

**Files:**
- Create: `src/bgai/engine/tm/legal.py`
- Test: `tests/test_legal.py`

**Interfaces:**
- Consumes: full engine.
- Produces: `legal_moves(state: GameState) -> tuple[ParsedCommand, ...]` for the active faction: enumerate pending-decision answers when queue nonempty (leech/decline pairs, favor picks from pool, town picks, ...); otherwise all legal main actions (builds on reachable+affordable hexes, upgrades, affordable power/special actions, digs/transforms, priest sends with open slots, advances, conversions — conversions enumerated coarsely: each single-unit exchange, pass with each available bonus tile).

- [ ] **Step 1: Red tests** — hand states: passed faction only generates nothing (skipped); leech pending → exactly `{leech n, decline}`; setup phase → only valid dwelling spots.
- [ ] **Step 2: Containment validation** — test: for 5 replayed games, every applied DECISION-kind move (modulo conversion amounts >1 modeled as repeats) is in `legal_moves` of its pre-state; and `apply` succeeds on every generated legal move for 200 sampled states (generative smoke: no exceptions, no negative resources).
- [ ] **Step 3: Implement → green → commit** — `git commit -m "feat: legal move generation validated by replay containment"`

---

## Execution notes

- Task order is strict: 1→2→3→4→(5,6 parallelizable)→7→8→9→10→11→12→13→14→15.
- Tasks 8-12 will not be perfectly right on first green — Task 13/14's oracle loop is where truth lands. Keep unit tests as pins, not as proofs.
- When a reference-Perl reading contradicts this plan's summary of a rule, **the Perl wins** — update the plan file in the same commit as the fix.

## Post-implementation corrections

The following items in this plan file were superseded by implementation and replay validation:

- **Task 3, line 222 — `active_faction` definition**: The plan specifies `active_faction(state) -> str` as "head of pending if any, else `turn_order[active_index]`". The actual implementation is `turn_order[active_index]` (pending does not affect active-faction slot). See `state.py`'s `active_faction` docstring.

- **Task 3, line 219 — `PendingDecision.kind` values**: The plan lists "halflings_spades", "bonus_choice", "spade_use" as produced kinds. These were plan-stage design artifacts; the actual live set is documented in `state.py`'s `PendingDecision` docstring (produced kinds: "leech", "gain_favor", "gain_town", "cult_choice", "convert_w_to_p", "bridge", "free_d", "free_tp", "free_tf", "cultist_leech_watch").

- **Task 15, line 95 onwards (Design decision 4)**: The plan's docstring originally stated the order-exempt baseline is suppressed when an outstanding blocking pending exists. The actual implementation is additive: `legal_moves_for` returns both the baseline (convert/wait) and pending answers, combined. See `legal.py`'s module docstring and `legal_moves_for` implementation.
