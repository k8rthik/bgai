# Phase 5: Imitation Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A faction-conditioned policy/value net trained on the Div 1–3 corpus's ~1.25M decision rows, wrapped as an `ImitationAgent`, that beats both Phase 4 baselines in mirrored arena play; held-out top-1/top-3 accuracy reported against expert moves.

**Architecture (design decisions, made 2026-08-04 under an autonomy mandate):**
- **Candidate-scoring policy**, not a fixed action space: the net embeds the encoded state and each legal candidate move, scores by dot product, softmaxes over the candidates actually on offer. This sidesteps TM's huge sparse command space, gives legal-move masking for free, and reuses the arena's canonical move ordering as a stable candidate indexing. Cross-entropy over candidates + a 4-seat value head (final-VP share, mover-relative) trained jointly.
- **Encoding v1** (`ENCODING_VERSION = 1`, stamped into every shard): per-hex int8 feature planes over the 113-hex base board (terrain one-hot, building type one-hot, owner as mover-relative seat one-hot) + global scalars (per-faction resources/power/tracks/cults/VP/passed + bonus/favor/town holdings, mover-relative seat order; round/phase one-hots; 6 score tiles; power-actions-taken; mover faction and opponent faction one-hots). Compact storage; featurization to float32 happens in the data loader.
- **Extraction by replay**: a capture callback added to the engine's `replay_game` yields (state, faction, chosen cmd) before each `Kind.DECISION` row command; candidates = `legal_moves_for` at that state, canonically sorted; rows with <2 candidates are skipped (no choice made). Only clean games (the arena sampler's 3,374).
- **Split/weights**: time-based by season parsed from game_id — seasons ≥ 67 are validation (no future leakage); sample weights Div1 1.0 / Div2 0.8 / Div3 0.6.
- **Model v1**: MLP torso (state vec → 2×1024 SiLU+LayerNorm → 256-d state embedding), move-feature MLP (→ 256-d), dot-product scores; value head 256→4. ~5M params, bf16-capable, runs on MPS (M3 Pro) now and CUDA later unchanged.
- **Cluster-portable**: plain PyTorch, config dataclass, checkpoint/resume, device auto-pick (cuda > mps > cpu), no hard-coded paths.

**Tech Stack:** Python 3.12, uv, pytest, polars, numpy (existing). New: `torch` (>=2.4).

## Global Constraints

- Tests: `uv run pytest <path> -x -q` from repo root; slow/GPU-dependent tests marked with the existing `slow` marker (`pyproject.toml` `addopts = "-m 'not slow'"`).
- IMMUTABILITY for domain types; training loop internals (optimizer state, buffers) are exempt by nature.
- Files < ~400 lines; `from __future__ import annotations`; full type annotations.
- Engine modified ONLY in Task 2 (`replay.py` capture hook). Arena untouched except `agents/__init__.py` export.
- Artifacts live under gitignored `data/`: shards in `data/datasets/imitation/`, checkpoints in `data/checkpoints/`.
- Reuse (do not reimplement): `bgai.arena.sim._canonical` — PROMOTE to public `canonical_moves` in Task 1 (sim keeps an alias); `bgai.engine.tm.legal.legal_moves_for`; `bgai.arena.setups.clean_game_ids`; `bgai.engine.tm.board.base_board()` for the hex universe; `bgai.data.ledger_parser.{ParsedCommand, Kind}`; `bgai.agents.base.Agent`.
- Verify at execution time (not assumed): exact `FactionState` holdings fields for bonus/favors/towns; `GameState` fields for score tiles/pool/power-actions; replay's row-command iteration point for the hook; season regex `S(\d+)` against all 3,374 clean ids.

## File Map

| File | Responsibility |
|---|---|
| `src/bgai/training/vocab.py` | Frozen vocabularies: hex index, verbs, tiles, colors, cults, resources, factions; `ENCODING_VERSION` |
| `src/bgai/training/encode_state.py` | `encode_state(state, faction) -> StateEncoding` (int8/int16 numpy) |
| `src/bgai/training/encode_move.py` | `encode_move(cmd, state, faction) -> np.ndarray` (int16 field ids), `MOVE_FEAT_DIM` |
| `src/bgai/training/extract.py` | replay-capture → per-game decision records |
| `src/bgai/training/dataset_build.py` | CLI: shards (npz) + manifest (split, weights, version) |
| `src/bgai/training/dataset.py` | torch `Dataset`/collate: memmap shards → float tensors + masks |
| `src/bgai/training/model.py` | `PolicyValueNet` (torso, move encoder, dot-product policy, value head) |
| `src/bgai/training/train.py` | CLI: config dataclass, loop, checkpoint/resume, device auto, val top-1/top-3 |
| `src/bgai/agents/imitation.py` | `ImitationAgent` (checkpoint → `Agent.choose`) |
| Modify | `src/bgai/engine/tm/replay.py` (+`on_decision` hook), `src/bgai/arena/sim.py` (promote `_canonical`), `src/bgai/agents/__init__.py` |

Tests: `tests/test_vocab.py`, `tests/test_encode_state.py`, `tests/test_encode_move.py`, `tests/test_extract.py`, `tests/test_dataset_build.py`, `tests/test_training_dataset.py`, `tests/test_model.py`, `tests/test_train_smoke.py`, `tests/test_imitation_agent.py`, gate in `tests/test_arena_verify.py` (extended, slow).

---

### Task 1: Vocabularies + canonical-move promotion (`training/vocab.py`)

**Interfaces produced:** `HEXES: tuple[str, ...]` (sorted `base_board()` land+river keys), `HEX_INDEX: dict[str,int]`; `VERBS`, `TILES` (BON/FAV/TW/ACT + faction ACTx), `COLORS`, `CULTS4`, `RESOURCES`, `FACTION_NAMES` with `*_INDEX` dicts; `ENCODING_VERSION = 1`. `bgai.arena.sim.canonical_moves` public (alias `_canonical` kept for callers).

Steps: (1) failing test — index round-trips, 113 land hexes present, every corpus verb in `VERBS` (assert against `moves.parquet` distinct verbs), every tile id in `TILES` (distinct non-null `tile` column); (2) RED; (3) implement (vocab built from engine constants — `FACTIONS`, `BONUS_TILES`/`FAVOR_TILES`/`TOWN_TILES`/`POWER_ACTIONS` — plus parquet-verified literals); (4) GREEN; (5) commit `feat: training vocabularies and canonical-move promotion`.

### Task 2: Replay capture hook (`engine/tm/replay.py`)

**Interfaces produced:** `replay_game(..., on_decision: Callable[[GameState, str, ParsedCommand], None] | None = None)` — invoked immediately before applying any command whose parquet `kind == "decision"`, with the pre-apply state and acting faction. Zero behavior change when `None` (default): assert one game's `ReplayResult` unchanged with hook set vs unset, and hook receives >100 calls for the reference game `4pLeague_S10_D1L1_G1` with first call during setup.

Steps: failing test (`tests/test_extract.py::test_capture_hook_transparent`) → RED → locate the single apply site in `_apply_row_commands` and thread the callback through `replay_game` → GREEN → full replay fast suite → commit `feat: decision-capture hook on replay_game`.

### Task 3: State encoder (`training/encode_state.py`)

**Interfaces produced:**
```python
@dataclass(frozen=True)
class StateEncoding:
    hex_planes: np.ndarray   # (113, HEX_FEAT_DIM) int8
    globals: np.ndarray      # (GLOBAL_DIM,) int16  (counts, not normalized)
HEX_FEAT_DIM: int; GLOBAL_DIM: int
def encode_state(state: GameState, faction: str) -> StateEncoding
```
Mover-relative convention: seat slots ordered `[mover, next clockwise, ...]` from `state.setup.factions` rotated to the mover. Tests: shapes/dtypes; determinism; spot-checks on the reference game's initial state (engineers mover: own D count 0 pre-setup, coin/worker exact values from `FACTIONS` data; a placed dwelling flips exactly one hex plane bit); encoding differs between movers.

Commit `feat: v1 state encoder (hex planes + global scalars, mover-relative)`.

### Task 4: Move featurizer (`training/encode_move.py`)

**Interfaces produced:** `encode_move(cmd, state, faction) -> np.ndarray` shape `(MOVE_FIELDS,)` int16 — categorical ids: verb, loc, loc2, building, tile, cult, color, target (mover-relative seat of target faction), res1, res2, plus clamped n1/n2; missing → 0 (reserved null id, real ids offset by 1). Tests: every distinct corpus decision verb encodes without KeyError (drive over one full game's captured commands); null handling; determinism.

Commit `feat: move featurizer`.

### Task 5: Extraction + shard builder (`training/extract.py`, `training/dataset_build.py`)

**Interfaces produced:**
```python
@dataclass(frozen=True)
class DecisionRecord:
    hex_planes: np.ndarray; globals: np.ndarray
    candidates: np.ndarray  # (n_cand, MOVE_FIELDS) int16, canonical order
    chosen: int             # index into candidates
    season: int; division: int; mover_faction_id: int
def extract_game(game_id: str) -> list[DecisionRecord]   # skips n_cand < 2
```
`dataset_build.py` CLI: `--limit N --out data/datasets/imitation/` → npz shard per 200 games (ragged candidates stored flat + offsets) + `manifest.json` (`ENCODING_VERSION`, game ids, record counts, season split boundary 67, division weights {1:1.0, 2:0.8, 3:0.6}). Multiprocessing over games. **Chosen-index invariant test:** for a full game, each record's `candidates[chosen]` equals `encode_move` of the actually-replayed command (establishes containment + canonical-order agreement end-to-end). Build a 20-game mini-set in tests (tmp dir); assert manifest counts.

Commit `feat: decision extraction and imitation shard builder`.

### Task 6: Torch dataset + model (`training/dataset.py`, `training/model.py`)

`uv add torch` first. **Interfaces produced:** `ImitationDataset(shard_dir, split: "train"|"val")` memmaps shards, returns dict of tensors; `collate(batch)` pads candidates to batch max with a bool mask. `PolicyValueNet(cfg)` with `forward(hex_planes, globals, faction_id, cand_feats, cand_mask) -> (logits, value4)`; logits masked with `-inf` at padding. Tests (CPU): forward shapes on synthetic batch; masked positions never win argmax; loss decreases over 50 optimizer steps on a 64-sample synthetic overfit task (learning sanity, seconds).

Commit `feat: imitation dataset loader and policy/value net`.

### Task 7: Training loop (`training/train.py`)

**Interfaces produced:** `TrainConfig` frozen dataclass (paths, lr 3e-4 AdamW, batch 512, epochs, val_every, device: `"auto"`, seed); CLI `python -m bgai.training.train --shards ... --out data/checkpoints/imitation_v1 --epochs 3`; checkpoint = `{model, optimizer, config, ENCODING_VERSION, step}` with resume; per-epoch val top-1/top-3 + weighted CE printed and appended to `metrics.jsonl`. Division/sample weights applied via per-sample CE weighting. Smoke test (fast suite): 200 steps on the Task 5 mini-set overfits to >60% train top-1 (device=cpu, <60s).

Commit `feat: cluster-portable imitation training loop`.

### Task 8: Build full dataset + train v1 (execution, no new code)

`uv run python -m bgai.training.dataset_build --out data/datasets/imitation/` (all clean games, ~10 min multiprocessed) then `uv run python -m bgai.training.train ... --epochs 3` on MPS in the background (`run_in_background`; expect tens of minutes). Record: dataset size, val top-1/top-3. **No fixed accuracy gate in the master plan — report honestly.** Reference points: random-among-candidates baseline (1/mean-branching) and a most-common-verb heuristic, both computed from the val shards for context.

### Task 9: ImitationAgent + arena gate (`agents/imitation.py`)

**Interfaces produced:** `ImitationAgent(checkpoint_path, name="imitation", temperature=0.0)` implementing `Agent.choose` — encodes state + offer (already canonical), batch-scores, argmax (temperature>0: softmax sample via the arena rng). Fast test with a randomly-initialized net (no checkpoint needed): returns a member of the offer, deterministic at temperature 0. Gate (extends `tests/test_arena_verify.py`, slow, auto-skip with a clear message if the checkpoint file is absent): imitation vs greedy vs random×2 base seats over ≥50 tables — imitation mean placement beats greedy's by ≥0.15 and random's by ≥0.5; zero errors. Also measure ms/decision (must stay <1s/game).

Commit `feat: ImitationAgent` + `test: pin Phase 5 arena gate`.

### Task 10: Docs + memory

README Phase 5 section (dataset command, training command, val metrics, arena numbers, MPS/CUDA note); master plan Phase 5 marked with results; memory file `tm-imitation-status`. Commit `docs: Phase 5 imitation results`.

---

## Self-Review

- Master-plan Phase 5 coverage: hex planes + globals ✓ (T3), faction-conditioned policy over legal moves ✓ (candidate scoring, faction id input, T6), 4-player value vector ✓ (T6), decision-level Div 1–3 filter ✓ (corpus is only Div 1–3; division weights T5), table-quality weighting ✓ (division weights), time-based split ✓ (season ≥67 val, T5), held-out top-1/top-3 ✓ (T7/T8), beats baselines in arena ✓ (T9 gate). Faction *draft* modeling explicitly out of scope (deferred past Phase 4 by standing decision; the arena assigns factions).
- Placeholder scan: interfaces named with exact signatures; numeric hyperparameters stated; no TBDs.
- Type consistency: `StateEncoding`/`DecisionRecord` field names match between T3/T5/T6; `encode_move` consumed by T5 and T9; `canonical_moves` promoted once (T1) and used by T5/T9.
- Known risk, stated: 18 GB RAM / MPS box — model sized ~5M params, shards memmapped; if 3-epoch training exceeds the session's patience it runs in the background and the gate test auto-skips until the checkpoint exists (honest-report requirement in T8).
