# bgai

Board game AI series. Game 1: **Terra Mystica**.

Two AI tracks, cross-validated against each other and against real humans:

1. **Engine AI** — imitation learning on high-quality human tournament games
   (tmtour Divisions 1–3, played on [terra.snellman.net](https://terra.snellman.net)),
   then MCTS search, then optional human-regularized self-play RL.
2. **LLM AI** — an LLM agent pushed up a capability ladder: bare prompt → knowledge-loaded
   context → engine tools → retrieval over expert games → test-time search → LLM-guided
   MCTS → fine-tuned models.

Layout:

```
src/bgai/
  data/       # bulk-feed ingest, polite log crawler, ledger parser, dataset builders
  engine/tm/  # rules engine: faction-agnostic core + factions/<name>.py hook modules
  agents/     # Agent interface + random / heuristic / imitation / mcts / llm
  arena/      # match runner, faction/seat rotation, TrueSkill reports
  training/   # encoders, nets, imitation + RL loops (cluster-portable)
tests/        # replay-oracle suite is the backbone
data/         # gitignored: raw game cache, parquet datasets
```

Data sources: game logs from terra.snellman.net (fetched politely: ≤1 req/s,
identifying User-Agent, resumable cache); tournament structure from tmtour.org's
open API. Thanks to Juho Snellman and the tmtour maintainers for keeping these
running.

## Replay harness

`src/bgai/engine/tm/replay.py` drives a crawled game's ledger
(`data/datasets/moves.parquet`) through `apply()` and cross-checks every row
against the `deltas.parquet` oracle. Normal `pytest` runs cover a fast
10-game smoke check (`tests/test_replay.py`) plus a curated 25-game
regression set with full faction/exotic-option coverage
(`tests/test_replay_corpus.py`, `.superpowers/sdd/2026-08-03-tm-engine-core/task-14-corpus-prep.md`
documents the selection).

To replay the **full corpus** (3563 games, ~70s pure Python, no
`--jobs`/parallelization needed) as a manual check outside normal CI:

```bash
uv run python -m bgai.engine.tm.replay --limit 4000 --report /tmp/replay_full.json
```

Prints per-faction row pass rates and a (verb, field) mismatch-frequency
summary; `--report` additionally writes the full per-game mismatch/error
detail as JSON. `--game-id <id>` replays a single game. 10 games in the
corpus have a `nofaction*` placeholder seat (`load_setup` raises `ValueError`
for them — an expected, documented exclusion, not an engine bug) and are
skipped automatically (reported with `error` set to the `ValueError` text,
`rows_checked=0`).

As of task 14 (phase 4), **3552/3553 loadable games** replay with zero
delta-oracle mismatches and zero errors. The 10 `nofaction*` games are
skipped as expected (fail to load with `ValueError`). One documented
anomaly, `4pLeague_S53_D1L1_G3`, is tracked in `tests/test_replay_corpus.py`'s
`KNOWN_ANOMALIES` (Cultists receive a duplicate cult-score row within the
same scoring block — snellman's own ledger contains a state that the
scoring algorithm cannot produce, treated as a server-side anomaly).

**Validated scope — 4 players only:** The entire corpus of 3,563 games
consists exclusively of 4-player games (every game_id starts with `4pLeague`).
Player-count-dependent machinery (bonus-tile pool size = `player_count + 3`,
turn rotation, leech seat order) is oracle-validated only at 4 players; the
reference-rules implementations for 2-player, 3-player, and 5-player play are
correct per specification but have never been replay-validated against real
tournament data.

## Arena (Phase 4)

`src/bgai/agents/` holds the `Agent` protocol (one method: pick a
`ParsedCommand` from the offered tuple — the engine's pending-decision
queue makes leech answers, favor/town picks, and setup placements
ordinary moves) plus the two baselines: `RandomAgent` and a one-ply
`GreedyAgent`. `src/bgai/arena/` plays complete headless games between
agents:

```
uv run python -m bgai.arena.run --tables 50 --seed 20260804 \
    --agents random,greedy --report /tmp/arena.html
```

One *table* = one corpus-sampled setup (real Div 1–3 game configuration
via `load_setup`, drop history cleared; no synthetic setup generator)
played once per mirrored seat rotation (4 games), so every agent
occupies every seat and faction equally often. Ratings are TrueSkill
(faction/seat covariates reported separately per the master plan); the
HTML report lists ratings, per-faction/per-seat mean placement, and
every error verbatim.

**Phase 4 gate** (pinned as a slow test, `tests/test_arena_verify.py`):
greedy ≫ random over 200 mirrored games — mean placement 1.02 vs 1.91
(0-based ranks), zero errored games, seed 20260804. Headless 4p games
run at ~50–60 ms (target was <1s; no optimization warranted yet).

Random-play arena fuzzing doubles as a `legal_moves` soundness check —
it found two offered-but-rejected move classes the corpus containment
sweep cannot see (Giants non-home transforms, ACTN fold-in builds
skipping the dwelling-cost check), both fixed and pinned in
`tests/test_legal_soundness.py`.
