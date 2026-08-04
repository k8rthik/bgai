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
