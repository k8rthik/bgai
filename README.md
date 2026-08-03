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
