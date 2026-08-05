# LLM Agent Track, Phase 1 — MCP Harness + Tool Ladder Design

**Date:** 2026-08-04
**Status:** Approved design, pending implementation plan
**Depends on:** TM engine core (merged 2026-08-04, corpus-validated 3552/3553, legal move generation validated by replay containment)

## Goal

Measure the playing strength of a frozen LLM (Claude) piloting Terra Mystica
seats, pushed up a ladder of tool rungs. Success metric: win rate / TrueSkill
against baseline opponents in the arena, per rung configuration. Strategic
commentary is incidental; strength is the deliverable.

## Core principle: facts vs. judgment

**The engine provides facts. The LLM provides all judgment.**

Engine-side tools never rank moves, score desirability, recommend actions, or
expose any evaluation signal (no value functions, no heuristic evals, no
rollout outcome statistics). They expose only:

- game state and history,
- legal move enumeration,
- deterministic arithmetic (resource deltas, power-leech offers, terraform
  costs),
- mechanically guaranteed projections (e.g., "if the round ended now, the
  round-scoring tile pays 6 VP; current cult standings award 4/2/1").

This is the bookkeeping a human expert does on scratch paper — the exact
thing LLMs are unreliable at — while every "is this good?" decision remains
the LLM's own reasoning. Rationale: this keeps the research object clean
(how strong is Claude's TM judgment when amplified by tools and compute?)
instead of laundering an engine evaluation through the LLM's mouth.

## Architecture

One MCP server is the single interface to the engine. Two frontends drive it:

1. **Interactive:** the user tells Claude Code "pilot this game"; Claude uses
   the MCP tools directly. This is the dev/debug loop.
2. **Headless arena:** a runner script spawns `claude -p "<pilot prompt>"`
   (Claude Code CLI with the MCP server configured) once per game per LLM
   seat, instead of raw API calls. Win rates are aggregated across batches.

Budget target: moderate — roughly $1–5 and ≤30 s/move per game, ~50–100
arena games per rung configuration.

### Components

```
src/bgai/
  agents/      # Agent protocol + random + heuristic baseline bots
  arena/       # match runner, faction/seat rotation, win-rate + TrueSkill reports
  mcp/         # MCP server: game sessions, tool rungs, config gating
  knowledge/   # corpus-stats mining + compendium generation pipeline
scripts/       # headless arena driver (spawns claude -p per game)
docs/knowledge/tm-compendium/   # generated faction playbooks (committed)
```

**`agents/` — baselines (prerequisite, currently empty):**

- `Agent` protocol: `choose_move(state, legal_moves) -> move` plus interrupt
  decisions (leech accept/decline).
- `RandomAgent`: uniform over legal moves. Floor baseline.
- `HeuristicAgent`: simple scripted bot (greedy VP + fixed resource-value
  weights). Arena opponent only — **never** exposed to the LLM as guidance.

**`arena/` — match runner:**

- Runs 4-player games with seat/faction rotation and fixed seeds for
  reproducibility. (Engine is oracle-validated at 4 players only.)
- A seat is either a local `Agent` or an *external* seat that blocks awaiting
  moves from an MCP session.
- Reports: win rate, mean VP margin, TrueSkill, cost and latency per game.

**`mcp/` — the server:**

- Session manager: create a game, configure which seats are bots, hold
  authoritative state. Sessions are identified so a headless `claude -p`
  invocation attaches to the game the arena created for it.
- Rungs are enabled per session config, so arena batches can ablate rungs
  without code changes.

Tool rungs:

| Rung | Tools | What it adds |
|------|-------|--------------|
| 1 — state + act | `new_game`, `get_state`, `legal_moves`, `play_move` | Legal play. `get_state` returns a compact token-efficient text render plus structured JSON. `play_move` advances bot seats until the LLM seat must act again (including leech-offer interrupts, which the LLM answers itself) and returns the events that happened in between. |
| 2 — factual analysis | `preview_move`, `score_projection` | Scratch-paper arithmetic: exact deltas of a move, leech offers it triggers, guaranteed scoring-tile / cult / area payouts. No desirability signal. |
| 3 — LLM-driven search | `branch`, `branch_state`, `branch_play`, `discard_branch` | Sandbox game trees. Claude forks the position, plays *all* seats in the branch itself (modeling opponents' replies with its own judgment, optionally via parallel subagent Claudes), reads factual states, and judges lines with its own reasoning. Both policy and evaluation are the LLM. |
| 4 — distilled knowledge | (context, not a tool) | The corpus-distilled compendium (below) is loaded into the pilot prompt. |

**`knowledge/` — corpus-distilled strategy compendium (rung 4):**

Learns from the greats *without* access to their moves at play time. An
offline pipeline mines the 3,553-game corpus for aggregate, win-conditioned
statistics — per-faction opening patterns, building/priest/favor timing
curves in wins vs. losses, bonus-tile pick rates by round and scoring tile,
matchup and board-region contest rates — then an LLM pass writes them up as
compact playbooks (principles with effect sizes, e.g. "Darklings winners
average 5.2 priests sacrificed by round 4; losers 3.1"). The pilot reads
principles and applies them with its own judgment; it never sees a single
retrieved expert move. Explicitly rejected alternatives: position-level kNN
retrieval (imitation risk, fails off the expert manifold) and any trained
value function (violates the facts/judgment boundary).

### Data flow (headless arena game)

1. Arena creates an MCP session (game + seat config + rung config), starts
   bot seats.
2. Arena spawns `claude -p` with the pilot prompt (+ compendium if rung 4)
   and the session id.
3. Claude loops: `get_state` → reason (→ rung 2/3 tools) → `play_move` →
   server advances bots, surfaces interrupts → repeat until game end.
4. Server reports the finished game; arena aggregates results, cost, latency.

## Error handling

- Invalid or illegal move submitted: structured error naming the violated
  rule, plus a pointer to `legal_moves`. The server never auto-picks a move
  for the LLM.
- Malformed tool input: schema validation with fail-fast messages.
- Claude session death / timeout mid-game: the game is marked invalid and
  excluded from win-rate stats (logged with full transcript); the arena
  retries the game up to 2 times with the same seed before flagging it.
- Engine exceptions inside a session are caught at the tool boundary and
  returned as structured errors with context logged server-side.

## Testing

- **Unit:** each MCP tool against engine fixtures (known positions with
  known legal moves, deltas, projections).
- **Replay-grounded integration:** pilot a corpus game's recorded moves
  through `play_move` and assert the resulting states match the delta
  oracle — reuses the replay harness as truth.
- **Full-loop integration:** a `RandomAgent` driving a complete game through
  the MCP interface (no LLM), proving session lifecycle, interrupts, and
  termination.
- **Slow/optional smoke:** one real headless `claude -p` game end-to-end.
- **Compendium pipeline:** stats-mining functions unit-tested against
  hand-computed values on a small fixture corpus.

## Measurement plan

- Rung configs measured cumulatively (1, 1+2, 1+2+3, 1+2+3+4) against the
  same opponent pool (RandomAgent ×3, then HeuristicAgent ×3), same seed
  set, faction rotation.
- ~50–100 games per config; report win rate, VP margin, TrueSkill, $/game,
  s/move. Ablation curves fall out of the rung flags for free.

## Out of scope (this phase)

- Fine-tuning or any weight updates (frozen models only).
- Position-level retrieval, value functions, engine-scored rollouts —
  rejected by design, not merely deferred.
- LLM-guided MCTS (engine-driven search with LLM priors).
- Self-improvement playbook (post-game loss analysis distilled into a
  persistent playbook) — compatible with this architecture, deferred to a
  later phase.
- 2/3/5-player games (engine not oracle-validated there).
- Engine-AI track (imitation/MCTS/RL) — proceeds separately.
