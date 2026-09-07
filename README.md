# bgai

Board game AI series. Game 1: **Terra Mystica**.

Two AI tracks, cross-validated against each other and against real humans:

1. **Engine AI** — imitation learning on high-quality human tournament games
   (tmtour Divisions 1–3, played on [terra.snellman.net](https://terra.snellman.net)),
   then MCTS search, then optional human-regularized self-play RL.
2. **LLM AI** — an LLM agent pushed up a capability ladder: bare prompt → knowledge-loaded
   context → engine tools → retrieval over expert games → test-time search → LLM-guided
   MCTS → fine-tuned models.

## Where it stands (2026-08-19)

The engine track is a complete, measured pipeline: a rules engine that
replays 3,552/3,553 tournament games with zero per-move mismatches, an
imitation net on 1.2M human decisions, a max^n MCTS with prior pruning
and batched leaf evaluation, and five verified generations of
human-regularized self-play. The champion agent (`selfplay_leg5b`, 512
sims / top-k 8 / depth 48 / c_puct 2.5) scores **~111 VP per player** on
its own tables: ahead of tmai's `ai_lode` (97–99, the only external bot
yardstick), at roughly the **22nd–24th percentile of human players**
(median 123, Div 1–3 average 136). Progress at local scale has been
exhausted by measurement (`docs/decisions.md` C10–C17); the next lever
is self-play at cluster scale (10^5–10^6 games, CPU only).

The LLM track is built and mock-tested but unmeasured for lack of API
credentials. Playing real humans (Phase 8) is prepared and deliberately
parked on the site operator's consent.

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
open API. Every request carries a contact address read from the
`BGAI_CRAWL_CONTACT` environment variable (copy `.env.example` to `.env`);
the crawler refuses to run anonymously. Thanks to Juho Snellman and the
tmtour maintainers for keeping these running.

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
`tests/test_legal_soundness.py`. (The LLM track's live driver, built in
parallel, independently hit the same two — also pinned in
`tests/test_legal.py`.)

## Imitation (Phase 5)

`src/bgai/training/` turns the corpus into a faction-conditioned
policy/value net; `src/bgai/agents/imitation.py` wraps a trained
checkpoint in the arena's `Agent` protocol.

```
uv run python -m bgai.training.dataset_build --out data/datasets/imitation
uv run python -m bgai.training.train --shards data/datasets/imitation \
    --out data/checkpoints/imitation_v1 --epochs 10
```

Extraction replays every clean game through the engine and captures each
`Kind.DECISION` command with the legal candidate set at that state:
**1,195,522 decisions** from 3,373 games (1.08M train / 117k val), split
by season (>= 67 is validation, so no future game informs an earlier
prediction) and weighted by division (Div 1 1.0 / Div 2 0.8 / Div 3 0.6).

The net scores *candidates* rather than a fixed action space: the state
embedding dots with each legal move's embedding, softmaxed over exactly
the moves the engine offers, so legality is structural and an untrained
net is still a legal player. See `docs/imitation-design.md`.

Training runs on MPS/CUDA/CPU (auto-detected), ~2 min/epoch on an M3 Pro.
Inference costs ~1 ms/decision, so arena games stay well under the <1s
target.

**Held-out accuracy** (10 epochs, seasons >= 67, 116,605 decisions):
**55.7% top-1 / 81.2% top-3**, against a random-among-candidates floor of
6.3% / 18.8% (the mean decision offers ~24 legal moves). The value head
predicts each seat's final-VP share to within 0.017 absolute. Accuracy is
even across factions -- every one of the 14 lands between 53.3%
(Chaos Magicians) and 58.0% (Cultists), so no faction is silently
unplayable. Per-verb it varies far more: `leech` 91%, `build` 62%,
`upgrade` 37%, `send` 12% -- the model is much better at *whether* to act
than at *where* and *how much*.

**Phase 5 gate** (`tests/test_arena_verify.py`, slow): imitation beats
both baselines over 100 mirrored games -- mean placement **1.08** vs
greedy 1.33 and random 2.41, zero errors.

**Absolute strength, honestly** (`docs/decisions.md` C1/C2). Placement
against our own baselines flatters the agent; final scores do not:

| | VP per player | table total |
|---|---|---|
| Div 1-3 humans (3,553 games) | **136.3** | 545.2 |
| all-population human median (304k scores) | **123** | — |
| our champion: deep search + self-play leg 5 (2026-08-17) | **~111** | — |
| our deep search on the imitation net, pre-RL (2026-08-13) | ~105 | — |
| tmai `ai_lode`, external heuristic (45 games) | **97–99** | 395.5 |
| our imitation net (after the driver fix) | ~95–99 | — |
| our imitation net (before the fix) | ~65 | ~256 |
| random | ~54 | — |

Reproduce the external baseline with
`node tools/tmai_headless.js <tmai_dir> 25 7`.

That jump from ~65 to ~96 was **not** a better model — it was a bug in
the arena driver, found only because these absolute, externally
referenced numbers existed. Convert and burn cost no action, so the turn
protocol re-offered them after every one, asking the agent "convert
again?" dozens of times per turn. Humans never see that prompt (they
submit a whole turn as one ledger row: 0.12 converts on average), so the
policy was being asked a question it was never trained on. The agent
spent **49% of its decisions converting** versus 8.3% for humans, and
built 29% of the structures and 2% of the towns humans build. Capping
free actions at 1/turn fixed it. Full diagnosis: `docs/decisions.md` C3.

The same bug had produced a confident, well-powered, and entirely
wrong conclusion (D5.6: "the policy must sample, not argmax") — argmax
had been re-picking the *same* convert forever. With the cap, argmax
wins by t = −11.44. Relative-only metrics cannot see a defect that
handicaps every agent equally.

The per-verb accuracy split explains the shape of the weakness: the net
is strong where options are few and conventions clear (`leech` 91%,
`gain_favor` 68%, `build` 62%) and weak exactly where the game is won
(`send` 12%, `dig` 29%, `transform` 35%, `upgrade` 37%). It learned the
game's grammar, not its strategy.


## Search (Phase 6)

`src/bgai/agents/mcts.py` is a max^n MCTS: every node carries a value
vector with one component per seat (a 4-player game is not zero-sum, so
a scalar would be a lie), selection maximises the *acting* seat's own
component, and leaves are evaluated by the net's value head rather than
by random rollouts. It runs on `arena/driver.py`'s immutable `SimState`,
which exists so positions can be cloned and branched (D6.1).

**The first version added nothing.** Over 1,000 paired games against the
imitation policy that supplies its priors, 64-simulation MCTS scored
**-0.005 ± 0.083 mean rank** at 75x the compute (D6.7, re-confirmed
after the driver fix in C4). Two diagnostics then separated the causes.
Blending a *computed* VP projection into the leaf evaluator produced an
inverted-U gain peaking at w=0.5 (C5): the learned value head goes blind
off the human distribution, and search had been faithfully amplifying
noise. A depth ladder (C6) showed 1024 simulations winning where 64 and
256 did not: the search was also too shallow.

**What fixed it (C7, 2026-08-13):** spend the budget deeper via **prior
pruning** (`top_k=8`, since policy top-3 is 84%), `max_depth=48`, batched
leaf evaluation with virtual loss, 512 simulations, on a net retrained
with a pairwise rank loss and a simplex value head over the
broad-population value split. Head-to-head vs the raw policy: **35.8% vs
15.0% win rate, 106.4 vs 96.6 VP** (n=120). Rank and VP moved together
for the first time. A one-variable sweep found more simulations help
while more depth or narrower width hurt; `c_puct` was later confirmed at
2.5 over the untuned 1.5 (C16, 28.7%/21.9%, n=320/side).

Tree reuse between moves, a confident-move fast path, and a disk-cached
setup list bring generation to **~650 games/hour on 10 laptop cores**
(~28 core-seconds per game). GPU inference was measured and refuted at
this model size: the IPC round-trip costs more than the 10 ms CPU
forward it replaces (C17), so the cluster ask is CPU-only. The batched
inference server (`bde9553`) stays for a future larger net.

## Self-play RL (Phase 6b)

`src/bgai/training/selfplay.py` + `selfplay_train.py` implement
human-regularized self-play: policy toward the search distribution,
value toward realised outcomes, KL toward the frozen imitation anchor
(piKL). Generation and arena share one `MCTSAgent.search()` entry point
so the teacher is exactly the searcher that was measured (D6.10).

**The first run made the agent worse** (+0.554 ± 0.163 mean rank,
D6.8): with search at parity with its policy, self-play distilled a
teacher no stronger than its student, and every iteration was a lossy
copy. Once C7 gave search an edge, the flywheel compounded. Each leg is
~600–770 games and is verified head-to-head against its predecessor at
the deep-search config (n=120, mirrored seats):

| leg | change | h2h vs predecessor |
|---|---|---|
| 1 | value MSE only | RL *policy* +, RL *search* − (value lost ordering) |
| 2 | + pairwise rank loss | **32.5% / 18.3%** vs pre-RL search |
| 3 | + winner-priority pairs, win CE | **30.8% / 20.8%** vs leg 2, 109.5 VP |
| 4 | 1024-sim targets, replay buffer | tie (25.8 / 27.5) — local scale spent |
| 5 | + aux head: final abs VP + town count | **30.8% / 20.8%** vs leg 4, +3.4 VP |
| 6 | KL leash halved | tie, clean drift canary — anchor is not the constraint |

The leg-4 plateau was diagnosed, not tuned away. VP-source decomposition
against rating-banded humans and a town-tile audit (C10–C13) found
**economy myopia**: the agent took cult town tiles 55% of the time vs
28% for top-5% humans and founded 1.28 towns per founding seat vs 2.3,
with identical expected VP per pick, so the error was invisible to
VP-greedy metrics. Share-based value targets are scale-blind to
compounding economy; adding auxiliary prediction of absolute final VP
and town count (C12) broke the plateau.

Also settled by measurement and refuted: round-scaled endgame budgets
(dead tie: the conversion weakness is evaluator knowledge, not thinking
time), max-child root selection (loses 15.8/34.2 to robust-child), and
in-training loss metrics as a proxy for strength (they never predicted
h2h; the human-validation ordering probe read flat across legs that
moved a full generation). Full record: `docs/decisions.md` C7–C17.

**What is left is scale.** Total self-play to date is ~4,500 games. No
leg has been run at more than one size, so the data-scaling curve is
unmeasured; a 5,000-game leg is the experiment that decides whether the
next 10^5 games buy strength or whether the lever is evaluator coverage.

## LLM ladder (Phase 7)

`src/bgai/llm/` + `src/bgai/agents/llm_agent.py` implement ladder rungs
L0–L4 (bare → knowledge → engine tools → corpus retrieval → propose /
critique / pick with a persistent game plan) behind a provider-agnostic
interface. All of it is mock-tested with no API key and no spend; **none
of it is measured**, because no LLM credentials exist in this
environment. See `docs/llm-track.md` for what each rung adds and why
retrieval deliberately avoids the imitation net's embedding.

## LLM agent track

The LLM plays through an MCP server whose tools expose **facts only**
(state, legality, arithmetic, guaranteed projections) — all judgment stays
with the model. Design: `docs/superpowers/specs/2026-08-04-llm-tm-mcp-harness-design.md`.

- **Interactive:** the repo's `.mcp.json` registers the `tm` server; tell
  Claude Code "pilot a game" and it plays a seat against bot opponents
  (random or greedy). Configure via a JSON file pointed at by
  `$BGAI_TM_SESSION` (`src/bgai/mcp/config.py` documents the schema).
- **Tool rungs** (per-session config, for ablations): 1 = state/legal/play,
  2 = factual analysis (`preview_move`, `score_projection`), 3 = sandbox
  branches where the LLM plays *all* seats for lookahead, 4 = the
  corpus-statistics compendium in the prompt
  (`docs/knowledge/tm-compendium/`, see `GENERATE.md` there).
- **Headless LLM arena** (spawns `claude -p` per game, aggregates win rate,
  VP, $/game):
  `uv run python scripts/arena_llm.py --games 4 --seed 100 --rungs 1,2,3 --out runs/r123/`

The live-game driver (`src/bgai/arena/live_driver.py`) is the *external-seat*
counterpart of `arena/sim.py`'s self-driving loop: same engine contract,
but it stops at any external seat's decision so the MCP session (or a
test) can supply the move, and resumes bots + bookkeeping afterwards.
`arena/setup_factory.py` provides seeded synthetic setups for MCP
determinism where `arena/setups.py` corpus-samples real ones.

Phase 8 (playing real humans) is prepared but deliberately not executed:
it requires contacting terra.snellman.net's operator and creating an
account on someone else's service. `docs/phase8-human-play.md` holds the
plan and a drafted request for the maintainer to review and send.
