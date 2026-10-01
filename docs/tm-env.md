# TM-Env — standard-API environments over the TM engine

`src/bgai/env/` wraps the oracle-verified rules engine in the two
interfaces the RL ecosystem actually consumes:

- **`bgai.env.aec`** — a PettingZoo **AEC** environment for the real
  2–5 player game.
- **`bgai.env.single`** — a **Gymnasium** environment where one seat
  learns and the rest are filled by a policy from `bgai.env.opponents`.

Both are thin adapters over one object, `bgai.env.core.Episode`, which is
itself a thin adapter over `bgai.arena.driver`. Legal action masks come
from `engine/tm/legal.py`'s own move generation by way of
`driver.decision`. **There is no second copy of the rules in this
package**, and the structure makes one impossible: the env never
enumerates a move, it only indexes what the engine offered.

Nothing in `src/bgai/engine/`, `src/bgai/training/` or `src/bgai/arena/`
was changed to add this.

```
src/bgai/env/
  config.py        constants, EnvConfig, the error types
  table.py         seeded 2–5 player GameSetup construction
  canonical.py     hash-independent offer ordering (see "Determinism")
  observation.py   mover-relative encoding, 2–5 seats, one fixed layout
  spaces.py        gymnasium.spaces definitions
  reward.py        four reward modes, all functions of engine VP alone
  core.py          Episode: the immutable engine<->env glue
  aec.py           PettingZoo AEC env
  single.py        Gymnasium single-agent env
  opponents.py     adapters for the existing bgai.agents family
  registry.py      gymnasium.register("bgai/TerraMystica-v0")
  codec.py         SimState <-> dict <-> bytes
  serialize.py     Observation <-> dict <-> bytes
  offline.py       logged human games -> offline-RL transitions
  bench.py         baselines, throughput, API conformance
```

`gymnasium` and `pettingzoo` are **dev-group** dependencies. The
engine-facing half (`config`, `core`, `observation`, `canonical`, `codec`,
`reward`, `serialize`, `offline`) imports and runs without either of them;
`tests/test_env_package.py` blocks both at import time and plays a whole
3-player game to prove it.

## Quick start

```python
# multi-agent, 2–5 players
from bgai.env.aec import env
from bgai.env.config import EnvConfig
import numpy as np

e = env(EnvConfig(player_count=4))
e.reset(seed=20261001)
for agent in e.agent_iter():
    obs, reward, terminated, truncated, info = e.last()
    if terminated or truncated:
        e.step(None)
        continue
    legal = np.flatnonzero(obs["action_mask"])
    e.step(int(np.random.choice(legal)))
```

```python
# single-agent, other seats played by the champion checkpoint
from bgai.env.single import make_single_env

e = make_single_env(
    player_count=4,
    opponent="mcts",
    checkpoint="data/checkpoints/selfplay_leg5b/current.pt",
    simulations=512, top_k=8, max_depth=48, leaf_batch=16, c_puct=2.5,
)
obs, info = e.reset(seed=1)
```

```python
# or by gymnasium id
import gymnasium
from bgai.env.registry import register_envs, ENV_ID
register_envs()
e = gymnasium.make(ENV_ID, player_count=3, opponent="greedy")
```

## Action space: an index into the offer

`Discrete(MAX_CANDIDATES)`, where action *i* names `offer[i]` and
`observation["candidates"][i]` carries that move's features.

**Why.** The engine offers a *variable* candidate set, and the repo's own
net already scores candidates rather than a fixed action vector
(`training/model.py` dots a state embedding with each legal move's
embedding), which is why an untrained imitation net is still a legal
player. The index space is the same shape.

**Rejected:** a flat, state-independent space over every conceivable
`ParsedCommand`. The grammar spans 113 hexes × 22 verbs ×
building/tile/cult/colour/target/resource/amount fields, so a faithful
flat space is six figures wide and >99.99 % masked at every decision.

**Cost, stated plainly:** action index *i* does not mean the same move in
two different states, so a policy **must** read
`observation["candidates"]`. An index-only policy would be learning noise.

`MAX_CANDIDATES = 320` is a padding width, not a rules constant. Measured
over 600 random-play games at 2/3/4/5 players (85,460 decisions) the
largest offer was **232** (a 2-player Mermaids turn); mean 10.7, p99 50.
An offer wider than the space raises `OfferOverflowError` rather than
truncating — a truncated mask would delete legal moves, i.e. be a
rules-correctness bug wearing a capacity nuisance's clothes.

## Observation

PettingZoo's masked-action convention,
`{"observation": {...}, "action_mask": ...}`:

| key | shape | dtype | contents |
|---|---|---|---|
| `observation.hex_planes` | (113, 18) | int8 | terrain colour one-hot (7) \| river (1) \| building one-hot (5) \| owner mover-relative seat one-hot (5) |
| `observation.globals` | (371,) | int16 | 5 seat blocks of 59, then seat-count one-hot (4) \| seat-present mask (5) \| round one-hot (7) \| phase one-hot (6) \| 6 score tiles × 6 \| bonus pool (10) \| power actions taken (6) \| pending count (1) \| offer size (1) |
| `observation.candidates` | (320, 12) | int16 | `training.encode_move` features per offered move, zero-padded |
| `action_mask` | (320,) | int8 | 1 for every index the engine currently offers |

Everything is **mover-relative**: seat blocks are ordered
`[mover, next in seat order, ...]` and hex ownership is a mover-relative
one-hot, so a policy is seat-equivariant for free.

`action_mask` is `int8` specifically because
`gymnasium.spaces.Discrete.sample(mask)` asserts that dtype; drifting to
`bool` would break every masked sampler.

**This is a sibling of `training.encode_state`, not a replacement.** That
encoder is pinned at four seats (`_CTX_ROUND = 4 * SEAT_BLOCK`, 4-wide
owner one-hot) because every one of the 3,563 corpus games is 4-player and
every existing checkpoint depends on its exact layout. TM-Env has to serve
2–5 players from one fixed space, so it carries `MAX_SEATS = 5` seat
blocks with a presence mask and absent seats left at zero. A per-count
layout would make a 2-player checkpoint structurally unusable at 4.

A seat that is not the one to decide — and every seat once the game is
over — gets an all-zero mask, the same convention `connect_four_v3` uses.

## Reward

All four modes are functions of the engine's own post-`final_scoring` VP
and nothing else. All are centred, so a table's rewards sum to 0 and a
seat finishing exactly average earns 0.

| mode | per step | terminal |
|---|---|---|
| `TERMINAL_VP_SHARE` (default) | 0 | `vp / table_total − 1/n` |
| `TERMINAL_RANK` | 0 | placement on [+1, −1], ties share their mean |
| `TERMINAL_WIN` | 0 | `1/len(winners) − 1/n` for a winner, `−1/n` otherwise |
| `DENSE_VP` | `Δ(own VP) / 100` | `+1.0 ×` the VP share |

`TERMINAL_VP_SHARE` is the default because it is *exactly* the target the
repo's value head already predicts (`training/dataset.py`'s `share`), so a
value function transfers between imitation and RL unchanged.

Honest limitations, each from this project's own measurement record:

- a **share** target is scale-blind to compounding economy. C12 found the
  leg-4 plateau broke only once *absolute* final VP was added as an
  auxiliary target. If you care about that, use `DENSE_VP` or add your own
  auxiliary head.
- **rank** throws away margin. C1/C3 is the cautionary tale: placement
  against our own baselines flattered the agent for months while its
  absolute VP sat at ~65.
- **`DENSE_VP` is deliberately not potential-based** with respect to the
  share objective. Its dense terms telescope to `(final_vp − 20) / 100`, a
  monotone affine function of absolute final VP, so the mode optimizes
  absolute VP *plus* share. That is C12's finding turned into a reward.
  *Cost if wrong:* it pays for VP that does not convert into placement —
  the C11 cult-town-hoarding failure mode — so it must be measured against
  `TERMINAL_VP_SHARE`, not assumed better.

Credit assignment is genuinely multi-agent: a build hands power to
neighbours and the scoring phase pays everyone, so `step` returns a reward
for *every* seat, not just the mover. The Gymnasium wrapper accumulates
what the learner earned during opponents' turns and pays it out on the
learner's next step.

## Determinism

Same seed + same action sequence ⇒ identical trajectory, in-process and
across a fresh interpreter with a different `PYTHONHASHSEED`
(`tests/test_env_determinism.py`, plus PettingZoo's own `seed_test`).

Getting there surfaced **a real pre-existing nondeterminism in the engine**,
which `bgai/env/canonical.py` documents in full.
`legal_actions.bridge_moves` builds its command from an unordered hex pair:

```python
a, b = tuple(pair)            # pair is a frozenset[str]
moves.append(cmd("bridge", loc=a, loc2=b))
```

`tuple()` over a two-element frozenset yields its elements in **hash**
order, and CPython randomizes string hashes per process. So the same
bridge is offered as `bridge loc=E4 loc2=G1` in one process and
`bridge loc=G1 loc2=E4` in another. `driver.canonical_moves` cannot repair
that: the two commands differ in their *field values*, not merely in their
position, so they sort to different slots and every subsequent action
index shifts.

Measured: two subprocesses at `PYTHONHASHSEED=1` and `987654`, same table
seed and same raw action stream, agreed for 55 decisions, diverged at the
first bridge offer, and ended on 191 vs 196 decisions with different final
VP.

**What TM-Env does about it.** Nothing to the engine — the rules engine's
behaviour is not this layer's to change, and `apply()` accepts both
spellings, so neither is wrong. Instead the env presents an
orientation-normalized offer (hex pair spelled low-to-high) in a stable
total order, and keeps the engine's own command objects alongside to step
with (`driver.advance` checks offer membership by equality, so the
original has to be what is applied). Consequences, deliberately:

- the env's action index is reproducible across processes and hash seeds;
- a bridge candidate's observation features are reproducible too;
- the env's offer order differs from `driver.canonical_moves`'s whenever a
  bridge is on offer. Harmless for every consumer: agents score candidates
  individually and are handed their chosen move's index back, so no
  checkpoint or dataset depends on the order itself.

**This is worth fixing upstream** (one line in `bridge_moves`:
`a, b = sorted(pair)`), which would make `arena` and self-play
cross-process reproducible too. It is left alone here because that changes
the engine's offers and therefore every pinned arena seed and gate.

## Serialization

Two independent round-trips, both tested at 2/3/4/5 players:

- **State.** `codec.sim_to_dict`/`sim_to_bytes` encode the whole paused
  game — `GameState` *and* the driver's turn bookkeeping. The bookkeeping
  matters as much as the board does: `fresh_taken` and `free_used` decide
  which moves the next offer contains, so a snapshot without them restores
  a *different* game. `Episode.to_bytes`/`from_bytes` adds the config and
  seed; the AEC and Gymnasium envs expose it as `snapshot()`/`restore()`.
- **Observation.** `serialize.obs_to_dict` (arrays), `obs_to_bytes`
  (version-stamped `.npz`), `obs_to_jsonable` (plain lists). Dtypes
  survive, so a decoded observation still satisfies
  `observation_space.contains`.

Not pickle: a pickled `SimState` is bound to this exact class layout and
this Python, so it cannot be a stored artefact, cannot cross a language
boundary, and cannot be diffed when an engine field is added. The explicit
codec also fails loudly on an unknown field instead of resurrecting a
stale shape, and writes sets **sorted**, so equal states encode to equal
bytes — which makes `sim_to_bytes` usable as a transposition key. A
4-player mid-game episode is ~7.3 kB of JSON.

## Illegal actions

`EnvConfig.illegal_action`:

- **`TERMINATE`** (default) — the episode ends, the offender collects
  `ILLEGAL_ACTION_REWARD = −1.0` (at or below the worst legitimate
  terminal reward in every mode), every other seat gets 0, and
  `info['illegal_action']` records the index. This is the standard masked
  env contract and it is what lets `gymnasium.utils.env_checker.check_env`
  pass at all: that checker samples the *whole* `Discrete` space with no
  mask, so an env that only raises cannot conform.
- **`RAISE`** — `IllegalActionError`, naming the legal index range. Better
  while debugging a policy: a masked policy can never reach it, so
  reaching it is a bug you want immediately rather than as a mysterious
  −1.0.

Either way the mask is **never silently repaired**. Projecting an illegal
index onto a legal one (modulo, clamp, nearest) would make every index
playable and quietly delete the mask's meaning;
`tests/test_env_core.py::test_the_mask_is_never_repaired_into_a_different_legal_move`
pins that.

A move the engine offered and then *refused* is a different thing — a
`legal_moves` soundness finding. The repo's rule there is visible-not-fatal
(random-play arena fuzzing is how two real ones were found and fixed,
`tests/test_legal_soundness.py`), so by default the episode ends truncated
with the rejection on every agent's `info`;
`EnvConfig.raise_on_engine_error=True` makes it an exception.

## Offline-RL export

```bash
uv run python -m bgai.env.offline --out data/datasets/offline_rl --limit 200
```

Every clean corpus game is replayed through `engine.tm.replay.replay_game`,
whose `on_decision` hook hands over the pre-apply `GameState`, the acting
faction, and the ledger command actually played — the same path
`training/extract.py` uses, so the decision set is directly comparable to
the imitation shards.

**Per-seat MDPs, not one interleaved stream.** A seat's `next_obs` is *its
own* next decision, not whatever seat happened to move next. One game
yields `player_count` trajectories, each terminal on that seat's last
decision. Interleaving them would make `next_obs` the wrong seat's board
and silently corrupt every bootstrapped value target.
`tests/test_env_offline.py` pins exactly one terminal transition per seat
per game, and that every `next_index` points at the same seat in the same
game, strictly forward.

**Honest caveat about the mask.** The offline mask is
`canonical_offer(legal_moves_for(state, faction))` — the engine's full
legal set, which is what the human was choosing among and what the
imitation shards record. The *online* env's mask is `driver.decision`'s
offer, a **subset**: the driver enforces a turn protocol (one fresh main
action, then continuations, at most `FREE_ACTIONS_PER_TURN` free converts)
that a human ledger row does not expose, because a human submits a whole
turn as one row. So offline and online action sets differ in a specific
way, and that difference is the same train/inference mismatch that cost
this project ~30 VP once already (C3). It is recorded in the export
manifest as `mask_source`/`mask_note`, not papered over.

On disk, `next_obs` is a `next_index` into the same shard (−1 when
terminal) rather than a second copy of the arrays: an observation is
~3.3 kB, so duplicating it would double a multi-gigabyte export for no
information. `offline.iter_transitions` materializes real
`(obs, action, mask, reward, next_obs, done)` records.

### How many logged transitions there actually are

| dataset | games | decision records |
|---|---|---|
| `data/datasets/imitation` (tmtour Div 1–3 only) | 3,374 | **1,195,522** (1,078,917 train / 116,605 val) |
| `data/datasets/imitation_v2` (+ broad population) | 64,411 | **20,515,803** (20,020,912 train / 494,891 val) |

Both figures are the `record_count` in each shard set's own
`manifest.json`, and both check out against the per-shard sums. 20.5 M is
the real total; the imitation net's headline accuracy is measured on the
1.2 M league set.

TM-Env's exporter rebuilds transitions from replay rather than reusing
those shards (it needs per-seat `next_obs` and episode boundaries, which
the shards do not carry). Measured on the first 10 clean league games:
**3,164 transitions, 316.4 per game, 0 unmatched commands, 0 skipped**,
2.8 s wall, ~27 kB compressed per game. So a full league export is
~1.1 M transitions and ~95 MB; the whole 64,411-game population corpus
projects to ~20 M transitions and ~1.7 GB compressed. Nothing of that size
is committed.

## Measured end to end

All numbers from `uv run python -m bgai.env.bench`, Apple M3 Pro, seed
20261001, one process. Episode lengths and VP are deterministic; the
throughput column was measured on an otherwise idle machine.

**Random masked play through the PettingZoo AEC env**, 50 episodes per
player count, 0 engine errors and 0 truncations at every count:

| players | episode length (decisions) | VP per player | table total |
|---|---|---|---|
| 2 | 81.2 [51–118] | 73.8 ± 8.5 [52–101] | 147.7 |
| 3 | 119.2 [83–170] | 65.3 ± 9.7 [41–97] | 195.9 |
| 4 | 158.3 [113–187] | 59.1 ± 10.3 [34–84] | 236.6 |
| 5 | 201.0 [163–251] | 56.2 ± 10.4 [32–88] | 281.1 |

Two 30-episode repeats agree closely. At the same seed base:
82.9 / 119.9 / 158.0 / 201.6 decisions and 74.4 / 65.5 / 59.7 / 56.8 VP.
At an independent seed base (4242, i.e. entirely different tables):
4-player 169.2 decisions / 59.4 VP, 5-player 204.4 decisions / 55.0 VP.
So VP per player is stable to ~±1 and episode length to ~±7 % across
table draws.

The 4-player figure (59.1 VP) sits where the arena's own random baseline
does (~54 VP on corpus-sampled tables), which is the sanity check that the
wrapper is not quietly changing the game.

**Existing agents through the new wrapper** (Gymnasium learner seat,
rotating across seats, three uniform-random opponents, checkpoint
`data/checkpoints/selfplay_leg5b/current.pt`):

| learner | n | episode length | learner VP | wins |
|---|---|---|---|---|
| `imitation`, argmax policy | 4 | 226.2 [191–289] | 119.0 | 4/4 |
| `mcts` 64 sims, top-k 8, depth 48, batch 16, c_puct 2.5 | 1 | 176.0 | 115.0 | 1/1 |
| `mcts` 512 sims, top-k 8, depth 48, batch 16, c_puct 2.5 | 12 | 222.0 [188–307] | 105.2 | 12/12 |

These n are small and the opponents are uniform random, so what this
establishes is that **the env is usable end to end with the project's real
agents** — tree reuse, batched leaf evaluation and all — not any strength
claim. The strength numbers that matter are in the README and
`docs/decisions.md` and are produced by the arena, not by this.

**Throughput**, one process, Apple M3 Pro, random masked play, in
decisions/sec. Reported as ranges over repeated runs because the
run-to-run spread on an unquiesced laptop is larger than any difference
being measured — the *same* 4-player configuration came out at 2,277 on
one run and 752 on another:

| path | 2p | 3p | 4p | 5p |
|---|---|---|---|---|
| through the AEC env | 2,688 | 2,410 | 752–2,277 | 786–892 |
| `Episode` only, no wrapper | 4,121 | 4,062 | 1,322–2,382 | 1,133–3,478 |

What survives the noise: TM-Env runs at **order 10³ decisions/sec**, and
the API wrapper costs well under 2× over the bare `Episode` path, so it is
not the bottleneck. Don't read these as a benchmark; re-measure on a quiet
machine if the number matters.

For scale, the arena's own headless 4-player game is ~50–60 ms, and search
dominates everything the moment a real agent is in the loop: the
512-simulation MCTS run above managed **~5 decisions/sec**.

## API conformance

```bash
uv run python -m bgai.env.bench --conformance --players 2,3,4,5
```

Passing at every player count, pinned in `tests/test_env_conformance.py`:

- `pettingzoo.test.api_test` on the raw env **and** on the wrapped
  `env()` (behind `AssertOutOfBoundsWrapper` + `OrderEnforcingWrapper`);
- `pettingzoo.test.seed_test`;
- `gymnasium.utils.env_checker.check_env` with `skip_render_check=False`.

`api_test` emits exactly two warnings — *"Observation is not a NumPy
array"* and *"Observation space for each agent probably should be
gymnasium.spaces.box or gymnasium.spaces.discrete"*. Both are structural
and shared with PettingZoo's own `chess_v6` and `connect_four_v3`: a
masked-action env has to use a `Dict` observation space. The test asserts
the warning set *exactly*, so a new one fails rather than scrolling past.
`check_env` through `gymnasium.make` is warning-free.

`TerminateIllegalWrapper` is deliberately **not** applied in `env()`: it
would end the game and penalize a seat for a masked action, which is
already `TERMINATE`'s job and is configurable.

## Scope and known limits

- **2/3/5-player tables are reference-rules only.** Every one of the 3,563
  corpus games is 4-player, so the player-count-dependent machinery
  (bonus-pool size, turn rotation, leech seat order) has never been
  replay-validated away from 4. TM-Env runs all four counts and 200
  random-play games at 2/3/4/5 complete with zero engine errors, which is
  evidence they *run*, not that they are right.
- **Checkpoint opponents are 4-player only.** `training.encode_state` is
  pinned at four seats and MCTS's max^n value vector is 4-wide, so
  `build_opponent("imitation"|"mcts", player_count != 4)` raises with that
  explanation rather than producing wrong numbers.
- **`setup_source=CORPUS` is 4-player only** for the same reason, and
  needs the gitignored `data/` tree.
- **No draft.** `GameSetup` takes factions as given (D4.3); faction
  assignment is the env's, not the agent's.
- **Synthetic tables simplify the lobby rules** exactly as
  `arena/setup_factory.py` does: uniform 6-of-SCORE1..8 (no "no spade tile
  in rounds 5/6"), uniform `player_count + 3` of BON1..9, default
  `GameOptions`.

## Decisions

Numbered to continue `docs/decisions.md`'s style, kept here because the
decision log's live tail belongs to the cluster self-play work.

**E1 — The action space is a candidate index, not a flat move space.**
The engine offers a variable candidate set and the repo's net already
scores candidates. *Rejected:* a six-figure flat space >99.99 % masked at
every decision. *Cost:* an index means different moves in different
states, so a policy must read the candidate features; stated in the
action-space section and in `spaces.action_space`'s own docstring.

**E2 — One 5-seat observation layout for all player counts.**
`MAX_SEATS` seat blocks plus a presence mask and seat-count one-hot.
*Rejected:* a layout per player count, which would make a 2-player
checkpoint structurally unusable at 4. *Cost:* ~25 % of the globals vector
is zeros at 4 players, the count the corpus actually validates.

**E3 — `TERMINAL_VP_SHARE` is the default reward.** It is the target the
existing value head already predicts, so a value function transfers
unchanged. *Rejected:* rank as the default (C1/C3 — placement flattered
the agent while its absolute VP was ~65). *Cost:* share is scale-blind to
compounding economy (C12); `DENSE_VP` exists for that and is documented as
optimizing absolute VP, not share.

**E4 — An illegal action terminates with a penalty by default; the mask is
never repaired.** *Rejected:* projecting an illegal index onto a legal
move (every index becomes playable, the mask stops meaning anything) and
raise-only (`check_env` samples the unmasked space, so it could never
pass). *Cost:* a buggy policy gets −1.0 instead of a stack trace;
`IllegalActionPolicy.RAISE` is one field away.

**E5 — Engine rejections are recorded, not raised, by default.** Matches
`arena.sim.run_game`: random-play fuzzing is how two real `legal_moves`
soundness bugs were found, and the repo's rule is visible-not-fatal.
*Cost:* a silent rules bug shows up as a truncated episode; it is on every
agent's `info`, on `Episode.engine_error`, and counted by `bench`.

**E6 — Bridge-orientation nondeterminism is normalized in the env, not
fixed in the engine.** See "Determinism". *Rejected:* the one-line engine
fix, which would change the engine's offers and therefore every pinned
arena seed and self-play gate. *Cost:* the env's offer order differs from
`driver.canonical_moves`'s when a bridge is on offer, and the engine stays
cross-process nondeterministic for every *other* consumer. Flagged for an
upstream fix.

**E7 — The offline export is rebuilt from replay, not from the existing
imitation shards.** It needs per-seat `next_obs` and episode boundaries,
which the shards do not carry. *Cost:* an export costs a replay pass
(~0.28 s/game) and its mask is the full legal set rather than the driver's
turn-protocol subset — recorded in the manifest, not hidden.

**E8 — `gymnasium`/`pettingzoo` are dev-group dependencies and imported
lazily.** The engine, training and arena tracks must not grow a hard
dependency on an RL API to keep working. *Cost:* `bgai.env.spaces`,
`aec`, `single`, `registry` and `bench` raise `ImportError` without them;
`tests/test_env_package.py` pins that the rest does not.
