# Terra Mystica engine design

Target: replay every crawled Div 1–3 tournament ledger with zero per-move
delta mismatches (snellman's `deltas.parquet` is the oracle), then serve as
the legal-move generator and simulator for agents (arena, MCTS, LLM tools).

## Scope (tournament config, exactly what the data plays)

Base map, 14 base factions, standard VP scoring, plus the tournament-standard
options observed in the corpus: `mini-expansion-1` (BON10 + town tiles
TW6–TW8 pool additions), `temple-scoring-tile` (SCORE9), `shipping-bonus`
(BON13-class tile), `variable-turn-order` (pass order defines next round's
turn order), `errata-cultist-power`, `strict-leech`,
`strict-darkling-sh`, `strict-chaosmagician-sh`, `maintain-player-order`.
Exact option set per game comes from `games_meta.parquet`; the engine takes
an options set at game construction and the replay harness passes each
game's own options.

## State model

Immutable public API; internal builder does controlled mutation during
`apply` then freezes.

- `GameState` (frozen): `round` (0=setup, 1–6), `phase` (SETUP_FACTIONS,
  SETUP_DWELLINGS, SETUP_BONUS, INCOME, ACTIONS, CLEANUP, FINISHED),
  `turn_order` (tuple of factions), `passed_order` (next round's order under
  variable-turn-order), `active_index`, `pending` (queue of forced
  sub-decisions, see below), `map_state` (hex → color/building/owner),
  `bridges`, `towns_founded`, `factions` (name → `FactionState`),
  `cult_tracks` (cult → position per faction + occupied 10-slot + priest
  slots 2/3/3/2), `round_scoring` (6 score tiles), `bonus_tiles` (pool +
  per-faction + accumulated coins), `favors_pool`, `towns_pool`,
  `power_actions_taken` (per-round ACT1–6 + faction/BON/FAV specials).
- `FactionState` (frozen): resources C/W/P, power bowls (p1,p2,p3),
  max_priests-in-play accounting, shipping/dig levels, buildings placed
  (counts + hex sets), favors/bonus/town tiles held, VP, passed flag,
  leech offers outstanding.

## Decision queue (the crux of replay fidelity)

Ledger rows interleave main actions with forced follow-ups (favor picks
`+FAV11`, town tile picks `+TW7`, cult advances, leech/decline by *other*
factions, Halflings SH 3 transforms, Darklings SH W→P conversion). Model:
`apply(state, move)` may push `PendingDecision(faction, kind, options)`
items onto a queue; the next mover is the queue head's faction, not the
turn-order successor, until the queue drains. Leech offers enqueue in
clockwise order from the triggering faction (with `strict-leech` semantics
and Cultists timing from `acting.pm`); `wait` rows are ordering no-ops.

## Move application pipeline

`legal_moves(state) -> tuple[Move, ...]` and
`apply(state, move) -> GameState`, with `Move` = the parser's
`ParsedCommand` types (same vocabulary; the engine consumes the parser's
normalized moves directly — no second grammar).

Order of a round: INCOME (per-faction: base + buildings + favors + bonus
tile; engine-generated ledger rows `cult_income_for_faction` /
`other_income_for_faction` mark the boundary in replay), then ACTIONS
(turn-taking with pending-queue interrupts until all pass), then CLEANUP
(round-tile cult rewards, spade/etc. bonuses, reset power actions, bonus
tile coin accumulation), then next round's turn order from pass order
(variable-turn-order). Round 6 → final scoring: cult tracks (8/4/2 with
tie-splitting), network size (largest connected group via
direct+bridge+shipping connectivity), plus per-faction leftovers
(`score_resources` rows: 3 resources → 1 VP conversions etc.).

## Faction hooks

`factions_data.py` covers static data. Behavior hooks (one module per
faction in `engine/tm/factions/`, ~8 real implementations needed):
Giants (all transforms cost exactly 2 spades → home terrain), Darklings
(priest-digging, +2 VP per dig, SH one-time up-to-3 W→P), Halflings (SH →
3 free spades sequence), Engineers (bridge build action for 2 W; pass-VP
3/bridge joining two own gray-hex buildings), Mermaids (river hex may join
a town; `connect r9` moves), Dwarves (tunneling: skip one hex for 2 W,
counts double for scoring tile), Fakirs (carpet flight: skip via priest,
+4 VP), Cultists (+1 cult when opponents accept leech, errata power when
all decline), Auren/Witches/Alchemists/Chaos Magicians/Nomads/Swarmlings
(stronghold actions ACTA/ACTW/ACTC/ACTN/ACTS — mostly data-driven via
`FACTION_SPECIAL_ACTIONS`).

## Replay harness (the test suite)

For each game: construct with the game's options + faction setup rows,
step through parsed moves; after every ledger row with recorded deltas,
assert exact match of VP/C/W/P deltas, PW bowl string, and CULT string for
the acting faction. On mismatch: emit (game_id, row, expected, actual) —
each becomes a pinned regression. Rollout: start with `test/testgame*.txt`
from the reference repo + 10 crawled games, then 100, then full corpus in
CI-style batches. Per-faction pass rates reported (asymmetry requirement).

## Performance posture

Correctness first in pure Python. The hot path for MCTS (legal_moves +
apply) gets optimized later (arrays/bitmasks or Rust port) only when arena
Phase demands it; the replay suite is the safety net for any rewrite.
