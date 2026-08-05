# Decision log

One entry per non-obvious choice, with the alternative rejected and the
cost of being wrong. Newest phase last. Design *rationale* lives in the
per-area docs (`engine-design.md`, `imitation-design.md`); this file is
the chronological record of what was decided, why, and what it costs.

---

## Phase 4 — Arena + baselines

**D4.1 — Agent interface is one method: `choose(state, faction, offer, rng)`.**
The engine's pending-decision queue already turns leech answers, favor
picks, town picks, and setup placements into ordinary moves, so no
separate callback protocol is needed.
*Rejected:* a multi-method interface (`on_leech`, `on_favor`, ...), which
would have to grow every time the engine gains a forced decision.
*Cost if wrong:* an agent that wants to treat forced decisions specially
must inspect `state.pending` itself. Acceptable — MCTS and the imitation
net both want the uniform view.

**D4.2 — Arena setups are sampled from the corpus, not generated.**
Each table reuses a real Div 1-3 game's score tiles, bonus pool, faction
lineup, and seat order via `load_setup`, with drop history cleared.
*Rejected:* a synthetic setup generator (random legal tile/faction
draws). Human lineups are pre-vetted for coherence, and evaluation stays
on the distribution the Phase 5 net trains on.
*Cost:* the arena inherits the human pick meta — rare factions (Auren,
Giants) get ~4 games per 200. Stratified oversampling of corpus games
containing rare factions is the fix if a report ever shows it matters;
deferred until then.

**D4.3 — Fixed faction assignment; the draft is not modeled.**
The arena assigns factions via mirrored rotation. Drafting is a separate
decision layer above the engine (the engine's `Phase` enum starts at
`SETUP_DWELLINGS`; `GameSetup` takes factions as given).
*Why:* mirrored rotation *requires* the arena to control assignment —
if agents drafted, faction strength and agent strength could not be
separated. TrueSkill would be measuring two things at once.
*Cost:* draft skill goes unmeasured. It matters for the Phase 8 human
comparison, where a draft-mode arena will be needed.

**D4.4 — Offers are canonically sorted before reaching an agent.**
Engine legal-move generation iterates sets in places, so raw order varies
with interpreter hash randomization.
*Why:* without it, a game is not reproducible from (setup, seats, seed)
across processes — which would undermine both debugging and the Phase 5
label indexing.
*Cost:* one sort per decision (~microseconds against a 48 ms game).

**D4.5 — Engine errors are recorded on the result, not raised.**
`run_game` catches `EngineError` and returns it in `GameResult.error`.
*Why:* random play doubles as a `legal_moves` soundness fuzzer; findings
must be visible and countable, not fatal. This immediately found two real
bugs (Giants non-home transforms, ACTN builds skipping affordability).
*Cost:* a caller that ignores `.error` silently rates broken games. The
gate test asserts `n_errors == 0` for exactly this reason.

**D4.6 — Greedy's building credit is per building type, not flat.**
First draft used a flat per-building weight and greedy preferred
`convert 3C -> 1VP` over a free dwelling.
*Why:* the eval must make development beat hoarding, or the "obviously
better than random" baseline isn't.
*Cost:* the weights are hand-tuned against one criterion (beat random
decisively), not calibrated to real TM value. Fine for a floor; it is
not a strategy reference.

---

## Phase 5 — Imitation

**D5.1 — Candidate scoring instead of a fixed action head.**
The net embeds the state and each *offered* move, and scores by dot
product over exactly that decision's candidates.
*Rejected:* a fixed output layer over the full command space (113 hexes x
43 tiles x 7 colors x ...), which is mostly dead weights and needs
hand-maintained legality masks.
*Cost:* every candidate must be featurized at inference (~1 ms/decision
for a ~24-candidate set — measured, and 20x under the arena's budget).
Also makes the policy's output dimension data-dependent, which the
padding mask in `collate` handles.

**D5.2 — Everything is mover-relative.**
Seat blocks are rotated to the acting faction; hex ownership, leech
targets, and the value head's 4 outputs are all in that rotated order.
*Why:* the arena rotates seats deliberately, so a seat-absolute encoding
would force the net to learn the same concept four times.
*Cost:* `encode_state` must rotate on every call (cheap), and any future
consumer must respect the convention — documented in
`imitation-design.md`.

**D5.3 — The label matcher is shared with the containment sweep.**
`engine/tm/legal_match.py` owns the per-verb identity keys and is used
by both `training.extract` and `tests/test_legal.py`.
*Why:* a ledger command and its generated candidate are not
field-identical (the ledger records a `decline`'s target and power, a
`convert`'s unit count; the generator emits a `transform`'s color the
ledger omits). Two copies of that mapping would drift, and drift means
silently mislabeled training data.
*Cost:* the engine now carries a module whose only consumers are a test
and the training pipeline. Worth it — the 29-game slow sweep is now also
the label matcher's regression test.

**D5.4 — Time-based split at season 67, division-weighted loss.**
*Rejected:* a random split, which leaks: two games from the same season
and league share context, so a random split would flatter validation.
*Cost:* validation is 117k decisions (9.8%) and is drawn from a
different era than training — later seasons may play a different meta,
so val accuracy is a slightly pessimistic estimate of same-era accuracy.
That is the honest direction to err in.

**D5.5 — Skip decisions with fewer than 2 candidates.**
A forced move carries no information about preference.
*Cost:* the model never learns "this is forced", but it never has to —
the engine offers exactly one move in those states, so the agent's choice
is determined regardless.

**D5.6 — The imitation agent SAMPLES (T=1.0) rather than taking argmax.**
Measured, 80 mirrored games per setting vs greedy: argmax loses
(-0.150 mean rank), T=0.5 wins (+0.069), T=1.0 wins most (+0.169), and
the trend is monotonic.
*Why:* a behavior-cloned policy's argmax is brittle. In states the expert
corpus never contains it commits to the single most-imitated move and
repeats that commitment every time the state recurs; sampling preserves
the diversity human play actually had.
*Cost:* the agent is stochastic, so single-game results vary. It still
draws from the arena's seeded rng, so a (setup, seats, seed) triple
remains exactly reproducible. For Phase 6, MCTS will want the raw
distribution as a prior anyway — argmax was never the end goal.

**D5.7 — `ImitationAgent` is not re-exported from `bgai.agents`.**
It imports torch; the arena and engine must stay usable without a
deep-learning dependency.
*Cost:* callers write `from bgai.agents.imitation import ImitationAgent`.

---

## Phase 6 — Search

**D6.1 — The driver's control state became explicit and immutable.**
`arena/driver.py` holds `SimState = (GameState, fresh_taken, prev_verb,
decisions, income_marker)`; `advance` is a pure function of
(SimState, choice).
*Why:* MCTS clones and branches positions. The turn protocol's
bookkeeping previously lived in `run_game`'s local variables, which
cannot be cloned — a search would have silently shared or lost it.
*Rejected:* pausing the old driver with generators/threads. Cloning a
paused coroutine is not something Python supports cleanly, and MCTS
needs to re-enter the same position many times.
*Cost:* one real refactor of a working, fuzz-validated component. Held
to zero behavior change: all arena tests plus a fresh 200-game random
fuzz (0 errors) pass on the rewrite.
*Detail worth remembering:* the income-batch marker lives on `SimState`,
not `GameState`. Stashing it on the shared game object (my first draft)
would have leaked across MCTS branches — the classic bug where search
results depend on visit order.

**D6.2 — max^n value vectors, not a scalar.**
Every node carries one value component per seat; selection maximizes the
*acting* seat's component; backup adds the whole vector.
*Rejected:* scalar minimax/negamax, which assumes zero-sum. In a
4-player game "my gain is your loss" is false — two trailing players can
both benefit from attacking the leader.
*Cost:* 4x the value storage per edge (trivial), and no alpha-beta-style
pruning is available in max^n.
*Implementation note:* vectors are re-based to the setup's absolute seat
order at every node. The net emits mover-relative values, so skipping the
re-base would make a component mean a different player at each depth —
pinned by `test_value_vectors_are_absolute_seat_order`.

**D6.3 — No random rollouts; the value head evaluates leaves.**
*Why:* a TM rollout is ~200 further decisions of noise, and the value
head was trained on 1.2M real positions. Random rollouts would be both
slower and worse.
*Cost:* search quality is bounded by the value head's quality. If the net
is weak, deeper search inherits that weakness rather than correcting it
— which is exactly what the arena measurement has to check rather than
assume.

**D6.4 — Search agents get an optional `choose_sim` hook.**
`run_game` calls `choose_sim(sim, ...)` when an agent exposes it, else
the ordinary `choose(state, ...)`.
*Why:* only search needs the driver's internals; making every agent take
a `SimState` would leak the driver into the LLM and heuristic agents.
*Cost:* two entry points to keep in sync. The protocol stays one method
for everyone who is not a searcher.

**D5.8 — Ten epochs, last checkpoint, no best-checkpoint selection.**
Val top-1 by epoch: 52.4, 53.8, 54.5, 55.0, 54.9, 55.2, 55.5, 55.4,
55.8, 55.7. The curve is flat after ~epoch 6.
*Decision:* ship the final epoch's checkpoint. It is 0.0005 below the
best epoch — inside run-to-run noise, and not worth the complexity of
early stopping at this stage.
*Cost:* a future longer run (or a bigger model) should add best-on-val
checkpointing; as written, a run that overfits late would ship the
overfit weights. Flagged rather than fixed because nothing here overfits
yet — train and val loss are still moving together.

**D5.9 — Report mean placement, not TrueSkill, as the headline.**
In the Phase 5 gate imitation shows mean rank 1.08 (best) but a *lower*
TrueSkill conservative estimate than greedy.
*Why:* the gate seats imitation twice and each baseline once. TrueSkill
rates them as separate teams and the conservative sigma-merge keeps the
less certain of the two seats, which penalizes the duplicated agent.
Mean placement over all seats has no such artifact.
*Cost:* TrueSkill stays in the report (it is the right tool for
many-agent round-robins) but is not the number a gate asserts on.

**D6.5 — Phase 6's search rung does NOT clear its gate. Reported, not buried.**
Paired measurements against the imitation policy it is built on (both
agents seated twice per game, so setup and seat effects cancel within
each game; negative favours MCTS):

    MCTS(64)  vs imitation:  -0.125 rank-sum, 40 games   (se ~0.28, t ~0.4)
    MCTS(128) vs imitation:  +0.330 rank-sum, 100 games  (se 0.281, t 1.18)

Neither is significant, and the *sign flips the wrong way* as
simulations increase: at 128 sims MCTS scores fewer VP (62.1 vs 64.2)
and places worse. Against greedy, MCTS wins comfortably (+0.850), so the
search is not broken -- it simply fails to add anything over the policy
that supplies its priors.

The master plan's Phase 6 verify clause is "each rung beats the
previous in mirrored matches". **It is not met.** Recording that plainly
rather than reporting the +0.850-vs-greedy number and moving on.

Leading hypotheses, in the order worth testing:
1. *Search amplifies value error.* The value head is trained on states
   from human games; MCTS deliberately explores states humans never
   reach, and then trusts the value head there. More simulations means
   more weight on the least reliable estimates -- which matches the sign
   flip between 64 and 128.
2. *The tree is far too shallow to matter.* One Terra Mystica turn is
   several decisions (fresh action, continuations, `done`), and the
   branching factor is ~24. 128 simulations is roughly one turn of
   lookahead spread across four players -- not enough to see a
   strategic consequence, but enough to inherit evaluation noise.
3. *Argmax brittleness, again.* The move is chosen by argmax visit
   count, and D5.6 established that argmax over this policy loses to
   sampling. Cheapest to test, so tested first.

*Cost of stopping here:* Phase 6 ships a correct, tested search that is
not yet an improvement. That is a real result about this value function,
not a bug to hide -- and it makes the case for the self-play phase
(which trains the value head on states the search actually visits)
rather than undermining it.

**D6.6 — MCTS samples its move too (temperature on visit counts).**
Applying D5.6's finding to search, paired vs imitation, 100 games each:

    MCTS(64), argmax visits:  -0.125 (40 games, underpowered)
    MCTS(128), argmax visits: +0.330 +/- 0.281  (t +1.18, wrong way)
    MCTS(64), T=1.0 sampling: -0.180 +/- 0.284  (t -0.63, right way)

Sampling puts the sign back in search's favour and recovers the VP lead
(63.7 vs 62.2), consistent with the argmax brittleness D5.6 found in the
raw policy. It is still not statistically significant, so the default is
set to sampling on the strength of *two* consistent findings rather than
on this one underpowered comparison.

*Power:* detecting an effect this small (~0.18 rank-sum, sd ~2.8) at
t=2 needs roughly 1,000 paired games -- ~2 hours of local compute. A
confirmatory run at that scale is the right way to settle it; anything
smaller re-measures noise.

**D6.7 — Settled: search adds nothing measurable. 1,000 paired games,
64 sims, sampled visits, 0 errors:**

    paired rank-sum diff: -0.005 +/- 0.083 (se)   t = -0.06
    MCTS      mean rank 1.464   mean VP 63.2
    imitation mean rank 1.466   mean VP 62.7

This is not "not significant" -- it is a tight zero. The 95% interval
(+/-0.16 rank-sum, i.e. +/-0.08 mean placement per seat) rules out even a
small benefit. Every earlier wobble (+0.125, -0.33, +0.18) was noise, as
the power analysis predicted.

*Conclusion:* 64-simulation max^n search over this value head is exactly
as strong as sampling the policy directly, and costs ~75x more compute
per game (3.6 s vs 48 ms). Nobody should run it in that configuration.

*What it isolates:* the bottleneck is the value function, not the search
code. The search machinery is correct (it beats greedy by +0.85, it
explores, its vectors re-base properly). It has nothing useful to
search *with*: a value head trained only on human-reached states cannot
rank the off-distribution states search generates, so deeper lookahead
averages noise. This is the precise failure mode self-play fixes --
training the value head on the states the search actually visits -- and
it is why Phase 6b is the indicated next step rather than a speculative
one. Repeating this measurement is the acceptance test for any future
value head.

*What this does not license:* claiming search works. Until a
significant result exists, the honest summary is "search is at best a
small gain over its own prior, and only when its move choice is
sampled."

**D6.8 — Self-play RAN (2,400 games, 789k records) and made the agent
WORSE. The diagnosis is that self-play cannot bootstrap from a search
with no edge.**

Configuration: 6 iterations x 400 games, 16 simulations, lambda_kl 1.0,
lr 5e-5, 9 parallel workers, ~2,600 games/hour. Training signals looked
healthy throughout -- value loss improved monotonically (0.0013 ->
0.0010), policy loss oscillated without diverging, KL stayed bounded
(0.055 -> 0.13 -> 0.11). Nothing in the loss curves predicted the
outcome, which is exactly why arena measurement is the gate.

Measured, paired, vs the imitation net it started from:

    self-play policy vs imitation:   +0.554 +/- 0.163  (t +3.40)  SIGNIFICANTLY WORSE
    self-play MCTS vs its own policy: -0.104 +/- 0.171  (t -0.61)  still zero
    VP: 61.1 vs 64.7

Drift curve (each iteration checkpoint vs imitation, 120 games each):

    iter 1: -0.075 +/- 0.250   (KL 0.055)  level
    iter 3: +0.458 +/- 0.235   (KL 0.132)  worse
    iter 6: +0.450 +/- 0.226   (KL 0.110)  worse, plateaued

Degradation is *progressive and tracks KL drift* -- iteration 1 is
harmless, damage appears as the policy leaves the human anchor, and both
plateau together.

*The diagnosis.* Self-play trains the policy toward the search's visit
distribution. D6.7 established that this search is exactly as strong as
the policy (-0.005 +/- 0.083 over 1,000 games). So the loop is
distilling a teacher with **no edge over the student** -- every iteration
is a lossy copy, and the KL anchor is the only thing keeping it from
collapsing outright. That predicts precisely what was observed: damage
proportional to distance from the anchor, plateauing when KL plateaus.

Training the value head on 789k search-visited states also failed to
make search pay (second row above). That was the leading hypothesis from
D6.5 for *why* search adds nothing, and this is evidence against it at
this scale.

*What this changes.* The fix ordering is now settled, and it is not the
one the master plan assumed:
1. **Search must acquire an edge first** -- deeper search, a better
   value target (e.g. bootstrapped n-step returns rather than final-VP
   shares only), or a stronger evaluator. Until MCTS beats the policy,
   self-play has nothing to teach with.
2. **Only then** does self-play bootstrap, because only then is the
   teacher stronger than the student.
Raising lambda_kl would reduce the damage but cannot create an edge --
at lambda -> infinity the policy simply stops changing.

*What is kept:* the loop itself is correct, parallel, checkpointed, and
cluster-portable (~2,600 games/hour on 9 local workers). The imitation
checkpoint remains the strongest agent and is what `data/checkpoints/
imitation/` still points at. The self-play run is preserved under
`data/checkpoints/selfplay_v1/` as the negative result it is.

---

## Calibration

**C1 — Absolute strength: the agents score less than half what the
humans they imitate score.**

Corpus (3,553 Div 1-3 games, from `games_meta.parquet`):

    human VP per player: mean 136.3  median 137  p10 116  p90 157
    human winning score: mean 153.4
    human table total:   mean 545.2

Our agents, same map and setups:

    imitation ~65 VP    greedy ~65 VP    random ~54 VP
    agent table total ~256 VP

The imitation net sits **below the 10th percentile of human scores**,
and its table totals are under half of a human table's. Some of the gap
is structural -- a weak table produces less total VP because nobody
develops an economy worth leeching from, so weakness compounds -- but
that is the diagnosis, not a defence: these agents do not build enough.

Note that greedy scores about the same ~65. The imitation net reliably
*beats* greedy head to head (1.08 vs 1.33 mean placement) without
extracting materially more value from the board; it is winning on
relative placement inside a weak field.

**Why 55.7% move-matching coexists with this.** The per-verb accuracy
breakdown is the useful artifact: the net is strong where options are
few and conventions clear (`leech` 91%, `gain_favor` 68%, `build` 62%)
and weak exactly where Terra Mystica is won (`send` 12%, `dig` 29%,
`transform` 35%, `upgrade` 37%). It has learned the game's *grammar* --
what a plausible move looks like -- without its *strategy*. Move-match
accuracy flatters it because easy decisions are numerous.

**Honest label:** advanced beginner. Legal, superficially sensible play;
far below the tournament players in its training data.

**How this reframes Phase 6.** D6.5-D6.8 treated "search adds nothing" as
a puzzle about the value head. A simpler reading is now available: when
the base policy is this far from competent, one turn of lookahead over
its own weak evaluations has very little to work with. Both readings
predict the same fix ordering (make the base agent stronger first), so
the plan does not change -- but the value head is no longer the only
suspect.

**What would replace inference with measurement:** an external baseline
nobody here wrote. `lvandeve/tmai`'s heuristic AI is the one the master
plan names. "Beats greedy" is currently measured against a yardstick I
wrote myself, which is exactly the kind of self-grading this calibration
exists to distrust.

**C2 — External baseline: tmai's `ai_lode` scores ~1.5x what our
imitation net scores.**

`tools/tmai_headless.js` runs lvandeve/tmai (an independent Terra
Mystica implementation with a hand-written 2013 heuristic AI) headless
in Node -- its game logic is DOM-free, so browser shims plus its own
synchronous `gameLoopBlocking` are enough. 25 four-player all-AI games,
deterministic seed, standard map, no expansions:

    Div 1-3 humans     136.3 VP/player   winner 153.4   table 545.2
    tmai ai_lode        98.9 VP/player   winner 115.5   table 395.5
    our imitation net  ~65   VP/player                  table ~256
    our greedy         ~65   VP/player
    random             ~54   VP/player

*Why this matters more than the arena numbers.* Every prior strength
claim was measured against a greedy heuristic written in this repo --
self-grading. An independent AI that does no learning, no search, and
never saw the corpus scores **1.5x our net** and gets roughly
three-quarters of the way to human level. Our net gets barely halfway,
and sits closer to random (54) than to ai_lode (99).

*Consequence.* "Beats greedy" was never evidence of much: greedy and the
net score identically in absolute terms (~65), and the net wins only on
relative placement inside an equally weak field.

This also gives D6.5-D6.8 a simpler explanation than the value-head
hypothesis: search over a policy this far below competent has little to
work with, and self-play distilling that search made things worse. **The
base agent is the bottleneck.** Work on the policy (bigger model,
better features, richer targets) before more work on search.

*Comparability caveat:* tmai uses its own engine and its own random
setups (ours are corpus-sampled), and its scoring differs in details.
Treat 98.9 as "roughly 100", not a precise figure. The gap is far too
large to be an artifact of that.
