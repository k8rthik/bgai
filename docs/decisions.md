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

---

## The free-action bug (2026-08-05) — and what it invalidated

**C3 — The agents were converting their economies away, because the
DRIVER kept asking them to.**

The calibration in C1/C2 gave a verdict (65 VP vs humans' 136, tmai's
99) but not a cause. `tools/diagnose_gap.py` compared end-of-game board
state against real corpus games and found the deficit was not uniform:

    metric        agent   human   ratio
    structures     3.74   12.87    0.29
    trading posts  0.17    3.32    0.05
    towns          0.04    2.33    0.02
    shipping       0.42    2.30    0.19

Not "slightly worse at everything" -- barely playing. `tools/
diagnose_temp.py` found why: **49% of all agent decisions were
`convert`**, against 8.3% for humans, and only 4.5% were builds.

*The cause was mine, not the policy's.* Convert and burn are free (no
action), so the turn protocol re-offered them after every one. The agent
was asked "convert again?" dozens of times per turn, each with a fresh
chance of yes. A human never faces that prompt: they submit an entire
turn as one ledger row (mean 0.12 converts, >2 in 0.12% of turns). This
was a train/inference mismatch introduced by the driver -- the policy
was fine.

*The fix* (`FREE_ACTIONS_PER_TURN = 1`) is monotonic in the cap:

    cap  VP    structures  towns  convert-share
    0    93.1  9.93        1.05   0.001
    1    96.0  9.93        0.90   0.183   <- chosen
    2    80.3  8.00        0.57   0.301
    3    70.9  6.15        0.30   0.379
    inf  61.0  3.71        0.04   0.485   <- old behaviour

**VP 65 -> 96, structures 3.7 -> 9.9** -- from "closer to random than to
a 2013 heuristic" to level with tmai's 98.9.

**D5.6 and D6.6 are SUPERSEDED.** Both concluded that sampling beats
argmax. The measurements were real; the explanation was wrong. Under the
uncapped loop a deterministic argmax agent re-picked the *same* convert
forever, so sampling looked better by escaping a trap the driver had
set. With the cap, argmax wins overwhelmingly:

    argmax vs sample: -1.938 +/- 0.169 mean rank (t = -11.44, 160 games)
    VP: 98.7 vs 81.2

Both defaults are argmax again.

*The methodological lesson, which is the real content of this entry:* a
statistically strong result (t = 3.40 for sampling, at the time) can
have entirely the wrong cause attached to it. What broke the tie was not
more games -- it was measuring an **absolute** quantity (VP, structures,
towns) against an **external** reference (human corpus, tmai), instead of
only ever comparing our agents to each other. Relative-only metrics
cannot see a defect that handicaps every agent equally, and every agent
in the arena shared this one.

*What must now be re-measured, because all of it was run on a crippled
agent:* the Phase 5 gate margins, D6.5/D6.7 (search vs policy), and D6.8
(the self-play regression). The self-play run in particular trained on
2,400 games generated by an agent that spent half its decisions
converting.

**C4 — Re-measured after the fix: Phase 6's conclusion SURVIVES, and is
now better founded.**

MCTS(64, argmax) vs the imitation policy, paired, on the fixed driver:

    400 games, 0 errors
    paired rank-sum diff: -0.105 +/- 0.134  (t = -0.78)
    MCTS      mean rank 1.445   mean VP 94.5
    imitation mean rank 1.498   mean VP 94.1

Still not significant. The sign favours search and the magnitude is
tiny -- the same verdict as D6.7, but now measured on an agent that
scores ~94 VP instead of ~61, i.e. one playing recognisable Terra
Mystica at roughly tmai's level.

*This weakens my own preferred explanation.* C3 suggested the earlier
null results might simply reflect "the base policy is too weak to search
over". The policy is now 50% stronger and search still adds nothing, so
that explanation is not sufficient. The remaining candidates from D6.5
stand: the value head cannot rank off-distribution states, and/or 64
simulations over a ~24-wide branching factor with multi-decision turns
is far too shallow to see a strategic consequence.

*Also re-measured:* the full slow suite (3,552-game replay oracle, legal
containment, both arena gates) passes unchanged on the fixed driver.

*Still stale:* D6.8's self-play regression. That run trained on 2,400
games generated by the pre-fix agent, so its conclusion ("self-play
distils a teacher with no edge") rests on data from a crippled
generator. The *reasoning* still holds -- C4 confirms search has no edge
over the policy even now -- but the experiment itself should be rerun
before the number is quoted.

**C5 — The value head, not the search, is Phase 6's bottleneck.**
C4 left two candidates from D6.5 standing: the value head cannot rank
off-distribution states (H1), and/or 64 simulations is too shallow to
see a strategic consequence (H2). They had never been separated.

Diagnostic A separates them without new data. `scoring.projected_vp`
computes a per-faction VP projection from any state -- banked VP plus
cult standings, network standings, and resource conversion, under the
real final-scoring rules -- and is exact against `final_scoring` on
terminal states by construction, so it stays correct in exactly the
off-distribution positions the learned head has never seen.
`agents/leaf_eval.blend` mixes it into MCTS's leaf evaluation as
`(1-w)*learned + w*computed`. Paired vs the imitation policy, 64 sims,
100 games per cell, mirrored seats (the D6.5-D6.7 protocol; negative
favours MCTS):

    w=0.00   +0.010 +/- 0.130   t=+0.08    MCTS 93.4 VP   policy 93.0
    w=0.25   -0.140 +/- 0.140   t=-1.00    MCTS 93.6 VP   policy 91.9
    w=0.50   -0.290 +/- 0.133   t=-2.18    MCTS 95.2 VP   policy 91.1
    w=1.00   -0.130 +/- 0.125   t=-1.04    MCTS 94.2 VP   policy 93.4

The w=0 control reproduces C4/D6.7's tight zero, so the harness is
measuring the same thing. Search does not change between cells -- only
the leaf evaluator does -- so **H1 is supported: the search was faithful
all along and was amplifying an evaluator that goes blind off the human
distribution.**

The curve is an inverted U peaking at w=0.5, and that shape is itself
informative. Pure computed value (w=1.0) beats the learned head alone
but loses to the blend, so the learned head does carry real information
-- position quality the projection cannot see -- it simply cannot be
trusted alone where it has no support. Neither evaluator is sufficient;
the blend is.

*Power, honestly.* Only w=0.5 crosses t=2, and that is one cell of four
at n=100 (se ~0.13). The monotone rise from the replicating control and
the coherent inverted-U are what carry the argument, not any single
cell. Treat the peak's magnitude as unconfirmed until a run at D6.7's
scale (~500-1,000 paired games at w=0.5 vs w=0, se ~0.06-0.08) settles
it. Default remains `value_blend_w=0.0`; nothing ships on a t=-2.18.

*What this implies for data (open, not yet measured).* The crutch works
because it is computed rather than learned. The corresponding repair is
a value head whose training states cover where search actually goes --
which is what broad-population games provide, since final VP share is an
objectively correct label regardless of who played the game, while the
same games are contaminated supervision for the *policy* head. That
argues for splitting their training distributions rather than retraining
both on one enlarged corpus.

*And it re-opens D6.8.* Self-play failed because it distilled a teacher
with no edge over its student. If blending gives search a real edge,
that precondition may now be satisfied -- so D6.8 should be rerun
against a blended-value searcher before its conclusion is quoted again.

**C6 — Depth is not innocent either: H1 and H2 both have support, and they
may compose.** Diagnostic B, the other half of C5: same value head (w=0),
same seed and setups across rungs, so only search depth varies. Paired vs
the imitation policy, negative favours MCTS:

    sims=  64   n=100   +0.000 +/- 0.126   t=+0.00   MCTS  93.6 VP   policy 92.8
    sims= 256   n= 48   +0.062 +/- 0.217   t=+0.29   MCTS  96.2 VP   policy 94.4
    sims=1024   n= 16   -0.500 +/- 0.428   t=-1.17   MCTS 102.6 VP   policy 92.2

The first two rungs say what C4 and C5's control said: at shallow depth,
search on this value head adds nothing. The 1024 rung does not. Its rank
diff favours search and its VP gap (+10.4) is more than double the best
the blend sweep produced (+4.1 at w=0.5) -- with no blending at all.

*This corrects a conclusion I had already drawn.* After the 64 and 256
rungs I recorded that depth alone buys nothing and read the ladder as
evidence against H2. The 1024 rung is the wrong shape for that. The
honest statement is that **both hypotheses have support**: the evaluator
is blind off-distribution (C5), *and* 64 simulations is too shallow to
see a strategic consequence (D6.5's hypothesis 2, now with a data point
behind it).

*Power, honestly.* n=16 at se 0.428 is the weakest cell in either
diagnostic -- a true zero produces -0.500 or better about 12% of the
time, so this is a lead, not a finding. It cost 199 s/game (22x the
64-sim cost); a confirmatory run at n=100 is ~5.5 hours.

*The experiment this implies, and nobody has run:* the interaction.
Every cell so far varies one factor -- C5 blends at fixed shallow depth,
C6 deepens at fixed blind value. If both effects are real they should
compose, so 1024 sims at w=0.5 is the cell worth measuring next, against
both the policy and against each single-factor arm. If it lands, Phase 6
has its edge and D6.8's self-play precondition (a teacher stronger than
its student) may finally be satisfied.

*Caution for whoever runs it:* prefer VP-and-rank together over rank
alone. Both diagnostics show cells where MCTS accumulates materially
more VP while placing no better (256 sims: +1.8 VP, worse rank), which
is the same "develops well, converts it badly" signature the per-verb
accuracy gaps show. Placement is the objective; VP is the earlier signal.

**C7 — Search finally has its edge: pruned, deepened, batched MCTS beats
its own policy 35.8% to 15.0% (2026-08-13).** The composition C6 asked
for was run — but the winning combination was not blending, it was
spending the budget deeper via **prior pruning**: `top_k=8` (policy
top3 is 84%, so the right move is almost always in the set),
`max_depth=48`, batched leaf evaluation with virtual loss, 512
simulations, on the pop_simplex net (rank loss + simplex value head,
trained on the broad-population value split). Head-to-head vs the raw
policy: 35.8% vs 15.0% win rate, 106.4 vs 96.6 VP, mean place 1.15 vs
1.81 (n=120, 0 errors; `data/h2h_depth.log`). Rank and VP move
*together* this time — the "develops well, converts badly" signature is
gone. Against the human distribution the agent stands at the 22.4th
percentile (from 12.9th).

*The one-variable sweep around that config* (96 games, 4-seat, all four
agents at 512/k8/d48 unless varied): more_sims=1024 won at 34.4%
vs base 25.0%, while deeper (d96, 21.9%) and narrow (k4, 21.9%) both
*lost* ground (`data/sweep_deep.log`). At n=96 the 1024 result is ~2σ —
directional, not settled — but the shape is consistent: k8/d48 is a
sweet spot, and additional budget should buy *simulations*, not depth
or narrowness.

**D6.10 — Self-play generation must search exactly like the arena does
(2026-08-13).** D6.8's failure was distilling a teacher with no edge;
C7 supplies the edge, so self-play is re-armed — but `selfplay.py`
predated pruning/batching and looped over `_simulate` directly, i.e. it
would have generated with precisely the shallow searcher D6.8 proved
worthless. `MCTSAgent.search()` is now the single entry point for
spending a simulation budget (arena `choose_sim` and self-play
`play_game` both call it), and `SelfPlayConfig` carries the C7 search
knobs. Two latent bugs fixed on the way: records stored the
mover-relative seat (always 0) as `faction_id`, so fine-tuning would
have conditioned every position on one faction embedding; and the
trainer rebuilt nets with a default `ModelConfig()`, which loads
simplex-trained weights into an unconstrained value head without error.
An AlphaZero-style temperature cutoff (`temp_decisions=30`) samples
openings ∝ visits and argmaxes after, so late-game value targets track
best play.

**C8 — The external yardstick has been passed (2026-08-14).** Fresh
20-game ai_lode batch (`tools/tmai_headless.js`, `data/ailode_fresh.log`):
97.1 VP/player, winner mean 119.3 — consistent with C2's 98.9/115.5. Our
deep-search agents now average ~103-105 VP/player on their own all-MCTS
tables (sweep_deep, h2h_leg2_search), versus ~65 when C2 was measured on
2026-08-06. Same caveat as C2 in reverse: cross-table absolute VP, not a
head-to-head — but by the only external yardstick this project has, the
learned agent has moved from "closer to random (54) than to ai_lode (99)"
to ahead of ai_lode, in eight days. The remaining calibration that
matters is the human distribution (median 123 VP; we stood at the 22.4th
percentile pre-RL — re-measure after leg 3).

**C9 — The flywheel compounds; the winner-priority objective works
(2026-08-14).** Three self-play legs, each h2h-verified against its
predecessor at the deep-search config (512/k8/d48, n=120 each):

    leg2 (rank term)          beats pre-RL search  32.5% / 18.3%
    leg3 (+winner-priority)   beats leg2           30.8% / 20.8%   109.5 VP

Leg 3 trained with winner_pair_weight=3 and win_weight=0.5 (the user's
prioritize-first-place directive): rank pairs involving the true winner
count 3x and the simplex value head trains winner-identification CE
directly. Held-out HUMAN-state ordering read flat vs leg 2 (0.722/0.562
vs 0.726/0.567) while the h2h moved a full generation -- confirming
(again, after the leg-2 flip) that the self-play distribution is where
search strength lives and the human-val probe is only a mismatched
proxy. Speed: leg-3's finale ran at 505-685 games/hour (2.6-3.1x leg 1)
via the cached setup ids, 10 workers, tree reuse, and the fast-prior
path.

**C10 — Two nulls that sharpen the diagnosis (2026-08-15).** Leg 4
(1024-sim targets, win_weight 0.25, window-2 replay buffer) ties leg 3
h2h (25.8%/27.5% win, better placement 1.41/1.54): the local flywheel's
per-leg gains have gone huge -> solid -> nil at ~750 games/leg. And the
round-scaled budget experiment (512 early/2048 late vs flat 1024, same
net, budget-matched) is a dead tie 25.0/25.0 -- the endgame conversion
weakness is NOT a search-depth problem. Together: local scale is spent,
and the remaining gap to the human median (~15 VP from the 23.7th
percentile) lives in evaluator knowledge, not thinking time. The levers
that remain are structural: cluster-scale games (10^5), value-head
training coverage, and win-prob-maximizing selection.

**C11 — The gap has a face: cult-hoarding myopia (2026-08-15).** Town
audit, agent (838 towns, 200 games) vs rating-banded corpus (610k
towns): the agent takes cult town tiles (TW5/TW6) 55% of the time vs
28% for top-5% humans, despite an already-elite endgame cult score, and
under-picks economy tiles (TW1/TW2/TW8) ~2x while founding 1.28
towns/founding-seat vs 2.3. Expected VP-per-pick is identical (6.73 vs
6.75) -- the error is invisible to VP-greedy metrics; it is a portfolio
error. Combined with C10's decomposition (towns -19, conversions -11 vs
elite; everything else at/above elite), one diagnosis covers all
findings: the evaluator prices immediate legible gains (cult positions
are literal state features) and cannot price compounding economy.
Human seat effect: seat1 29.3%/seat3 21.7% win over 76k games (se
0.16pp) -- mirrored eval design retroactively essential; agent inherits
the seat-1 edge but not the seat-3 penalty (n=200, needs the
agent-games parquet store to settle).

**C12 — Trajectory targets broke the plateau (2026-08-17).** Leg 5
(aux head: per-seat [final abs VP/150, towns/3] predicted from every
recorded state, aux_weight 0.5, win_weight 0.25) beats leg 4 h2h
30.8%/20.8%, place 1.37/1.59, +3.4 VP -- a full generation, after leg 4
tied. The aux loss converged 0.042 -> 0.019 (~+-19 VP final-score
prediction from mid-game states). The diagnostic chain that produced
it: C10 decomposition (towns -19, conversions -11 vs elite) -> C11
cult-hoarding portfolio -> economy-myopia diagnosis -> share targets
are scale-blind -> absolute-VP + town-count auxiliary prediction.
Behavioral before/after (tile portfolio, towns/seat, conversions) via
the agent-games parquet store: batch leg5_256.

**C13 — Erratum to C10/C11's town numbers (2026-08-17, user-caught).**
The corpus `town` bucket bundled each founding row's ENTIRE VP delta
(tile + founding move's score-tile/favor VP + SCORE2 round bonus +
Witches passive) while the agent tracer split tile-only -- inflating
"human town VP" to an arithmetically impossible 27.2/seat (11.7/tile).
Corrected tile-only: all 14.4 / top25 15.6 / top5 16.8, with the excess
reassigned to `action` (elite action VP is therefore ~49.6, ABOVE the
agent's 42.9, reversing C11's "agent exceeds elite action" claim).
Revised gaps vs elite: towns -8, action -7, conversions -11 -- an even
economy-wide spread rather than one hole, which fits the C11 myopia
diagnosis more cleanly and explains C12's broad strength gain without
any single behavior jumping. Lesson pinned: cross-side bucket
comparisons require identical row-splitting rules; conservation checks
catch missing VP, not misattributed VP.

**C14 — The KL leash is not the constraint (2026-08-18).** Leg 6 (leg-5
recipe, lambda_kl 1.0 -> 0.5, sole change) ties leg 5 h2h (25.0/27.5,
identical placement) with a CLEAN drift canary: KL re-equilibrated
0.03 -> ~0.05 and human-val ordering stayed exactly at leg-5 level
(0.711/0.561 vs 0.714/0.560). The policy took 60% more room and neither
gained strength nor drifted -- the anchor was not what blocks
behavioral change. Kills hypothesis B; the evidence concentrates
further on data quantity (cluster scale). In-training metric records
(best policy/rank/win CEs) did not translate to h2h -- as with every
leg, own-distribution training metrics are not an arbiter. Champion
stays selfplay_leg5b.

**C15 — Max-child root choice refuted (2026-08-18).** root=q (argmax
own-seat Q, 5% visit floor) loses to robust-child argmax-visits
15.8%/34.2%, -6 VP, same net both sides. Low-visit Q is noisy and
selection-biased upward; the winner-priority objective belongs in the
LOSS (C12, where it won a generation), not in the decision rule.

**C16 — c_puct belongs at 2.5 (2026-08-18).** Never tuned since 1.5 was
set for the pre-simplex value scale, through two head-regime changes.
Directional 4-way sweep, then pre-registered confirmation h2h: 2.5
beats 1.5 28.7%/21.9% (n=320/side, ~2.8 sigma) with identical mean VP
and placement -- more exploration converts equal positions into more
outright wins (higher-variance, more decisive play; the right trade
for win-maximization). Champion eval config: leg5 ckpt, 512 sims,
top_k 8, depth 48, batch 16, c_puct 2.5.

**C17 — GPU inference refuted at this model size (2026-08-19).** The
centralized MPS server (1.7ms forwards, flat to batch 256) HALVES
end-to-end throughput (329 vs ~650 games/hour): a CPU forward at batch
16 costs ~10ms, less than the queue round-trip that would replace it.
In-process MPS (no IPC): b16 worse than CPU, b64 ~5% better -- noise,
pre virtual-loss cost. The 11M-param MLP is too small for GPUs to pay.
Consequences: (1) the cluster request needs CPU nodes only (easier
allocation); (2) the server code stays for a future bigger net, where
the arithmetic flips. Pre-cluster optimization checklist is COMPLETE:
every knob measured, adopted (c_puct 2.5, aux targets, rank/winner
terms, tree reuse, fast-prior, setup cache), or refuted (KL loosening,
max-child, endgame budget, 1024-sim targets, GPU inference).
