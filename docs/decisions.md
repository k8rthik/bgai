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
