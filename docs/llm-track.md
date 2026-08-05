# LLM track design (Phase 7)

The master plan's thesis is that "LLM vs hand-built AI" is only
interesting if the LLM is pushed to its actual ceiling, so tool use is
rung 2 of 7, not the finish line. Each rung is measured before the next
is added, producing a capability-vs-strength curve.

## Status

| Rung | What it adds | Built | Measured |
|---|---|---|---|
| L0 | serialized state + numbered legal moves | yes | no |
| L1 | rules digest, faction identity, strategy priors | yes | no |
| L2 | engine tools (`apply`, `neighbors`) | yes | no |
| L3 | retrieval over the Div 1-3 corpus | yes | no |
| L4 | propose->critique->pick, persistent game plan | yes | no |
| L5 | LLM as MCTS policy prior | not built | no |
| L6 | fine-tuned open-weights model | not built | no |

**Nothing is measured, and that is a credentials problem, not an
oversight.** No `ANTHROPIC_API_KEY` or Azure OpenAI credentials exist in
this environment, so every rung's *plumbing* is verified against
`MockProvider` (prompts, tool loop, answer protocol, arena integration,
fallback accounting) while its *playing strength* is unknown. Reporting
a strength number here would mean inventing one. When keys exist, the
arena run is a single command per rung — the harness is finished.

## The answer protocol, and why it is an index

The model never names a move; it returns the **index** of one of the
legal moves the engine offered. An illegal move is therefore
inexpressible, which gives the LLM track the same structural legality
the imitation net gets from candidate scoring. This matters for
fairness: a bare LLM that hallucinated illegal moves would score badly
for a reason that has nothing to do with Terra Mystica judgement.

Malformed answers fall back to the previous valid index and are counted
in `LLMAgent.fallbacks`. A rung that cannot follow the protocol shows up
as a fallback count in the report rather than as a mysteriously weak
player. (An early parser bug matched at most three digits, so "9999"
became move 9 — a protocol failure silently scoring as a choice. Fixed;
the parser now range-checks whole integers.)

## Fairness classes

The master plan defines Class P (pure LLM: engine tools for
legality/simulation/arithmetic are fine, nothing derived from our
trained nets) and Class C (centaur: anything, including LLM+MCTS).

This shaped a real design decision in L3. The obvious way to build a
"similar positions" index is to embed states with the imitation net's
256-d torso output — it is right there and it is good. Doing so would
quietly move the LLM into Class C, because its retrieval quality would
then depend on our trained engine AI. So L3 indexes the **raw encoded
features** instead: it uses the corpus (public data) but not the model.
Slightly worse retrieval, honest classification.

L5 (LLM as MCTS prior) is Class C by construction and will be reported
separately, never merged into the headline "LLM vs engine AI" number.

## Cost control

`Usage` tracks tokens and calls per provider, because the rungs differ
by two orders of magnitude in cost per move: L0 is one call, L4 is
1 + N + 1 calls plus periodic plan updates. The arena's mirrored design
means expensive rungs can play fewer, better-chosen games (paired
comparisons on identical setups) rather than needing thousands.

## What is left

- **L5** needs `MCTSAgent` to accept an external policy prior. The hook
  is small — `_expand` already isolates prior computation — but it needs
  an API budget to be worth building, since every node expansion becomes
  an LLM call.
- **L6** (fine-tuning) needs either the local GPU box or the university
  Azure allocation, and a (state, move) SFT export. The extraction
  pipeline already produces exactly those pairs for the imitation net;
  the export is a formatting change, not new machinery.
