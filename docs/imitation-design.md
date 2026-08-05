# Imitation learning design (Phase 5)

Companion to `engine-design.md`. Records the decisions behind
`src/bgai/training/` and `src/bgai/agents/imitation.py`, and the
invariants that must hold if any of it is changed.

## Candidate scoring, not a fixed action space

Terra Mystica's command space is huge and sparse: a move can name any of
113 hexes, 43 tiles, 7 colors, 4 cults, and several amounts. A fixed
output head over that product would be mostly dead weights and would
need hand-maintained legality masks.

Instead the net scores *the candidates the engine actually offers*. The
torso embeds the state into a vector; a small MLP embeds each candidate
move's field ids into the same space; the policy logit for a candidate is
their dot product, softmaxed over that decision's candidate set. Three
consequences:

- **Legality is structural, never learned.** The policy is defined only
  over `legal_moves_for`'s output, so an untrained net is still a legal
  player (`tests/test_imitation_agent.py` plays a full game with random
  weights).
- **No action-space maintenance.** A new verb or tile changes the move
  featurizer's vocabulary, not the network's output shape.
- **The arena's canonical ordering is load-bearing.** Training indexes
  candidates in `arena.sim.canonical_moves` order and the agent scores
  the arena's offer in that same order, so train and play agree.

## Mover-relative encoding

Every encoding is from the acting faction's point of view: seat blocks
are `state.setup.factions` rotated to the mover, hex ownership is a
mover-relative seat one-hot, a leech/decline target is a relative seat
id, and the value head predicts the 4 seats' final-VP *shares* in that
same rotated order. The net therefore never has to learn "seat 3 means
me this time" -- one function generalizes across seats, which matters
because the arena rotates seats deliberately.

Faction identity still enters explicitly, as an embedding plus a
per-seat faction one-hot: TM's asymmetry is the point, not noise to be
normalized away (master plan asymmetry requirement 4).

## Labels come from the shared matcher

A ledger command and the generated candidate denoting the same decision
are not field-identical -- the ledger records resolved bookkeeping the
generator never enumerates (a `decline`'s target and power amount, a
`convert`'s unit count) and omits detail the generator produces (a
`transform`'s target color). `engine/tm/legal_match.py` owns the per-verb
identity keys, and is used by BOTH the training label lookup and the
corpus containment sweep, so the 29-game slow sweep is also the label
matcher's regression test. Duplicating that logic is how labels silently
drift off by one.

## Split and weighting

- **Time-based split:** seasons >= 67 are validation. A random split
  would let a game from 2024 inform a prediction about 2019 through
  shared table context; a time split cannot.
- **Division weights** 1.0 / 0.8 / 0.6 for Div 1 / 2 / 3, applied as
  per-sample cross-entropy weights (master plan: table-quality
  weighting).
- **Version stamp:** every shard manifest and checkpoint carries
  `ENCODING_VERSION`. The loader and the agent both refuse a mismatch
  rather than silently feeding garbage into a trained net.

## Known data notes

- 1 game (`4pLeague_S53_D1L1_G3`) cannot be extracted: it is the same
  pinned server-side scoring anomaly the replay suite documents.
- 22 commands out of ~1.2M are `burn n1=0` -- a zero-power no-op the
  ledger records and the generator (correctly) never offers. Those
  decisions are skipped.
- Decisions with a single candidate carry no choice and are skipped;
  the mean surviving decision has ~24 candidates (median 22, max 216),
  so random-among-candidates scores 6.3% top-1 / 18.8% top-3 on the
  validation split. That is the floor any reported accuracy beats.
