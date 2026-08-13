# Faction ability audit — official rulebook vs engine (2026-08-13)

Cross-check of all 14 base-game factions against the official Z-Man
rulebook, Appendix VI (plus the power-action / favor / bonus / town /
scoring appendices, where faction nuances hide in footnotes). Ground
truth note: where the printed rulebook and snellman's implementation
disagree, **snellman wins** — the corpus, the opponents, and the
evaluation distribution all come from it, and the engine is
replay-validated against 3,552 corpus games.

Engine-side facts verified against the working tree at `fafbb0c`
(faction data in `engine/tm/factions_data.py`, special-case logic per
the table in that file's audit trail).

## Verdict summary

| Faction | Ability | Stronghold | Notes |
|---|---|---|---|
| Alchemists | ✅ | ✅ | VP↔C rates incl. 2C/VP endgame; SH 12PW once + 2PW/spade *gained* (all 5 grant paths) |
| Auren | ✅ (none) | ⚠️ **ACTA grants nothing in engine play** | see Gap 2 |
| Chaos Magicians | ✅ | ✅ (after `a480f76`) | 2 favors per TE/SA; 1 dwelling placed last; ACTC pass semantics fixed |
| Cultists | ⚠️ partial | ✅ (+7 VP once) | all-declined→+1PW never fires in engine play; see Gap 3 |
| Darklings | ✅ | ✅ | 1P/spade +2VP/step; dig track unadvanceable; SH ≤3 W→P, priest-pool clamped |
| Dwarves | ✅ | ✅ | 2W tunnel +4VP → 1W after SH; no shipping (BON4 +1 correctly refused); tunnels count in area scoring |
| Engineers | ✅ | ✅ | ACTE 2W bridge, unlimited/round, ≤3 bridges; SH 3VP/own-bridge on every pass |
| Fakirs | ✅ | ✅ | carpet 1P +4VP; range 1→2 via SH (and TW7); counts in area scoring; no shipping |
| Giants | ✅ | ✅ | flat 2 spades to home color (engine *stricter* than Perl: rejects non-home targets outright) |
| Halflings | ✅ | ✅* | +1VP/spade incl. cult-income spades; SH grants 3 spades immediately (*dwelling-on-one-space flow untested in agent play) |
| Mermaids | ✅ | ✅ | river-hex towns (exact 1-river BFS); SH free ship step at 0 VP; endgame river-skip is a documented over-approximation |
| Nomads | ✅ | ✅ | 3rd setup dwelling before CM's 1st; Sandstorm: plain adjacency (no bridges), home color, not a spade, dwelling still paid |
| Swarmlings | ✅ | ✅ | +3W/town (incl. multi-town rows); ACTS free D→TP still triggers leech (snellman-correct) |
| Witches | ✅ | ✅ | +5VP/town; Ride: free + unreachable OK but **forest-only enforced**; leech still fires |

Cross-cutting rulebook footnotes verified: Darklings pay 1P per missing
spade on ACT5/6 (dig cost is priests everywhere); Giants forfeit a
single Phase-III cult-bonus spade (no legal 1-spade transform → settled
as forfeit); Dwarves/Fakirs get no benefit from BON10/BON4 shipping
(SHIP_NONE, `effective_shipping` refuses the passive); Halflings/
Alchemists spade bonuses fire "regardless of the way you get the Spade"
(dig, build-implicit, power actions, cult income — five grant sites).

## Gaps found

### Gap 1 — ACTC ticket survived a pass in the sim driver (FIXED, `a480f76`)

Rulebook: "take 2 Actions one after another — passing is also considered
an Action." The engine offers `pass` as a double-turn action correctly,
but `round_flow._advance_actions`' extra-action branch checks neither
`passed` nor `dropped`, so a CM who passed with a banked action stayed
structurally active; legal moves degenerate to conversions (once) or
nothing → the empty offer that crashed self-play iteration 5.
`live_driver` has clamped this since task 14; the sim driver now has the
same clamp. Defensive guards in MCTS/self-play (from `fafbb0c`) remain
as backstops.

### Gap 2 — Auren ACTA grants nothing in engine-driven play (FIXED, `8be2cce`)

Fixed as designed below — and the fix healed BON2 and FAV6 as well,
which rode the identical no-op path. Validated by unit tests plus a
320-game stratified corpus replay sample (319 clean; the single failure
reproduces identically pre-fix and is an unrelated population-corpus
edge).

Original finding:

The SH special action `ACTA` (+2 cult on one track) is *offered* to
agents, and the engine accepts it, but both gain keys are replay-only
no-ops: in corpus replay the cult steps arrive as companion `+2CULT`
ledger rows; in engine-driven play nothing generates them. **Auren's
stronghold is a rules-legal no-op in self-play and the arena**, so every
Auren game in training data undervalues the SH. Fix shape: have ACTA
push a `cult_choice`-style pending (amount 2, same-track), mirroring the
Cultists flow that already works in both replay and live play; must keep
the 3,552-game replay validation green (companion rows must consume the
pending, not double-grant).

### Gap 3 — Cultists' all-declined bonus never fires in engine play (FIXED, `8be2cce`)

Fixed via a ``cultist_bonus_due`` pending pushed by the resolving
decline (errata-gated): replay's bracket row and the live driver's
forced-answer path consume it through the same granting handler. The
cached-amount-vs-live-actual trigger divergence remains open (matches
validated corpus behaviour; revisit only if replay evidence appears).

Original finding:

Rulebook: if every opponent refuses the power, Cultists gain exactly
1 power (and nothing if no one *could* take). The engine implements this
only as a replay verb triggered by the ledger's literal
"[all opponents declined power]" row, additionally gated on the
`errata_cultist_power` option (default off). In self-play/arena the
Cultists get their cult step when someone accepts, but never the
consolation power when all decline. Milder than Gap 2 (the common case
works) but biases the agent against Cultists builds that get declined.
Also: the accept-side trigger keys on the offer's cached amount rather
than the recomputed actual (Perl gates on actual > 0) — an edge-case
divergence.

## Deliberate divergences (documented in code, snellman-side or benign)

- Giants: engine rejects explicit non-home transform targets (Perl
  allowed; 202/202 corpus rows agree with the engine's choice).
- Dwarves/Fakirs: a second teleport in one turn is allowed-but-charged
  (Perl dies). Unreachable via legal-move generation.
- Mermaids: endgame network `river_skip` is an over-approximation of
  Perl's `{skip}` table; town founding uses the exact 1-river BFS.
  Replay-validated on every corpus mermaids game to date.
- ACTA same-track enforcement skipped in the replay path (corpus never
  splits the +2).
- `strict_darkling_sh` and `strict_leech` options parsed but inert.

## Rulebook sources

- Z-Man rulebook (2016 printing), Appendix VI "The Factions" + App. I
  (power actions), II (favors), III (scoring tiles), IV (bonus cards),
  V (town tiles). PDF: cdn.1j1ju.com `c8-terra-mystica-rulebook.pdf`.
