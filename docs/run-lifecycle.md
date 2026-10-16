# Run lifecycle: checkpoints as the contract between code versions

A self-play leg is a sequence of iterations. Each iteration reads
`current.pt`, generates games, trains, and writes a new `current.pt`.
Nothing else is state. That makes the checkpoint the only contract a
code upgrade has to honour, and iteration boundaries the only places
an upgrade may land.

## Rules

1. **A leg is resumable from its own directory.** `--resume RUN_DIR`
   restores iteration number, base seed, anchor, and config from the
   checkpoint. Numbering and seeds continue; nothing is overwritten.
   (Before this existed, a restart replayed the same seeds from
   iteration 1 -- never restart the old way.)
2. **Stop only at boundaries.** `touch RUN_DIR/STOP`; the trainer
   finishes the in-flight iteration, saves, logs `stopped`, exits.
   `tools/spark_upgrade.sh <host> <leg>` does stop -> sync -> resume.
3. **Every checkpoint and metrics row carries provenance:** git sha,
   dirty flag, encoding version, hostname. A `code_changed` row marks
   the iteration where an upgrade landed, so a games/hour or loss
   shift is attributable.
4. **Only semantics-preserving optimizations roll into a running leg.**
   The bar is parity: identical legal-move sets on corpus replay and,
   for search changes, identical visit distributions on fixed seeds
   (or a quantified, negligible difference). Anything that changes
   what the searcher does -- `c_puct`, pruning, a transposition table
   that alters visits -- starts a *new* leg from the last checkpoint,
   named for the change, and is verified head-to-head. Mixing it into
   a running leg destroys the scaling curve that leg exists to measure.
5. **Pipelined runs upgrade rolling.** Actors are stateless (they
   reload weights per game), so upgrade them one at a time with no
   learner interruption; the learner honours STOP at step boundaries
   and is upgraded last.

## Procedure for a perf change

- Land it on a branch, run the parity gate, measure seconds/game.
- `tools/spark_upgrade.sh aim-spark0N <leg>` per box, staggered so
  the fleet never fully stops.
- Watch the next metrics row: `games_per_hour` should move,
  losses should not.
