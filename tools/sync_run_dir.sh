#!/usr/bin/env bash
# Keep a self-play run directory in sync between a learner host and a
# generation host (sketch; not exercised over a network in tests).
#
#   tools/sync_run_dir.sh LEARNER_HOST:/abs/path/to/R /abs/local/R [INTERVAL_SECS]
#
# Run this ON EACH GENERATION HOST next to its actors. Every interval it
#   1. PUSHES  local  R/shards/*.npz  -> learner  R/shards/   (append-only;
#      rsync --ignore-existing so nothing is ever rewritten, *.tmp excluded
#      because an actor may still be writing it)
#   2. PULLS   learner R/current.pt   -> local    R/current.pt  via a temp
#      file + mv so actors never observe a half-copied checkpoint (rsync's
#      own --temp-dir/rename does the same; the mtime change is what the
#      actors key on, hence --times).
#
# The learner host runs nothing extra: it reads whatever lands in
# R/shards/ and writes R/current.pt atomically. Multiple generation hosts
# never collide because shard names carry the actor id -- give every
# actor process across the cluster a distinct --actor-id.
#
# Bandwidth: shards are ~69 B/record (~20 KB/game); current.pt is ~45 MB
# and changes once per learner step (minutes apart), so a 30-60 s
# interval is generous. Stop with Ctrl-C or by creating R/STOP on the
# learner host (it is pulled too, and the actors honour it).
set -euo pipefail

REMOTE="${1:?usage: sync_run_dir.sh LEARNER_HOST:/path/R /local/R [interval]}"
LOCAL="${2:?usage: sync_run_dir.sh LEARNER_HOST:/path/R /local/R [interval]}"
INTERVAL="${3:-30}"

mkdir -p "$LOCAL/shards"

while true; do
  # 1. push shards (never overwrite, never send partial writes)
  rsync -a --ignore-existing --exclude '*.tmp' \
      "$LOCAL/shards/" "$REMOTE/shards/" || echo "sync: shard push failed (retrying)" >&2

  # 2. pull the checkpoint atomically (rsync writes a temp file and renames)
  rsync -a --times --inplace=no --temp-dir="$LOCAL" \
      "$REMOTE/current.pt" "$LOCAL/current.pt" 2>/dev/null \
      || echo "sync: checkpoint pull failed (retrying)" >&2

  # 3. propagate a STOP request, if any
  rsync -a --ignore-missing-args "$REMOTE/STOP" "$LOCAL/STOP" 2>/dev/null || true

  sleep "$INTERVAL"
done
