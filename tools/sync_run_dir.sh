#!/usr/bin/env bash
# Keep a self-play run directory in sync between a learner host and a
# generation host (exercised locally in tests/test_selfplay_review_fixes.py;
# not exercised over a network).
#
#   tools/sync_run_dir.sh LEARNER_HOST:/abs/path/to/R /abs/local/R [INTERVAL_SECS]
#
# Run this ON EACH GENERATION HOST next to its actors. Every interval it
#   1. PUSHES  local  R/shards/*.npz  -> learner  R/shards/   (append-only:
#      --ignore-existing so nothing is ever rewritten; *.tmp excluded
#      because an actor may still be writing it)
#   2. PULLS   learner R/current.pt   -> local    R/current.pt  (rsync writes
#      a temp file in --temp-dir and renames it, so actors never observe a
#      half-copied checkpoint; --times carries the mtime the actors key on)
#   3. MIRRORS learner R/STOP         <-> local   R/STOP  INCLUDING REMOVAL:
#      --delete restricted to that one file, so when the learner host's
#      owner removes STOP (the launcher, or you, once the learner stopped)
#      the mirror drops it too and a later actor start does not exit at
#      once. Actors also discard a STOP older than their own start.
#   4. PULLS   learner R/DONE         -> local    R/DONE  (completion marker
#      for run_watchdog.sh).
#
# The learner host runs nothing extra. Multiple generation hosts never
# collide because shard names carry the actor id -- give every actor
# process across the cluster a distinct --actor-id.
#
# Bandwidth: shards are ~69 B/record (~20 KB/game); current.pt is ~45 MB
# and changes once per learner step (minutes apart), so a 30-60 s
# interval is generous. rsync errors are logged to stderr, never hidden.
# SYNC_MAX_ROUNDS=N (env) stops after N rounds (tests); default 0 = forever.
set -uo pipefail

REMOTE="${1:?usage: sync_run_dir.sh LEARNER_HOST:/path/R /local/R [interval]}"
LOCAL="${2:?usage: sync_run_dir.sh LEARNER_HOST:/path/R /local/R [interval]}"
INTERVAL="${3:-30}"
MAX_ROUNDS="${SYNC_MAX_ROUNDS:-0}"

mkdir -p "$LOCAL/shards"

# run_rsync LABEL ARGS...: rsync with its stderr captured and logged.
run_rsync() {
  local label="$1"; shift
  local err
  if ! err=$(rsync "$@" 2>&1 >/dev/null); then
    echo "sync: $label failed (rc=$?): $err" >&2
    return 1
  fi
}

round=0
while true; do
  # 1. push shards (never overwrite, never send partial writes)
  run_rsync "shard push" -a --ignore-existing --exclude '*.tmp' \
      "$LOCAL/shards/" "$REMOTE/shards/" || true

  # 2. pull the checkpoint atomically (temp file + rename on our side)
  run_rsync "checkpoint pull" -a --times --temp-dir="$LOCAL" \
      "$REMOTE/current.pt" "$LOCAL/current.pt" || true

  # 3. mirror STOP, including its removal
  run_rsync "STOP mirror" -a --delete --include='/STOP' --exclude='*' \
      "$REMOTE/" "$LOCAL/" || true

  # 4. pull DONE (the learner's completion marker, read by run_watchdog.sh)
  run_rsync "DONE pull" -a --include='/DONE' --exclude='*' \
      "$REMOTE/" "$LOCAL/" || true

  round=$((round + 1))
  if [ "$MAX_ROUNDS" -gt 0 ] && [ "$round" -ge "$MAX_ROUNDS" ]; then
    break
  fi
  sleep "$INTERVAL"
done
