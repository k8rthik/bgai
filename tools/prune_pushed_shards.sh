#!/usr/bin/env bash
# Delete a generation host's LOCAL shards once the learner has them.
#
#   tools/prune_pushed_shards.sh LEARNER_HOST:/abs/path/to/R /abs/local/R [MIN_AGE_HOURS] [--apply]
#
# Run this ON A GENERATION HOST, beside its sync_run_dir.sh loop.
# tools/sync_run_dir.sh pushes shards with --ignore-existing and never
# deletes anything, so a gen host keeps every shard it ever wrote: ~20 KB
# per game, ~0.5 GB per host per day. On a box whose disk is shared (and
# nearly full) that is what eventually stops generation.
#
# A shard is deleted only when ALL hold:
#   1. the learner's shards/ directory already lists that exact name --
#      the authoritative copy exists, so --ignore-existing would skip it
#      on every future round anyway and dropping it re-pushes nothing;
#   2. it is older than MIN_AGE_HOURS (default 6) -- slack for a shard
#      written just now, or a sync round still in flight;
#   3. it is not its actor's newest local shard -- a restarted actor reads
#      its seq and game index (its seeds) from that one.
# The remote listing is fetched ONCE per run and treated as the only
# evidence; a remote that cannot be listed aborts without deleting.
# ``*.tmp`` never matches ``is_shard_file`` and is never considered.
#
# Dry run unless --apply is passed, because this deletes training data.
# Typical cron on a tight box (hourly, quiet, real deletions):
#   17 * * * * cd ~/code/bgai && tools/prune_pushed_shards.sh \
#       learner:/home/murug030/code/bgai/runs/scale_1m \
#       /home/murug030/code/bgai/runs/scale_1m 6 --apply >> scratch/prune.log 2>&1
set -euo pipefail

REMOTE="${1:?usage: prune_pushed_shards.sh LEARNER_HOST:/path/R /local/R [min_age_hours] [--apply]}"
LOCAL="${2:?usage: prune_pushed_shards.sh LEARNER_HOST:/path/R /local/R [min_age_hours] [--apply]}"
MIN_AGE_HOURS="${3:-6}"
APPLY="${4:-}"

case "$REMOTE" in
  *:*) host="${REMOTE%%:*}"; path="${REMOTE#*:}" ;;
  *) echo "prune: REMOTE must be HOST:/abs/path, got '$REMOTE'" >&2; exit 2 ;;
esac
case "$MIN_AGE_HOURS" in
  ''|*[!0-9]*) echo "prune: min_age_hours must be a non-negative integer" >&2; exit 2 ;;
esac
[ -d "$LOCAL/shards" ] || { echo "prune: no local shards dir at $LOCAL/shards" >&2; exit 2; }
if [ -n "$APPLY" ] && [ "$APPLY" != "--apply" ]; then
  echo "prune: unexpected argument '$APPLY' (did you mean --apply?)" >&2; exit 2
fi

stamp() { date -u +%FT%TZ; }

# 1. The learner's inventory, fetched once. An empty or failed listing is
#    fatal: "the learner has nothing" and "I could not ask" look identical
#    from here, and only one of them makes deletion safe.
remote_list=$(mktemp) || exit 1
trap 'rm -f "$remote_list"' EXIT
if ! ssh -o BatchMode=yes "$host" "ls -1 '$path/shards' 2>/dev/null" > "$remote_list"; then
  echo "$(stamp) prune: cannot list $REMOTE/shards -- aborting, nothing deleted" >&2
  exit 1
fi
remote_n=$(grep -c '\.npz$' "$remote_list" || true)
if [ "$remote_n" -eq 0 ]; then
  echo "$(stamp) prune: learner lists 0 shards -- aborting, nothing deleted" >&2
  exit 1
fi

# 2. Local candidates: shard files older than the age floor. -mmin keeps
#    this to one traversal and avoids parsing ls.
candidates=$(mktemp) || exit 1
trap 'rm -f "$remote_list" "$candidates"' EXIT
find "$LOCAL/shards" -maxdepth 1 -type f -name '*.npz' \
     -mmin "+$(( MIN_AGE_HOURS * 60 ))" -printf '%f\n' | sort > "$candidates"

# 3. Never delete an actor's NEWEST local shard. A restarted actor derives
#    its shard seq and game index (hence its game seeds) from the shards in
#    its local dir only; pruning them all made a restart begin again at
#    index 0 and replay seeds it had already played.
keep=$(mktemp) || exit 1
trap 'rm -f "$remote_list" "$candidates" "$keep"' EXIT
find "$LOCAL/shards" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' \
  | awk -F'[_.]' '{ seq = $3 + 0; if (!($2 in best) || seq > best[$2]) { best[$2] = seq; name[$2] = $0 } }
                  END { for (a in name) print name[a] }' | sort > "$keep"

# 4. Delete only: local candidate AND present on the learner AND not kept.
confirmed=$(grep '\.npz$' "$remote_list" | sort | comm -12 - "$candidates" | comm -23 - "$keep")
n=$(printf '%s' "$confirmed" | grep -c . || true)
if [ "$n" -eq 0 ]; then
  echo "$(stamp) prune: nothing to do (learner has $remote_n shards, $(wc -l < "$candidates") local candidates)"
  exit 0
fi

bytes=0
if [ "$APPLY" = "--apply" ]; then
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    f="$LOCAL/shards/$name"
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if rm -f -- "$f"; then bytes=$(( bytes + sz )); fi
  done <<< "$confirmed"
  echo "$(stamp) prune: deleted $n shards ($(( bytes / 1024 / 1024 )) MiB) confirmed on $host; $(df -h "$LOCAL" | tail -1 | awk '{print $4}') free"
else
  echo "$(stamp) prune: DRY RUN -- $n shards older than ${MIN_AGE_HOURS}h are confirmed on $host and would be deleted"
  # sed -n reads its input to the end; `head` would close the pipe early and
  # trip pipefail, failing a run that did exactly what it was asked to.
  printf '%s\n' "$confirmed" | sed -n '1,3p'
  echo "prune: re-run with --apply to delete them"
fi
