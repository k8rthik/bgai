#!/usr/bin/env bash
# Copy a pipelined run's per-iteration checkpoints to an eval host, each
# with a provenance.json, so tools/eval_harness.sh can evaluate them.
#
#   tools/snap_ckpts.sh <eval-host> <learner-alias> <abs-run-dir> <prefix> [--base-games N] ITER...
#
# Runs ON THE EVAL HOST (via ssh), which must reach the learner as
# <learner-alias> (gen hosts have `Host learner`). Each ITER lands at
#   ~/code/bgai/data/checkpoints/<prefix>_it<ITER>/current.pt
# ITER "latest" means the newest metrics row; its iter_NNNN.pt is copied
# (not current.pt, which the learner may replace between the two reads).
#
# provenance.json carries the metrics row's iteration and total_games,
# plus approx_games_trained = base_games + sum over steps of
# min(new_games, window_games): the games that actually entered a training
# window. total_games alone is NOT a training axis -- it counts every shard
# present, including backlogs the learner skipped (scale_1m: ~180k around
# it275-300). --base-games covers steps with no metrics rows (scale_1m:
# ~26000 for it1-274, whose metrics died with spark02).
set -euo pipefail
evalhost="${1:?eval host}"; learner="${2:?learner alias as seen from the eval host}"
run="${3:?absolute run dir on the learner}"; prefix="${4:?checkpoint name prefix}"; shift 4
base=0
if [ "${1:-}" = "--base-games" ]; then base="${2:?--base-games N}"; shift 2; fi
[ "$#" -ge 1 ] || { echo "snap: no iterations given" >&2; exit 2; }
case "$base" in ''|*[!0-9]*) echo "snap: --base-games must be an integer" >&2; exit 2 ;; esac
for it in "$@"; do
  case "$it" in latest) ;; ''|*[!0-9]*) echo "snap: bad iteration '$it'" >&2; exit 2 ;; esac
done

# %q: ssh re-splits its joined arguments on the remote side
# shellcheck disable=SC2029  # client-side expansion is the point
ssh "$evalhost" "bash -s -- $(printf '%q ' "$learner" "$run" "$prefix" "$base" "$@")" <<'REMOTE'
set -euo pipefail
learner="$1"; run="$2"; prefix="$3"; base="$4"; shift 4
cd ~/code/bgai
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
scp -q "$learner:$run/metrics.jsonl" "$tmp"
for it in "$@"; do
  if [ "$it" = latest ]; then
    it=$(python3 - "$tmp" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"total_games"' in l]
print(rows[-1]["iteration"])
PY
)
  fi
  src="$run/iter_$(printf %04d "$it").pt"
  dest="data/checkpoints/${prefix}_it$it"
  mkdir -p "$dest"
  scp -q "$learner:$src" "$dest/current.pt"
  python3 - "$tmp" "$it" "$base" "$learner" "$run" > "$dest/provenance.json" <<'PY'
import json, sys
path, it, base, learner, run = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(path) if '"total_games"' in l]
upto = [r for r in rows if r["iteration"] <= it]
if not upto or upto[-1]["iteration"] != it:
    sys.exit(f"snap: no metrics row for iteration {it}")
trained = base + sum(min(r["new_games"], r["window_games"]) for r in upto)
print(json.dumps({"iteration": it, "total_games_counted": upto[-1]["total_games"],
                  "approx_games_trained": trained, "learner": learner, "run": run}))
PY
  echo "[$(hostname)] ${prefix}_it$it <- $src : $(cat "$dest/provenance.json")"
done
REMOTE
