#!/usr/bin/env bash
# Attach a generation host to a self-play run whose LEARNER runs elsewhere.
# Starts, in one tmux session "gen": a sync_run_dir.sh loop (pushes shards to
# the learner, pulls current.pt) plus N actor processes with globally unique
# actor ids. See selfplay_launcher.py's docstring for the two-host design.
#
# Usage: tools/add_gen_host.sh <gen-host> <learner-host> <abs-run-dir> <first-actor-id> [actors] [workers] [c_puct]
set -euo pipefail
gen="${1:?gen host}"; learner="${2:?learner host}"; run="${3:?absolute run dir}"
base="${4:?first actor id}"; actors="${5:-4}"; workers="${6:-20}"; cpuct="${7:-2.5}"
per=$(( workers / actors ))
[ "$per" -ge 1 ] || { echo "workers ($workers) < actors ($actors)" >&2; exit 1; }
pin="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1"

ssh "$gen" "
set -euo pipefail
export PATH=\$HOME/.local/bin:/snap/bin:\$PATH
cd ~/code/bgai
if tmux has-session -t gen 2>/dev/null; then echo '[$gen] tmux session gen already exists' >&2; exit 1; fi
mkdir -p '$run/shards' scratch
# sync first: actors wait for current.pt, which the sync loop pulls in
tmux new-session -d -s gen \"tools/sync_run_dir.sh $learner:$run $run 30 > scratch/sync_gen.log 2>&1\"
for i in \$(seq 0 $(( actors - 1 ))); do
  id=\$(( $base + i ))
  tmux new-window -t gen -d \"$pin uv run python -u -m bgai.training.selfplay_actor \
    --run-dir '$run' --actor-id \$id --games-per-shard 8 --workers $per \
    --simulations 512 --top-k 8 --max-depth 48 --leaf-batch 16 \
    --temp-decisions 30 --fast-prior 0.95 --c-puct $cpuct \
    > scratch/actor_\$id.log 2>&1\"
done
echo \"[$gen] attached: $actors actors (ids $base..$(( base + actors - 1 ))), $per workers each, c_puct $cpuct\"
"
