#!/usr/bin/env bash
# Head-to-head a finished self-play leg against the leg-5b champion, at the
# C16 champion eval config (512 sims, top_k 8, depth 48, batch 16, c_puct 2.5).
# Runs detached in tmux session "h2h" so it survives a dropped VPN.
# Usage: tools/launch_spark_h2h.sh <ssh-host> <leg-name> [tables] [workers]
set -euo pipefail
host="${1:?ssh host}"; name="${2:?leg name}"; tables="${3:-120}"; workers="${4:-16}"

base="${BGAI_H2H_BASELINE:-data/checkpoints/selfplay_leg5b/current.pt}"
cand="data/checkpoints/$name/current.pt"
# champion eval config (docs/decisions.md C16)
cfg="sims=512,top_k=8,max_depth=48,batch=16,c_puct=2.5,temperature=0"

ssh "$host" "
set -euo pipefail
export PATH=\$HOME/.local/bin:\$PATH
cd ~/code/bgai
for f in '$cand' '$base'; do
  [ -f \"\$f\" ] || { echo \"[$host] missing checkpoint: \$f\" >&2; exit 1; }
done
if tmux has-session -t h2h 2>/dev/null; then echo '[$host] tmux session h2h already exists' >&2; exit 1; fi
mkdir -p scratch
tmux new-session -d -s h2h \"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
  uv run python -u -m bgai.arena.run \
  --tables $tables --seed 20260910 --workers $workers \
  --report scratch/h2h_$name.html \
  --agents 'mcts@$name:ckpt=$cand,$cfg,mcts@leg5b:ckpt=$base,$cfg' \
  > scratch/h2h_$name.log 2>&1\"
echo \"[$host] h2h launched: $name vs leg5b, $tables tables ($(( tables * 4 )) games), $workers workers\"
"
