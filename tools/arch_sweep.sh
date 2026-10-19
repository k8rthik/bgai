#!/usr/bin/env bash
# Architecture sweep, stage 1: train each candidate from scratch with the
# champion base's recipe (imitation_v2, value_simplex, rank_weight 1,
# lr 3e-4, batch 512) for the SAME step budget, validating every VAL steps,
# so held-out top1 / pairwise / winner_top1 curves compare like for like.
#
#   tools/arch_sweep.sh [lanes] [steps] [seed]      (run inside tmux on the GPU box)
#
# Runs LANES trainings at once; a run whose out dir already has last.pt at
# the full step count is skipped, so the queue is resumable. Results:
# data/checkpoints/arch/<name>/{metrics.jsonl,last.pt,best.pt}, logs in
# scratch/arch_<name>.log.
set -uo pipefail
LANES="${1:-2}"; STEPS="${2:-40000}"; SEED="${3:-11}"; VAL="${VAL:-4000}"
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
cd "$(dirname "$0")/.."

# name|--arch overrides
CANDIDATES="${CANDIDATES:-
mlp_h1024|
mlp_h2048|hidden=2048
mlp_h4096|hidden=4096
resmlp_h1024_d4|torso=resmlp,hidden=1024,depth=4
resmlp_h2048_d4|torso=resmlp,hidden=2048,depth=4
hexgnn_d64_l4|torso=hexgnn,hex_dim=64,depth=4
hexformer_d64_l4|torso=hexformer,hex_dim=64,depth=4,heads=4
}"

done_at_steps() {  # done_at_steps DIR -> 0 when last.pt holds STEPS
  [ -f "$1/last.pt" ] && uv run python -c "
import sys, torch
sys.exit(0 if torch.load('$1/last.pt', map_location='cpu', weights_only=False).get('step') == $STEPS else 1)
" 2>/dev/null
}

run_one() {
  local name="$1" arch="$2" out="data/checkpoints/arch/${1}_s${SEED}"
  if done_at_steps "$out"; then echo "$(date -Is) skip $name (done)"; return; fi
  rm -rf "$out"
  echo "$(date -Is) start $name arch=[$arch]"
  uv run python -u -m bgai.training.train --shards data/datasets/imitation_v2 --out "$out" \
    --epochs 2 --max-steps "$STEPS" --val-every-steps "$VAL" --rank-weight 1.0 \
    --value-simplex --seed "$SEED" --device cuda --arch "$arch" \
    > "scratch/arch_${name}_s${SEED}.log" 2>&1
  echo "$(date -Is) end $name exit=$? $(tail -n 1 "$out/metrics.jsonl" 2>/dev/null | cut -c1-200)"
}

mkdir -p scratch data/checkpoints/arch
running=0
while IFS='|' read -r name arch; do
  [ -z "$name" ] && continue
  run_one "$name" "$arch" &
  running=$((running + 1))
  if [ "$running" -ge "$LANES" ]; then wait -n; running=$((running - 1)); fi
done <<< "$CANDIDATES"
wait
echo "$(date -Is) sweep complete"
