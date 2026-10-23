#!/usr/bin/env bash
# Launch a self-play leg on a DGX Spark inside tmux, from the leg-5b champion with the
# leg-5b training config (96 games/iteration), thread pools pinned to 1 per worker.
# Usage: tools/launch_spark_leg.sh <ssh-host> <leg-name> <iterations> <seed> [workers] [c_puct]
# c_puct defaults to 1.5 (what every leg through C18 used); pass 2.5 to make
# generation search at the config the champion is actually evaluated at (C16).
# env CKPT / ANCHOR override the starting checkpoint and the KL anchor (defaults:
# leg5b and snap/simplex_best.pt), e.g. to fine-tune a new imitation base anchored to itself.
set -euo pipefail
host="${1:?host}"; name="${2:?leg name}"; iters="${3:?iterations}"; seed="${4:?seed}"; workers="${5:-20}"; c_puct="${6:-1.5}"
ckpt="${CKPT:-data/checkpoints/selfplay_leg5b/current.pt}"; anchor="${ANCHOR:-data/checkpoints/snap/simplex_best.pt}"
ssh "$host" "export PATH=\$HOME/.local/bin:\$PATH; cd ~/code/bgai && mkdir -p scratch && tmux has-session -t leg 2>/dev/null && { echo 'tmux session leg already exists on $host'; exit 1; }
tmux new-session -d -s leg \"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
  uv run python -u -m bgai.training.selfplay_train \
  --checkpoint $ckpt --anchor $anchor \
  --out data/checkpoints/$name --iterations $iters --games-per-iteration 96 --workers $workers \
  --simulations 512 --top-k 8 --max-depth 48 --leaf-batch 16 --temp-decisions 30 --fast-prior 0.95 \
  --c-puct $c_puct \
  --lambda-kl 1.0 --rank-weight 1.0 --winner-pair-weight 3.0 --win-weight 0.25 --aux-weight 0.5 --lr 5e-5 \
  --seed $seed > scratch/$name.log 2>&1\"
echo \"[$host] launched $name from $ckpt (anchor $anchor): $iters x 96 games, seed $seed, $workers workers, c_puct $c_puct\""
