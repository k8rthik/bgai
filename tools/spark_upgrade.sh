#!/usr/bin/env bash
# Roll a code upgrade into a RUNNING self-play leg on a DGX Spark without losing work.
#   1. drop RUN_DIR/STOP -> trainer finishes its in-flight iteration, saves, exits
#   2. rsync the new code, uv sync
#   3. relaunch with --resume RUN_DIR (numbering, seeds, anchor come from the checkpoint)
# Requires the --resume / STOP-file support from branch feat/actor-learner-split.
# Usage: tools/spark_upgrade.sh <ssh-host> <leg-name> [workers]
set -euo pipefail
host="${1:?host}"; name="${2:?leg name}"; workers="${3:-20}"
run="~/code/bgai/data/checkpoints/$name"
echo "[$host] requesting graceful stop of $name"
ssh "$host" "touch $run/STOP"
echo "[$host] waiting for the in-flight iteration to finish (up to 30 min)"
ssh "$host" 'for i in $(seq 1 180); do tmux has-session -t leg 2>/dev/null || exit 0; sleep 10; done; echo "still running after 30 min" >&2; exit 1'
echo "[$host] syncing code"
rsync -az --exclude .venv --exclude /data --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude .DS_Store --exclude .env --exclude .claude ./ "$host:~/code/bgai/"
ssh "$host" 'export PATH=$HOME/.local/bin:$PATH; cd ~/code/bgai && uv sync --no-dev 2>&1 | tail -1 && uv run pytest -q -x tests/test_engine_smoke.py 2>/dev/null | tail -1 || true'
echo "[$host] resuming $name"
ssh "$host" "export PATH=\$HOME/.local/bin:\$PATH; cd ~/code/bgai && tmux new-session -d -s leg \"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run python -u -m bgai.training.selfplay_train --resume $run --workers $workers >> scratch/$name.log 2>&1\""
echo "[$host] $name resumed on new code ($(git rev-parse --short HEAD))"
