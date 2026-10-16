#!/usr/bin/env bash
# Prepare a DGX Spark host for self-play runs: sync code + data, install env, smoke-test.
# Usage: tools/prep_spark.sh aim-spark02   (run from the bgai repo root; needs key-based SSH)
set -euo pipefail
host="${1:?usage: prep_spark.sh <ssh-host>}"
remote="$host:~/code/bgai"
ex=(--exclude .venv --exclude /data --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude .DS_Store --exclude .env --exclude .claude)
ssh "$host" 'mkdir -p ~/code/bgai/data/raw/games ~/code/bgai/data/datasets ~/code/bgai/data/checkpoints/selfplay_leg5b ~/code/bgai/data/checkpoints/snap ~/code/bgai/scratch'
echo "[$host] code";        rsync -az "${ex[@]}" ./ "$remote/"
echo "[$host] datasets";    rsync -az data/datasets/moves.parquet data/datasets/clean_setup_ids.json "$remote/data/datasets/"
echo "[$host] checkpoints"; rsync -az data/checkpoints/selfplay_leg5b/current.pt "$remote/data/checkpoints/selfplay_leg5b/"
                            rsync -az data/checkpoints/snap/simplex_best.pt "$remote/data/checkpoints/snap/"
echo "[$host] raw games";   rsync -az data/raw/games/ "$remote/data/raw/games/"
echo "[$host] uv sync";     ssh "$host" 'export PATH=$HOME/.local/bin:/snap/bin:$PATH; cd ~/code/bgai && uv sync --no-dev 2>&1 | tail -1'
echo "[$host] smoke test";  ssh "$host" 'export PATH=$HOME/.local/bin:/snap/bin:$PATH; cd ~/code/bgai && uv run python -c "
import random, torch; from bgai.arena.setups import sample_setup; from bgai.agents.mcts import MCTSAgent; from pathlib import Path
s = sample_setup(random.Random(1)); MCTSAgent(Path(\"data/checkpoints/selfplay_leg5b/current.pt\"), device=\"cpu\")
print(\"OK games:\", len(__import__(\"os\").listdir(\"data/raw/games\")), \"torch\", torch.__version__)" 2>&1 | grep -v -i warning | tail -1'
