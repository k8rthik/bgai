#!/usr/bin/env bash
# Snapshot a running learner's checkpoint and head-to-head it against the
# leg5b champion on a separate evaluation host, so the scaling curve can be
# read WHILE the run continues instead of after it finishes.
#
# The learner host and the eval host need no ssh trust between them: the
# checkpoint is relayed through this machine.
#
# Usage: tools/eval_milestone.sh <learner-host> <abs-run-dir> <eval-host> <label> [tables] [workers]
set -euo pipefail
learner="${1:?learner host}"; run="${2:?absolute run dir}"; evalhost="${3:?eval host}"
label="${4:?label, e.g. 25k}"; tables="${5:-120}"; workers="${6:-20}"
name="scale_${label}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

games=$(ssh -n "$learner" "tail -1 '$run/metrics.jsonl'" | python3 -c \
  "import json,sys; print(json.loads(sys.stdin.read())['total_games'])")
echo "[$label] learner is at $games games; snapshotting current.pt"

scp -q "$learner:$run/current.pt" "$tmp/current.pt"
ssh -n "$evalhost" "mkdir -p ~/code/bgai/data/checkpoints/$name"
scp -q "$tmp/current.pt" "$evalhost:~/code/bgai/data/checkpoints/$name/current.pt"
# record provenance next to the checkpoint so the number is never orphaned
echo "{\"label\":\"$label\",\"total_games\":$games,\"learner\":\"$learner\",\"run\":\"$run\",\"snapped\":\"$(date -u +%FT%TZ)\"}" \
  | ssh "$evalhost" "cat > ~/code/bgai/data/checkpoints/$name/provenance.json"

"$(dirname "$0")/launch_spark_h2h.sh" "$evalhost" "$name" "$tables" "$workers"
echo "[$label] h2h launched on $evalhost against leg5b ($games games of training)"
