#!/usr/bin/env bash
# One block per DGX Spark for a PIPELINED run (learner + generation hosts).
#
#   tools/spark_status.sh [run-dir] [hosts...]
#
# The previous version of this script looked for the serial legs' tmux
# session "leg" and scratch/spark_leg_*.log. A pipelined run has neither,
# so it reported STOPPED on hosts that were pulling load 20 -- exactly
# the wrong answer during an incident. This one asks what is actually
# running and names the role it finds:
#
#   LEARNER  consumes shards, publishes current.pt -- the run's progress
#            is its metrics.jsonl tail (iteration, staleness, losses).
#   GEN      actors + a sync loop; its health is "are actors playing" and
#            "is the sync clean", not an iteration count.
#   IDLE     neither -- the thing you want to see when a box has dropped out.
#
# Unreachable hosts are reported as DOWN rather than silently skipped.
set -uo pipefail

RUN="${1:-/home/murug030/code/bgai/runs/scale_1m}"
shift 2>/dev/null || true
HOSTS=("${@:-}")
[ -z "${HOSTS[0]:-}" ] && HOSTS=(aim-spark01 aim-spark02 aim-spark03 aim-spark04)

for h in "${HOSTS[@]}"; do
  out=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$h" "
    cd ~/code/bgai 2>/dev/null || { echo 'NOREPO'; exit 0; }
    run='$RUN'
    # Match the real processes, anchored at argv[0], NOT with pgrep -f:
    # this very command line contains the strings 'selfplay_learner' and
    # 'selfplay_actor', so pgrep -f finds itself and every host reports
    # whatever role you asked about.
    learner=\$(ps -eo args= | grep -cE '^[^ ]*python[0-9.]* +-u +-m +bgai\.training\.selfplay_learner')
    actors=\$(ps -eo args= | grep -cE '^[^ ]*python[0-9.]* +-u +-m +bgai\.training\.selfplay_actor')
    sync=\$(ps -eo args= | grep -cE '^bash +tools/sync_run_dir\.sh')
    disk=\$(df -h / | tail -1 | awk '{print \$5\" used, \"\$4\" free\"}')
    load=\$(cut -d' ' -f1 /proc/loadavg)
    shards=\$(ls \"\$run/shards\" 2>/dev/null | wc -l)
    echo \"HOST \$(hostname) load=\$load disk=\$disk shards_local=\$shards\"
    if [ \"\$learner\" -gt 0 ]; then
      echo 'ROLE LEARNER'
      tail -1 \"\$run/metrics.jsonl\" 2>/dev/null | python3 -c '
import sys, json
line = sys.stdin.read().strip()
if not line:
    print(\"  metrics: none yet\"); raise SystemExit
r = json.loads(line)
print(\"  iter={} total_games={} gph={} staleness_mean={} loss={} window={}g/{}sh\".format(
    r.get(\"iteration\"), r.get(\"total_games\"), r.get(\"games_per_hour\"),
    r.get(\"staleness_mean\"), r.get(\"total\"), r.get(\"window_games\"), r.get(\"window_shards\")))
' 2>/dev/null || echo '  metrics: unreadable'
      echo \"  ckpt: \$(ls -l \"\$run/current.pt\" 2>/dev/null | awk '{print \$6,\$7,\$8}')\"
      # grep -c always prints a count; '|| echo 0' would print a SECOND
      # line on the (exit 1) no-match case and split the report.
      echo \"  errors in learner.log: \$(grep -cE 'Traceback|CUDA|Error' scratch/learner.log 2>/dev/null)\"
    elif [ \"\$actors\" -gt 0 ]; then
      echo \"ROLE GEN actors=\$actors sync=\$([ \"\$sync\" -gt 0 ] && echo up || echo DOWN)\"
      last=\$(ls -t scratch/actor_*.log 2>/dev/null | head -1)
      echo \"  newest: \$(grep -h ' done in ' \"\$last\" 2>/dev/null | tail -1 | sed 's/^[0-9-]* //')\"
      echo \"  sync errors (last 20 lines): \$(tail -20 scratch/sync_gen.log 2>/dev/null | grep -c 'failed')\"
    else
      echo 'ROLE IDLE (no learner, no actors)'
    fi
  " 2>/dev/null)
  if [ -z "$out" ]; then
    printf '%-13s DOWN (unreachable)\n' "$h"
  elif [ "$out" = "NOREPO" ]; then
    printf '%-13s no ~/code/bgai\n' "$h"
  else
    printf '%-13s %s\n' "$h" "$(echo "$out" | head -1)"
    echo "$out" | tail -n +2 | sed 's/^/              /'
  fi
done
