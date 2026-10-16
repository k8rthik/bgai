#!/usr/bin/env bash
# Cron watchdog for a pipelined self-play run: push an ntfy alert when the
# run stalls, and stop this host's actors once the learner is clearly gone.
#
#   tools/run_watchdog.sh RUN_DIR LEARNERS ACTORS [STALE_MIN] [STOP_MIN]
#
# Install on EVERY live host (the learner host and each generation host):
#   */10 * * * * cd $HOME/code/bgai && tools/run_watchdog.sh \
#       /home/murug030/code/bgai/runs/scale_1m 1 4 >> scratch/watchdog.log 2>&1
#
# Why: a dead learner is silent. Twice now (spark02 on 2026-09-14, spark03 on
# 2026-09-29) the learner died, actors on other boxes kept playing against a
# frozen checkpoint for days, and the games (150k, then 188k) could never be
# trained on. Nothing told anyone.
#
# Checks, each run:
#   1. current.pt age. On a generation host this is the copy the sync loop
#      pulls, so one check covers learner crashed, learner hung, learner BOX
#      down and sync broken -- the failure modes a box cannot report itself.
#   2. Process counts on THIS host vs. what it should run (LEARNERS 0|1,
#      ACTORS n), matched anchored at argv[0]: `pgrep -f selfplay_actor` would
#      also match any command line that merely contains the string.
# Alerting: topic from ~/.config/bgai/ntfy_topic (kept out of the repo: on
# the public ntfy.sh server the topic name is the only access control).
# Alerts when the set of problems changes, repeats every REMIND_H hours while
# it persists, and sends one "recovered" note when it clears.
# Action: once current.pt is older than STOP_MIN, this host's actors get
# SIGTERM -- they finish the in-flight game and exit -- so a dead learner
# stops costing a shared box games nobody will train on.
# A finished run (RUN_DIR/DONE, written by the learner) is reported once.
# `touch RUN_DIR/WATCH_OFF` silences everything.
set -uo pipefail

RUN="${1:?usage: run_watchdog.sh RUN_DIR LEARNERS ACTORS [stale_min] [stop_min]}"
WANT_LEARNERS="${2:?expected learner count (0 or 1)}"
WANT_ACTORS="${3:?expected actor count}"
STALE_MIN="${4:-30}"
STOP_MIN="${5:-60}"
REMIND_H="${REMIND_H:-6}"
TOPIC_FILE="${NTFY_TOPIC_FILE:-$HOME/.config/bgai/ntfy_topic}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
STATE_DIR="${WATCHDOG_STATE_DIR:-$HOME/.cache/bgai_watchdog}"

LEARNER_RE='^[^ ]*python[0-9.]* +-u +-m +bgai\.training\.selfplay_learner'
ACTOR_RE='^[^ ]*python[0-9.]* +-u +-m +bgai\.training\.selfplay_actor'

for v in "$WANT_LEARNERS" "$WANT_ACTORS" "$STALE_MIN" "$STOP_MIN" "$REMIND_H"; do
  case "$v" in ''|*[!0-9]*) echo "watchdog: numeric arguments only, got '$v'" >&2; exit 2 ;; esac
done

stamp() { date -u +%FT%TZ; }
host=$(hostname)
run_name=$(basename "$RUN")
state_file="$STATE_DIR/$run_name.state"
mkdir -p "$STATE_DIR"

[ -e "$RUN/WATCH_OFF" ] && { echo "$(stamp) watchdog: WATCH_OFF present, silent"; exit 0; }

notify() {  # notify PRIORITY TITLE BODY
  local topic
  if [ ! -r "$TOPIC_FILE" ] || ! topic=$(tr -d ' \n' < "$TOPIC_FILE") || [ -z "$topic" ]; then
    echo "$(stamp) watchdog: no ntfy topic in $TOPIC_FILE -- alert NOT sent: $2 | $3" >&2
    return 1
  fi
  if ! curl -fsS -m 20 -o /dev/null -H "Title: $2" -H "Priority: $1" -H "Tags: bgai" \
       -d "$3" "$NTFY_SERVER/$topic"; then
    echo "$(stamp) watchdog: ntfy POST failed -- alert NOT sent: $2 | $3" >&2
    return 1
  fi
}

# shellcheck disable=SC2009  # pgrep -f would match command lines merely containing the pattern
count_matching() { ps -eo args= | grep -cE "$1" || true; }
# Actors running for at least $1 seconds. A just-(re)started actor is spared:
# after a relaunch current.pt keeps its old mtime until the learner's first
# step, and that must not read as a dead learner.
actor_pids() {
  # The regex goes through ENVIRON: `awk -v` would process its backslashes.
  ps -eo pid=,etimes=,args= | sed -E 's/^ *//' | RE="$ACTOR_RE" awk -v min="$1" \
    '{p=$1; e=$2; $1=""; $2=""; sub(/^ +/,""); if ($0 ~ ENVIRON["RE"] && e >= min) print p}'
}

# 0. finished run: the learner writes DONE at max_iterations (gen hosts get
# it from the sync loop). Its current.pt then ages exactly like a dead
# learner's, so without this the run "needs attention" forever. Stop any
# actors still playing and say so once; the note is not retried, since a
# missed "finished" costs nothing and the line below stays in the log.
if [ -f "$RUN/DONE" ]; then
  pids=$(actor_pids 0 | tr '\n' ' ')
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
  if [ "$(sed -n 1p "$state_file" 2>/dev/null)" != "DONE" ]; then
    notify default "bgai $host: run finished" \
      "$run_name complete: $(tr -d '\n' < "$RUN/DONE")${pids:+; stopped actors ${pids% }}" || true
    printf 'DONE\n%s\n' "$(date +%s)" > "$state_file"
  fi
  echo "$(stamp) watchdog: finished $run_name${pids:+; sent SIGTERM to actor pids ${pids% }}"
  exit 0
fi

# 1. checkpoint age
now=$(date +%s)
if ckpt_mtime=$(stat -c %Y "$RUN/current.pt" 2>/dev/null); then
  age_min=$(( (now - ckpt_mtime) / 60 ))
else
  age_min=-1
fi
last_iter=$(tail -1 "$RUN/metrics.jsonl" 2>/dev/null | grep -oE '"iteration": [0-9]+' | grep -oE '[0-9]+' || true)

# 2. processes on this host
learners=$(count_matching "$LEARNER_RE")
actors=$(count_matching "$ACTOR_RE")

problems=()
if [ "$age_min" -lt 0 ]; then
  problems+=("no current.pt in $RUN")
elif [ "$age_min" -ge "$STALE_MIN" ]; then
  problems+=("current.pt not updated for ${age_min} min")
fi
[ "$learners" -lt "$WANT_LEARNERS" ] && problems+=("learner not running here")
[ "$actors" -lt "$WANT_ACTORS" ] && problems+=("actors ${actors}/${WANT_ACTORS} running")

# Action: stop actors feeding a learner that is gone.
stopped=""
if [ "$age_min" -ge "$STOP_MIN" ] && [ "$actors" -gt 0 ]; then
  pids=$(actor_pids $(( STOP_MIN * 60 )) | tr '\n' ' ')
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    stopped="sent SIGTERM to actor pids ${pids% }"
    echo "$(stamp) watchdog: $stopped"
  fi
fi

key=$(printf '%s|' "${problems[@]:-}")
prev_key=""; prev_sent=0
if [ -f "$state_file" ]; then
  prev_key=$(sed -n 1p "$state_file"); prev_sent=$(sed -n 2p "$state_file")
  case "$prev_sent" in ''|*[!0-9]*) prev_sent=0 ;; esac
fi
summary="iter=${last_iter:--} ckpt_age=${age_min}m learners=${learners}/${WANT_LEARNERS} actors=${actors}/${WANT_ACTORS}"

if [ "${#problems[@]}" -eq 0 ]; then
  if [ -n "$prev_key" ] && [ "$prev_key" != "|" ]; then
    notify default "bgai $host: recovered" "$run_name healthy again. $summary" || true
  fi
  printf '|\n%s\n' "$now" > "$state_file"
  echo "$(stamp) watchdog: ok $summary"
  exit 0
fi

body="$run_name on $host: $(printf '%s; ' "${problems[@]}")${stopped:+$stopped; }$summary"
if [ "$key" != "$prev_key" ] || [ $(( now - prev_sent )) -ge $(( REMIND_H * 3600 )) ] || [ -n "$stopped" ]; then
  if notify high "bgai $host: run needs attention" "$body"; then
    printf '%s\n%s\n' "$key" "$now" > "$state_file"
  fi
fi
echo "$(stamp) watchdog: PROBLEM $body"
