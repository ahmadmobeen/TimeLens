#!/usr/bin/env bash
# Restart the tab:prior queue whenever the GPUs go idle with work still outstanding.
#
#   setsid nohup ./watchdog_tabprior.sh >> logs/tabprior_watchdog.log 2>&1 &
#
# WHY THIS EXISTS. The queue ran all 11 jobs back-to-back and exited cleanly at 21:21 with
# three runs short of target: two VideoChat jobs hit a relative-path bug (0 and 1840 of
# 3679) and two ActivityNet jobs stopped at 475/500 because their harnesses hardcoded .mp4.
# The queue logged "END ... -> 0" and moved on, because finishing its LIST is not the same
# as finishing the WORK. Nothing was watching, so four GPUs sat idle for about 7.5 hours.
#
# The queue is idempotent -- every job skips when its output already meets target -- so the
# correct repair is simply to run it again. This loop does that, and only that:
#
#   GPUs idle  AND  `run_tabprior_queue.sh status` reports a shortfall  ->  relaunch.
#
# It deliberately does NOT kill or pre-empt anything. If any eval process is alive, whether
# started by the queue or by hand, it waits. That keeps it safe to leave running while a
# human is also working on the box.
#
# LOOP GUARD. A job that can never reach target -- a genuinely missing video, a broken
# harness -- would otherwise be relaunched forever. After MAX_RESTARTS the watchdog stops
# and says so, because an infinite retry loop hides a real defect instead of surfacing it.
set -uo pipefail
cd "$(dirname "$0")"

POLL=${POLL:-300}
MAX_RESTARTS=${MAX_RESTARTS:-5}
IDLE_MIB=${IDLE_MIB:-2000}
restarts=0

say() { printf '[%s] watchdog %s\n' "$(date -u '+%F %T')" "$*"; }

gpus_idle() {
  local g m
  for g in 0 1 2 3; do
    m=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "$m" -gt "$IDLE_MIB" ] && return 1
  done
  return 0
}

# Any eval process at all, including ones a human started outside the queue.
work_running() {
  [ "$(ps -eo args | grep -cE '[.]venv/bin/python (eval_bench|run_ns_p1|run_trace)')" -gt 0 ]
}

queue_running() { [ "$(ps -eo args | grep -c '[r]un_tabprior_queue.sh')" -gt 0 ]; }

say "started; poll=${POLL}s max_restarts=$MAX_RESTARTS"
while :; do
  sleep "$POLL"

  if work_running || queue_running; then continue; fi
  if ! gpus_idle; then continue; fi

  short=$(./run_tabprior_queue.sh status 2>/dev/null)
  if [ -z "$short" ]; then
    say "all targets met and GPUs idle -- nothing left to do, exiting"
    exit 0
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "STOPPING: $MAX_RESTARTS restarts already used and work is still short:"
    printf '%s\n' "$short" | sed 's/^/    /'
    say "This is a defect to diagnose, not something more retries will fix."
    exit 1
  fi

  restarts=$((restarts + 1))
  say "IDLE GPUs with outstanding work (restart $restarts/$MAX_RESTARTS):"
  printf '%s\n' "$short" | sed 's/^/    /'
  setsid nohup ./run_tabprior_queue.sh >> logs/tabprior_queue.log 2>&1 < /dev/null &
  sleep 120   # let it claim cards before the next poll sees an idle box
done
