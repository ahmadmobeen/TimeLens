#!/usr/bin/env bash
# Queue driver for the two 8B bench runs. Fills each GPU as it frees up instead
# of waiting for the whole machine, then hands ActivityNet the full machine.
#
#   setsid nohup ./chain_8b_bench.sh >> logs/scale8b_chain.log 2>&1 &
#
# Phases:
#   0. (already launched by hand) qvh sub-2s, shards 0..5, 3 per GPU.
#   1. wait for all six                     -> anet sub-2s, shards 0..5, 3 per GPU
#
# Both phases are SUB-2 S ONLY. See run_8b_bench.sh for why: 8B runs ~420 s/clip
# on these long-video splits versus ~36 s/clip on Charades, so the full splits
# would cost ~30 h and ~80 h, and 94% / 87.5% of that would be spent on moments
# the paper's claim does not concern. The strata are 92 and 518 clips, roughly
# 2 h and 10 h at six-way concurrency.
#
# QVHighlights goes first because it is 92 clips against ActivityNet's 518:
# finishing it converts the 420 s/clip estimate into a measured rate before the
# longer run commits the machine.
#
# Deliberately NOT queued here: the full-split aggregate runs (mIoU, beta). They
# are ~4.6 days of both GPUs together and should be an explicit decision, not
# something a driver script starts while nobody is watching. The stopped
# full-split shards left 3 records in logs/scale8b_qvh/ and are resumable at
# NSHARDS=6 if that decision goes the other way.
#
# Waiting is done with kill -0 on explicit PIDs read from the pidfiles, never
# with pgrep: a watcher whose pattern matches its own cmdline waits forever.
# This script only ever reads those PIDs -- it does not signal them -- so a
# router that finishes normally is untouched, and if this driver is killed the
# already-running shards keep going (they are setsid'd, in their own sessions).
set -uo pipefail
cd "$(dirname "$0")"

QVH_PIDS=logs/scale8b_qvh_short/run.pids
POLL=300
QVH_N=92     # sub-2s clips in qvhighlights-timelens.json, of 1541 on disk
ANET_N=518   # sub-2s clips in activitynet-timelens.json, of 4140 on disk

stamp() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

# Block until every PID in a pidfile is gone. A missing pidfile means nothing to
# wait for, which is the correct reading for a phase that never launched.
wait_pids() {
  local file=$1 label=$2 pids alive
  [ -f "$file" ] || { stamp "$label: no pidfile ($file), nothing to wait for"; return 0; }
  pids=$(cat "$file")
  while :; do
    alive=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
    [ "$alive" -eq 0 ] && break
    stamp "$label: $alive/$(wc -w <<< "$pids") still alive"
    sleep "$POLL"
  done
  stamp "$label: all exited"
}

progress() {
  local dir=$1 n=0
  for f in "$dir"/preds_s*of6.jsonl; do
    [ -f "$f" ] && n=$((n + $(wc -l < "$f")))
  done
  echo "$n"
}

stamp "chain start; qvh sub-2s shards 0-5 assumed already running (3 per GPU)"

wait_pids "$QVH_PIDS" "qvh-short"
stamp "qvh sub-2s done: $(progress logs/scale8b_qvh_short)/$QVH_N records"

stamp "launching anet sub-2s shards 0-5, 3 per GPU"
SHORT=2.0 SHARDS=0,2,4 GPUS=0 NSHARDS=6 ./run_8b_bench.sh anet
SHORT=2.0 SHARDS=1,3,5 GPUS=1 NSHARDS=6 ./run_8b_bench.sh anet
wait_pids logs/scale8b_anet_short/run.pids "anet-short"
stamp "anet sub-2s done: $(progress logs/scale8b_anet_short)/$ANET_N records"
stamp "chain complete"
