#!/usr/bin/env bash
# Launch one reward-1 arm's eval across both GPUs, one shard each.
#
#   ./run_eval_arm.sh <adapter-dir> <out-prefix>
#
# eval_lora.py resumes from keys already present in its --out file, so re-running
# this against a partial arm tops it up rather than redoing the work.
#
# Two deliberate choices, both learned the hard way (NS-P1 ledger #80):
#
#   * The launch lives in a FILE, not inline in a Monitor command. Inline, the
#     watcher's own /proc cmdline contains "eval_lora.py --adapter <name>", so a
#     `pgrep -f "eval_lora.py.*<name>"` wait loop matches the watcher itself and
#     never clears. That silently swallowed one arm's completion signal.
#   * setsid puts each shard in its OWN session and process group, so stopping a
#     watcher cannot take the eval down with it. nohup is not enough: it ignores
#     SIGHUP, not the SIGTERM a process-group stop delivers. Stopping a watcher
#     once cost us 103 records of an almost-finished arm.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <adapter-dir> <out-prefix>" >&2
  exit 2
fi
ADAPTER=$1
PREFIX=$2

cd "$(dirname "$0")"
OUTDIR=logs/lora_eval
PIDFILE=$OUTDIR/$PREFIX.pids

[ -d "$ADAPTER" ] || { echo "no such adapter dir: $ADAPTER" >&2; exit 1; }
mkdir -p "$OUTDIR"
: > "$PIDFILE"
for i in 0 1; do
  have=0
  [ -f "$OUTDIR/${PREFIX}_s$i.jsonl" ] && have=$(wc -l < "$OUTDIR/${PREFIX}_s$i.jsonl")
  echo "shard $i: $have records already on disk, resuming"
  CUDA_VISIBLE_DEVICES=$i setsid nohup env -u VIRTUAL_ENV .venv/bin/python eval_lora.py \
    --adapter "$ADAPTER" \
    --out "$OUTDIR/${PREFIX}_s$i.jsonl" \
    --shard "$i" --num_shards 2 \
    >> "$OUTDIR/${PREFIX}_s$i.log" 2>&1 < /dev/null &
  echo $! >> "$PIDFILE"
done
echo "launched $PREFIX on GPU 0,1 -> pids $(tr '\n' ' ' < "$PIDFILE")"
