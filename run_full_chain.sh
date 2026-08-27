#!/usr/bin/env bash
# Chain the remaining full-split 8B runs.
#
# MUST live in a FILE, not an inline `bash -c`. An inline driver's command line
# contains the literal string "eval_bench.py" (from the launch function body), so a
# wait-loop that greps for eval_bench matches the DRIVER ITSELF and blocks forever.
# That is exactly what happened on 2026-08-14: QVHighlights finished 1541/1541 and the
# chain sat waiting on its own process while both GPUs idled. Same self-match trap
# documented in run_eval_arm.sh, third occurrence in this project.
set -uo pipefail
cd "$(dirname "$0")"

launch() {
  local d=$1 json=$2 vdir=$3
  mkdir -p "logs/$d"
  for i in 0 1 2; do
    local g=$((i % 2))
    CUDA_VISIBLE_DEVICES=$g setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
      .venv/bin/python eval_bench.py --model TencentARC/TimeLens-8B \
      --json "$json" --video_dir "$vdir" --patch_embed matmul \
      --out "logs/$d/preds_s${i}of3.jsonl" --shard "$i" --num_shards 3 \
      >> "logs/$d/run_s${i}of3.log" 2>&1 < /dev/null &
  done
  echo "[$(date -u '+%F %T')] launched $d"
}

# Wait on the PYTHON workers by their interpreter path, which this script's own command
# line does not contain.
wait_workers() {
  local label=$1
  while [ "$(ps -eo args | grep -c '[.]venv/bin/python eval_bench.py')" -gt 0 ]; do sleep 60; done
  echo "[$(date -u '+%F %T')] $label: all workers exited"
}

launch scale8b_charades_full data/TimeLens-Bench/charades-timelens.json data/TimeLens-Bench/videos/charades
sleep 120; wait_workers charades
echo "CHARADES: $(cat logs/scale8b_charades_full/preds_s*of3.jsonl 2>/dev/null | wc -l)/3363"

launch scale8b_anet_full_tl data/TimeLens-Bench/activitynet-timelens.json /gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos
sleep 120; wait_workers anet_tl
echo "ANET-TL: $(cat logs/scale8b_anet_full_tl/preds_s*of3.jsonl 2>/dev/null | wc -l)/4140"
echo "CHAIN COMPLETE"
