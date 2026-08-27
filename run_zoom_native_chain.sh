#!/usr/bin/env bash
# Post-fix re-runs: zoom self-placed, then native-ActivityNet for tab:prior.
# File-based for the reason documented in run_full_chain.sh: an inline `bash -c` driver's
# command line contains the strings its own wait-loop greps for, so it blocks on itself.
set -uo pipefail
cd "$(dirname "$0")"
wait_workers() { while [ "$(ps -eo args | grep -c '[.]venv/bin/python eval_')" -gt 0 ]; do sleep 45; done; }

wait_workers
echo "[$(date -u '+%F %T')] ceiling done: $(cat logs/lora_eval/tl8b_zoom_ceiling_fx_s*of3.jsonl | wc -l)/590"

# Self-placed uses the model's OWN coarse centers. Those must come from the POST-FIX
# Charades run, not the pre-fix one, or the row mixes kernels internally.
for i in 0 1 2; do
  g=$((i % 2))
  CUDA_VISIBLE_DEVICES=$g setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
    .venv/bin/python eval_zoom.py --mode realistic --model TencentARC/TimeLens-8B \
    --patch_embed matmul --window_len 8 --place 0.6 --fps 8 --max_moment_s 2.0 \
    --coarse_jsonl logs/scale8b_charades_full/merged.jsonl \
    --out "logs/lora_eval/tl8b_zoom_realistic_fx_s${i}of3.jsonl" --shard "$i" --num_shards 3 \
    >> "logs/lora_eval/tl8b_zoom_realistic_fx_s${i}of3.log" 2>&1 < /dev/null &
done
echo "[$(date -u '+%F %T')] launched zoom self-placed (post-fix coarse)"
sleep 90; wait_workers
echo "[$(date -u '+%F %T')] self-placed done: $(cat logs/lora_eval/tl8b_zoom_realistic_fx_s*of3.jsonl 2>/dev/null | wc -l)/590"

mkdir -p logs/scale8b_anet_native500
for i in 0 1 2; do
  g=$((i % 2))
  CUDA_VISIBLE_DEVICES=$g setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
    .venv/bin/python eval_bench.py --model TencentARC/TimeLens-8B \
    --json data/TimeLens-Bench/activitynet-native-canonical500.json \
    --video_dir /gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos \
    --patch_embed matmul \
    --out "logs/scale8b_anet_native500/preds_s${i}of3.jsonl" --shard "$i" --num_shards 3 \
    >> "logs/scale8b_anet_native500/run_s${i}of3.log" 2>&1 < /dev/null &
done
echo "[$(date -u '+%F %T')] launched native-ActivityNet (canonical 500, FLOAT GT -> score UNROUNDED)"
sleep 90; wait_workers
echo "[$(date -u '+%F %T')] native-AN done: $(cat logs/scale8b_anet_native500/preds_s*of3.jsonl 2>/dev/null | wc -l)/500"
echo "CHAIN COMPLETE"
