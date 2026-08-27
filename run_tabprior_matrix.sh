#!/usr/bin/env bash
# Fill every remaining cell of tab:prior. Smoke-test each model path first, then run the
# full matrix. Target: 11 runs across 4 models x 3 dataset versions.
#
#   setsid nohup ./run_tabprior_matrix.sh >> logs/tabprior_matrix.log 2>&1 &
#
# WHAT IS MISSING (12 \pending tokens, 11 runs -- one run fills beta/R@.7/|dc| together):
#   Qwen2.5-VL-7B  0-shot  x charades_sta, anet_native, qvh
#   Qwen2.5-VL-72B 0-shot  x charades_sta, anet_native, qvh
#   VideoChat-R1           x charades_sta, anet_native, qvh
#   TRACE                  x anet_native, qvh            (its charades_sta cell is filled)
#
# AUTHENTICITY. Each model is run through the SAME harness that produced its published
# Charades cell, not a new one, so the new cells are comparable to the old by construction:
#   Qwen 7B / 72B  -> eval_bench.py, which produced logs/qwen_zeroshot/ and logs/scale72b/
#                     (their shard-naming is eval_bench's)
#   VideoChat-R1   -> run_ns_p1.py, which already takes --fmt {timelens,anet,charades_sta}
#   TRACE          -> run_trace_charades.py, which is parameterised by --bench/--videos
# Using a different harness per cell is exactly how the #86 population mix-up happened.
#
# SCHEDULING. TRACE is the long pole: its script has no --shard, so it is single-process
# and cannot be spread over GPUs. It therefore gets GPU 3 to itself, starting immediately,
# while everything shardable uses GPUs 0-2. The 72B needs a whole card each (~145 GB of
# 183 GB), so its shards map one-to-one onto GPUs 0-2.
#
# SMOKE FIRST. Every model runs 2 clips before its full splits are queued. A model whose
# environment is broken is skipped with a loud line rather than silently producing nothing,
# and the others still proceed -- discovering a broken venv on Monday would cost the
# deadline. Runs are resumable (eval_bench and run_ns_p1 both skip keys already in --out),
# so an interrupted matrix continues rather than restarting.
set -uo pipefail
cd "$(dirname "$0")"

TL=$PWD
VC=../VideoChat-R1
TR=../TRACE
BENCH=data/TimeLens-Bench
QVH_V=/gpfs/public/datasets/qvhighlights/videos
ANET_V=/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos
CHAR_V=$BENCH/videos/charades

say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }
# Wait on python eval workers by interpreter path; this script's own command line is just
# its filename, so it cannot match itself (the trap that stalled an earlier inline driver).
wait_gpu() { while [ "$(ps -eo args | grep -cE '[.]venv/bin/python (eval_|run_ns_p1|run_trace)')" -gt 0 ]; do sleep 60; done; }
wait_pat() { while [ "$(ps -eo args | grep -c "$1")" -gt 0 ]; do sleep 60; done; }

# ---------------------------------------------------------------- qwen via eval_bench
qwen() {  # model outdir json videodir gpus
  local model=$1 out=$2 json=$3 vids=$4 gpus=$5
  IFS=',' read -r -a g <<< "$gpus"; local n=${#g[@]}
  mkdir -p "logs/$out"
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${g[$i]} setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
      .venv/bin/python eval_bench.py --model "$model" --json "$json" --video_dir "$vids" \
      --patch_embed matmul --decode_workers 3 \
      --out "logs/$out/preds_s${i}of${n}.jsonl" --shard "$i" --num_shards "$n" \
      >> "logs/$out/run_s${i}of${n}.log" 2>&1 &
  done
  say "launched $out on gpus $gpus"
}

# ---------------------------------------------------------------- videochat-r1
vchat() {  # outdir anno videoroot fmt gpus
  local out=$1 anno=$2 vids=$3 fmt=$4 gpus=$5
  IFS=',' read -r -a g <<< "$gpus"; local n=${#g[@]}
  mkdir -p "logs/$out"
  for i in $(seq 0 $((n - 1))); do
    ( cd "$VC" && CUDA_VISIBLE_DEVICES=${g[$i]} setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
      .venv/bin/python run_ns_p1.py --anno "$anno" --video-root "$vids" --fmt "$fmt" \
      --out "$TL/logs/$out/preds_s${i}of${n}.jsonl" --shard "$i" --nshards "$n" \
      >> "$TL/logs/$out/run_s${i}of${n}.log" 2>&1 & )
  done
  say "launched $out (videochat-r1) on gpus $gpus"
}

# ---------------------------------------------------------------- trace
trace_run() {  # outdir bench videos gpu
  local out=$1 bench=$2 vids=$3 gpu=$4
  mkdir -p "logs/$out"
  ( cd "$TR" && CUDA_VISIBLE_DEVICES=$gpu setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
    .venv/bin/python run_trace_charades.py --bench "$bench" --videos "$vids" \
    --out "$TL/logs/$out/preds.jsonl" >> "$TL/logs/$out/run.log" 2>&1 & )
  say "launched $out (TRACE) on gpu $gpu"
}

count() { cat logs/"$1"/*.jsonl 2>/dev/null | wc -l; }

say "=== waiting for any in-flight eval to finish"
wait_gpu
say "=== GPUs free; starting smoke tests (2 clips each)"

# ---- smoke ---------------------------------------------------------------
SMOKE_OK_QWEN7=0 SMOKE_OK_QWEN72=0 SMOKE_OK_VC=0 SMOKE_OK_TRACE=0
mkdir -p logs/smoke_matrix

CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_bench.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --json $BENCH/qvhighlights-timelens.json --video_dir $QVH_V \
  --decode_workers 2 --limit 2 --out logs/smoke_matrix/qwen7b.jsonl --shard 0 --num_shards 1 \
  > logs/smoke_matrix/qwen7b.log 2>&1
[ "$(wc -l < logs/smoke_matrix/qwen7b.jsonl 2>/dev/null || echo 0)" -ge 2 ] && SMOKE_OK_QWEN7=1
say "smoke qwen7b: ok=$SMOKE_OK_QWEN7"

CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_bench.py \
  --model Qwen/Qwen2.5-VL-72B-Instruct --json $BENCH/qvhighlights-timelens.json --video_dir $QVH_V \
  --decode_workers 2 --limit 2 --out logs/smoke_matrix/qwen72b.jsonl --shard 0 --num_shards 1 \
  > logs/smoke_matrix/qwen72b.log 2>&1
[ "$(wc -l < logs/smoke_matrix/qwen72b.jsonl 2>/dev/null || echo 0)" -ge 2 ] && SMOKE_OK_QWEN72=1
say "smoke qwen72b: ok=$SMOKE_OK_QWEN72"

( cd "$VC" && CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV .venv/bin/python run_ns_p1.py \
  --anno "$TL/$BENCH/qvhighlights-timelens.json" --video-root "$QVH_V" --fmt timelens \
  --limit 2 --out "$TL/logs/smoke_matrix/vc.jsonl" > "$TL/logs/smoke_matrix/vc.log" 2>&1 )
[ "$(wc -l < logs/smoke_matrix/vc.jsonl 2>/dev/null || echo 0)" -ge 2 ] && SMOKE_OK_VC=1
say "smoke videochat-r1: ok=$SMOKE_OK_VC"

( cd "$TR" && CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV .venv/bin/python run_trace_charades.py \
  --bench "$TL/$BENCH/qvhighlights-timelens.json" --videos "$QVH_V" --limit 2 \
  --out "$TL/logs/smoke_matrix/trace.jsonl" > "$TL/logs/smoke_matrix/trace.log" 2>&1 )
[ "$(wc -l < logs/smoke_matrix/trace.jsonl 2>/dev/null || echo 0)" -ge 1 ] && SMOKE_OK_TRACE=1
say "smoke trace: ok=$SMOKE_OK_TRACE"

say "=== SMOKE SUMMARY qwen7b=$SMOKE_OK_QWEN7 qwen72b=$SMOKE_OK_QWEN72 videochat=$SMOKE_OK_VC trace=$SMOKE_OK_TRACE"

# ---- full matrix ---------------------------------------------------------
# TRACE first and alone on GPU 3: single-process, so it is the long pole.
if [ "$SMOKE_OK_TRACE" = 1 ]; then
  trace_run trace_anet_native "$TL/$BENCH/activitynet-native-canonical500.json" "$ANET_V" 3
else
  say "SKIP trace (smoke failed) -- see logs/smoke_matrix/trace.log"
fi

if [ "$SMOKE_OK_QWEN7" = 1 ]; then
  for spec in "qwen7b_qvh:$BENCH/qvhighlights-timelens.json:$QVH_V" \
              "qwen7b_anet_native:$BENCH/activitynet-native-canonical500.json:$ANET_V" \
              "qwen7b_charades_sta:$BENCH/charades-sta-native.json:$CHAR_V"; do
    IFS=':' read -r o j v <<< "$spec"
    qwen Qwen/Qwen2.5-VL-7B-Instruct "$o" "$j" "$v" 0,1,2
    sleep 120; wait_pat '[e]val_bench.py'
    say "$o done: $(count "$o")"
  done
else
  say "SKIP qwen7b (smoke failed)"
fi

if [ "$SMOKE_OK_VC" = 1 ]; then
  for spec in "vc_qvh:$TL/$BENCH/qvhighlights-timelens.json:$QVH_V:timelens" \
              "vc_anet_native:$TL/$BENCH/activitynet-native-canonical500.json:$ANET_V:timelens" \
              "vc_charades_sta:$TL/$BENCH/charades-sta-native.json:$CHAR_V:timelens"; do
    IFS=':' read -r o j v f <<< "$spec"
    vchat "$o" "$j" "$v" "$f" 0,1,2
    sleep 120; wait_pat '[r]un_ns_p1.py'
    say "$o done: $(count "$o")"
  done
else
  say "SKIP videochat-r1 (smoke failed)"
fi

if [ "$SMOKE_OK_QWEN72" = 1 ]; then
  for spec in "qwen72b_qvh:$BENCH/qvhighlights-timelens.json:$QVH_V" \
              "qwen72b_anet_native:$BENCH/activitynet-native-canonical500.json:$ANET_V" \
              "qwen72b_charades_sta:$BENCH/charades-sta-native.json:$CHAR_V"; do
    IFS=':' read -r o j v <<< "$spec"
    qwen Qwen/Qwen2.5-VL-72B-Instruct "$o" "$j" "$v" 0,1,2
    sleep 180; wait_pat '[e]val_bench.py'
    say "$o done: $(count "$o")"
  done
else
  say "SKIP qwen72b (smoke failed)"
fi

# TRACE's second dataset last: by now GPU 3 is free of its first run, or still busy, in
# which case this waits rather than stacking two TRACE processes on one card.
if [ "$SMOKE_OK_TRACE" = 1 ]; then
  wait_pat '[r]un_trace_charades.py'
  say "trace_anet_native done: $(count trace_anet_native)"
  trace_run trace_qvh "$TL/$BENCH/qvhighlights-timelens.json" "$QVH_V" 3
  wait_pat '[r]un_trace_charades.py'
  say "trace_qvh done: $(count trace_qvh)"
fi

say "=== MATRIX COMPLETE"
for d in qwen7b_qvh qwen7b_anet_native qwen7b_charades_sta vc_qvh vc_anet_native \
         vc_charades_sta qwen72b_qvh qwen72b_anet_native qwen72b_charades_sta \
         trace_anet_native trace_qvh; do
  printf '  %-24s %6d\n' "$d" "$(count "$d")"
done
