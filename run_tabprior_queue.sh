#!/usr/bin/env bash
# Per-GPU job queues for the remaining tab:prior cells. Each GPU owns a list and works
# through it back-to-back, so no card sits idle waiting for a human to launch the next job.
#
#   setsid nohup ./run_tabprior_queue.sh >> logs/tabprior_queue.log 2>&1 &
#   tail -f logs/tabprior_queue.log
#
# WHY THIS REPLACES run_tabprior_matrix.sh. That script had two defects that cost most of a
# working day. Its wait function blocked on *every* eval process, so one straggler shard held
# three idle GPUs; and its smoke tests were serialised onto GPU 0, idling the rest. Here each
# GPU is independent: it waits only for ITS OWN card to be free, then starts.
#
# STAGGERED STARTS ARE DELIBERATE, not politeness. Launching two Qwen-72B shards at once
# means two ~145 GB cold reads from GPFS in parallel, which starved every other job on the
# box for hours -- measured, not guessed: aggregate throughput was 0.89 clips/s in steady
# state versus roughly 0.002 clips/s while those loads ran. GPU_START_STAGGER spaces the
# first launch on each card so weight loads do not collide. Once weights are in page cache
# the later jobs on the same queue start quickly.
#
# RESUMABLE. eval_bench.py and run_ns_p1.py both skip keys already present in --out, so
# re-running this script continues rather than restarting. run_trace_charades.py does NOT
# resume; its jobs are therefore guarded by a target count and skipped if already complete.
#
# EXPECTED per-clip costs, measured on this box: Qwen-7B ~5 s, Qwen-72B ~8 s/shard,
# TRACE ~2.3 s, VideoChat-R1 unmeasured (assume ~5 s). Remaining work is ~13k clip-evals.
set -uo pipefail
cd "$(dirname "$0")"

TL=$PWD
B=data/TimeLens-Bench
QVH_V=/gpfs/public/datasets/qvhighlights/videos
ANET_V=/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos
# ABSOLUTE. The VideoChat runner is invoked after `cd ../VideoChat-R1`, so a relative
# video root resolved into that repo, matched nothing, and every item was filtered out by
# the runner's os.path.exists() check -- a job that "finished" in 13 s with 0 records.
# QVHighlights and ActivityNet escaped it only because their roots are absolute /gpfs paths.
CHAR_V=$TL/$B/videos/charades
GPU_START_STAGGER=${GPU_START_STAGGER:-420}

say() { printf '[%s] gpu%s %s\n' "$(date -u '+%F %T')" "${GPUID:-?}" "$*"; }
count() { cat logs/"$1"/*.jsonl 2>/dev/null | wc -l; }

# Compare what a job produced against what it owed. WITHOUT THIS the queue logs
# "END ... -> 0" and moves on, which is exactly what happened: two jobs finished with 0
# and 475 records, the queue completed "successfully", and nothing acted on it for ~7.5 h.
# A run that under-delivers is a failure, not a completion, and must say so.
verify() {  # outdir target
  local out=$1 target=$2 n
  n=$(count "$out")
  if [ "$n" -ge "$target" ]; then
    say "END   $out -> $n/$target OK"
    return 0
  fi
  say "SHORTFALL $out -> $n/$target -- retrying once"
  return 1
}

# The queue's own to-do list, usable from outside. The watchdog calls this to decide
# whether idle GPUs mean "finished" or "stalled with work outstanding".
if [ "${1:-}" = "status" ]; then
  cd "$(dirname "$0")"
  short=0
  while read -r d t; do
    [ -z "$d" ] && continue
    n=$(cat logs/"$d"/*.jsonl 2>/dev/null | wc -l)
    if [ "$n" -lt "$t" ]; then printf 'SHORT %-24s %6d/%s\n' "$d" "$n" "$t"; short=1; fi
  done <<'TARGETS'
qwen7b_qvh 1541
qwen7b_anet_native 500
qwen7b_charades_sta 3679
qwen72b_qvh 1541
qwen72b_anet_native 500
qwen72b_charades_sta 3679
vc_qvh 1541
vc_anet_native 500
vc_charades_sta 3679
trace_qvh 1541
TARGETS
  exit $short
fi

# A card is free when almost nothing is resident on it. Checking the card rather than a
# process list means this cannot be fooled by a job the script did not start.
wait_card() {
  while [ "$(nvidia-smi -i "$GPUID" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do
    sleep 60
  done
}

# ---- job runners. Each blocks until its job finishes, so a queue is strictly sequential.
qwen() {  # outdir model json videodir shard nshards target
  local out=$1 model=$2 json=$3 vids=$4 sh=$5 ns=$6 target=$7
  mkdir -p "logs/$out"
  if [ "$(count "$out")" -ge "$target" ]; then say "SKIP $out (already $target)"; return; fi
  _go() {
    CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
      .venv/bin/python eval_bench.py --model "$model" --json "$json" --video_dir "$vids" \
      --patch_embed matmul --decode_workers 3 \
      --out "logs/$out/preds_s${sh}of${ns}.jsonl" --shard "$sh" --num_shards "$ns" \
      >> "logs/$out/run_s${sh}of${ns}.log" 2>&1
  }
  say "START $out shard $sh/$ns"
  _go
  verify "$out" "$target" && return
  say "RETRY $out shard $sh/$ns"
  _go
  verify "$out" "$target" || say "STILL SHORT $out -- needs a human; not silently accepted"
}

vchat() {  # outdir anno videoroot fmt shard nshards target
  local out=$1 anno=$2 vids=$3 fmt=$4 sh=$5 ns=$6 target=$7
  mkdir -p "logs/$out"
  if [ "$(count "$out")" -ge "$target" ]; then say "SKIP $out (already $target)"; return; fi
  _go() {
    ( cd ../VideoChat-R1 && CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
      .venv/bin/python run_ns_p1.py --anno "$anno" --video-root "$vids" --fmt "$fmt" \
      --out "$TL/logs/$out/preds_s${sh}of${ns}.jsonl" --shard "$sh" --nshards "$ns" \
      >> "$TL/logs/$out/run_s${sh}of${ns}.log" 2>&1 )
  }
  say "START $out shard $sh/$ns (videochat-r1)"
  _go
  verify "$out" "$target" && return
  say "RETRY $out shard $sh/$ns"
  _go
  verify "$out" "$target" || say "STILL SHORT $out -- needs a human; not silently accepted"
}

trace_job() {  # outdir bench videos target
  local out=$1 bench=$2 vids=$3 target=$4
  mkdir -p "logs/$out"
  if [ "$(count "$out")" -ge "$target" ]; then say "SKIP $out (already $target)"; return; fi
  _go() {
    ( cd ../TRACE && CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
      .venv/bin/python run_trace_charades.py --bench "$bench" --videos "$vids" \
      --out "$TL/logs/$out/preds.jsonl" >> "$TL/logs/$out/run.log" 2>&1 )
  }
  say "START $out (TRACE, single-process, no resume)"
  _go
  verify "$out" "$target" && return
  # TRACE has no resume, so a retry redoes the whole split. Truncate first, otherwise the
  # rerun appends to a partial file and the count looks right while the contents are mixed.
  say "RETRY $out (truncating: TRACE appends and cannot resume)"
  : > "logs/$out/preds.jsonl"
  _go
  verify "$out" "$target" || say "STILL SHORT $out -- needs a human; not silently accepted"
}

Q7=Qwen/Qwen2.5-VL-7B-Instruct
Q72=Qwen/Qwen2.5-VL-72B-Instruct

# ---- the four queues -------------------------------------------------------
# Balanced by measured cost. The 72B is the pole (~8 s/clip/shard), so it keeps two cards
# and nothing else is stacked behind it until its three splits are done.
queue0() {
  GPUID=0; wait_card
  qwen qwen72b_qvh          "$Q72" "$B/qvhighlights-timelens.json"            "$QVH_V" 0 2 1541
  qwen qwen72b_anet_native  "$Q72" "$B/activitynet-native-canonical500.json"  "$ANET_V" 0 2 500
  qwen qwen72b_charades_sta "$Q72" "$B/charades-sta-native.json"              "$CHAR_V" 0 2 3679
}
queue1() {
  GPUID=1; wait_card
  qwen qwen72b_qvh          "$Q72" "$B/qvhighlights-timelens.json"            "$QVH_V" 1 2 1541
  qwen qwen72b_anet_native  "$Q72" "$B/activitynet-native-canonical500.json"  "$ANET_V" 1 2 500
  qwen qwen72b_charades_sta "$Q72" "$B/charades-sta-native.json"              "$CHAR_V" 1 2 3679
}
queue2() {
  GPUID=2; wait_card
  trace_job trace_anet_native "$TL/$B/activitynet-native-canonical500.json" "$ANET_V" 500
  trace_job trace_qvh         "$TL/$B/qvhighlights-timelens.json"           "$QVH_V"  1541
  vchat vc_qvh          "$TL/$B/qvhighlights-timelens.json"           "$QVH_V"  timelens 0 2 1541
  vchat vc_anet_native  "$TL/$B/activitynet-native-canonical500.json" "$ANET_V" timelens 0 2 500
  vchat vc_charades_sta "$TL/$B/charades-sta-native.json"             "$CHAR_V" timelens 0 2 3679
}
queue3() {
  GPUID=3; wait_card
  qwen qwen7b_qvh          "$Q7" "$B/qvhighlights-timelens.json"           "$QVH_V" 2 3 1541
  qwen qwen7b_anet_native  "$Q7" "$B/activitynet-native-canonical500.json" "$ANET_V" 0 1 500
  qwen qwen7b_charades_sta "$Q7" "$B/charades-sta-native.json"             "$CHAR_V" 0 1 3679
  vchat vc_qvh          "$TL/$B/qvhighlights-timelens.json"           "$QVH_V"  timelens 1 2 1541
  vchat vc_anet_native  "$TL/$B/activitynet-native-canonical500.json" "$ANET_V" timelens 1 2 500
  vchat vc_charades_sta "$TL/$B/charades-sta-native.json"             "$CHAR_V" timelens 1 2 3679
}

printf '[%s] queue start; stagger=%ss\n' "$(date -u '+%F %T')" "$GPU_START_STAGGER"
queue0 &
sleep "$GPU_START_STAGGER"; queue1 &
sleep 60; queue2 &
sleep 60; queue3 &
wait

printf '[%s] ==== ALL QUEUES COMPLETE ====\n' "$(date -u '+%F %T')"
for d in qwen7b_qvh qwen7b_anet_native qwen7b_charades_sta \
         qwen72b_qvh qwen72b_anet_native qwen72b_charades_sta \
         vc_qvh vc_anet_native vc_charades_sta trace_anet_native trace_qvh; do
  printf '  %-24s %6d\n' "$d" "$(count "$d")"
done
