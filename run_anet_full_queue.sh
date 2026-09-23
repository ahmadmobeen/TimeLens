#!/usr/bin/env bash
# Full-split native ActivityNet (17,505 pairs) for the four tab:prior models that only
# ever ran on the 500-pair subsample. Makes tab:prior's ActivityNet column uniform:
# the two TimeLens rows are already on the full split, these four are not, and a column
# whose rows sit on different populations cannot support a cross-model claim.
#
#   setsid nohup ./run_anet_full_queue.sh >> logs/anet_full_queue.log 2>&1 &
#   ./run_anet_full_queue.sh status
#
# Structure and job runners are lifted from run_tabprior_queue.sh, including the reasons
# recorded there: per-GPU queues so one straggler cannot idle the other card, a staggered
# first launch so two ~145 GB Qwen-72B cold reads do not collide on GPFS, and verify()
# treating an under-delivering run as a failure rather than a completion.
#
# TWO GPUs, not four. Work is balanced to ~49 GPU-h per card:
#   Qwen-72B  2 shards x 8753 @ ~8 s  = 38.8 GPU-h
#   Qwen-7B   2 shards x 8753 @ ~5 s  = 24.4
#   VideoChat-R1 2 shards    @ ~5 s   = 24.4
#   TRACE     2 halves x ~8.7k @ ~2.3 s = 11.2
#
# TRACE IS SPLIT BY MANIFEST, NOT BY --shard, because run_trace_charades.py has neither a
# shard flag nor resume. Two half-manifests give it parallelism and halve what is lost to a
# crash. It is scheduled second so a failure surfaces around hour 25 with slack to redo it,
# rather than at hour 44.
set -uo pipefail
cd "$(dirname "$0")"

TL=$PWD
B=data/TimeLens-Bench
ANET_V=/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos
FULL=$B/activitynet-native-full17505.json
GPU_START_STAGGER=${GPU_START_STAGGER:-420}
N=17505

say() { printf '[%s] gpu%s %s\n' "$(date -u '+%F %T')" "${GPUID:-?}" "$*"; }
count() { cat logs/"$1"/*.jsonl 2>/dev/null | wc -l; }          # the WHOLE arm
count_shard() { wc -l < "logs/$1/preds_s${2}of${3}.jsonl" 2>/dev/null || echo 0; }

# Both sharded runners partition with items[shard::nshards] (eval_bench.py:199,
# run_ns_p1.py:105), so shard sh of ns over N items owns exactly this many.
# N=17505, ns=2 -> 8753 and 8752, which is what both arms actually wrote.
shard_target() { echo $(( ($1 - $2 + $3 - 1) / $3 )); }         # N sh ns

# verify() is a WHOLE-ARM check and is only safe for a job that is the arm's
# only writer (trace_job: one process, one file, its own directory).
#
# A sharded job must NOT use it. It did until 2026-09-23 and cost a full rerun:
# two shards write into one directory, count() sums the directory, and each
# shard called verify() against the whole-arm target. On 2026-09-22 gpu0
# finished vc_anet_full shard 0 complete at 8753 while gpu1 was still 40
# records from done; the directory summed to 17465, gpu0 read that as its own
# shortfall and retried. run_ns_p1.py opens --out with "w", so the retry
# truncated a finished shard and re-decoded 8753 clips -- 10.5 GPU-hours.
# Then gpu1 finished and its verify() summed the directory back to 17505 and
# printed "OK" while shard 0 was being rewritten from zero.
verify() {  # outdir target
  local out=$1 target=$2 n
  n=$(count "$out")
  if [ "$n" -ge "$target" ]; then say "END   $out -> $n/$target OK"; return 0; fi
  say "SHORTFALL $out -> $n/$target -- retrying once"
  return 1
}

verify_shard() {  # outdir shard nshards arm-total
  local out=$1 sh=$2 ns=$3 n t
  t=$(shard_target "$4" "$sh" "$ns")
  n=$(count_shard "$out" "$sh" "$ns")
  if [ "$n" -ge "$t" ]; then say "END   $out shard $sh/$ns -> $n/$t OK"; return 0; fi
  say "SHORTFALL $out shard $sh/$ns -> $n/$t -- retrying once"
  return 1
}

if [ "${1:-}" = "status" ]; then
  short=0
  while read -r d t; do
    [ -z "$d" ] && continue
    n=$(cat logs/"$d"/*.jsonl 2>/dev/null | wc -l)
    if [ "$n" -lt "$t" ]; then printf 'SHORT %-28s %6d/%s\n' "$d" "$n" "$t"; short=1
    else printf 'OK    %-28s %6d/%s\n' "$d" "$n" "$t"; fi
  done <<'TARGETS'
qwen72b_anet_full 17505
qwen7b_anet_full 17505
vc_anet_full 17505
trace_anet_full_h0 8843
trace_anet_full_h1 8662
TARGETS
  exit $short
fi

wait_card() {
  while [ "$(nvidia-smi -i "$GPUID" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do
    sleep 60
  done
}

qwen() {  # outdir model shard nshards target
  local out=$1 model=$2 sh=$3 ns=$4 target=$5
  mkdir -p "logs/$out"
  if [ "$(count_shard "$out" "$sh" "$ns")" -ge "$(shard_target "$target" "$sh" "$ns")" ]; then
    say "SKIP $out shard $sh/$ns (already complete)"; return
  fi
  _go() {
    CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
      .venv/bin/python eval_bench.py --model "$model" --json "$FULL" --video_dir "$ANET_V" \
      --patch_embed matmul --decode_workers 3 \
      --out "logs/$out/preds_s${sh}of${ns}.jsonl" --shard "$sh" --num_shards "$ns" \
      >> "logs/$out/run_s${sh}of${ns}.log" 2>&1
  }
  say "START $out shard $sh/$ns"
  _go
  verify_shard "$out" "$sh" "$ns" "$target" && return
  say "RETRY $out shard $sh/$ns (eval_bench resumes from --out)"
  _go
  verify_shard "$out" "$sh" "$ns" "$target" \
    || say "STILL SHORT $out shard $sh/$ns -- needs a human; not silently accepted"
}

vchat() {  # outdir shard nshards target
  local out=$1 sh=$2 ns=$3 target=$4
  mkdir -p "logs/$out"
  if [ "$(count_shard "$out" "$sh" "$ns")" -ge "$(shard_target "$target" "$sh" "$ns")" ]; then
    say "SKIP $out shard $sh/$ns (already complete)"; return
  fi
  _go() {
    ( cd ../VideoChat-R1 && CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
      .venv/bin/python run_ns_p1.py --anno "$TL/$FULL" --video-root "$ANET_V" --fmt timelens \
      --out "$TL/logs/$out/preds_s${sh}of${ns}.jsonl" --shard "$sh" --nshards "$ns" \
      >> "$TL/logs/$out/run_s${sh}of${ns}.log" 2>&1 )
  }
  say "START $out shard $sh/$ns (videochat-r1)"
  _go
  verify_shard "$out" "$sh" "$ns" "$target" && return
  say "RETRY $out shard $sh/$ns"
  _go
  verify_shard "$out" "$sh" "$ns" "$target" \
    || say "STILL SHORT $out shard $sh/$ns -- needs a human; not silently accepted"
}

trace_job() {  # outdir half target
  local out=$1 half=$2 target=$3
  mkdir -p "logs/$out"
  if [ "$(count "$out")" -ge "$target" ]; then say "SKIP $out (already $target)"; return; fi
  _go() {
    ( cd ../TRACE && CUDA_VISIBLE_DEVICES=$GPUID env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 \
      .venv/bin/python run_trace_charades.py \
      --bench "$TL/$B/activitynet-native-full17505_${half}.json" --videos "$ANET_V" \
      --out "$TL/logs/$out/preds.jsonl" >> "$TL/logs/$out/run.log" 2>&1 )
  }
  say "START $out ($half, TRACE single-process, no resume)"
  _go
  verify "$out" "$target" && return
  # TRACE appends and cannot resume, so a retry must start from an empty file or the count
  # looks right while the contents are a mix of two partial passes.
  say "RETRY $out (truncating first)"
  : > "logs/$out/preds.jsonl"
  _go
  verify "$out" "$target" || say "STILL SHORT $out -- needs a human; not silently accepted"
}

Q7=Qwen/Qwen2.5-VL-7B-Instruct
Q72=Qwen/Qwen2.5-VL-72B-Instruct

queue0() {
  GPUID=0; wait_card
  qwen      qwen72b_anet_full  "$Q72" 0 2 $N
  trace_job trace_anet_full_h0 h0 8843
  qwen      qwen7b_anet_full   "$Q7"  0 2 $N
  vchat     vc_anet_full       0 2 $N
}
queue1() {
  GPUID=1; wait_card
  qwen      qwen72b_anet_full  "$Q72" 1 2 $N
  trace_job trace_anet_full_h1 h1 8662
  qwen      qwen7b_anet_full   "$Q7"  1 2 $N
  vchat     vc_anet_full       1 2 $N
}

printf '[%s] anet-full queue start; stagger=%ss; target=%s pairs/model\n' "$(date -u '+%F %T')" "$GPU_START_STAGGER" "$N"
queue0 &
sleep "$GPU_START_STAGGER"; queue1 &
wait

printf '[%s] ==== ALL QUEUES COMPLETE ====\n' "$(date -u '+%F %T')"
for d in qwen72b_anet_full qwen7b_anet_full vc_anet_full trace_anet_full_h0 trace_anet_full_h1; do
  printf '  %-28s %6d\n' "$d" "$(count "$d")"
done
