#!/usr/bin/env bash
# Ego4D NLQ crop arms (issue #96), on the sub-2s stratum: 1145 clips of 4467.
#
# WHAT THIS TESTS. The NLQ baseline places moments ~60-92s away from truth on 480s clips
# while predicting roughly the right LENGTH (median 6.3s against GT 4.0s). If that failure
# is a search-space cost rather than a representational one, shrinking the search space
# should fix it. The crop shrinks it two ways at once, and the second is specific to this
# dataset: an 8s window at 8fps samples 64 frames where the full-video pass samples 384
# frames over 480s. Effective fps goes 0.8 -> 8, a 10x. Sub-1.25s moments that could fall
# BETWEEN sampled frames in the baseline are, for the first time, actually sampled.
#
# TWO ARMS, and the gap between them is the result:
#   ceiling   - window placed on the GT centre (off-centre by `place` so localization
#               inside the crop is still non-trivial). Upper bound: can the frozen model
#               resolve the moment at all once the haystack is small?
#   realistic - window placed on the model's OWN first-pass centre, from the baseline
#               predictions. The deployable two-pass method, bounded by how often the
#               first pass lands near the moment. On a 6-clip smoke test containment was
#               1 full / 2 miss, so expect this to be much weaker than ceiling here.
#
# COMPARE ONLY THE SUB-2s METRICS against the baseline. The arms cover 1145 clips and the
# baseline covers 4467, so aggregate mIoU and overall_Rs7 are computed on different
# populations and are NOT comparable. The sub-2s cells are, because both are the same 1145
# GT items.
#
# ATTENTION IMPLEMENTATION DIFFERS FROM THE BASELINE and is left alone deliberately:
# eval_zoom.py defaults to flash_attention_2 where eval_bench.py used sdpa. Same maths,
# different reduction order. Noted so nobody reads a difference into it later; if it ever
# matters, verify_preds_identical.py is the tool.
set -uo pipefail
cd "$(dirname "$0")"

MODEL=TencentARC/TimeLens-7B
DATASET=ego4d-nlq
BASE_PREDS=logs/nlq_7b_val
COARSE=logs/nlq_crop/coarse_merged.jsonl
WINDOW_LEN=8
CROP_FPS=8
MAX_MOMENT=2.0
EXPECT=1145
NSHARD=4
# Same I/O discipline as the baseline run, which is what let the pod stay up 10.5h:
# weights local (NFS at concurrency ran at 110s/shard against 3s local), and reader count
# held down. See run_nlq_baseline_chain.sh for the measurements.
MODEL_CACHE=/var/tmp/hf-cache

mkdir -p logs/nlq_crop
PATH="$PWD/.venv/bin:$PATH"; export PATH
say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

# ---- coarse centres: one file, because _load_coarse_centers() opens a PATH, not a glob.
# Passing a single shard silently halves the routable population; on the smoke test that
# showed up as "3 skipped: no coarse center" out of 6.
if [ ! -s "$COARSE" ]; then
  cat "$BASE_PREDS"/preds_s*of4.jsonl > "$COARSE"
  say "merged coarse centres: $(wc -l < "$COARSE") records from $BASE_PREDS"
fi
[ "$(wc -l < "$COARSE")" -ge 4000 ] || { say "ABORT: coarse file looks truncated"; exit 5; }

count() {  # $1 = arm
  local t=0 i f
  for i in $(seq 0 $((NSHARD-1))); do
    f="logs/nlq_crop/${1}_s${i}of${NSHARD}.jsonl"
    [ -f "$f" ] && t=$((t + $(wc -l < "$f")))
  done
  echo "$t"
}

run_arm() {
  local arm=$1 extra=$2
  for attempt in 1 2 3; do
    n=$(count "$arm")
    if [ "$n" -ge "$EXPECT" ]; then say "$arm complete: $n/$EXPECT"; return 0; fi
    say "$arm attempt $attempt: $n/$EXPECT on disk, launching $NSHARD shards"
    pids=()
    for i in $(seq 0 $((NSHARD-1))); do
      CUDA_VISIBLE_DEVICES=$i setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
        TORCH_DISABLE_ADDR2LINE=1 HF_HOME="$MODEL_CACHE" \
        .venv/bin/python eval_zoom.py \
        --mode "$arm" --dataset "$DATASET" --model "$MODEL" \
        --window_len "$WINDOW_LEN" --fps "$CROP_FPS" --max_moment_s "$MAX_MOMENT" $extra \
        --out "logs/nlq_crop/${arm}_s${i}of${NSHARD}.jsonl" \
        --shard "$i" --num_shards "$NSHARD" \
        >> "logs/nlq_crop/${arm}_s${i}of${NSHARD}.log" 2>&1 < /dev/null &
      pids+=($!)
    done
    say "  $arm shards: ${pids[*]}"
    # Explicit PIDs, never `pgrep -f eval_zoom.py`: this script's own command line contains
    # that string and such a loop waits on itself forever.
    for p in "${pids[@]}"; do wait "$p"; done
    say "  $arm attempt $attempt finished ($(count "$arm")/$EXPECT)"
  done
  n=$(count "$arm")
  [ "$n" -ge "$EXPECT" ] || { say "$arm INCOMPLETE $n/$EXPECT after 3 attempts"; return 3; }
}

say "CEILING arm (oracle-placed window)"
run_arm ceiling "" || exit 3
say "REALISTIC arm (window from the model's own first pass)"
run_arm realistic "--coarse_jsonl $COARSE" || exit 3

# ---------------------------------------------------------------- score
say "scoring (validating the scorer against its published landmarks first)"
SCORER="env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python score_short_stratum.py"
if ! $SCORER --validate > logs/nlq_crop/score.txt 2>&1; then
  say "ABORT: scorer fails its own landmarks."; cat logs/nlq_crop/score.txt; exit 4
fi
for arm in ceiling realistic; do
  $SCORER --preds "logs/nlq_crop/${arm}_s*of${NSHARD}.jsonl" --label "nlq-crop-${arm}" \
    >> logs/nlq_crop/score.txt 2>&1 || { say "ABORT: scoring $arm failed"; exit 4; }
done
# Baseline for the only comparison that is fair: its sub-2s cells are the same 1145 items.
$SCORER --preds "$BASE_PREDS/preds_s*of4.jsonl" --label "nlq-baseline-full" \
  >> logs/nlq_crop/score.txt 2>&1
tail -20 logs/nlq_crop/score.txt
say "DONE -> logs/nlq_crop/score.txt"
say "REPORT: effective fps 0.8 (baseline, 384 frames/480s) vs 8.0 (crop, 64 frames/8s)."
say "Compare SUB-2s cells only; aggregate cells are on different populations."
