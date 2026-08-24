#!/usr/bin/env bash
# Ego4D NLQ val, zero-shot TimeLens-7B. The FIRST population in this project that can
# falsify the paper's central number rather than confirm it.
#
# WHY. NS-P1 claims a residual ~1s centre error that no backbone update crosses. Every
# split the paper currently uses is too coarse to test that: native Charades-STA has NO
# sub-2s moments, and TACoS test has 7 sub-1s moments out of 4001 (measured 2026-08-21).
# Ego4D NLQ val has 386 sub-1s and 1145 sub-2s moments out of 4467, on 480s clips whose
# median moment is 0.76% of the video. If the ~1s floor is real it should appear here; if
# it is an artifact of 30s Charades clips, here is where that shows.
#
# THE SAMPLING FACT THIS RUN WILL DEMONSTRATE, and it must be reported, not buried.
# At total_tokens=14336 a 480s clip is sampled at 384 frames, i.e. 0.8 effective fps, one
# frame every 1.25s. A 0.41s moment is therefore SHORTER THAN THE GAP BETWEEN SAMPLED
# FRAMES: the model can miss it entirely without ever being wrong about what it saw. That
# is not a flaw in the measurement, it is the search-space thesis in its strongest form,
# and the crop lever's prediction here is sharp: cropping raises effective fps inside the
# window until the moment becomes sampled at all. Report effective fps alongside recall,
# or a reader will read a sampling limit as a representation limit -- the exact confusion
# the paper exists to correct.
#
# Baseline only. The crop/zoom pass is a separate run and should be built on whatever this
# one shows, not queued blind behind it.
set -uo pipefail
cd "$(dirname "$0")"

MODEL=TencentARC/TimeLens-7B
ANNO=data/TimeLens-Bench/ego4d-nlq-val.json
VIDEOS=/gpfs/public/datasets/ego4d_data/v2/clips
OUT=logs/nlq_7b_val
EXPECT=4467
NSHARD=4
# I/O BUDGET, REVISED 2026-08-24 after this run took the pod down twice.
#
# The first configuration was NSHARD=4 x DECODE_WORKERS=3. That is four concurrent 16G
# model reads over NFS plus twelve subprocesses streaming 480s videos off GPFS, while
# writing predictions back to NFS. Measured symptom: checkpoint loading fell to 110s PER
# SHARD against ~1s for a single loader, a 100x slowdown that was pure NFS contention, and
# load average sat at 60 with the GPUs at 0%. The pod restarted twice (PID 1 age confirmed
# it; cgroup oom_kill was 0 and the 400G limit was never approached, so this was I/O
# pressure and not memory). CLAUDE.md warns about exactly this for GPFS.
#
# Three changes, all aimed at I/O rather than compute:
#   1. HF_HOME points at a LOCAL copy of the weights (see MODEL_CACHE below). Model
#      loading went 440s -> 8.2s and NFS leaves the model path entirely.
#   2. DECODE_WORKERS 3 -> 1, cutting GPFS video readers from 12 to CONCURRENCY.
#   3. CONCURRENCY caps how many shards run AT ONCE. The shard COUNT stays 4 so the
#      existing preds_s*of4.jsonl files and their 1285 records remain resumable; dropping
#      to NSHARD=2 would rename the outputs AND repartition which clips each shard owns,
#      discarding that work. Same halved concurrency, none of the cost.
DECODE_WORKERS=1
# 2026-08-24, second revision. CONCURRENCY=2 was MEASURED at 2.9 rec/min against 12.8 for
# the original 4x3 config: an 18.3h ETA, not the ~2x slowdown I predicted. Decode is the
# bottleneck (GPUs sat at 0% util), so cutting decode parallelism costs about 2x per shard,
# and I had cut it twice over. 4 shards x 1 decode worker keeps readers at 4, still a third
# of the original 12, while using all four GPUs: ~5.8 rec/min, ~9h.
# The reader count, not the shard count, is what to hold down if the pod misbehaves again.
CONCURRENCY=4
# Staged with: cp -a ~/.cache/huggingface/hub/models--TencentARC--TimeLens-7B \
#                    /var/tmp/hf-cache/hub/
# Verified 26 files / 16600376152 bytes identical on both sides. This is the container's
# overlay filesystem, so it is LOST on pod restart; if loading is slow again, re-copy
# rather than assuming the cache is warm.
MODEL_CACHE=/var/tmp/hf-cache

mkdir -p "$OUT"
PATH="$PWD/.venv/bin:$PATH"; export PATH
say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

say "MODEL=$MODEL  anno=$ANNO  expect=$EXPECT pairs over $NSHARD shards"

count() {
  local t=0 i f
  for i in $(seq 0 $((NSHARD-1))); do
    f="$OUT/preds_s${i}of${NSHARD}.jsonl"
    [ -f "$f" ] && t=$((t + $(wc -l < "$f")))
  done
  echo "$t"
}

for attempt in 1 2 3; do
  n=$(count)
  if [ "$n" -ge "$EXPECT" ]; then say "complete: $n/$EXPECT"; break; fi
  say "attempt $attempt: $n/$EXPECT on disk, $NSHARD shards in waves of $CONCURRENCY (resume)"
  # WAVES, not all-at-once. Each shard still owns the same clips and writes the same file,
  # so resume is unaffected; only the number of simultaneous readers changes. A shard that
  # finishes early leaves its GPU idle until the wave ends, which is accepted: the point of
  # this loop is to bound I/O concurrency, and the shards are within ~4% of each other in
  # size (1117/1117/1117/1116) so the imbalance is negligible.
  for wave_start in $(seq 0 "$CONCURRENCY" $((NSHARD-1))); do
    pids=()
    for i in $(seq "$wave_start" $((wave_start + CONCURRENCY - 1))); do
      [ "$i" -ge "$NSHARD" ] && break
      CUDA_VISIBLE_DEVICES=$i setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
        TORCH_DISABLE_ADDR2LINE=1 HF_HOME="$MODEL_CACHE" \
        .venv/bin/python eval_bench.py --model "$MODEL" \
        --json "$ANNO" --video_dir "$VIDEOS" \
        --decode_workers "$DECODE_WORKERS" \
        --out "$OUT/preds_s${i}of${NSHARD}.jsonl" --shard "$i" --num_shards "$NSHARD" \
        >> "$OUT/run_s${i}of${NSHARD}.log" 2>&1 < /dev/null &
      pids+=($!)
    done
    say "  wave from shard $wave_start: pids ${pids[*]}"
    # Explicit PIDs, never `pgrep -f eval_bench.py`: this script's own command line contains
    # that string, so such a loop waits on itself forever. Documented in run_full_chain.sh
    # and hit three times in this project already.
    for p in "${pids[@]}"; do wait "$p"; done
    say "  wave from shard $wave_start finished ($(count)/$EXPECT on disk)"
  done
  say "attempt $attempt finished"
done

n=$(count)
for i in $(seq 0 $((NSHARD-1))); do
  f="$OUT/preds_s${i}of${NSHARD}.jsonl"
  say "  shard $i: $([ -f "$f" ] && wc -l < "$f" || echo 0) records, skips=$(grep -c '^  \[skip ' "$OUT/run_s${i}of${NSHARD}.log" 2>/dev/null || echo 0)"
done
if [ "$n" -lt "$EXPECT" ]; then
  say "INCOMPLETE after 3 attempts: $n/$EXPECT. NOT scoring: a partial population changes"
  say "every number, and on a new dataset there is no published value to catch it."
  exit 3
fi
say "all $n/$EXPECT records present"

# --validate still guards the SCORER even though its landmarks are Charades runs: it is a
# regression check on the metric code, which is dataset-independent. There is no published
# NLQ value to check the RESULT against, which is the point of running it.
say "scoring (validating the scorer against its published Charades landmarks first)"
SCORER="env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python score_short_stratum.py"
if ! $SCORER --validate > "$OUT/score.txt" 2>&1; then
  say "ABORT: scorer fails its own landmarks; nothing from this run is trustworthy."
  cat "$OUT/score.txt"; exit 4
fi
if ! $SCORER --preds "$OUT/preds_s*of${NSHARD}.jsonl" --label "nlq-7b-val" >> "$OUT/score.txt" 2>&1; then
  say "ABORT: scoring failed. See $OUT/score.txt"; tail -20 "$OUT/score.txt"; exit 4
fi
tail -14 "$OUT/score.txt"
say "DONE -> $OUT/score.txt"
say "REPORT EFFECTIVE FPS WITH THESE NUMBERS: 384 frames / 480s = 0.8 fps, so sub-1.25s"
say "moments can fall between sampled frames. Recall alone will overstate a representation"
say "limit and understate the search-space one."
