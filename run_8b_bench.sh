#!/usr/bin/env bash
# TimeLens-8B full-video grounding on the two TimeLens-Bench splits we have 7B
# numbers for but no 8B ones: QVHighlights and ActivityNet.
#
#   SHORT=2.0 SHARDS=0,1,2 GPUS=0 ./run_8b_bench.sh qvh     # sub-2s stratum only
#   SHARDS=0,1,2,3,4,5 GPUS=0,1 ./run_8b_bench.sh anet      # full split
#
# THE SUB-2 S STRATUM RUNS FIRST, and on these two splits it may be all we run.
# 8B costs ~420 s/clip here against ~36 s/clip on Charades, because Charades
# videos average 30 s and these average 135-150 s. At fps 2 that is 299 frames
# instead of 59, and the Qwen3-VL ViT in 8B has no windowed attention (no
# window_size, no fullatt_block_indexes -- see issue #82), so all 27 layers
# attend over every patch of every frame at once and cost grows quadratically in
# video length. 5x the frames is ~25x the ViT attention. 7B's windowed ViT keeps
# this near-linear, which is why the penalty is 8B-specific and why the cheap 8B
# Charades run does not predict the cost here.
#
# At that rate the full splits are ~30 h (qvh) and ~80 h (anet). But sub-2 s is
# the only stratum the paper's claim concerns, and it is 6.0% of QVHighlights
# (92/1541) and 12.5% of ActivityNet (564/4500). SHORT=2.0 buys the decisive
# numbers -- emission rate, sub-2 s recall, the |dc| floor -- for a tenth of the
# compute. The aggregates (mIoU, beta) need the full split and are a separate,
# much more expensive decision; do not start one by accident.
#
# Why these two and why now: the ladder and the crop reversal are TimeLens-7B
# only, so every interventional claim rests on one checkpoint. 8B already beats
# 7B on Charades aggregates while keeping the same sub-2 s center floor, and it
# emits whole-second timestamps (3363/3363 in the #84 audit), so it is
# convention-invariant -- UNROUNDED and ROUNDED agree on it by construction and
# neither reading can flatter it. Extending it to the other two splits turns a
# one-dataset, one-model observation into a 2x3 grid.
#
# MATCHING THE 7B RUNS EXACTLY. These reuse the same annotation JSONs and video
# roots as the existing 7B predictions, so the comparison is same-split rather
# than same-benchmark-name:
#   qvh  -> qvhighlights-timelens.json,  1541 clips, 7B in logs/qvh_timelens/
#   anet -> activitynet-timelens.json,   4140 clips, 7B in logs/anet_timelens_robust/
# ActivityNet lists 4500 (key, query) pairs; 360 resolve to videos that are not
# on disk and load_annos drops them, which is where 4140 comes from. The 7B run
# dropped the same 360, so the sets line up without any post-hoc intersection.
# Every generation knob is left at the eval_bench default (fps 2, 14336 tokens,
# 512 new tokens) because that is what produced both the 7B splits and the 8B
# Charades baseline.
#
# SCORING: both cells are ROUNDED, and the reason is worth recording because it
# corrects a rule stated too coarsely elsewhere. Grid alignment is a property of
# the FILE, not of the dataset. qvhighlights-timelens.json is integer 1541/1541,
# as expected. activitynet-timelens.json is ALSO integer, 4500/4500 -- TimeLens
# rounded ActivityNet's float ground truth onto a 1 s grid when it repackaged the
# split (spans read [0, 22] and [5, 10] while durations keep full precision at
# 38.5 and 48.3483). So "ActivityNet is UNROUNDED" holds for the omniembed
# packaging behind the 7B activitynet-omniembed run and NOT for this one. Trust
# audit_precision.py's per-file gt_grid over any per-dataset claim, including the
# one in that script's own docstring. 8B emits whole seconds anyway, so the two
# conventions agree on these runs by construction -- but the rule still decides
# how the 7B files they are compared against get read.
#
# NSHARDS IS FROZEN AT 6. eval_bench resumes by reading --out and skipping keys
# it already contains, while the work it iterates is annos[shard::num_shards].
# Resume a shard under a different NSHARDS and it owns a different slice than
# the file it is skipping against, so it silently stops early and looks
# complete. The count is baked into the filename (_s0of6) so a mismatch shows
# up as a new empty file rather than a quietly truncated old one.
#
# setsid + launch-from-a-file for the reasons documented in run_eval_arm.sh: a
# watcher that pattern-matches its own cmdline never clears, and stopping a
# watcher SIGTERMs its nohup'd children.
set -euo pipefail

BENCH=${1:?usage: $0 <qvh|anet>   [env: SHORT=2.0 SHARDS=0,1,2 GPUS=0 NSHARDS=6]}
NSHARDS=${NSHARDS:-4}
SHARDS=${SHARDS:-0,1,2,3}
GPUS=${GPUS:-0,1,2,3}
SHORT=${SHORT:-0}          # GT-length cutoff in seconds; 0 = full split
DECODE_WORKERS=${DECODE_WORKERS:-3}
# WHY ONE SHARD PER GPU WITH DECODE WORKERS, rather than many shards per GPU.
# Measured on a 150 s clip: decode 1.77 s, generate 1.52 s -- roughly balanced, so the
# ceiling is GPU generate time and ~2 workers already saturate a card. Overlapping decode
# inside ONE process per GPU reaches that ceiling (2.8 s/clip -> 1.4 s/clip, verified
# byte-identical on 12/12 records) while loading the model once per GPU instead of once
# per shard. Piling 4 shards on a card would hit the same ceiling with 4x the weights
# resident and 4x the cold-start cost, which on a cold GPFS cache is ~150 s each.
# Raising DECODE_WORKERS past ~3 buys nothing: generate, not decode, is the limit.

cd "$(dirname "$0")"
case "$BENCH" in
  qvh)
    JSON=data/TimeLens-Bench/qvhighlights-timelens.json
    VIDEO_DIR=/gpfs/public/datasets/qvhighlights/videos ;;
  anet)
    JSON=data/TimeLens-Bench/activitynet-timelens.json
    VIDEO_DIR=/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos ;;
  *) echo "usage: $0 <qvh|anet>" >&2; exit 2 ;;
esac
[ -f "$JSON" ] || { echo "missing annotations: $JSON" >&2; exit 1; }

MODEL=TencentARC/TimeLens-8B
# A filtered run and a full run disagree about which clips a shard index owns, so
# they get separate directories. Sharing one would let a resume skip against a
# file describing different work and stop early looking complete.
OUTDIR=logs/scale8b_$BENCH
extra=()
if [ "$SHORT" != "0" ]; then
  OUTDIR=logs/scale8b_${BENCH}_short
  extra=(--max_moment_s "$SHORT")
fi
PIDFILE=$OUTDIR/run.pids
mkdir -p "$OUTDIR"

IFS=',' read -r -a gpu_arr <<< "$GPUS"
IFS=',' read -r -a shard_arr <<< "$SHARDS"
# Enforce the one-process-per-GPU design instead of only documenting it. Round-robin
# assignment silently stacks processes when there are more shards than GPUs (6 shards on
# 4 GPUs puts two on each of GPU 0 and 1), which contradicts the rationale above and
# makes throughput depend on which shards you happened to select. Flagged by an external
# review of this change. Override deliberately with ALLOW_STACKING=1.
if [ "${#shard_arr[@]}" -gt "${#gpu_arr[@]}" ] && [ "${ALLOW_STACKING:-0}" != "1" ]; then
  echo "refusing to launch ${#shard_arr[@]} shards on ${#gpu_arr[@]} gpu(s): that stacks" >&2
  echo "  processes unevenly. Run them in waves, or set ALLOW_STACKING=1 if intended." >&2
  exit 2
fi
n=0
for i in "${shard_arr[@]}"; do
  [ "$i" -lt "$NSHARDS" ] || { echo "shard $i >= NSHARDS $NSHARDS" >&2; exit 2; }
  gpu=${gpu_arr[$((n % ${#gpu_arr[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
    .venv/bin/python eval_bench.py \
    --model "$MODEL" --json "$JSON" --video_dir "$VIDEO_DIR" \
    --decode_workers "$DECODE_WORKERS" \
    "${extra[@]}" \
    --out "$OUTDIR/preds_s${i}of${NSHARDS}.jsonl" \
    --shard "$i" --num_shards "$NSHARDS" \
    >> "$OUTDIR/run_s${i}of${NSHARDS}.log" 2>&1 < /dev/null &
  echo $! >> "$PIDFILE"
  n=$((n + 1))
done
note=$([ "$SHORT" != "0" ] && echo " (GT<${SHORT}s only)" || echo " (full split)")
echo "launched 8b/$BENCH shards $SHARDS of $NSHARDS on gpu(s) $GPUS -> $OUTDIR$note"
tail -n "$n" "$PIDFILE" | tr '\n' ' '; echo
