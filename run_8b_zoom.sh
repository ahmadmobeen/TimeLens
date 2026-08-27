#!/usr/bin/env bash
# Temporal zoom on TimeLens-8B -- does the training-free crop transfer across
# backbone generations?
#
#   ./run_8b_zoom.sh ceiling     oracle window, sub-2s clips only (the decisive run)
#   ./run_8b_zoom.sh realistic   router-gated, coarse centers from the 8B baseline
#
# Why this matters (review-claude-2026-08-10, weakness 5): the finetuning ladder
# and the crop reversal are TimeLens-7B only, so the interventional evidence is
# single-model. 8B is both the harder case and the more interesting one: it
# improves every aggregate over 7B (mIoU 0.490 -> 0.554, beta 0.60 -> 0.73) while
# keeping the SAME 1.00 s sub-2 s center floor and dropping sub-2 s strict recall
# from 0.041 to 0.007. If the crop recovers the center there too, the
# search-space account holds across a backbone generation instead of describing
# one checkpoint.
#
# setsid + launch-in-a-file for the reasons documented in run_eval_arm.sh: a
# watcher that pattern-matches its own cmdline never clears, and stopping a
# watcher SIGTERMs nohup'd children.
set -euo pipefail

MODE=${1:-ceiling}
# Finer sharding. With NSHARDS=8 and NSHARD_RUN=2 we run two of eight slices, so
# ~148 of the 590 sub-2 s clips across the two GPUs. That answers "does the crop
# move the 1.00 s floor" in a few hours rather than ~15. Remaining slices can be
# filled in later with the same command and a higher NSHARD_RUN; each slice is
# scored independently, so a partial sweep is a smaller sample, not a biased one
# (eval_zoom shards by list position over a fixed annotation order).
NSHARDS=${NSHARDS:-8}
NSHARD_RUN=${NSHARD_RUN:-2}
case "$MODE" in
  ceiling|realistic|router) ;;
  *) echo "usage: $0 <ceiling|realistic|router>   [env: NSHARDS=8 NSHARD_RUN=2]" >&2; exit 2 ;;
esac
[ "$NSHARD_RUN" -le "$NSHARDS" ] || { echo "NSHARD_RUN must be <= NSHARDS" >&2; exit 2; }

cd "$(dirname "$0")"
MODEL=TencentARC/TimeLens-8B
OUTDIR=logs/lora_eval
PREFIX=tl8b_zoom_$MODE
PIDFILE=$OUTDIR/$PREFIX.pids
# The 8B baseline behind the paper's 0.554 mIoU / 1.00 s floor; re-verified with
# score_official_stratified.py on 2026-08-10 (mIoU 0.554, sub-2s R@0.7 0.007).
COARSE=logs/scale8b/charades_8b_merged.jsonl

mkdir -p "$OUTDIR"
: > "$PIDFILE"
extra=()
GT_FILTER=2.0
# eval_zoom.py only knows --mode ceiling|realistic. "router" is a mode of THIS
# launcher: it runs the realistic (self-placed) window and adds the predicted-length
# gate via --max_pred_len. Passing MODE straight through fails argparse.
ZMODE=$MODE
[ "$MODE" = "router" ] && ZMODE=realistic
if [ "$MODE" = "realistic" ] || [ "$MODE" = "router" ]; then
  [ -f "$COARSE" ] || { echo "missing coarse baseline: $COARSE" >&2; exit 1; }
  extra=(--coarse_jsonl "$COARSE")
fi
if [ "$MODE" = "router" ]; then
  # The deployable gate: zoom fires only where the coarse pass PREDICTED a short
  # moment (< tau), with no knowledge of true length. So the GT-length filter must
  # be lifted -- otherwise we would gate on ground truth and measure an oracle.
  # tau=4.0 matches LENGTH_ONLY_TAU in experiments/nsp1-zoom/ratio_router.py, the
  # value behind the paper's 7B router row. On the 8B baseline this selects
  # 1297/3363 clips (38.6%), including 397 of the 590 sub-2s ones (67.3%),
  # closely tracking 7B's 1396/422.
  #
  # Scoring note: this run produces zoom predictions for the ROUTED clips only.
  # The reported number is the FULL-SET composite -- these predictions substituted
  # into the 8B baseline, then the sub-2s stratum read out of it, scored on the
  # ROUNDED convention (see the registry's SCORING CONVENTION note). Scoring the
  # routed subset alone conditions on predicted length and overstates the method.
  extra+=(--max_pred_len 4.0)
  GT_FILTER=1000000
fi

for i in $(seq 0 $((NSHARD_RUN - 1))); do
  gpu=$((i % 2))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_zoom.py \
    --mode "$ZMODE" --model "$MODEL" \
    --window_len 8 --place 0.6 --fps 8 --max_moment_s "$GT_FILTER" \
    "${extra[@]}" \
    --out "$OUTDIR/${PREFIX}_s$i.jsonl" \
    --shard "$i" --num_shards "$NSHARDS" \
    >> "$OUTDIR/${PREFIX}_s$i.log" 2>&1 < /dev/null &
  echo $! >> "$PIDFILE"
done
echo "launched $PREFIX ($MODEL): shards 0..$((NSHARD_RUN - 1)) of $NSHARDS -> pids $(tr '\n' ' ' < "$PIDFILE")"
