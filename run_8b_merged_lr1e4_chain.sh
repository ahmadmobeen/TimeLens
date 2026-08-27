#!/usr/bin/env bash
# TimeLens-8B replication of the paper's merged supply arm, at the adapter rate 1e-4.
# Generated 2026-08-18 from run_8b_armc_lr1e4_chain.sh, which is the proven driver: only
# the corpus, output root, prefix and log dir differ. Every guard it carries is retained --
# adapter non-zero check, 8B-base check, 3-attempt eval resume, scorer --validate gate.
# WHY: issue #90 Exp 1 replicated ONE supply arm (natural-short) on 8B. The paper has three
# on 7B. These two take the replication to three and answer the codex review's
# generalisation objection (#92 issue 4) with arms rather than with prose.
#
# WHY THIS RUN EXISTS. The first 8B replication (a6_natshort_8b) used arm C's corpus
# (data/charades_full/anet_train_shorttail.json) and arm C's adapter (r=64, alpha=64, 200
# steps) but trained at lr 1e-5 -- the rate the paper uses for its FULL-PARAMETER arms.
# Arm C is an adapter arm and used 1e-4; the paper's own words are that 1e-4 "destabilizes
# full-parameter training", which is why the two rates exist. So that run is a 10x-under-
# trained cousin of arm C, not a replication of it, and "reproduced on a second grounder"
# cannot rest on it. Same corpus, same adapter, same step budget, matched rate.
#
# The 1e-5 run is NOT discarded: it is a usable lr ablation, and its emission still moved
# +27.5 points, so it already shows the effect is not rate-specific.
#
# CHAINED so no GPU time is lost between stages, and VERIFIED at each boundary. A queue that
# finishes its list without finishing its work is the failure mode this repo has hit before.
set -uo pipefail
cd "$(dirname "$0")"

ARM_ANNO=data/charades_full/mix_shorttail_speedaug.json
OUT_ROOT=output/TimeLens-8B/a6_merged_8b_lr1e4
PREFIX=a6_merged_8b_lr1e4
LOG=logs/8b_merged_lr1e4
EXPECT=3363
NSHARD=4

mkdir -p "$LOG" logs/lora_eval

# The venv is not activated in the shells this repo is driven from -- every call site uses an
# explicit .venv/bin/python path. run_sft_timelens7b.sh invokes `deepspeed` by bare name, so
# without this the training step dies instantly with "command not found" while reporting a
# perfectly normal-looking output directory first.
PATH="$PWD/.venv/bin:$PATH"
export PATH
say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

# ---------------------------------------------------------------- step 1: train
if [ -f "$LOG/adapter_path.txt" ] && [ -d "$(cat "$LOG/adapter_path.txt")" ]; then
  ADAPTER=$(cat "$LOG/adapter_path.txt")
  say "adapter already trained, reusing: $ADAPTER"
else
  say "training arm C on TimeLens-8B at lr 1e-4, 4 GPUs, 200 steps"
  bash train_scripts/run_sft_timelens7b.sh \
    --model_path TencentARC/TimeLens-8B \
    --model_id "$PREFIX" \
    --datasets filtered_hybrid \
    --raw_anno_path "$ARM_ANNO" \
    --output_root "$OUT_ROOT" \
    --learning_rate 1e-4 \
    --max_steps 200 \
    --num_devices 4 > "$LOG/train.log" 2>&1
  rc=$?
  # Read the run directory from the trainer's own announcement rather than globbing the
  # output tree: the tag is a timestamp, and a glob would pick up any sibling attempt.
  RUN_DIR=$(grep -m1 '^Output directory: ' "$LOG/train.log" | sed 's/^Output directory: //')
  if [ $rc -ne 0 ] || [ -z "$RUN_DIR" ]; then
    say "ABORT: training failed (rc=$rc). tail of $LOG/train.log:"; tail -25 "$LOG/train.log"; exit 2
  fi
  ADAPTER="$RUN_DIR/lora"
  [ -f "$ADAPTER/adapter_config.json" ] || { say "ABORT: no adapter at $ADAPTER"; exit 2; }
  echo "$ADAPTER" > "$LOG/adapter_path.txt"
  say "trained: $ADAPTER"
fi

# ------------------------------------------- step 2: prove the adapter is not a no-op
# A silently zero lora_B decodes bit-identically to the frozen model and raises nothing;
# that cost this project a month of base-model numbers (ledger #80). eval_lora.py aborts on
# it too, but checking here means we learn it before occupying 4 GPUs for hours.
say "verifying adapter is non-zero and declares the 8B base"
env -u VIRTUAL_ENV .venv/bin/python - "$ADAPTER" <<'PYEOF' | tee "$LOG/adapter_check.txt"
import json, os, sys
from safetensors import safe_open
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "adapter_config.json")))
print("base:", cfg.get("base_model_name_or_path"), "r:", cfg.get("r"), "alpha:", cfg.get("lora_alpha"))
f = os.path.join(d, "adapter_model.safetensors")
nz = tot = 0
with safe_open(f, framework="pt") as h:
    for k in h.keys():
        if "lora_B" in k:
            tot += 1
            nz += float(h.get_tensor(k).abs().sum()) > 0
print(f"lora_B non-zero: {nz}/{tot}")
print("VERDICT:", "OK" if nz else "ZERO-ADAPTER")
PYEOF
grep -q "VERDICT: OK" "$LOG/adapter_check.txt" || { say "ABORT: adapter is a no-op"; exit 2; }
grep -q "base: TencentARC/TimeLens-8B" "$LOG/adapter_check.txt" || { say "ABORT: adapter does not declare the 8B base"; exit 2; }

# ------------------------------------------------- step 3: eval, one shard per GPU
for attempt in 1 2 3; do
  total=0
  for i in $(seq 0 $((NSHARD-1))); do
    f=logs/lora_eval/${PREFIX}_s${i}of${NSHARD}.jsonl
    [ -f "$f" ] && total=$((total + $(wc -l < "$f")))
  done
  if [ "$total" -ge "$EXPECT" ]; then say "eval complete: $total/$EXPECT records"; break; fi

  say "eval attempt $attempt: $total/$EXPECT on disk, launching $NSHARD shards (resume)"
  pids=()
  for i in $(seq 0 $((NSHARD-1))); do
    CUDA_VISIBLE_DEVICES=$i setsid nohup env -u VIRTUAL_ENV .venv/bin/python eval_lora.py \
      --adapter "$ADAPTER" \
      --out logs/lora_eval/${PREFIX}_s${i}of${NSHARD}.jsonl \
      --shard "$i" --num_shards "$NSHARD" \
      >> "$LOG/eval_s${i}.log" 2>&1 < /dev/null &
    pids+=($!)
  done
  say "shards running: ${pids[*]}"
  # Wait on the PIDs, never on `pgrep -f eval_lora.py`: this script's own cmdline matches
  # that pattern, so such a loop waits on itself forever. Third time that trap bit us.
  for p in "${pids[@]}"; do wait "$p"; done
  say "attempt $attempt finished"
done

total=0
for i in $(seq 0 $((NSHARD-1))); do
  f=logs/lora_eval/${PREFIX}_s${i}of${NSHARD}.jsonl
  n=0; [ -f "$f" ] && n=$(wc -l < "$f")
  say "  shard $i: $n records"
  total=$((total + n))
done
if [ "$total" -lt "$EXPECT" ]; then
  say "INCOMPLETE after 3 attempts: $total/$EXPECT. Scoring anyway is NOT safe -- a partial"
  say "population changes every number. Inspect $LOG/eval_s*.log."
  exit 3
fi

# ------------------------------------------------------------------ step 4: score
say "scoring (validating the scorer against published values first)"
# PYTHONPATH=. is REQUIRED: the scorer imports timelens.utils for the official IoU and parse
# primitives, and research/TimeLens is not an installed package. Omitting it cost this run its
# scoring step -- 3h of training and 70min of eval completed, then ModuleNotFoundError.
# set -o pipefail is on, so `| tee` no longer hides the scorer's exit status either. The first
# version of this block printed DONE through that failure, which is the one thing a chain must
# never do: it reported success while producing no numbers.
SCORER="env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python score_short_stratum.py"
if ! $SCORER --validate > "$LOG/score.txt" 2>&1; then
  say "ABORT: the scorer does not reproduce its published landmarks. Numbers from this run are"
  say "not trustworthy until that passes. See $LOG/score.txt"; cat "$LOG/score.txt"; exit 4
fi
cat "$LOG/score.txt"
if ! $SCORER --pair 'logs/scale8b_charades_full/preds_s*of3.jsonl' \
       "logs/lora_eval/${PREFIX}_s*of${NSHARD}.jsonl" >> "$LOG/score.txt" 2>&1; then
  say "ABORT: scoring failed. See $LOG/score.txt"; tail -20 "$LOG/score.txt"; exit 4
fi
tail -12 "$LOG/score.txt"
say "DONE -> $LOG/score.txt"
