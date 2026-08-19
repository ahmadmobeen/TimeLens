#!/usr/bin/env bash
# ISSUE #90 EXPERIMENT 2: arm C on VideoChat-R1_7B, the second training PARADIGM.
#
# WHAT THIS ARM ASKS. Every supply-side arm so far has been run on a TimeLens grounder.
# The open question is whether "more short moments in the corpus does not move the CENTRE"
# is a fact about TimeLens or a fact about grounding finetunes. This runs arm C's exact
# recipe -- same corpus, same lr, same steps, same LoRA rank -- on a DIFFERENT grounder.
#
# NOTHING WAS BUILT FOR THIS. I previously reported it unrunnable because VideoChat-R1's
# own repo has no SFT entry point for temporal grounding (src/open_r1 is grpo*.py only).
# That was the wrong frame. VideoChat-R1_7B is Qwen2_5_VLForConditionalGeneration, 28
# layers, visual.merger present, and the 7B recipe's freezes leave exactly the same
# trainable set as on TimeLens-7B -- verified on a meta device, the same audit that
# cleared the 8B. TimeLens's own SFT trainer takes it as a --model_path. This mirrors what
# the project already does in the other direction: reward1_* is TimeLens-7B trained
# THROUGH the VideoChat-R1 GRPO harness. Harness and model were always separable.
#
# KNOWN CONFOUND, state it in any writeup: VideoChat-R1 is RL-tuned and emits its own
# answer format, so finetuning it on this corpus teaches format AND supply together. The
# EMISSION number is therefore not a like-for-like against 7B/8B. The CENTRE is, and the
# centre is what this arm exists to test.
#
# ---------------------------------------------------------------------------------------
# HISTORY OF THIS FILE, so nobody re-derives it from a stale log. Three launches failed
# before this version; their logs are still on disk and MUST NOT be read as results:
#
#   logs/vcr1_armc_chain.log           launch 1: died instantly. training/model_loader.py
#                                      rejected the model by NAME. _SUPPORTED_MARKERS is a
#                                      substring allowlist and every accepted branch hands
#                                      back the same Auto* classes, so it gated a name, not
#                                      an architecture. Widened (2026-08-19).
#   logs/vcr1_armc_chain_relaunch.log  launch 2: died ~4 min in, inside HF's own video
#   logs/vcr1_armc_lr1e4/train.crash-32grid.log
#                                      processor. training/data/grounding.py::_is_timelens_7b
#                                      is the same species of name check; VideoChat-R1 spells
#                                      neither "timelens-7b" nor "qwen2.5", so it fell to the
#                                      Qwen3-VL branch, which resizes on a 32 px grid
#                                      (image_patch_size=16) then passes do_resize=False --
#                                      a patch-14 processor patchifying a 32-grid tensor:
#                                        RuntimeError: shape '[1,28,2,3,8,2,14,10,2,14]' is
#                                        invalid for input of size 10838016
#                                      Fixed by _is_plain_qwen25_arch (2026-08-19).
#   logs/vcr1_armc_chain_relaunch2.log launch 3: training was CORRECT and running, killed on
#                                      purpose at ~20 min before it could reach scoring,
#                                      because step 4 below was wrong (see next block). No
#                                      adapter, no predictions and no score.txt were written.
#
# THE BUG THAT KILLED LAUNCH 3, because it is the dangerous kind. This file was derived from
# run_8b_armc_lr1e4_chain.sh, and the scoring step still carried the 8B's baseline path:
#     --pair 'logs/scale8b_charades_full/preds_s*of3.jsonl'  <arm preds>
# --pair is (BASE, ARM). That would have compared VideoChat-R1+adapter against TimeLens-8B
# ZERO-SHOT. Same Charades population, so the join succeeds and prints entirely plausible
# deltas that actually mix "8B -> VideoChat-R1" with "the finetune". It would not have
# crashed; it would have produced a number. A run that reports a wrong number is worse than
# one that fails. Step 0 below now builds the arm's OWN baseline and step 4 pairs against it.
#
# Three stale strings from the same copy-paste are also fixed: the training banner and two
# abort messages said "TimeLens-8B" / "the 8B base" while training VideoChat-R1. The base
# ASSERTION was always correct; only its message was wrong. Cosmetic, but a log that names
# the wrong model is exactly how a result gets misattributed six months later.
# ---------------------------------------------------------------------------------------
#
# CHAINED so no GPU time is lost between stages, and VERIFIED at each boundary. A queue that
# finishes its list without finishing its work is the failure mode this repo has hit before.
set -uo pipefail
cd "$(dirname "$0")"

MODEL=OpenGVLab/VideoChat-R1_7B
ARM_ANNO=data/charades_full/anet_train_shorttail.json
EVAL_JSON=data/TimeLens-Bench/charades-timelens.json
EVAL_VIDEOS=data/TimeLens-Bench/videos/charades
OUT_ROOT=output/VideoChat-R1/a6_armc_vcr1_lr1e4
PREFIX=a6_armc_vcr1_lr1e4
LOG=logs/vcr1_armc_lr1e4
BASE_DIR=logs/vcr1_base_charades_full     # the arm's OWN zero-shot baseline
EXPECT=3363
NSHARD=4

mkdir -p "$LOG" "$BASE_DIR" logs/lora_eval

# The venv is not activated in the shells this repo is driven from -- every call site uses an
# explicit .venv/bin/python path. run_sft_timelens7b.sh invokes `deepspeed` by bare name, so
# without this the training step dies instantly with "command not found" while reporting a
# perfectly normal-looking output directory first.
PATH="$PWD/.venv/bin:$PATH"
export PATH
say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

say "MODEL=$MODEL   arm=$PREFIX   baseline=$BASE_DIR"

# ------------------------------------------- step 0: the arm's OWN zero-shot baseline
# Produced with eval_bench.py, which is how logs/scale8b_charades_full (the 8B baseline) was
# produced by run_full_chain.sh. Baseline-from-eval_bench paired with arm-from-eval_lora is
# the convention every published arm in this paper already uses, so this pairing is matched
# to precedent rather than invented here.
#
# A partial VideoChat-R1 baseline already exists at
# logs/convert_vcr1/charades_vcr1_tlformat.jsonl, but it holds 609 of 3363 records (its
# run_baseline.log stops at [600/3363]) and it is a single 1-shard file. It is NOT used: a
# partial population changes every number.
#
# --patch_embed matmul is kept for flag-parity with run_full_chain.sh. eval_bench.py
# documents it as "No effect on 7B" -- it patches the Qwen3-VL vision stem, so on this
# Qwen2.5-VL model it is a no-op, not a silent numerical change.
base_count() {
  local t=0 i f
  for i in $(seq 0 $((NSHARD-1))); do
    f="$BASE_DIR/preds_s${i}of${NSHARD}.jsonl"
    [ -f "$f" ] && t=$((t + $(wc -l < "$f")))
  done
  echo "$t"
}

for attempt in 1 2 3; do
  n=$(base_count)
  if [ "$n" -ge "$EXPECT" ]; then say "baseline complete: $n/$EXPECT records"; break; fi
  say "baseline attempt $attempt: $n/$EXPECT on disk, launching $NSHARD shards (resume)"
  pids=()
  for i in $(seq 0 $((NSHARD-1))); do
    CUDA_VISIBLE_DEVICES=$i setsid nohup env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. \
      .venv/bin/python eval_bench.py --model "$MODEL" \
      --json "$EVAL_JSON" --video_dir "$EVAL_VIDEOS" --patch_embed matmul \
      --out "$BASE_DIR/preds_s${i}of${NSHARD}.jsonl" --shard "$i" --num_shards "$NSHARD" \
      >> "$BASE_DIR/run_s${i}of${NSHARD}.log" 2>&1 < /dev/null &
    pids+=($!)
  done
  say "baseline shards running: ${pids[*]}"
  # Wait on explicit PIDs, never `pgrep -f eval_bench.py`: this script's own command line
  # contains that string, so such a loop waits on itself forever. Third occurrence of that
  # trap in this project; it is documented in run_full_chain.sh and run_eval_arm.sh too.
  for p in "${pids[@]}"; do wait "$p"; done
  say "baseline attempt $attempt finished"
done

n=$(base_count)
if [ "$n" -lt "$EXPECT" ]; then
  say "ABORT: baseline INCOMPLETE after 3 attempts: $n/$EXPECT. Scoring against a partial"
  say "baseline changes every number. Inspect $BASE_DIR/run_s*.log."
  exit 5
fi

# ---------------------------------------------------------------- step 1: train
if [ -f "$LOG/adapter_path.txt" ] && [ -d "$(cat "$LOG/adapter_path.txt")" ]; then
  ADAPTER=$(cat "$LOG/adapter_path.txt")
  say "adapter already trained, reusing: $ADAPTER"
else
  say "training arm C on $MODEL at lr 1e-4, 4 GPUs, 200 steps"
  bash train_scripts/run_sft_timelens7b.sh \
    --model_path "$MODEL" \
    --model_id "$PREFIX" \
    --datasets filtered_hybrid \
    --raw_anno_path "$ARM_ANNO" \
    --output_root "$OUT_ROOT" \
    --learning_rate 1e-4 \
    --max_steps 200 \
    --num_devices 4 > "$LOG/train.log" 2>&1
  rc=$?
  # Read the run directory from the trainer's own announcement rather than globbing the
  # output tree: the tag is a timestamp, and a glob would pick up any sibling attempt --
  # including the dlever-20260819-0126_* directory left behind by the crashed launch 2.
  RUN_DIR=$(grep -m1 '^Output directory: ' "$LOG/train.log" | sed 's/^Output directory: //')
  if [ $rc -ne 0 ] || [ -z "$RUN_DIR" ]; then
    say "ABORT: training failed (rc=$rc). See $LOG/train.log"; exit 1
  fi
  ADAPTER="$RUN_DIR/lora"
  [ -d "$ADAPTER" ] || { say "ABORT: no adapter at $ADAPTER"; exit 1; }
  echo "$ADAPTER" > "$LOG/adapter_path.txt"
  say "trained: $ADAPTER"
fi

# ------------------------------------------------- step 2: the adapter is real and ours
say "verifying adapter is non-zero and declares the $MODEL base"
env -u VIRTUAL_ENV .venv/bin/python - "$ADAPTER" > "$LOG/adapter_check.txt" 2>&1 <<'PYEOF'
import json, os, sys
from safetensors import safe_open
ad = sys.argv[1]
cfg = json.load(open(os.path.join(ad, "adapter_config.json")))
print("base:", cfg.get("base_model_name_or_path"), "r:", cfg.get("r"), "alpha:", cfg.get("lora_alpha"))
nz = tot = 0
for fn in os.listdir(ad):
    if fn.endswith(".safetensors"):
        with safe_open(os.path.join(ad, fn), framework="pt") as h:
            for k in h.keys():
                if "lora_B" in k:
                    tot += 1
                    nz += float(h.get_tensor(k).abs().sum()) > 0
print(f"lora_B non-zero: {nz}/{tot}")
print("VERDICT:", "OK" if nz else "ZERO-ADAPTER")
PYEOF
cat "$LOG/adapter_check.txt"
grep -q "VERDICT: OK" "$LOG/adapter_check.txt" || { say "ABORT: adapter is a no-op"; exit 2; }
grep -q "base: $MODEL" "$LOG/adapter_check.txt" || { say "ABORT: adapter does not declare the $MODEL base"; exit 2; }

# ------------------------------------------------- step 3: eval the arm, one shard per GPU
# eval_lora.py reads base_model_name_or_path out of the adapter and loads THAT, so the arm
# is evaluated on VideoChat-R1 even though C.MODEL_PATH defaults to TimeLens-7B.
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
# primitives, and research/TimeLens is not an installed package. Omitting it cost an earlier
# run its scoring step -- 3h of training and 70min of eval completed, then ModuleNotFoundError.
# set -o pipefail is on, so `| tee` no longer hides the scorer's exit status either. The first
# version of this block printed DONE through that failure, which is the one thing a chain must
# never do: it reported success while producing no numbers.
SCORER="env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python score_short_stratum.py"
if ! $SCORER --validate > "$LOG/score.txt" 2>&1; then
  say "ABORT: the scorer does not reproduce its published landmarks. Numbers from this run are"
  say "not trustworthy until that passes. See $LOG/score.txt"; cat "$LOG/score.txt"; exit 4
fi
cat "$LOG/score.txt"

# BASE is this arm's OWN zero-shot baseline, NOT logs/scale8b_charades_full. See the header.
if ! $SCORER --pair "$BASE_DIR/preds_s*of${NSHARD}.jsonl" \
       "logs/lora_eval/${PREFIX}_s*of${NSHARD}.jsonl" >> "$LOG/score.txt" 2>&1; then
  say "ABORT: scoring failed. See $LOG/score.txt"; tail -20 "$LOG/score.txt"; exit 4
fi
tail -12 "$LOG/score.txt"
say "DONE -> $LOG/score.txt"
say "REMINDER for the writeup: emission is NOT comparable to 7B/8B (format+supply are"
say "confounded on an RL-tuned base). The CENTRE is the comparable quantity."
