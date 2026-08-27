#!/usr/bin/env bash
# Issue #90 Exp 1: replicate the natural-short training arm on TimeLens-8B.
#
#   setsid nohup ./run_8b_arms_queue.sh >> logs/8b_arms_queue.log 2>&1 &
#
# QUEUED, NOT STARTED IMMEDIATELY, for the reason #90 itself gives: the tab:prior cell
# measurements block check-submission-ready.sh and are on the critical path; this is not.
# This script waits until every card is free before doing anything.
#
# STEP 0 IS AN AUDIT, AND IT IS A HARD GATE. The 7B recipe is
#   --lora_enable True --freeze_llm True --freeze_vision_tower True, r=64 alpha=64,
#   lr=1e-5, max_steps=200, datasets=filtered_hybrid,
#   raw_anno_path=data/charades_full/anet_train_shorttail.json
# and the trainer's freeze logic is architecture-sensitive in two ways that 7B never
# exercised:
#
#   1. configure_llm() freezes model.model.parameters(). On Qwen3-VL (8B) model.model IS
#      Qwen3VLModel, which CONTAINS .visual -- so that call also touches the vision tower,
#      which is not true of the 7B tree.
#   2. configure_vision_tower() then sets requires_grad on model.visual.merger only. 8B's
#      vision model additionally has deepstack_merger_list (3 mergers, indexes 8/16/24)
#      that 7B has no counterpart for, so those are left in whatever state step 1 produced.
#
# Which parameters end up trainable therefore depends on call order and cannot be settled
# by reading the source. If the 8B trainable set is not structurally the same as 7B's
# (LoRA adapters on LLM linears, vision frozen), then this is not a replication of the arm
# and the number would be misleading rather than merely noisy. So the audit prints both
# sets and ABORTS on divergence instead of training. A wrong replication is worse than a
# missing one, because it would be quoted.
#
# STEP 2 VERIFIES THE ADAPTER IS NON-ZERO before any eval. This project has already lost
# time to a silent PEFT no-op: target_modules matched by suffix while the restore matched
# full paths, so every lora_B stayed at its zero init and the "finetuned" run was measuring
# the frozen base model. Non-strict loading raised nothing. eval_lora.py carries an
# abort-on-zero guard for exactly this; the guard is the point, not a formality.
set -uo pipefail
cd "$(dirname "$0")"

ARM_ANNO=data/charades_full/anet_train_shorttail.json
OUT_TAG=a6_natshort_8b
LOG=logs/8b_arms_queue

say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }
mkdir -p "$LOG"

say "waiting for all 4 GPUs to be free (tab:prior queue has priority)"
while :; do
  busy=0
  for g in 0 1 2 3; do
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$m" -gt 2000 ] && busy=1
  done
  [ "$busy" = 0 ] && break
  sleep 300
done
say "all cards free; starting step 0 audit"

# ---- step 0: trainable-parameter audit, 7B vs 8B -------------------------------
env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python - > "$LOG/audit.txt" 2>&1 <<'PYEOF'
"""Print the trainable-parameter structure of each model under the 7B arm's freeze flags.

Loads on meta device: this asks a structural question about which parameters the recipe
marks trainable, which needs no weights and no GPU.
"""
import torch
from transformers import AutoConfig, AutoModelForImageTextToText

def summarise(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        m = AutoModelForImageTextToText.from_config(cfg)
    # The 7B arm's flags: freeze_llm=True, freeze_vision_tower=True. Reproduce the
    # trainer's call ORDER exactly, because that is what the concern is about.
    for p in m.model.parameters():          # configure_llm: freeze_llm=True
        p.requires_grad = False
    if hasattr(m, "lm_head"):
        for p in m.lm_head.parameters():
            p.requires_grad = False
    for p in m.visual.parameters():         # configure_vision_tower: freeze_vision_tower=True
        p.requires_grad = False
    groups = {}
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        key = "visual" if "visual" in n else ("lm_head" if "lm_head" in n else "llm")
        groups[key] = groups.get(key, 0) + 1
    has_ds = hasattr(m.visual, "deepstack_merger_list")
    print(f"{model_id}")
    print(f"  trainable groups after the 7B recipe's freezes: {groups or 'NONE (all frozen)'}")
    print(f"  visual has .merger={hasattr(m.visual,'merger')}  "
          f".deepstack_merger_list={has_ds}"
          + (f" (n={len(m.visual.deepstack_merger_list)})" if has_ds else ""))
    print(f"  model.model contains visual: {'visual' in dict(m.model.named_children())}")

for mid in ("TencentARC/TimeLens-7B", "TencentARC/TimeLens-8B"):
    try:
        summarise(mid)
    except Exception as ex:
        print(f"{mid}: AUDIT FAILED {type(ex).__name__}: {ex}")
PYEOF

say "audit written to $LOG/audit.txt:"
sed 's/^/    /' "$LOG/audit.txt"

# The gate: 8B must show the same trainable-group structure as 7B, and its extra
# deepstack mergers must be frozen. Anything else and the arm is not the same experiment.
if ! grep -q "TimeLens-8B" "$LOG/audit.txt"; then
  say "ABORT: audit did not reach 8B. Not training -- a wrong replication would be quoted."
  exit 2
fi
if grep -q "AUDIT FAILED" "$LOG/audit.txt"; then
  say "ABORT: audit raised. Fix the trainer for the Qwen3-VL tree before spending GPU hours."
  exit 2
fi

say "audit produced output; STOPPING HERE BY DESIGN."
say "The audit is a human-review gate, not an automatic pass. Compare the two trainable"
say "structures in $LOG/audit.txt. If they match (all frozen except the LoRA adapters added"
say "later, deepstack mergers frozen), run the arm with:"
cat <<EOS

  bash train_scripts/run_sft_timelens7b.sh \\
    --model_path TencentARC/TimeLens-8B \\
    --model_id $OUT_TAG \\
    --datasets filtered_hybrid \\
    --raw_anno_path $ARM_ANNO \\
    --learning_rate 1e-5 \\
    --max_steps 200 \\
    --num_devices 4

then, BEFORE scoring anything, verify the adapter is not all-zero:

  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_lora.py \\
    --adapter output/TimeLens-8B/$OUT_TAG/<run-dir> \\
    --out logs/lora_eval/${OUT_TAG}.jsonl
  # eval_lora.py aborts if every lora_B is zero. That guard exists because a previous
  # arm silently measured the frozen base model for days.

EOS
say "queued-and-gated: no training started."
