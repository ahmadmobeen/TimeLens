#!/usr/bin/env bash
# Queue the two remaining 8B supply arms, sequentially. Each needs all four GPUs.
#
#   setsid nohup ./chain_8b_two_arms.sh >> logs/8b_two_arms_chain.log 2>&1 &
#
# WHY. Issue #90 Exp 1 replicated ONE supply arm (natural-short) on TimeLens-8B; the
# paper has three on 7B. These two take it to three, which answers the codex review's
# generalisation objection (issue #92, its issue 4) with arms instead of prose.
#
# WHY NOT Exp 2. #90's Exp 2 is the same arm on VideoChat-R1 and is not runnable:
# src/open_r1 has no SFT entry point at all (grpo*.py only), training_scripts has SFT for
# cls_qa/gqa/track but not tg, and every ckpt/reward1_* is TimeLens-7B trained THROUGH the
# VideoChat-R1 harness, not a VideoChat-R1 model -- #85's registry says so in as many
# words. Exp 2 needs a new SFT-TG trainer for a model never trained here. That is not a
# queued run, and it is a poor thing to attempt unattended.
#
# Each arm runs the proven driver, which self-gates: it aborts on a zero adapter, on an
# adapter that does not declare the 8B base, on an incomplete eval after three attempts,
# and on a scorer that fails --validate. A failure in arm one therefore does not
# contaminate arm two, so the second still runs.
set -uo pipefail
cd "$(dirname "$0")"
say() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }

for arm in speedaug merged; do
  d=./run_8b_${arm}_lr1e4_chain.sh
  [ -x "$d" ] || { say "SKIP $arm: $d missing or not executable"; continue; }
  say "=== arm $arm: starting ==="
  if bash "$d"; then
    say "=== arm $arm: COMPLETE ==="
    tail -6 logs/8b_${arm}_lr1e4/score.txt 2>/dev/null
  else
    rc=$?
    say "=== arm $arm: FAILED rc=$rc; continuing to the next arm ==="
    tail -12 logs/8b_${arm}_lr1e4/train.log 2>/dev/null
  fi
done
say "both arms attempted; scores in logs/8b_{speedaug,merged}_lr1e4/score.txt"
