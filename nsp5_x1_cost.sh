#!/usr/bin/env bash
# Study X1 (NeedleScan METHODS-LOG 2026-11-12, wave 2, registered in 9e08ae6): cost measurements on
# one GPU of the first node that the runner leaves alone (logs/nsp5/runner/skip_gpus).
#
# Waits until that GPU is empty, then runs the five selectors of the paper's selector table with
# --timing on the same 50 LVBench questions (--sample 50 --seed 0; --prefetch 0, so reading the
# frames is timed inside each question), then nsp5_x1_cost.py index and score, then hands the
# GPU back to the runner. Every step has a time limit, and a failed step does not stop the rest.
#
#   setsid nohup bash nsp5_x1_cost.sh > /dev/null 2>&1 &      (progress in logs/nsp5/x1_cost.log)
set -u
cd "$(dirname "$0")"
GPU=${GPU:-3}
SKIP=logs/nsp5/runner/skip_gpus
LOG=logs/nsp5/x1_cost.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

until [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  sleep 60
done
say "GPU $GPU free; X1 starts"

timed_arm() {   # name, then the arm's own arguments
  local name=$1; shift
  say "arm $name"
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 .venv/bin/python -u eval_lvbench_foveate.py \
    --bench lvbench "$@" --timing --prefetch 0 --cpu_threads 8 --sample 50 --seed 0 \
    --out "logs/nsp5/x1cost_$name.jsonl" >> "$LOG" 2>&1 || say "arm $name exited with $?"
}
timed_arm uniform --arm uniform
timed_arm q8_kelly_eps0 --arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz \
  --alloc kelly --tau 3.58 --eps 0 --refill nearest
timed_arm f2c_topk_nr --arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_concat.npz \
  --alloc topk --refill nearest --topk_short uniform
timed_arm bolt_own --arm curve --curves data/t1/priors/clip_l14/curves_stem.npz --alloc its --power 3
timed_arm aks_own --arm curve --curves data/t1/priors/blip_itm_l/curves_stem.npz --alloc aks

for mode in index score; do
  say "nsp5_x1_cost.py $mode"
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 .venv/bin/python -u nsp5_x1_cost.py "$mode" \
    >> "$LOG" 2>&1 || say "nsp5_x1_cost.py $mode exited with $?"
done

if [ "$(tr -d ' \n' < "$SKIP")" = "$GPU" ]; then : > "$SKIP"; fi
say "X1 done; GPU $GPU back to the runner"
