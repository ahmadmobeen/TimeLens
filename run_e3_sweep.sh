#!/usr/bin/env bash
# NS-P5 E3: hour-long oracle-ceiling sweep on LVBench. One GPU, arms in sequence.
#
#   run_e3_sweep.sh <gpu> <arm> [<arm> ...]
#
# Pre-registered design and the pass/kill ladder live in
# docs/research/projects/NS-P5-adaptive-sampling/METHODS-LOG.md, entry 2026-10-20.
#
# Two conventions this repo learned the hard way:
#  * absolute cd, because this runs from a frozen copy outside the repo -- editing a
#    bash script while it runs corrupts its byte offset and it then executes fragments
#    of the new file;
#  * not `set -e`: one arm that dies must not take the rest of the sweep with it.
#
# The harness resumes: each arm skips questions already present in its jsonl, so a
# killed sweep continues where it stopped instead of starting over.

set -uo pipefail

cd /home/mobeen/codes/active_repos/NeedleScan/research/TimeLens

GPU="${1:?usage: run_e3_sweep.sh <gpu> <arm> [<arm> ...]}"
shift
ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || { echo "no arms given"; exit 1; }

mkdir -p logs/nsp5
CHAIN="logs/nsp5/e3-chain-gpu${GPU}.log"

echo "chain gpu${GPU} start $(date -Is): ${ARMS[*]}" >> "$CHAIN"

for arm in "${ARMS[@]}"; do
  echo "=== ${arm} started $(date -Is)" >> "$CHAIN"

  # The placebo must be displaced far enough to leave the evidence behind. The
  # harness reports the displacement it actually achieved, because a placebo clamped
  # back onto the evidence would read as a successful null.
  extra=()
  [[ "$arm" == offpeak_* ]] && extra=(--offpeak_shift 600)

  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. OMP_NUM_THREADS=8 \
    .venv/bin/python -u eval_lvbench_foveate.py \
      --arm "$arm" "${extra[@]}" \
      --n_frames 64 \
      --out "logs/nsp5/e3_${arm}.jsonl" \
      > "logs/nsp5/e3_${arm}.log" 2>&1
  rc=$?

  echo "=== ${arm} finished $(date -Is) rc=${rc}" >> "$CHAIN"
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED -- continuing; see logs/nsp5/e3_${arm}.log" >> "$CHAIN"
  fi
done

echo "CHAIN GPU${GPU} COMPLETE $(date -Is)" >> "$CHAIN"
