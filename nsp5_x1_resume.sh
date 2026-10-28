#!/usr/bin/env bash
# nsp5_x1_resume.sh -- bring NS-P5's X1/G2 runs back after a restart, and keep them fed.
#
#   bash nsp5_x1_resume.sh status          # one line per arm: rows/target and state
#   bash nsp5_x1_resume.sh run --dry-run   # one scheduling pass, printing what it would do
#   setsid nohup bash nsp5_x1_resume.sh run >> logs/nsp5/x1_scheduler.log 2>&1 < /dev/null &
#
# Safe to rerun at any time. Eval arms resume from their jsonl (eval_lvbench_foveate.py skips
# questions already written) and an arm that is already running is left alone. prior_encode.py
# does NOT resume (it saves at the end), so a missing 8B prior is encoded from scratch; with four
# or more GPUs each 8B encoder gets a GPU of its own. Once a benchmark's 8B curves exist they are
# fused (nsp5_fuse_curves.py, frozen temperatures) and that benchmark's method arms become ready.
# Arms and constants are the frozen protocol in METHODS-LOG (X1, the X1 amendment, G2); the order
# below is launch priority, decisive comparisons first.
set -u
cd "$(dirname "$(readlink -f "$0")")"
L=logs/nsp5
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
Q3=Qwen/Qwen3-VL-8B-Instruct
declare -A TOTAL=([videomme]=2700 [lvb]=1337 [mlvu]=2174 [cgbench]=3000 [lvbench]=1534)
# the same-video placebo skips CG-Bench's 317 single-question videos (no partner question)
declare -A ARM_TOTAL=([x1_cgbench_a3_kelly_placebo]=2683 [x1_cgbench_q8_kelly_placebo]=2683)
ts() { date '+%F %T'; }
sig() { echo "data/t1/priors/siglip_so400m/$1_curves_maxopt.npz"; }
q8b() { echo "data/t1/priors/qwen3vl_emb_8b/$1_curves_maxopt.npz"; }
fused() { echo "data/t1/priors/fused_siglip_q8b/$1_curves_maxopt.npz"; }

arms() {  # name|bench|venv|prerequisite curves|eval args
  local b
  # confirmation runs for the simplified method M (8B with options, Kelly tau 3.58, eps 0.5)
  echo "q8_kelly_q3|lvbench|.venv|data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz|--arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz --alloc kelly --tau 3.58 --eps 0.5 --model $Q3"
  echo "q8_kelly_q35|lvbench|.venv-q35|data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz|--arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz --alloc kelly --tau 3.58 --eps 0.5 --model Qwen/Qwen3.5-4B"
  echo "q8_kelly_placebo|lvbench|.venv|data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz|--arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz --alloc kelly --tau 3.58 --eps 0.5 --curve_placebo"
  echo "x1_cgbench_q8_kelly_placebo|cgbench|.venv|$(q8b cgbench)|--arm curve --curves $(q8b cgbench) --alloc kelly --tau 3.58 --eps 0.5 --curve_placebo"
  for b in videomme lvb mlvu cgbench; do
    echo "x1_${b}_q8_kelly_q3|$b|.venv|$(q8b $b)|--arm curve --curves $(q8b $b) --alloc kelly --tau 3.58 --eps 0.5 --model $Q3"
  done
  # X1 amendment 2: the 8B scorer with options under Kelly (its own CV temperature 3.58)
  echo "q8_kelly|lvbench|.venv|data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz|--arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz --alloc kelly --tau 3.58 --eps 0.5"
  for b in videomme lvb mlvu cgbench; do
    echo "x1_${b}_q8_kelly|$b|.venv|$(q8b $b)|--arm curve --curves $(q8b $b) --alloc kelly --tau 3.58 --eps 0.5"
  done
  for b in videomme lvb mlvu cgbench; do
    echo "x1_${b}_a3_kelly|$b|.venv|$(fused $b)|--arm curve --curves $(fused $b) --alloc kelly --tau 1.76 --eps 0.5"
    echo "x1_${b}_b1|$b|.venv||--arm topk_siglip"
  done
  for b in videomme lvb mlvu cgbench; do
    echo "x1_${b}_q8opt_topk|$b|.venv|$(q8b $b)|--arm curve --curves $(q8b $b) --alloc topk"
    echo "x1_${b}_a3_kelly_q3|$b|.venv|$(fused $b)|--arm curve --curves $(fused $b) --alloc kelly --tau 1.76 --eps 0.5 --model $Q3"
    echo "x1_${b}_uniform|$b|.venv||--arm uniform"
    echo "x1_${b}_uniform_q3|$b|.venv||--arm uniform --model $Q3"
    echo "x1_${b}_a2_kelly|$b|.venv|$(sig $b)|--arm curve --curves $(sig $b) --alloc kelly --tau 1.59 --eps 0.5"
  done
  echo "x1_cgbench_a3_kelly_placebo|cgbench|.venv|$(fused cgbench)|--arm curve --curves $(fused cgbench) --alloc kelly --tau 1.76 --eps 0.5 --curve_placebo"
  echo "g2_q35_uniform|lvbench|.venv-q35||--arm uniform --model Qwen/Qwen3.5-4B"
  echo "g2_q35_b1|lvbench|.venv-q35||--arm topk_siglip --model Qwen/Qwen3.5-4B"
  echo "g2_q35_a3_kelly|lvbench|.venv-q35|data/t1/curves/fused_siglip_q8b_maxopt.npz|--arm curve --curves data/t1/curves/fused_siglip_q8b_maxopt.npz --alloc kelly --tau 1.76 --eps 0.5 --model Qwen/Qwen3.5-4B"
}

rows() { if [ -f "$L/$1.jsonl" ]; then wc -l < "$L/$1.jsonl"; else echo 0; fi; }
pyprocs() { ps -eo pid=,comm=,args= | awk '$2 ~ /^python/'; }
running() { pyprocs | grep -qF -- "--out $L/$1.jsonl"; }
gpu_of() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'; }
prior_gpus() { pyprocs | grep -F "prior_encode.py" | awk '{print $1}' | while read -r p; do gpu_of "$p"; done | sort -u; }
NGPU=$(nvidia-smi -L | wc -l)
free_gpus() {  # "free_MiB index" for GPUs not in the skip list, most free first
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' |
    while IFS=, read -r i f; do case " $1 " in *" $i "*) ;; *) echo "$f $i";; esac; done | sort -n -r
}
declare -A RECENT   # GPU -> launch times this process issued; a model takes minutes to load
reserved() {        # MiB still to be claimed on GPU $1: 30 GB per launch in the last 10 minutes
  local now e c=0; now=$(date +%s)
  for e in ${RECENT[$1]:-}; do [ $((now - e)) -lt 600 ] && c=$((c + 1)); done
  echo $((c * 30000))
}
eval_gpu() {  # most free memory net of reservations, at least 40 GB; 8B-encoder GPUs stay free when 4+
  local skip=""; [ "$NGPU" -ge 4 ] && skip=$(prior_gpus | tr '\n' ' ')
  free_gpus "$skip" | while read -r f i; do echo "$((f - $(reserved "$i"))) $i"; done |
    sort -n -r | awk '$1 >= 40000 {print $2; exit}'
}

covered() {  # bench curves-file: every question has a curve (result cached beside the file)
  [ -e "$2.covered" ] && return 0
  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python - "$1" "$2" > /dev/null 2>&1 <<'PY' && touch "$2.covered"
import sys, numpy as np, bench
keys = set(np.load(sys.argv[2]).files)
assert all(it.key in keys for it in bench.load(sys.argv[1]))
PY
}

encode_missing_priors() {
  local shm workers b busy g
  shm=$(df -BG --output=size /dev/shm | tail -1 | tr -dc 0-9)
  workers=12; [ "$shm" -ge 48 ] && workers=24
  for b in cgbench mlvu videomme lvb; do
    [ -e "data/t1/priors/qwen3vl_emb_8b/${b}_img.npz" ] && continue
    pyprocs | grep -qF -- "--name qwen3vl_emb_8b --bench $b" && continue
    busy=$(prior_gpus | tr '\n' ' ')
    g=$(free_gpus "$busy" | awk '{print $2; exit}')
    [ -z "$g" ] && { echo "$(ts) no GPU free of 8B encoders for $b"; continue; }
    echo "$(ts) encode 8B prior $b on GPU $g (workers $workers, /dev/shm ${shm}G)"
    [ "$DRY" = 1 ] && continue
    CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python -u prior_encode.py --model Qwen/Qwen3-VL-Embedding-8B \
      --name qwen3vl_emb_8b --bench "$b" --workers "$workers" >> "$L/x1_prior_q8b_${b}_r3.log" 2>&1 < /dev/null &
    sleep 60
  done
}

fuse_ready() {
  local b
  for b in videomme lvb mlvu cgbench; do
    [ -e "$(fused $b)" ] && continue
    [ -e "$(q8b $b)" ] || continue
    pyprocs | grep -qF -- "--name qwen3vl_emb_8b --bench $b" && continue   # still being written
    echo "$(ts) fuse curves for $b"
    [ "$DRY" = 1 ] && continue
    env -u VIRTUAL_ENV .venv/bin/python nsp5_fuse_curves.py --siglip "$(sig $b)" --q8b "$(q8b $b)" --out "$(fused $b)"
  done
}

heal() {  # drop lines torn by an abrupt stop (resume parses every line), keeping the original
  [ -f "$L/$1.jsonl" ] || return 0
  python3 - "$L/$1.jsonl" "$DRY" <<'PY'
import json, shutil, sys
path, dry = sys.argv[1], sys.argv[2] == "1"
body = [l for l in open(path, encoding="utf-8").read().split("\n") if l.strip()]
good = []
for line in body:
    try:
        json.loads(line)
        good.append(line)
    except json.JSONDecodeError:
        pass
if len(good) < len(body):
    print(f"torn lines in {path}: {len(body) - len(good)}" + ("" if dry else "; original kept as .torn"))
    if not dry:
        shutil.copy(path, path + ".torn")
        open(path, "w", encoding="utf-8").write("".join(l + "\n" for l in good))
PY
}

launch_arms() {
  local name b venv prereq args n t g
  while IFS='|' read -r name b venv prereq args; do
    n=$(rows "$name"); t=${ARM_TOTAL[$name]:-${TOTAL[$b]}}
    [ "$n" -ge "$t" ] && continue
    running "$name" && continue
    heal "$name"
    if [ -n "$prereq" ]; then
      [ -e "$prereq" ] || continue
      covered "$b" "$prereq" || { echo "$(ts) COVERAGE INCOMPLETE in $prereq: $name not launched"; continue; }
    fi
    g=$(eval_gpu)
    [ -z "$g" ] && { echo "$(ts) no GPU with 40 GB free; $name and later arms wait"; return; }
    echo "$(ts) launch $name ($n/$t) on GPU $g"
    [ "$DRY" = 1 ] && continue
    CUDA_VISIBLE_DEVICES=$g setsid nohup env -u VIRTUAL_ENV PYTHONPATH=. "$venv/bin/python" -u eval_lvbench_foveate.py \
      --bench "$b" $args --out "$L/$name.jsonl" >> "$L/$name.log" 2>&1 < /dev/null &
    RECENT[$g]="${RECENT[$g]:-} $(date +%s)"
    sleep 90
  done < <(arms)
}

case "${1:-}" in
  status)
    while IFS='|' read -r name b venv prereq args; do
      n=$(rows "$name"); t=${ARM_TOTAL[$name]:-${TOTAL[$b]}}
      if [ "$n" -ge "$t" ]; then s=done; elif running "$name"; then s=running
      elif [ -z "$prereq" ] || [ -e "$prereq" ]; then s=ready; else s="blocked (needs $prereq)"; fi
      printf "%-30s %5s/%-5s %s\n" "$name" "$n" "$t" "$s"
    done < <(arms) ;;
  run)
    while :; do
      encode_missing_priors; fuse_ready; launch_arms
      [ "$DRY" = 1 ] && break
      left=0
      while IFS='|' read -r name b venv prereq args; do [ "$(rows "$name")" -lt "${ARM_TOTAL[$name]:-${TOTAL[$b]}}" ] && left=$((left + 1)); done < <(arms)
      [ "$left" -eq 0 ] && { echo "$(ts) every arm complete"; break; }
      sleep 300
    done ;;
  *) echo "usage: $0 status | run [--dry-run]"; exit 2 ;;
esac
