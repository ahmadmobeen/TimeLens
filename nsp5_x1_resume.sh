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
declare -A ARM_TOTAL=([x1_cgbench_a3_kelly_placebo]=2683 [x1_cgbench_q8_kelly_placebo]=2683
                      [x1_cgbench_q8_kelly_eps0_placebo]=2683 [x1_cgbench_q8_kelly_eps010_placebo]=2683
                      [x1_cgbench_q8_kelly_eps025_placebo]=2683 [x1_cgbench_q8_kelly_placebo_nr]=2683)
ts() { date '+%F %T'; }
sig() { echo "data/t1/priors/siglip_so400m/$1_curves_maxopt.npz"; }
q8b() { echo "data/t1/priors/qwen3vl_emb_8b/$1_curves_maxopt.npz"; }
fused() { echo "data/t1/priors/fused_siglip_q8b/$1_curves_maxopt.npz"; }

tau_of() { python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))['tau'], 2))" "$1" 2>/dev/null; }

arms() {  # name|bench|venv|prerequisite curves|eval args
  local b pre tag pw ak q8 cc cl bl tc tb sc qs
  # Prior art in our harness (METHODS-LOG 2026-10-28, pre-registered): F2C's query form, BOLT and
  # AKS as released, each published rule on M's scores, Kelly on each published scorer. Kelly
  # temperatures come from the LVBench likelihood fits; an arm whose fit is missing is not listed.
  tc=$(tau_of data/t1/priors/clip_l14/tau_stem.json)
  tb=$(tau_of data/t1/priors/blip_itm_l/tau_stem.json)
  for b in lvbench videomme lvb mlvu cgbench; do
    pre=""; tag=""; [ $b != lvbench ] && { pre="x1_${b}_"; tag="${b}_"; }
    pw=3; [ $b = lvb ] && pw=2.5                              # BOLT's script: 2.5 on LongVideoBench
    ak=""; [ $b = lvb ] && ak=" --aks_t1 0.2 --aks_depth 3"   # AKS's own LongVideoBench setting
    q8=data/t1/priors/qwen3vl_emb_8b/${tag}curves_maxopt.npz
    cc=data/t1/priors/qwen3vl_emb_8b/${tag}curves_concat.npz
    cl=data/t1/priors/clip_l14/${tag}curves_stem.npz
    bl=data/t1/priors/blip_itm_l/${tag}curves_stem.npz
    echo "${pre}f2c_kelly|$b|.venv|$cc|--arm curve --curves $cc --alloc kelly --tau 3.58 --eps 0.5"
    echo "${pre}bolt_its_q8|$b|.venv|$q8|--arm curve --curves $q8 --alloc its --power $pw"
    echo "${pre}aks_ada_q8|$b|.venv|$q8|--arm curve --curves $q8 --alloc aks$ak"
    echo "${pre}bolt_own|$b|.venv|$cl|--arm curve --curves $cl --alloc its --power $pw"
    echo "${pre}aks_own|$b|.venv|$bl|--arm curve --curves $bl --alloc aks$ak"
    [ -n "$tc" ] && echo "${pre}kelly_clip|$b|.venv|$cl|--arm curve --curves $cl --alloc kelly --tau $tc --eps 0.5"
    [ -n "$tb" ] && echo "${pre}kelly_blip|$b|.venv|$bl|--arm curve --curves $bl --alloc kelly --tau $tb --eps 0.5"
    # Calibration and the hedge separated (METHODS-LOG 2026-10-30, pre-registered and amended):
    # M's scores and placement with no uniform share, and at half and twice the fitted
    # temperature; crowded seconds take the nearest unused frames. The _nr arms are the
    # corrected references (seeded by nsp5_refill.py; only refilled questions rerun).
    echo "${pre}q8_kelly_eps0|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest"
    echo "${pre}q8_kelly_tau179|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 1.79 --eps 0.5 --refill nearest"
    echo "${pre}q8_kelly_tau716|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 7.16 --eps 0.5 --refill nearest"
    echo "${pre}q8_kelly_nr|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0.5 --refill nearest"
    echo "${pre}q8opt_topk_nr|$b|.venv|$q8|--arm curve --curves $q8 --alloc topk --refill nearest --topk_short uniform"
    # M2 = M without the hedge (METHODS-LOG 2026-11-02, pre-registered): its confirmation with
    # Qwen3-VL-8B, the temperature check without the hedge, and M with Qwen3-VL under the nearest
    # refill (seeded by nsp5_refill.py; only crowded questions rerun).
    echo "${pre}q8_kelly_eps0_q3|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest --model $Q3"
    echo "${pre}q8_kelly_eps0_tau179|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 1.79 --eps 0 --refill nearest"
    echo "${pre}q8_kelly_eps0_tau716|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 7.16 --eps 0 --refill nearest"
    echo "${pre}q8_kelly_q3_nr|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0.5 --refill nearest --model $Q3"
    # F2C's appended query placed as M2 (METHODS-LOG 2026-11-02, pre-registered: P1')
    echo "${pre}f2c_kelly_eps0|$b|.venv|$cc|--arm curve --curves $cc --alloc kelly --tau 3.58 --eps 0 --refill nearest"
    # Studies D and E (METHODS-LOG 2026-11-05, pre-registered; run by nsp5_runner.py). D: M2's
    # placement with a small hedge. E: each weaker scorer placed as M2 (eps 0) and as M (eps 0.5),
    # both with the nearest refill; SigLIP's LVBench curve predates the priors layout (A2).
    echo "${pre}q8_kelly_eps010|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0.1 --refill nearest"
    echo "${pre}q8_kelly_eps025|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0.25 --refill nearest"
    sc=$(sig "$b"); [ "$b" = lvbench ] && sc=data/t1/curves/zeroshot_siglip_maxopt.npz
    echo "${pre}a2_kelly_eps0|$b|.venv|$sc|--arm curve --curves $sc --alloc kelly --tau 1.59 --eps 0 --refill nearest"
    echo "${pre}a2_kelly_nr|$b|.venv|$sc|--arm curve --curves $sc --alloc kelly --tau 1.59 --eps 0.5 --refill nearest"
    [ -n "$tc" ] && echo "${pre}kelly_clip_eps0|$b|.venv|$cl|--arm curve --curves $cl --alloc kelly --tau $tc --eps 0 --refill nearest"
    [ -n "$tc" ] && echo "${pre}kelly_clip_nr|$b|.venv|$cl|--arm curve --curves $cl --alloc kelly --tau $tc --eps 0.5 --refill nearest"
    [ -n "$tb" ] && echo "${pre}kelly_blip_eps0|$b|.venv|$bl|--arm curve --curves $bl --alloc kelly --tau $tb --eps 0 --refill nearest"
    [ -n "$tb" ] && echo "${pre}kelly_blip_nr|$b|.venv|$bl|--arm curve --curves $bl --alloc kelly --tau $tb --eps 0.5 --refill nearest"
    # Study F (METHODS-LOG 2026-11-05, pre-registered): GMM-EVA's selection (its top-K) with the
    # options in the query, i.e. top-k on the 8B appended-query curve
    echo "${pre}f2c_topk_nr|$b|.venv|$cc|--arm curve --curves $cc --alloc topk --refill nearest --topk_short uniform"
  done
  # Study C (METHODS-LOG 2026-11-05, pre-registered; run by nsp5_runner.py): the published
  # setting, 32 frames at lmms-eval's Qwen2.5-VL defaults (qwen_vl_utils caps its request at 768
  # tokens per frame), on LVBench and Video-MME.
  local pub="--n_frames 32 --min_pixels 200704 --max_pixels 1605632 --total_pixels 0"
  for b in lvbench videomme; do
    pre=""; tag=""; [ $b != lvbench ] && { pre="x1_${b}_"; tag="${b}_"; }
    q8=data/t1/priors/qwen3vl_emb_8b/${tag}curves_maxopt.npz
    qs=data/t1/priors/qwen3vl_emb_8b/${tag}curves_stem.npz
    cl=data/t1/priors/clip_l14/${tag}curves_stem.npz
    echo "${pre}pub32_uniform|$b|.venv||--arm uniform $pub"
    echo "${pre}pub32_ge1_stem_topk|$b|.venv|$qs|--arm curve --curves $qs --alloc topk --refill nearest --topk_short uniform $pub"
    echo "${pre}pub32_bolt_own|$b|.venv|$cl|--arm curve --curves $cl --alloc its --power 3 --refill nearest $pub"
    echo "${pre}pub32_q8_kelly_eps0|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest $pub"
    cc=data/t1/priors/qwen3vl_emb_8b/${tag}curves_concat.npz
    echo "${pre}pub32_f2c_topk|$b|.venv|$cc|--arm curve --curves $cc --alloc topk --refill nearest --topk_short uniform $pub"
  done
  # Study F at GMM-EVA's token budget: 32 frames of 200,704 pixels (256 tokens per frame pair,
  # about 4,032 visual tokens, half of its 32 x 256 reference), on GMM-EVA's three benchmarks
  local gb="--n_frames 32 --min_pixels 200704 --max_pixels 200704 --total_pixels 0"
  for b in lvbench lvb videomme; do
    pre=""; tag=""; [ $b != lvbench ] && { pre="x1_${b}_"; tag="${b}_"; }
    q8=data/t1/priors/qwen3vl_emb_8b/${tag}curves_maxopt.npz
    cc=data/t1/priors/qwen3vl_emb_8b/${tag}curves_concat.npz
    echo "${pre}gb32_uniform|$b|.venv||--arm uniform $gb"
    echo "${pre}gb32_f2c_topk|$b|.venv|$cc|--arm curve --curves $cc --alloc topk --refill nearest --topk_short uniform $gb"
    echo "${pre}gb32_q8_kelly_eps0|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest $gb"
  done
  # Study G (METHODS-LOG 2026-11-05, pre-registered): M2 against GMM-EVA's selection at other frame
  # budgets, under the harness's shared token budget (14,336; qwen_vl_utils caps a frame pair at
  # 768 tokens, so N=16 realizes about 6,144). N=64 is the existing q8_kelly_eps0 / f2c_topk_nr.
  local n
  for n in 16 32 128; do
    for b in lvbench videomme lvb mlvu cgbench; do
      pre=""; tag=""; [ $b != lvbench ] && { pre="x1_${b}_"; tag="${b}_"; }
      q8=data/t1/priors/qwen3vl_emb_8b/${tag}curves_maxopt.npz
      cc=data/t1/priors/qwen3vl_emb_8b/${tag}curves_concat.npz
      echo "${pre}n${n}_q8_kelly_eps0|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest --n_frames $n"
      echo "${pre}n${n}_f2c_topk|$b|.venv|$cc|--arm curve --curves $cc --alloc topk --refill nearest --topk_short uniform --n_frames $n"
    done
  done
  # Study H (METHODS-LOG 2026-11-09, pre-registered): answer-model scale. Uniform and M2 on LVBench
  # with Qwen2.5-VL-32B and -72B answering; each eval needs a GPU to itself (nsp5_plan.py --exclusive).
  local m
  for m in 32 72; do
    echo "h_uniform_q25_${m}b|lvbench|.venv||--arm uniform --model Qwen/Qwen2.5-VL-${m}B-Instruct"
    echo "h_q8_kelly_eps0_q25_${m}b|lvbench|.venv|data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz|--arm curve --curves data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz --alloc kelly --tau 3.58 --eps 0 --refill nearest --model Qwen/Qwen2.5-VL-${m}B-Instruct"
  done
  # Study J (METHODS-LOG 2026-11-10, pre-registered): Study H's 32B answer model on the other four
  # benchmarks, uniform and M2, one GPU per eval (nsp5_plan.py --exclusive). --cpu_threads and
  # --prefetch change no output (checked 2026-11-10 on logged questions).
  for b in videomme lvb mlvu cgbench; do
    q8=data/t1/priors/qwen3vl_emb_8b/${b}_curves_maxopt.npz
    echo "x1_${b}_h_uniform_q25_32b|$b|.venv||--arm uniform --model Qwen/Qwen2.5-VL-32B-Instruct --cpu_threads 4 --prefetch 2"
    echo "x1_${b}_h_q8_kelly_eps0_q25_32b|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau 3.58 --eps 0 --refill nearest --model Qwen/Qwen2.5-VL-32B-Instruct --cpu_threads 4 --prefetch 2"
  done
  # Study I (METHODS-LOG 2026-11-09, pre-registered): a review's controls. BOLT's rule on M2's scores
  # at the exponent fitted by M2's likelihood (8.68, frozen from LVBench), M2 at four more
  # temperatures, and uniform at Study G's other budgets.
  local tt
  for b in lvbench videomme lvb mlvu cgbench; do
    pre=""; tag=""; [ $b != lvbench ] && { pre="x1_${b}_"; tag="${b}_"; }
    q8=data/t1/priors/qwen3vl_emb_8b/${tag}curves_maxopt.npz
    echo "${pre}q8_its_fit|$b|.venv|$q8|--arm curve --curves $q8 --alloc its --power 8.68"
    for tt in 100:1.0 250:2.5 500:5.0 1000:10.0; do
      echo "${pre}q8_kelly_eps0_tau${tt%%:*}|$b|.venv|$q8|--arm curve --curves $q8 --alloc kelly --tau ${tt##*:} --eps 0 --refill nearest"
    done
    for n in 16 32 128; do
      echo "${pre}n${n}_uniform|$b|.venv||--arm uniform --n_frames $n"
    done
  done
  local lv=data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz cg
  cg=$(q8b cgbench)
  echo "q8_kelly_eps0_q35|lvbench|.venv-q35|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0 --refill nearest --model Qwen/Qwen3.5-4B"
  echo "q8_kelly_q35_nr|lvbench|.venv-q35|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0.5 --refill nearest --model Qwen/Qwen3.5-4B"
  echo "q8_kelly_eps0_placebo|lvbench|.venv|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0 --refill nearest --curve_placebo"
  echo "x1_cgbench_q8_kelly_eps0_placebo|cgbench|.venv|$cg|--arm curve --curves $cg --alloc kelly --tau 3.58 --eps 0 --refill nearest --curve_placebo"
  # Study D's placebos: each hedge size on the same-video placebo; M's rerun with the nearest refill
  echo "q8_kelly_eps010_placebo|lvbench|.venv|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0.1 --refill nearest --curve_placebo"
  echo "q8_kelly_eps025_placebo|lvbench|.venv|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0.25 --refill nearest --curve_placebo"
  echo "q8_kelly_placebo_nr|lvbench|.venv|$lv|--arm curve --curves $lv --alloc kelly --tau 3.58 --eps 0.5 --refill nearest --curve_placebo"
  echo "x1_cgbench_q8_kelly_eps010_placebo|cgbench|.venv|$cg|--arm curve --curves $cg --alloc kelly --tau 3.58 --eps 0.1 --refill nearest --curve_placebo"
  echo "x1_cgbench_q8_kelly_eps025_placebo|cgbench|.venv|$cg|--arm curve --curves $cg --alloc kelly --tau 3.58 --eps 0.25 --refill nearest --curve_placebo"
  echo "x1_cgbench_q8_kelly_placebo_nr|cgbench|.venv|$cg|--arm curve --curves $cg --alloc kelly --tau 3.58 --eps 0.5 --refill nearest --curve_placebo"
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
    grep -qF "\"$name\"" "$L/runner/plan.json" 2>/dev/null && continue   # nsp5_runner.py shards it
    # the eval is CPU-bound (~10 cores each, decoding long video): cap the concurrent processes
    [ "$(pyprocs | grep -c 'eval_lvbench_foveate.py')" -ge "${MAX_EVALS:-16}" ] && return
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
  arms)   # name|bench|venv|prerequisite|eval args|population, for nsp5_runner.py's plan
    while IFS='|' read -r name b venv prereq args; do
      echo "$name|$b|$venv|$prereq|$args|${ARM_TOTAL[$name]:-${TOTAL[$b]}}"
    done < <(arms) ;;
  *) echo "usage: $0 status | run [--dry-run] | arms"; exit 2 ;;
esac
