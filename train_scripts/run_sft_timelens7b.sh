#!/usr/bin/env bash
#
# NS-P1 D-lever SFT recipe for the reproduced TimeLens-7B (Qwen2.5-VL), adapted from
# run_sft_qwen3_8b.sh. Key differences vs the 8B recipe:
#   * model_path defaults to TencentARC/TimeLens-7B (Qwen2.5-VL); model_loader allowlist widened.
#   * LoRA finetune by default (--lora_enable True + --freeze_llm True, required by the trainer).
#   * disable_flash_attn2=True -> sdpa (flash-attn2 is ABI-incompatible on B200/sm_100).
#   * use_liger=False (liger-kernel not installed in the training venv).
#   * datasets=filtered_hybrid + --raw_anno_path <manifest> (the length-rebalanced mix);
#     filtered_hybrid bypasses the duration-resampler (manifest is the final curated set).
#   * --max_steps passthrough (default -1) so a small smoke test can run before the full pilot.
#
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

model_path="TencentARC/TimeLens-7B"
datasets="filtered_hybrid"
raw_anno_path="data/charades_full/charades_train_rebalanced.json"
model_id="timelens-7b"
min_tokens=64
total_tokens=14336
fps=2
fps_max_frames=""
seed=42

global_batch_size=64
batch_per_device=1
num_devices=2
epochs=1
max_steps=-1
target_size=30000
learning_rate=1e-4
lora_rank=64
lora_alpha=64
deepspeed_config="scripts/zero3.json"
# Overridable via --output_root. It was hardcoded to a TimeLens-7B path, so running this
# recipe against a different backbone (issue #90 replicates the arms on TimeLens-8B)
# silently filed the checkpoint under output/TimeLens-7B/, mislabelling which model
# produced it. That is the artifact-identity confusion the provenance README calls its
# fourth axis, and it is worth a flag rather than a later forensic exercise.
output_root="output/TimeLens-7B/dlever_pilot"
report_to="none"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --raw_anno_path) raw_anno_path="$2"; shift 2 ;;
    --model_id) model_id="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --min_tokens) min_tokens="$2"; shift 2 ;;
    --total_tokens) total_tokens="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --fps_max_frames) fps_max_frames="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --num_devices) num_devices="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --max_steps) max_steps="$2"; shift 2 ;;
    --target_size) target_size="$2"; shift 2 ;;
    --learning_rate) learning_rate="$2"; shift 2 ;;
    --lora_rank) lora_rank="$2"; shift 2 ;;
    --lora_alpha) lora_alpha="$2"; shift 2 ;;
    --deepspeed_config) deepspeed_config="$2"; shift 2 ;;
    --output_root) output_root="$2"; shift 2 ;;
    --report_to) report_to="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
if [[ -z "${fps_max_frames}" ]]; then
  fps_max_frames=$((total_tokens / min_tokens * 2))
fi
run_tag="$(date +%Y%m%d-%H%M)"
run_name="dlever-${run_tag}_MAXFRAMES-${fps_max_frames}_FPS-${fps}_TOTALtokens-${total_tokens}_MINtokens-${min_tokens}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"
echo "Model: ${model_path} | datasets: ${datasets} | raw_anno_path: ${raw_anno_path}"
echo "LoRA: rank=${lora_rank} alpha=${lora_alpha} lr=${learning_rate} | max_steps=${max_steps}"

deepspeed --num_gpus "${num_devices}" training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger False \
  --lora_enable True \
  --vision_lora False \
  --lora_rank "${lora_rank}" \
  --lora_alpha "${lora_alpha}" \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${model_path}" \
  --model_id "${model_id}" \
  --conv_type "chatml" \
  --datasets "${datasets}" \
  --raw_anno_path "${raw_anno_path}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --target_size "${target_size}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm True \
  --freeze_merger True \
  --learning_rate "${learning_rate}" \
  --merger_lr "${learning_rate}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${epochs}" \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${epochs}" \
  --dataloader_num_workers 4 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${model_id}-dlever/${run_name}" \
  --logging_dir wandb \
  --save_only_model True
