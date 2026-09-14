#!/usr/bin/env bash
# 单次微调入口（真实训练）。
#
# 例子：
#   bash scripts/train_lora.sh                       # 基线：LoRA rank32
#   RUN_NAME=e2_rank8 LORA_RANK=8 bash scripts/train_lora.sh
#   RUN_NAME=e4_dac USE_DAC=1 bash scripts/train_lora.sh
set -euo pipefail

PY="${PY:-/root/autodl-tmp/venv-train/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-/root/autodl-tmp/models/sd15}"
MOTION_ADAPTER_ID="${MOTION_ADAPTER_ID:-/root/autodl-tmp/models/adapter-v15-2}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/anime_shooter}"
CSV_PATH="${CSV_PATH:-$DATA_DIR/video_metadata.csv}"
VIDEO_FOLDER="${VIDEO_FOLDER:-$DATA_DIR}"

RUN_NAME="${RUN_NAME:-lora_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/runs/$RUN_NAME}"
STEPS="${STEPS:-1200}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
TRAIN_MODE="${TRAIN_MODE:-lora}"
RESOLUTION="${RESOLUTION:-512}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PARTIAL_LAST_N="${PARTIAL_LAST_N:-4}"
USE_DAC="${USE_DAC:-0}"
DAC_LOSS_WEIGHT="${DAC_LOSS_WEIGHT:-0.01}"

DAC_FLAG=""
if [ "$USE_DAC" = "1" ]; then DAC_FLAG="--use_dac --dac_loss_weight $DAC_LOSS_WEIGHT"; fi

echo "[train] run=$RUN_NAME mode=$TRAIN_MODE rank=$LORA_RANK steps=$STEPS res=$RESOLUTION frames=$NUM_FRAMES samples=${MAX_SAMPLES:-all} dac=$USE_DAC"
cd "$REPO"
$PY animatediff_sft.py \
  --model_id_or_path "$MODEL_ID" \
  --motion_adapter_id_or_path "$MOTION_ADAPTER_ID" \
  --csv_path "$CSV_PATH" \
  --video_folder "$VIDEO_FOLDER" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --train_mode "$TRAIN_MODE" \
  --partial_last_n "$PARTIAL_LAST_N" \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" --lr_scheduler_type cosine --warmup_steps 50 \
  --max_train_steps "$STEPS" --num_train_epochs 100 \
  --resolution "$RESOLUTION" --num_frames "$NUM_FRAMES" \
  --max_samples "$MAX_SAMPLES" \
  --gradient_checkpointing True --mixed_precision bf16 \
  --logging_steps 20 --save_steps 400 \
  --report_to csv,tensorboard \
  $DAC_FLAG
