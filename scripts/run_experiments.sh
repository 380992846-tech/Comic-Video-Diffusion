#!/usr/bin/env bash
# E1–E4 消融驱动：等预算（同 steps / 同数据 / 同分辨率）逐臂串行训练。
#
#   bash scripts/run_experiments.sh                 # 跑 ARMS 里全部
#   ARMS="e2_rank8 e2_rank16" bash scripts/run_experiments.sh
#
# 每臂的原始指标在 /root/autodl-tmp/runs/<arm>/metrics.csv（论文数据源）。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-400}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
RESOLUTION="${RESOLUTION:-384}"
NUM_FRAMES="${NUM_FRAMES:-16}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/anime_clips}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/root/autodl-tmp/data/anime_clips_all}"
RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/runs}"
ARMS="${ARMS:-e1_lora_r32 e1_partial_last4 e1_motion_full e2_rank8 e2_rank16 e2_rank64 e3_data50 e3_data150 e3_data300 e4_dac_r32}"

mkdir -p "$RUNS_DIR"
echo "[experiments] steps=$STEPS grad_accum=$GRAD_ACCUM res=$RESOLUTION frames=$NUM_FRAMES data=$DATA_DIR"
echo "[experiments] arms: $ARMS"

for arm in $ARMS; do
  out="$RUNS_DIR/$arm"
  if [ -f "$out/metrics.csv" ] && [ "$(wc -l < "$out/metrics.csv")" -gt 3 ]; then
    echo "[skip] $arm (already has metrics)"
    continue
  fi
  case "$arm" in
    e1_lora_r32)      TRAIN_MODE=lora;    LORA_RANK=32; MAX_SAMPLES=0;   USE_DAC=0 ;;
    e1_partial_last4) TRAIN_MODE=partial; LORA_RANK=32; MAX_SAMPLES=0;   USE_DAC=0 ;;
    e1_motion_full)   TRAIN_MODE=motion;  LORA_RANK=32; MAX_SAMPLES=0;   USE_DAC=0 ;;
    e2_rank8)         TRAIN_MODE=lora;    LORA_RANK=8;  MAX_SAMPLES=0;   USE_DAC=0 ;;
    e2_rank16)        TRAIN_MODE=lora;    LORA_RANK=16; MAX_SAMPLES=0;   USE_DAC=0 ;;
    e2_rank64)        TRAIN_MODE=lora;    LORA_RANK=64; MAX_SAMPLES=0;   USE_DAC=0 ;;
    e3_data50)        TRAIN_MODE=lora;    LORA_RANK=32; MAX_SAMPLES=50;  USE_DAC=0 ;;
    e3_data150)       TRAIN_MODE=lora;    LORA_RANK=32; MAX_SAMPLES=150; USE_DAC=0 ;;
    e3_data300)       TRAIN_MODE=lora;    LORA_RANK=32; MAX_SAMPLES=300; USE_DAC=0 ;;
    e4_dac_r32)       TRAIN_MODE=lora;    LORA_RANK=32; MAX_SAMPLES=0;   USE_DAC=1 ;;
    *) echo "[warn] unknown arm $arm"; continue ;;
  esac
  echo "=================================================================="
  echo "[arm] $arm  mode=$TRAIN_MODE rank=$LORA_RANK samples=${MAX_SAMPLES:-all} dac=$USE_DAC"
  echo "=================================================================="
  START=$(date +%s)
  RUN_NAME="$arm" TRAIN_MODE="$TRAIN_MODE" LORA_RANK="$LORA_RANK" MAX_SAMPLES="$MAX_SAMPLES" \
    USE_DAC="$USE_DAC" STEPS="$STEPS" GRAD_ACCUM="$GRAD_ACCUM" RESOLUTION="$RESOLUTION" \
    NUM_FRAMES="$NUM_FRAMES" DATA_DIR="$DATA_DIR" VIDEO_FOLDER="$VIDEO_FOLDER" \
    bash "$REPO/scripts/train_lora.sh" || echo "[warn] arm $arm exited non-zero"
  echo "[arm] $arm finished in $(( ($(date +%s) - START) / 60 )) min"
done

echo "[experiments] all arms done"
for d in "$RUNS_DIR"/*/metrics.csv; do
  [ -f "$d" ] || continue
  printf "%-22s %s rows\n" "$(basename "$(dirname "$d")")" "$(($(wc -l < "$d") - 1))"
done
