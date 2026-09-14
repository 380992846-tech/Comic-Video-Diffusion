#!/usr/bin/env bash
# 训练完成后统一出片：每臂用同一批 prompt/seed 生成固定数量的视频，供评测与论文配图。
#
#   bash scripts/run_inference_all_arms.sh            # 全部臂，每臂 16 条
#   KEY=1 bash scripts/run_inference_all_arms.sh      # 关键臂（base / e1_lora_r32 / e4_dac_r32）52 条
set -uo pipefail

PY="${PY:-/root/autodl-tmp/venv-train/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/runs}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/samples}"
MODEL_ID="${MODEL_ID:-/root/autodl-tmp/models/sd15}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/models/adapter-v15-2}"
NUM_VIDEOS="${NUM_VIDEOS:-16}"
STEPS="${STEPS:-25}"
RES="${RES:-384}"
FRAMES="${FRAMES:-16}"
ALL_ARMS="${ALL_ARMS:-e1_lora_r32 e1_partial_last4 e1_motion_full e2_rank8 e2_rank16 e2_rank64 e3_data50 e3_data150 e3_data300 e4_dac_r32}"

PROMPTS=(
  "anime style animation, 2d cel animation, comic style, a quiet seaside town at sunrise, slow camera pan"
  "anime style animation, 2d cel animation, comic style, cherry blossoms falling over a river, gentle breeze"
  "anime style animation, 2d cel animation, comic style, night sky with shooting stars over a mountain, drifting clouds"
  "anime style animation, 2d cel animation, comic style, rain falling on a neon city street, reflections moving"
  "anime style animation, 2d cel animation, comic style, a field of sunflowers swaying, warm afternoon light"
  "anime style animation, 2d cel animation, comic style, snow falling in a forest, camera slowly pushing in"
  "anime style animation, 2d cel animation, comic style, clouds moving fast over a school rooftop, wide shot"
  "anime style animation, 2d cel animation, comic style, lanterns floating on a lake at night, soft ripples"
)

cd "$REPO"
mkdir -p "$OUT_ROOT"

run_arm () {
  local arm="$1" n="$2"
  local ckpt="$RUNS_DIR/$arm/checkpoints/final.safetensors"
  local out="$OUT_ROOT/$arm"
  if [ -f "$out/manifest.json" ] && [ "$(ls "$out"/*.mp4 2>/dev/null | wc -l)" -ge "$n" ]; then
    echo "[skip] $arm already has $n videos"; return 0
  fi
  if [ "$arm" != "base" ] && [ ! -f "$ckpt" ]; then
    echo "[warn] $arm: no checkpoint at $ckpt"; return 0
  fi
  mkdir -p "$out"
  echo "[infer] $arm -> $out (n=$n, steps=$STEPS, res=$RES)"
  local i=0 seed=1000
  while [ "$i" -lt "$n" ]; do
    local p="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
    local ckpt_arg=""
    [ "$arm" != "base" ] && ckpt_arg="--ckpt_dir $ckpt"
    $PY animatediff_infer.py \
      --model_id_or_path "$MODEL_ID" \
      --motion_adapter_id_or_path "$ADAPTER" \
      $ckpt_arg \
      --prompt "$p" \
      --num_frames "$FRAMES" --resolution "$RES" \
      --num_inference_steps "$STEPS" --guidance_scale 7.5 \
      --seed "$seed" --num_videos 1 --fps 8 \
      --output_dir "$out" --prefix "${arm}_$(printf %03d $i)" > "$out/log_$(printf %03d $i).txt" 2>&1 \
      || echo "[warn] $arm #$i failed (see $out/log_$(printf %03d $i).txt)"
    i=$((i+1)); seed=$((seed+1))
  done
  echo "{\"arm\": \"$arm\", \"n\": $n, \"steps\": $STEPS, \"resolution\": $RES, \"frames\": $FRAMES, \"ckpt\": \"$ckpt\"}" > "$out/manifest.json"
  echo "[done] $arm: $(ls "$out"/*.mp4 2>/dev/null | wc -l) videos"
}

for arm in $ALL_ARMS; do
  run_arm "$arm" "$NUM_VIDEOS"
done

if [ "${KEY:-0}" = "1" ]; then
  for arm in base e1_lora_r32 e4_dac_r32; do
    run_arm "$arm" 52
  done
fi

echo "[all inference done]"
for d in "$OUT_ROOT"/*/; do
  [ -d "$d" ] || continue
  printf "%-20s %s videos\n" "$(basename "$d")" "$(ls "$d"/*.mp4 2>/dev/null | wc -l)"
done
