#!/usr/bin/env bash
# 收尾第二轮：等 partial 补齐后，做 DAC 时序饱和分析 + 定性对比图 + 结果表。
set -u
V=/root/autodl-tmp/venv-train/bin/python
REPO=/root/comic_video_diffusion
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for rerun_partial.sh"
while pgrep -f "rerun_partial.sh" >/dev/null 2>&1; do sleep 60; done
log "rerun finished"

mkdir -p paper/figures results

log "DAC temporal saturation analysis (real trained DAC checkpoint)"
CKPT=/root/autodl-tmp/runs/e4_dac_r32/checkpoints/final.safetensors
if [ -f "$CKPT" ]; then
  $V scripts/analyze_dac.py --ckpt "$CKPT" \
    --output_json paper/figures/dac_analysis.json \
    --output_fig paper/figures/dac_temporal_saturation.png 2>&1 | tail -25
else
  log "[warn] DAC checkpoint missing: $CKPT"
fi

log "qualitative grid"
$V scripts/make_qualitative_grid.py \
  --gen base=/root/autodl-tmp/samples/base \
  --gen e1_lora_r32=/root/autodl-tmp/samples/e1_lora_r32 \
  --gen e1_partial_last4=/root/autodl-tmp/samples/e1_partial_last4 \
  --gen e1_motion_full=/root/autodl-tmp/samples/e1_motion_full \
  --gen e4_dac_r32=/root/autodl-tmp/samples/e4_dac_r32 \
  --output paper/figures/qualitative_grid.png --frames 4 2>&1 | tail -12

log "tables"
$V scripts/write_results.py --results_dir results --out results/tables.md 2>&1 | tail -80

log "FINAL REPORT DONE"
ls -la results/ paper/figures/ 2>/dev/null
