#!/usr/bin/env bash
# E4 的 DAC 臂在 dac_loss_weight=0.01 下 tau 基本没动（实测 tau_mean 恒为 1.0）。
# 这里补一个"强损失权重"臂 (w=10) 来证明机制本身是可驱动的，而不是没接上。
set -u
V=/root/autodl-tmp/venv-train/bin/python
REPO=/root/comic_video_diffusion
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for bench_speed to finish"
while pgrep -f "bench_speed" >/dev/null 2>&1; do sleep 60; done
log "chain finished; training the strong-DAC arm"

RUN_NAME=e4_dac_strong TRAIN_MODE=lora LORA_RANK=32 USE_DAC=1 DAC_LOSS_WEIGHT=10.0 \
  STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 \
  DATA_DIR=/root/autodl-tmp/data/anime_clips \
  VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
  bash scripts/train_lora.sh 2>&1 | tail -12

log "refresh results + figures"
$V scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results 2>&1 | tail -8
$V scripts/make_figures.py --results_dir results 2>&1 | tail -10
$V scripts/write_results.py --results_dir results --out results/tables.md 2>&1 | tail -50
bash scripts/eval_all_val_loss.sh 2>&1 | tail -20
log "STRONG DAC DONE"
