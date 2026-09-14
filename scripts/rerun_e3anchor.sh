#!/usr/bin/env bash
# E3 曲线需要一个"干净"的全量锚点：验证切片取数据集末尾 20 条，只有把训练数据也限制在
# 前 324 条，这条切片才对**所有** E3 点都是 hold-out。否则 344 条臂的验证集是 in-sample，
# 会让"数据越多越好"看起来比实际更强。
set -u
V=/root/autodl-tmp/venv-train/bin/python
REPO=/root/comic_video_diffusion
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for eval_all_val_loss to finish"
while pgrep -f "eval_all_val_loss" >/dev/null 2>&1; do sleep 60; done
log "chain finished; training the clean E3 anchor (max_samples=324)"

RUN_NAME=e3_data324 TRAIN_MODE=lora LORA_RANK=32 MAX_SAMPLES=324 \
  STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 \
  DATA_DIR=/root/autodl-tmp/data/anime_clips \
  VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
  bash scripts/train_lora.sh 2>&1 | tail -15

log "refresh results tables + val-loss table"
$V scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results 2>&1 | tail -15
$V scripts/make_figures.py --results_dir results 2>&1 | tail -12
$V scripts/write_results.py --results_dir results --out results/tables.md 2>&1 | tail -60
bash scripts/eval_all_val_loss.sh 2>&1 | tail -30

log "E3 ANCHOR RERUN DONE"
