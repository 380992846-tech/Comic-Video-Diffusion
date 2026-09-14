#!/usr/bin/env bash
# 补齐失败臂 e1_partial_last4（class 名不匹配导致一开始 unfroze 0 个 block），
# 并在主流水线跑完后重新汇总/出图/出片。
set -u
V=/root/autodl-tmp/venv-train/bin/python
REPO=/root/comic_video_diffusion
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for after_sweep.sh to finish"
while pgrep -f "after_sweep.sh" >/dev/null 2>&1; do sleep 60; done
log "after_sweep finished; re-running the partial arm"

RUN_NAME=e1_partial_last4 TRAIN_MODE=partial PARTIAL_LAST_N=4 \
  STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 \
  DATA_DIR=/root/autodl-tmp/data/anime_clips \
  VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
  bash scripts/train_lora.sh 2>&1 | tail -20

log "re-collect + figures + tables"
$V scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results 2>&1 | tail -20
$V scripts/make_figures.py --results_dir results 2>&1 | tail -12
$V scripts/write_results.py --results_dir results 2>&1 | tail -60

log "inference for the partial arm"
ALL_ARMS=e1_partial_last4 NUM_VIDEOS=16 bash scripts/run_inference_all_arms.sh 2>&1 | tail -10

log "PARTIAL RERUN DONE"
