#!/usr/bin/env bash
# 训练跑完后自动执行的收尾流水线：汇总 → 出图 → 逐臂出片 → 评测。
# 以 nohup 方式挂在实例上，避免依赖本地会话是否在线。
set -u
V=/root/autodl-tmp/venv-train/bin/python
REPO=/root/comic_video_diffusion
RUNS=/root/autodl-tmp/runs
LOG=/root/autodl-tmp/logs/after_sweep.log
cd "$REPO"

log() { echo "[$(date '+%F %T')] $*"; }

log "watcher started; waiting for run_experiments.sh to finish"
while pgrep -f "run_experiments.sh" >/dev/null 2>&1; do sleep 60; done
log "training finished"

log "step 1/5: collect results"
$V scripts/collect_results.py --runs_dir "$RUNS" --output_dir results 2>&1 | tail -25

log "step 2/5: figures"
$V scripts/make_figures.py --results_dir results 2>&1 | tail -20

log "step 3/5: per-arm inference (16 videos each)"
bash scripts/run_inference_all_arms.sh 2>&1 | tail -30

log "step 4/5: key-arm inference (52 videos each)"
KEY=1 NUM_VIDEOS=52 bash scripts/run_inference_all_arms.sh 2>&1 | tail -20

log "step 5/5: evaluation"
mkdir -p /root/autodl-tmp/reference_clips
i=0
for f in $(ls /root/autodl-tmp/data/anime_clips_all/*.mp4 2>/dev/null | head -60); do
  cp -n "$f" "/root/autodl-tmp/reference_clips/ref_$(printf %03d $i).mp4" 2>/dev/null || true
  i=$((i+1))
done
log "reference clips: $(ls /root/autodl-tmp/reference_clips/*.mp4 2>/dev/null | wc -l)"

GEN_ARGS=""
for arm in base e1_lora_r32 e1_partial_last4 e1_motion_full e2_rank8 e2_rank16 e2_rank64 \
           e3_data50 e3_data150 e3_data300 e4_dac_r32; do
  d="/root/autodl-tmp/samples/$arm"
  [ -d "$d" ] && GEN_ARGS="$GEN_ARGS --gen $arm=$d"
done
# shellcheck disable=SC2086
$V scripts/eval_fvd.py $GEN_ARGS --real_dir /root/autodl-tmp/reference_clips \
  --output_json results/eval_all_arms.json --output_csv results/eval_all_arms.csv 2>&1 | tail -40

log "ALL DONE"
ls -la "$REPO/results" 2>/dev/null
