#!/usr/bin/env bash
# 采样：用微调后的 LoRA 权重生成漫画风格视频
set -e
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path runwayml/stable-diffusion-v1-5 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --ckpt_dir logs/lora_experiments/run_001/checkpoints/iter-20000 \
  --prompt "a comic style character walking in the rain, 2d animation, studio ghibli style" \
  --num_frames 16 \
  --num_samples 4 \
  --output_dir paper/demo_videos
