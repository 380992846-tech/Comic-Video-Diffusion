#!/usr/bin/env bash
# 启动 AnimateDiff LoRA 微调（单卡 A800 主流程）
set -e
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
  --model_id_or_path runwayml/stable-diffusion-v1-5 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --csv_path /path/to/video_metadata.csv \
  --video_folder /path/to/videos \
  --sft_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --gradient_checkpointing True \
  --mixed_precision bf16 \
  --num_frames 16 \
  --report_to wandb tensorboard \
  --wandb_project comic_video_diffusion \
  --logging_steps 50 \
  --save_steps 5000 \
  --output_dir logs/lora_experiments/run_001
