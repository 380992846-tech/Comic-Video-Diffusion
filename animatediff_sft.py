"""AnimateDiff LoRA / 全量微调入口（骨架）。

本文件统一解析训练参数，委托给 `models/animatediff/`（git submodule）的实际训练实现。
真实场景：把参数转发到 submodule 的训练脚本（具体入口因 fork 而异，见 models/README.md）。

用法：bash scripts/train_lora.sh
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnimateDiff LoRA / 全量微调")
    p.add_argument("--model_id_or_path", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--motion_adapter_id_or_path", default="Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--video_folder", required=True)
    p.add_argument("--sft_type", choices=["lora", "full"], default="lora")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", default="bf16")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--report_to", nargs="+", default=["tensorboard"])
    p.add_argument("--wandb_project", default="comic_video_diffusion")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # TODO: 将 args 组装成 submodule 训练脚本的命令行并委托运行。
    # 例（fork 相关）：
    #   import subprocess
    #   subprocess.run(["python", "models/animatediff/train.py", ...], check=True)
    print("[骨架] 已解析训练参数：")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print("[骨架] 请按 models/README.md 将参数转发给 models/animatediff/ 的训练实现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
