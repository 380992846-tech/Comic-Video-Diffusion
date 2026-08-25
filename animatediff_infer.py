"""AnimateDiff 采样入口（骨架）。

用微调后的 LoRA 权重生成漫画风格视频；实际推理委托给 `models/animatediff/`（submodule）。

用法：bash scripts/inference.sh
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnimateDiff 采样")
    p.add_argument("--model_id_or_path", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--motion_adapter_id_or_path", default="Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2")
    p.add_argument("--ckpt_dir", required=True, help="微调 checkpoint 目录")
    p.add_argument("--prompt", required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--output_dir", default="paper/demo_videos")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # TODO: 委托给 models/animatediff/ 的推理脚本，加载 ckpt_dir 的 LoRA 权重 + prompt 采样。
    print("[骨架] 采样参数：")
    for k, v in vars(args).items():
        print(f"  {k} = {v}")
    print("[骨架] 请按 models/README.md 委托给 models/animatediff/ 推理实现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
