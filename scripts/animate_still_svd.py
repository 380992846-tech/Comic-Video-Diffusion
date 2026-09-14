"""用 Stable Video Diffusion 把「鲸鱼娘」静图变成短片（I2V）。

为什么用 SVD 而不是 AnimateDiff：
* AnimateDiff 的动模块是 **SD1.5** 且用真实视频训练，接到动漫插画上会明显糊化；
* SVD（img2vid-xt）本身就是 **1024 级**的图像到视频模型，输入一张干净插画 → 输出
  25 帧带镜头/发丝/衣物微动的短片，画质保持得好得多。
正确顺序：SDXL 出静图 → 挑一张 → SVD 动起来。

用法：
    python scripts/animate_still_svd.py \
      --image /root/autodl-tmp/whale_still/02_chibi_plush_s20262914.png \
      --model_id_or_path /root/autodl-tmp/models/svd-xt \
      --out_dir /root/autodl-tmp/whale_video --num_frames 25 --motion_bucket_id 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, nargs="+", help="一张或多张静图")
    ap.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/svd-xt")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/whale_video")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--num_frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--motion_bucket_id", type=int, default=100, help="越大动得越厉害（80-127 常用）")
    ap.add_argument("--noise_aug_strength", type=float, default=0.02)
    ap.add_argument("--decode_chunk_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            args.model_id_or_path, torch_dtype=torch.float16, variant="fp16")
    except Exception as exc:
        print(f"[warn] variant=fp16 不可用（{type(exc).__name__}），改用默认权重名")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            args.model_id_or_path, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    print(f"[model] SVD ready: {args.model_id_or_path}")

    for img_path in args.image:
        image = load_image(img_path).resize((args.width, args.height))
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            frames = pipe(image, height=args.height, width=args.width,
                          num_frames=args.num_frames, decode_chunk_size=args.decode_chunk_size,
                          motion_bucket_id=args.motion_bucket_id,
                          noise_aug_strength=args.noise_aug_strength,
                          generator=g).frames[0]
        arr = np.stack([np.asarray(f.convert("RGB")) for f in frames], axis=0)
        stem = Path(img_path).stem
        out = out_dir / f"{stem}_svd{args.motion_bucket_id}_{args.num_frames}f.mp4"
        imageio.mimwrite(out, list(arr), fps=args.fps, quality=8, macro_block_size=8)
        print(f"[ok] {out}  {len(arr)} frames  {len(arr)/args.fps:.1f}s  {arr.shape}")
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
