"""用动漫向 SDXL 模型生成"鲸鱼娘"静图候选（这是出好画面的关键一步）。

为什么单独做静图：AnimateDiff + SD1.5 的动模块会明显糊化画面，而**好看的二次元插画**
本来是文生图的强项。正确顺序是「先出一张高质量静图 → 再做 I2V / v2v 让它动起来」，
而不是直接从文本生视频。

用法：
    python scripts/make_whale_girl_still.py \
      --model_id_or_path /root/autodl-tmp/models/animagine-xl-3.1 \
      --out_dir /root/autodl-tmp/whale_still --variants 6 --seeds_per_variant 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

BASE = ("masterpiece, best quality, very aesthetic, absurdres, 1girl, solo, "
        "whale girl, blue hair, very long hair, blue eyes, fin ears, whale tail, "
        "maid, maid headdress, white apron, frilled apron, cute, kawaii")

NEG = ("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
       "fewer digits, cropped, worst quality, low quality, low score, bad score, "
       "average score, signature, watermark, username, blurry, nsfw, realistic, 3d")

# 每个变体只改"体型/姿态/背景"这几段，角色描述保持一致
VARIANTS = [
    ("chibi_fullbody", "chibi, super deformed, 3 heads tall, full body, standing, waving, one hand up, smile, open mouth, closed eyes, blush, white background, simple background"),
    ("chibi_seated", "chibi, super deformed, full body, sitting, seiza, hands on lap, smile, blush, looking at viewer, blue background, gradient background"),
    ("chibi_plush", "chibi, super deformed, full body, holding small whale plush, hugging, happy, sparkling eyes, pastel background"),
    ("chibi_portrait", "chibi, super deformed, upper body, portrait, head tilt, smile, blush, hands up, hearts, white background"),
    ("chibi_sea", "chibi, super deformed, full body, floating, underwater, bubbles, small fish, jellyfish, blue gradient background, serene"),
    ("chibi_maid_serve", "chibi, super deformed, full body, holding a tray with a tiny cake, maid uniform, smile, blush, cafe background"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/animagine-xl-3.1")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/whale_still")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--variants", type=int, default=len(VARIANTS))
    ap.add_argument("--seeds_per_variant", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()

    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id_or_path, torch_dtype=dtype,
                                                     use_safetensors=True, add_watermarker=False)
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception as exc:
        print(f"[warn] scheduler override failed: {exc}")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"[model] {args.model_id_or_path} on {device} {args.dtype}")

    made = []
    for vi, (name, tail) in enumerate(VARIANTS[: args.variants]):
        prompt = f"{BASE}, {tail}"
        for k in range(args.seeds_per_variant):
            seed = args.seed + vi * 1000 + k
            g = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                img = pipe(prompt=prompt, negative_prompt=NEG, width=args.width, height=args.height,
                           num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                           generator=g).images[0]
            path = out_dir / f"{vi:02d}_{name}_s{seed}.png"
            img.save(path)
            made.append((name, seed, path, np.asarray(img)))
            print(f"  [{vi}] {path.name}")

    # 拼一张总览图方便挑
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(made)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 6.0 * nrow), squeeze=False)
    for i, (name, seed, path, arr) in enumerate(made):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(arr)
        ax.set_title(f"{name}  seed={seed}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    grid = out_dir / "_candidates.png"
    fig.savefig(grid, dpi=95)
    plt.close(fig)
    print(f"[ok] {n} images -> {out_dir}\n[ok] contact sheet -> {grid}")
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
