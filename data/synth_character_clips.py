#!/usr/bin/env python3
"""合成「角色演出」的动画/漫画片段（用于预算内的域适配训练）。

为什么用合成：真实剧集（冥王神话 LC / 银魂）有版权问题，且公开的动画视频数据集要么是
风景空镜（svjack/Anime_Landscape_Videos），要么是 gated（Sakuga-42M / AnimeShooter）。
合成数据的优势是**风格可控、可无限扩、无版权风险**；劣势是分布偏窄，所以论文里必须
如实写成「合成角色片段」，并把 FVD 的参考域单独说明。

用法（A800）：

    python data/synth_character_clips.py \
        --model_path /root/autodl-tmp/models/sd15 \
        --motion_adapter /root/autodl-tmp/models/adapter-v15-2 \
        --style lc --num_clips 120 --frames 16 --resolution 512 --steps 12 \
        --out_dir /root/autodl-tmp/data/synth_lc

产物：`<out_dir>/clips/*.mp4` + `<out_dir>/video_metadata.csv`（列见下），
可直接喂 `data/prepare_comic_dataset.py`。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

# 风格描述包：用**视觉特征**而不是作品名（更稳，也不会把 IP 名写进数据卡）
STYLE_PACKS = {
    "lc": {
        "name": "2000s shonen armor anime",
        "prompt": (
            "anime style, 2000s shonen series, young male warrior in bronze armor and cape, "
            "dramatic rim lighting, deep blue night sky, dynamic pose, cel shading, "
            "detailed lineart, film grain, cinematic composition"
        ),
        "motion": "slow camera pan, hair and cape flowing in the wind",
        "characters": [
            "young man with short dark hair and determined eyes",
            "young woman with long silver hair and armored dress",
            "teenage boy with white hair, red scarf, serious expression",
        ],
    },
    "gintama": {
        "name": "comedy gag anime",
        "prompt": (
            "anime style, comedy gag series, expressive exaggerated faces, silver hair samurai "
            "in black kimono, wooden sword, warm sunlight, town street background, "
            "vibrant colors, cel shading, clean lineart"
        ),
        "motion": "snappy comedic gesture, head turn, camera shake",
        "characters": [
            "young man with messy silver hair and lazy eyes",
            "young woman with long dark hair and red hairpin, smiling",
            "small boy with glasses and serious deadpan face",
        ],
    },
    "generic": {
        "name": "generic anime character acting",
        "prompt": (
            "anime style, cinematic lighting, detailed character design, expressive face, "
            "cel shading, clean lineart, 2d animation still"
        ),
        "motion": "subtle body motion, breathing, slight head turn",
        "characters": [
            "young woman with long black hair and school uniform",
            "young man in a dark coat with short hair",
        ],
    },
}

NEGATIVE = "low quality, blurry, watermark, text, logo, extra fingers, deformed face, photo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合成角色动画片段")
    p.add_argument("--model_path", required=True)
    p.add_argument("--motion_adapter", required=True)
    p.add_argument("--style", default="generic", choices=sorted(STYLE_PACKS))
    p.add_argument("--num_clips", type=int, default=50)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--min_motion", type=float, default=0.004, help="帧间差阈值，低于它的丢弃")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    style = STYLE_PACKS[args.style]
    out = Path(args.out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    adapter = MotionAdapter.from_pretrained(args.motion_adapter, torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(args.model_path, motion_adapter=adapter, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler",
                                                  beta_schedule="linear", clip_sample=False,
                                                  timestep_spacing="linspace", steps_offset=1)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.num_clips):
        character = rng.choice(style["characters"])
        motion = rng.choice([style["motion"], "walking forward, camera tracking", "turning head and smiling"])
        prompt = f"{character}, {motion}, {style['prompt']}"
        seed = args.seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            num_frames=args.frames, height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen,
        )
        frames = np.asarray(result.frames[0], dtype=np.float32) / 255.0
        gray = frames.mean(axis=-1)
        motion_score = float(np.abs(gray[1:] - gray[:-1]).mean()) if len(gray) > 1 else 0.0
        keep = motion_score >= args.min_motion
        name = f"{args.style}_{i:04d}.mp4"
        path = out / "clips" / name
        if keep:
            from diffusers.utils import export_to_video
            export_to_video(result.frames[0], str(path), fps=8)
        rows.append({
            "video": str(path), "caption": prompt, "style": args.style,
            "character": character, "motion": motion, "seed": seed,
            "frames": args.frames, "resolution": args.resolution,
            "motion_score": round(motion_score, 6), "kept": int(keep),
        })
        print(f"[{i + 1}/{args.num_clips}] {'kept' if keep else 'drop'} motion={motion_score:.5f} {name}", flush=True)

    csv_path = out / "video_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(r["kept"] for r in rows)
    summary = {"style": args.style, "requested": args.num_clips, "kept": kept,
               "median_motion": sorted(r["motion_score"] for r in rows)[len(rows) // 2],
               **{k: v for k, v in vars(args).items() if k not in {"num_clips"}}}
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：{kept}/{args.num_clips} 条保留 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
