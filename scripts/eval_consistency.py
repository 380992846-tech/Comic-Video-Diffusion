#!/usr/bin/env python3
"""Anchor-LoRA 的轻量一致性评测（不装 detectron2 / VBench）。

对每个生成的片段，与它的参考帧（默认首帧）比：

- **subject similarity**：CLIP 图像特征的余弦相似度（每帧 vs 参考帧，取均值）
- **edge-SSIM**：Sobel 边缘图的 SSIM（结构/线稿保持）
- **hsv distance**：HSV 空间均值 + 标准差距离（配色稳定）
- **motion**：相邻帧平均绝对差（确认没有退化成静图）

用法：

    python scripts/eval_consistency.py --pred_dir samples/E_full --ref_dir data/refs \
        --output_json results/eval_E_full.json

目录约定：`pred_dir` 下每个 `*.mp4`，同名参考图 `ref_dir/<stem>.png|jpg`。
没有参考图时退化为「首帧当参考」，并在 JSON 里标 `ref_source: first_frame`。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--ref_dir", default=None)
    p.add_argument("--output_json", required=True)
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_clips", type=int, default=0)
    return p.parse_args()


def read_frames(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
        arr = iio.imread(path, plugin="pyav")
        if arr.ndim == 3:
            arr = arr[None]
        return np.asarray(arr[..., :3], dtype=np.uint8)
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(out)


def sobel(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = kx.T
    from scipy.ndimage import convolve
    return np.sqrt(convolve(g, kx) ** 2 + convolve(g, ky) ** 2 + 1e-6)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 1e-4, 9e-4
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def hsv_stats(rgb: np.ndarray) -> np.ndarray:
    import colorsys
    px = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px[:: max(1, len(px) // 5000)]])
    return np.concatenate([hsv.mean(axis=0), hsv.std(axis=0)])


def main() -> int:
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel

    device = args.device if torch.cuda.is_available() else "cpu"
    processor = CLIPImageProcessor.from_pretrained(args.clip_model)
    model = CLIPVisionModel.from_pretrained(args.clip_model).to(device).eval()

    @torch.no_grad()
    def clip_embed(img: np.ndarray) -> np.ndarray:
        inputs = processor(images=Image.fromarray(img), return_tensors="pt").to(device)
        feat = model(**inputs).pooler_output
        return (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    pred_dir = Path(args.pred_dir)
    videos = sorted(pred_dir.glob("*.mp4"))
    if args.max_clips:
        videos = videos[: args.max_clips]
    results = []
    for vid in videos:
        frames = read_frames(vid)
        ref = None
        ref_source = "first_frame"
        if args.ref_dir:
            for ext in (".png", ".jpg", ".jpeg"):
                cand = Path(args.ref_dir) / f"{vid.stem}{ext}"
                if cand.exists():
                    ref = np.asarray(Image.open(cand).convert("RGB"))
                    ref_source = "file"
                    break
        if ref is None:
            ref = frames[0]
        gray_ref = ref.mean(axis=-1)
        gray_frames = frames.mean(axis=-1)

        emb_ref = clip_embed(ref)
        emb_frames = np.stack([clip_embed(f) for f in frames[:: max(1, len(frames) // 6)]])
        subj = float(np.mean(emb_frames @ emb_ref))

        edge_ref = sobel(gray_ref)
        edge_ssims = [ssim(sobel(g), edge_ref) for g in gray_frames[:: max(1, len(gray_frames) // 6)]]
        motion = float(np.abs(np.diff(gray_frames, axis=0)).mean()) if len(gray_frames) > 1 else 0.0
        # 颜色：拿「后续帧的平均统计」对参考帧比 —— 只比 frames[0] 会恒等于 0（自己比自己）
        later = frames[1:] if len(frames) > 1 else frames
        hsv = float(np.linalg.norm(hsv_stats(np.concatenate([f[None] for f in later]).mean(axis=0)) - hsv_stats(ref)))

        results.append({
            "video": str(vid), "frames": int(len(frames)), "ref_source": ref_source,
            "subject_similarity": round(subj, 4),
            "edge_ssim": round(float(np.mean(edge_ssims)), 4),
            "hsv_distance": round(hsv, 4),
            "motion": round(motion, 5),
        })
        print(f"{vid.name}: subject={results[-1]['subject_similarity']} edge={results[-1]['edge_ssim']} "
              f"hsv={results[-1]['hsv_distance']}", flush=True)

    agg = {}
    if results:
        for key in ("subject_similarity", "edge_ssim", "hsv_distance", "motion"):
            agg[key] = round(float(np.mean([r[key] for r in results])), 4)
    payload = {"pred_dir": str(pred_dir), "n_clips": len(results),
               "reliable": len(results) >= 50, "aggregate": agg, "per_clip": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False), "reliable:", payload["reliable"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
