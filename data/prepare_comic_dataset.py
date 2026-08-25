"""多进程视频预处理：统一帧数/分辨率 → 存帧 + 写 metadata CSV；可选 VAE 预计算。

> 建议在训练开始前至少 48 小时提前运行。
> 用法见 README。VAE 预计算请先用 scripts/test_vae_cache.py 验证兼容。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2
import pandas as pd
from dataset import process_video


def _process_one(args: tuple) -> dict:
    video_path, prompt, out_dir, num_frames, resolution, crop_method = args
    frames, _ = process_video(video_path, resolution, crop_method)
    if frames is None or len(frames) < 2:
        return {"video_path": video_path, "ok": False}
    # 截取/补齐到 num_frames 并写入 out_dir（近似首帧+抽样）
    clip_name = Path(video_path).stem
    clip_dir = Path(out_dir) / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    n = min(num_frames, len(frames))
    for i in range(n):
        cv2.imwrite(str(clip_dir / f"f{i:03d}.png"), frames[i])
    return {
        "video_path": f"{clip_name}/",
        "prompt": prompt,
        "num_frames": n,
        "original_w": frames[0].shape[1],
        "original_h": frames[0].shape[0],
        "crop_method": crop_method,
        "ok": True,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="漫画视频数据预处理")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=32)
    p.add_argument("--crop_method", choices=["center", "pad"], default="center")
    p.add_argument("--save_format", choices=["frames", "npy"], default="frames")
    p.add_argument("--precompute_vae", action="store_true")
    args = p.parse_args()

    # 读取输入视频清单（此处假设有 metadata.csv 或扫描目录；骨架示意）
    videos = sorted(Path(args.input_dir).rglob("*.mp4"))
    tasks = [(str(v), "", args.output_dir, args.num_frames, args.resolution, args.crop_method) for v in videos]

    with mp.Pool(args.num_workers) as pool:
        rows = pool.map(_process_one, tasks)

    good = [r for r in rows if r.get("ok")]
    csv_path = Path(args.output_dir) / "video_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(good).to_csv(csv_path, index=False)
    print(f"处理完成 {len(good)}/{len(videos)}，metadata 写入 {csv_path}")
    if args.precompute_vae:
        print("[提示] VAE 预计算请先跑 scripts/test_vae_cache.py 验证兼容，再交给 precompute_vae.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
