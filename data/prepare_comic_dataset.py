"""多进程视频索引/清洗：扫描视频 → 计算元数据与运动强度 → 写 metadata CSV。

设计要点（相对早期骨架版的改动）
--------------------------------
早期版本把每个 clip 落成 16 张 PNG 帧，等于把数据集放大 20~50 倍
（512×512×16 帧 ≈ 12.6 MB/clip，几千条就撑爆磁盘）。真实训练并不需要
materialize：``data/dataset.py`` 会直接从源视频里随机时间裁剪。因此本脚本默认
只做 **索引 + 过滤 + 写 CSV**，磁盘占用接近于零。

运动强度 ``motion`` = 相邻采样帧的平均绝对差（0~1）。AnimateDiff 的时序模块学的是
帧间变化，纯静止镜头（动漫里非常多）几乎没有梯度信号，默认按 ``--min_motion`` 过滤。

可选 ``--materialize npy`` 仍支持把固定时长的 clip 落盘（用于复现/交付小样例集）。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import probe_video, read_clip  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def motion_score(path: str, resolution: int = 256, samples: int = 8) -> float:
    """相邻采样帧平均绝对差（0~1）；解码失败返回 -1。"""
    frames = read_clip(path, num_frames=samples, resolution=resolution,
                       crop_method="center", frame_stride=1, start_frame=0)
    if frames is None or len(frames) < 2:
        return -1.0
    f = frames.astype(np.float32) / 255.0
    diffs = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))
    return float(diffs.mean())


def _worker(task: tuple) -> dict:
    path, prompt, resolution, min_frames = task
    info = probe_video(path)
    if info["frames"] and info["frames"] < min_frames:
        return {"ok": False, "path": path, "reason": "too_short"}
    if not info["frames"]:
        # 帧数未知（部分容器不写 nb_frames）：实际解码一次确认
        got = read_clip(path, num_frames=min_frames, resolution=64)
        if got is None:
            return {"ok": False, "path": path, "reason": "undecodable"}
    m = motion_score(path, resolution=resolution)
    if m < 0:
        return {"ok": False, "path": path, "reason": "decode_fail"}
    w, h = info["width"], info["height"]
    return {
        "ok": True,
        "video_path": path,
        "prompt": prompt,
        "num_frames": info["frames"],
        "fps": round(info["fps"], 3),
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4) if h else 0.0,
        "motion": round(m, 6),
    }


def load_captions(path: str | None) -> dict[str, str]:
    """caption 表：json {relative_or_basename: caption} 或 csv(video_path,prompt)。"""
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row.get("video_path") or row.get("path") or row.get("video")
            if key:
                out[key] = row.get("prompt", "") or row.get("caption", "")
    return out


def materialize(rows: list[dict], out_dir: Path, num_frames: int, resolution: int,
                stride: int, crop: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        frames = read_clip(row["video_path"], num_frames, resolution, crop, stride, 0)
        if frames is None:
            continue
        name = Path(row["video_path"]).stem
        if fmt == "npy":
            np.save(out_dir / f"{name}.npy", frames)
        else:
            import imageio.v2 as imageio
            imageio.mimwrite(out_dir / f"{name}.mp4", frames, fps=8, quality=8)
        row["video_path"] = f"{name}.{'npy' if fmt == 'npy' else 'mp4'}"


def main() -> int:
    p = argparse.ArgumentParser(description="漫画/动画视频数据集索引与清洗")
    p.add_argument("--input_dir", required=True, help="源视频目录（递归扫描）")
    p.add_argument("--output_dir", required=True, help="输出目录（写 video_metadata.csv）")
    p.add_argument("--captions", default=None, help="captions json/csv（键为相对路径或文件名）")
    p.add_argument("--num_frames", type=int, default=16, help="最短帧数门槛 / materialize 长度")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--crop_method", choices=["center", "pad"], default="center")
    p.add_argument("--min_motion", type=float, default=0.008,
                   help="运动强度下限（0~1），滤掉近乎静止的镜头；设 0 关闭")
    p.add_argument("--max_videos", type=int, default=0, help="只处理前 N 个（快速验证）")
    p.add_argument("--default_prompt", default="anime style animation, 2d cel animation, comic style")
    p.add_argument("--materialize", choices=["none", "npy", "mp4"], default="none")
    p.add_argument("--materialize_limit", type=int, default=200)
    args = p.parse_args()

    src = Path(args.input_dir)
    videos = sorted(v for v in src.rglob("*") if v.suffix.lower() in VIDEO_EXT)
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        print(f"[error] 未在 {src} 找到视频文件")
        return 1

    caps = load_captions(args.captions)
    tasks = []
    for v in videos:
        rel = str(v.relative_to(src)).replace("\\", "/")
        prompt = caps.get(rel) or caps.get(v.name) or caps.get(v.stem) or args.default_prompt
        tasks.append((str(v), prompt, args.resolution, args.num_frames))

    print(f"[index] {len(tasks)} videos, workers={args.num_workers}")
    with mp.Pool(args.num_workers) as pool:
        results = pool.map(_worker, tasks, chunksize=8)

    good = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    drop_static = 0
    if args.min_motion > 0:
        kept = [r for r in good if r["motion"] >= args.min_motion]
        drop_static = len(good) - len(kept)
        good = kept

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_rel = out_dir.resolve()

    if args.materialize != "none":
        materialize(good[: args.materialize_limit], out_dir / "clips", args.num_frames,
                    args.resolution, 1, args.crop_method, args.materialize)
        root = out_dir / "clips"
    else:
        root = src
    for r in good:
        try:
            r["video_path"] = str(Path(r["video_path"]).resolve().relative_to(root.resolve()))
        except ValueError:
            r["video_path"] = str(Path(r["video_path"]).resolve())
        r.pop("ok", None)

    csv_path = out_dir / "video_metadata.csv"
    fields = ["video_path", "prompt", "num_frames", "fps", "width", "height", "aspect_ratio", "motion"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(good)

    motions = np.array([r["motion"] for r in good]) if good else np.array([0.0])
    stats = {
        "input_videos": len(videos), "kept": len(good), "dropped_decode": len(bad),
        "dropped_static": drop_static, "min_motion": args.min_motion,
        "motion_mean": float(motions.mean()), "motion_p50": float(np.percentile(motions, 50)),
        "motion_p90": float(np.percentile(motions, 90)),
        "video_folder": str(root), "csv_path": str(csv_path),
    }
    (out_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] kept {len(good)}/{len(videos)} (dropped decode={len(bad)}, static={drop_static})")
    print(f"[done] csv -> {csv_path}\n[done] stats -> {out_dir / 'prepare_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
