"""VAE latents 预计算（真实实现）。

把 ``data/prepare_comic_dataset.py`` 产出的 ``video_metadata.csv`` 里的每条视频
解码成 ``--num_frames`` 帧、过一次 SD 的 VAE，把 latent 落盘成

    <cache_dir>/<stem>_<num_frames>f_<resolution>.npz   （键名 ``latents``）

这个命名和 ``data/dataset.py::ComicVideoDataset`` 里的查找逻辑一一对应：

    cache_file = self.latent_cache / f"{Path(video_rel).stem}_{self.num_frames}f_{self.resolution}.npz"
    ...  z["latents"]

预计算的目的是在训练时省掉 VAE 前向，从而省下 8–10GB 显存（``dataset.py``
在 ``latents`` 存在时把它一并 collate 出来）。因此 latent 的形状必须和训练
侧的写法一致：``(C, T, H, W)``（``animatediff_sft.py`` 里是
``latents.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)``）。

工程细节
--------
* **多进程**：解码是 CPU/IO 密集，编码是 GPU 密集。这里用
  ``DataLoader(num_workers=N)`` 让 N 个进程并行解码，主进程串行跑 VAE，
  避免 N 个进程抢同一张卡。
* **断点续跑**：目标文件已存在即跳过（``--overwrite`` 可强制重算），
  所以中断后重跑只补缺失的部分。
* **统计**：结束打印 encoded/skipped/failed、耗时、缓存体积，并写
  ``<cache_dir>/precompute_stats.json``。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import frames_to_tensor, probe_video, read_clip  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="预计算 AnimateDiff 训练用的 VAE latents")
    p.add_argument("--csv_path", "--metadata_csv", dest="csv_path", required=True,
                   help="prepare_comic_dataset.py 产出的 video_metadata.csv")
    p.add_argument("--video_folder", required=True, help="video_path 列的根目录")
    p.add_argument("--cache_dir", required=True, help="latents 输出目录")
    p.add_argument("--model_id_or_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="SD 主模型（只用其中的 vae 子目录）")
    p.add_argument("--vae_path", default=None, help="直接指定 VAE 目录（覆盖 --model_id_or_path/subfolder）")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--crop_method", choices=["center", "pad"], default="center")
    p.add_argument("--latent_mode", choices=["mean", "sample"], default="mean",
                   help="mean=后验均值（确定性，推荐）；sample=后验采样（与在线训练一致但不可复现）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                   help="落盘精度；fp16 省一半磁盘，训练侧读取时会 .float()")
    p.add_argument("--compress", action="store_true", help="np.savez_compressed（慢一点，磁盘更省）")
    p.add_argument("--batch_size", type=int, default=4, help="每次送进 VAE 的 clip 数")
    p.add_argument("--encode_chunk", type=int, default=4,
                   help="单次 VAE 前向的帧数上限；显存不够就调小（512px fp32 建议 4）")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟测试用）")
    p.add_argument("--overwrite", action="store_true", help="已存在的缓存也重算")
    p.add_argument("--verify", action="store_true", help="写完后重新读回校验形状/统计量")
    return p.parse_args()


def cache_name(stem: str, num_frames: int, resolution: int) -> str:
    """与 data/dataset.py 的查找规则严格一致。"""
    return f"{stem}_{num_frames}f_{resolution}.npz"


class ClipDataset(Dataset):
    """只负责解码：返回 (T,3,H,W) float 张量 + 元信息，供 DataLoader 多进程并行。"""

    def __init__(self, rows: list[dict], video_folder: Path, num_frames: int, resolution: int,
                 crop_method: str, frame_stride: int):
        self.rows = rows
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.resolution = resolution
        self.crop_method = crop_method
        self.frame_stride = frame_stride

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        rel = row["video_path"]
        path = Path(rel)
        if not path.is_absolute():
            path = self.video_folder / rel
        frames = read_clip(path, self.num_frames, self.resolution, self.crop_method,
                           self.frame_stride, 0)
        if frames is None:
            # 解码失败：返回零张量并置 ok=False，由主进程统计后跳过
            return {
                "pixels": torch.zeros(self.num_frames, 3, self.resolution, self.resolution),
                "ok": False, "video_path": rel, "prompt": row.get("prompt", ""),
            }
        return {
            "pixels": frames_to_tensor(frames),
            "ok": True, "video_path": rel, "prompt": row.get("prompt", ""),
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        return {
            "pixels": torch.stack([b["pixels"] for b in batch], dim=0),
            "ok": [b["ok"] for b in batch],
            "video_paths": [b["video_path"] for b in batch],
            "prompts": [b["prompt"] for b in batch],
        }


def _worker_init(worker_id: int = 0) -> None:
    """限制每个解码进程的线程数，避免 N 个 worker × M 线程把 CPU 打满。"""
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    torch.set_num_threads(1)


def load_vae(args: argparse.Namespace, device: torch.device):
    from diffusers import AutoencoderKL

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_pretrained(args.model_id_or_path, subfolder="vae", torch_dtype=dtype)
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_batch(vae, pixels: torch.Tensor, device: torch.device, dtype: torch.dtype,
                 mode: str, chunk: int = 4) -> torch.Tensor:
    """``(B,T,3,H,W)`` in [-1,1] -> ``(B,C,T,h,w)``（已乘 scaling_factor）。

    帧按 ``chunk`` 切片过 VAE：512×512 的 SD VAE 解码器在 fp32 下，一次前向
    64 帧会吃掉 20GB+ 显存，切成几帧一次峰值就降下来了。
    """
    b, t = pixels.shape[0], pixels.shape[1]
    flat = pixels.reshape(b * t, *pixels.shape[2:])
    outs = []
    for i in range(0, flat.shape[0], max(1, chunk)):
        piece = flat[i:i + max(1, chunk)].to(device=device, dtype=dtype)
        dist = vae.encode(piece).latent_dist
        lat = dist.mean if mode == "mean" else dist.sample()
        outs.append((lat * vae.config.scaling_factor).to("cpu"))
    lat = torch.cat(outs, dim=0)
    lat = lat.reshape(b, t, *lat.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
    return lat


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_path"):
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"[error] {args.csv_path} 里没有带 video_path 的行")
        return 1

    video_folder = Path(args.video_folder)
    todo, skipped = [], 0
    for row in rows:
        stem = Path(row["video_path"]).stem
        if not args.overwrite and (cache_dir / cache_name(stem, args.num_frames, args.resolution)).exists():
            skipped += 1
            continue
        todo.append(row)

    print(f"[precompute] rows={len(rows)} todo={len(todo)} skipped(existing)={skipped} "
          f"device={device} dtype={args.dtype} frames={args.num_frames} res={args.resolution} "
          f"workers={args.num_workers} batch={args.batch_size}")
    if not todo:
        print("[precompute] 没有需要计算的样本，缓存已是最新")
        return 0

    vae = load_vae(args, device)
    dataset = ClipDataset(todo, video_folder, args.num_frames, args.resolution,
                          args.crop_method, args.frame_stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=ClipDataset.collate,
                        worker_init_fn=_worker_init, pin_memory=(device.type == "cuda"))

    save = np.savez_compressed if args.compress else np.savez
    encoded = failed = 0
    out_dtype = np.float16 if args.dtype == "fp16" else np.float32
    bytes_written = 0
    t0 = time.time()
    verify_report: list[dict] = []
    seen_stems: set[str] = set()

    for batch in loader:
        pixels = batch["pixels"]
        oks = batch["ok"]
        paths = batch["video_paths"]
        good = [i for i, ok in enumerate(oks) if ok]
        if not good:
            failed += len(paths)
            for p in paths:
                print(f"[warn] 解码失败: {p}")
            continue
        latents = encode_batch(vae, pixels[good], device, dtype, args.latent_mode,
                               args.encode_chunk)
        latents = latents.to("cpu").numpy().astype(out_dtype)
        for j, i in enumerate(good):
            rel = paths[i]
            stem = Path(rel).stem
            if stem in seen_stems:
                print(f"[warn] 文件名冲突（不同目录同名视频），跳过: {rel}")
                continue
            seen_stems.add(stem)
            dst = cache_dir / cache_name(stem, args.num_frames, args.resolution)
            save(dst, latents=latents[j])
            bytes_written += dst.stat().st_size
            encoded += 1
            if args.verify:
                with np.load(dst) as z:
                    arr = z["latents"]
                verify_report.append({
                    "file": dst.name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "finite": bool(np.isfinite(arr).all()),
                })
        n_bad = len(paths) - len(good)
        failed += n_bad
        for i, ok in enumerate(oks):
            if not ok:
                print(f"[warn] 解码失败: {paths[i]}")
        done = encoded + failed
        print(f"[precompute] {done}/{len(todo)} encoded={encoded} failed={failed} "
              f"elapsed={time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    stats = {
        "csv_path": str(args.csv_path), "video_folder": str(video_folder),
        "cache_dir": str(cache_dir), "num_frames": args.num_frames,
        "resolution": args.resolution, "frame_stride": args.frame_stride,
        "crop_method": args.crop_method, "latent_mode": args.latent_mode,
        "dtype": args.dtype, "rows": len(rows), "encoded": encoded,
        "skipped_existing": skipped, "failed": failed,
        "elapsed_s": round(elapsed, 2),
        "sec_per_item": round(elapsed / max(encoded, 1), 3),
        "cache_bytes": bytes_written, "cache_mb": round(bytes_written / 2**20, 2),
        "device": str(device),
    }
    (cache_dir / "precompute_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    latent_hw = (args.resolution // 8, args.resolution // 8)
    print(f"[done] encoded={encoded} skipped={skipped} failed={failed} in {elapsed:.1f}s "
          f"({stats['sec_per_item']}s/item)")
    print(f"[done] latent 形状 (C={4}, T={args.num_frames}, H={latent_hw[0]}, W={latent_hw[1]})，"
          f"缓存 {stats['cache_mb']} MB -> {cache_dir}")
    print(f"[done] stats -> {cache_dir / 'precompute_stats.json'}")
    if args.verify and verify_report:
        r0 = verify_report[0]
        print(f"[verify] {len(verify_report)} files ok; first: {r0['file']} shape={r0['shape']} "
              f"dtype={r0['dtype']} mean={r0['mean']:.4f} std={r0['std']:.4f} finite={r0['finite']}")
        if not all(r["finite"] for r in verify_report):
            print("[verify] !! 存在非有限值，缓存不可用")
            return 2
    return 0


def hard_exit(code: int) -> None:
    """绕开解释器退出阶段的 SIGABRT（实测环境缺陷，不是本脚本的问题）。

    在本实例上（diffusers 0.40.0 + torch 2.14.0.dev+cu130）只要在 CUDA 上跑过
    ``AutoencoderKL.encode``，解释器正常退出时就会 ``Aborted (core dumped)``
    （exit 134），最小复现：

        python -c "import torch; from diffusers import AutoencoderKL; \\
                   v=AutoencoderKL.from_pretrained(SD, subfolder='vae').eval().cuda(); \\
                   v.encode(torch.zeros(1,3,512,512,device='cuda'))"

    显式 del / synchronize / empty_cache 都无效，CPU 路径不复现；所有产物在
    崩溃前都已落盘，因此这里在所有工作完成后直接 ``os._exit``，
    让退出码如实反映脚本结果（不影响任何输出）。设 ``CVD_NO_HARD_EXIT=1`` 可关闭。
    """
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("CVD_NO_HARD_EXIT") == "1":
        raise SystemExit(code)
    os._exit(code)


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        hard_exit(1)
    hard_exit(_code)
