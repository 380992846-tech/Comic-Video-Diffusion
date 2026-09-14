"""生成定性对比图：各臂在同一 prompt / 同一 seed 下的等间隔帧网格。

用法：
    python scripts/make_qualitative_grid.py \
      --gen base=/root/autodl-tmp/samples/base \
      --gen e1_lora_r32=/root/autodl-tmp/samples/e1_lora_r32 \
      --gen e4_dac_r32=/root/autodl-tmp/samples/e4_dac_r32 \
      --output paper/figures/qualitative_grid.png --frames 4

每个目录里按文件名排序取第一个 mp4（各臂出片脚本用的前缀与 prompt 顺序一致，
所以同一 index 的视频是同一 prompt / 同一 seed）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import read_clip  # noqa: E402


def load_video_frames(path: Path, n: int, resolution: int = 256) -> np.ndarray | None:
    """从视频里等间隔抽 n 帧（用解码器真实读，不用缩略图）。"""
    import av

    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    picked = [frames[i] for i in idx]
    # 统一到 resolution
    out = []
    for fr in picked:
        h, w = fr.shape[:2]
        s = resolution / min(h, w)
        if s != 1.0:
            import cv2
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        y0 = max(0, (fr.shape[0] - resolution) // 2)
        x0 = max(0, (fr.shape[1] - resolution) // 2)
        out.append(fr[y0:y0 + resolution, x0:x0 + resolution])
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--output", default="paper/figures/qualitative_grid.png")
    ap.add_argument("--frames", type=int, default=4, help="每行抽几帧")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--video_index", type=int, default=0, help="取目录里排序后的第几个视频")
    args = ap.parse_args()

    entries = []
    for spec in args.gen:
        if "=" not in spec:
            print(f"[skip] bad --gen spec: {spec}")
            continue
        label, d = spec.split("=", 1)
        files = sorted(Path(d).glob("*.mp4"))
        if not files:
            print(f"[skip] {label}: no mp4 in {d}")
            continue
        idx = min(args.video_index, len(files) - 1)
        frames = load_video_frames(files[idx], args.frames, args.resolution)
        if frames is None:
            print(f"[skip] {label}: decode failed for {files[idx]}")
            continue
        entries.append((label, files[idx].name, frames))
        print(f"[ok] {label}: {files[idx].name} -> {frames.shape}")

    if not entries:
        print("[error] nothing to plot")
        return 1

    rows, cols = len(entries), args.frames
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    for r, (label, fname, frames) in enumerate(entries):
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(frames[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                ax.set_title(f"frame {c + 1}/{cols}", fontsize=9)
    fig.suptitle("Qualitative comparison - same prompt & seed across arms", fontsize=11)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
