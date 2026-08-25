"""WebVid 数据集工具（骨架）。

WebVid-10M 为通用视频-字幕对，可作为漫画风格预训练补充数据。
真实实现建议：用 huggingface datasets 加载 webvid，或按官方 ts 分片解析。
"""

from __future__ import annotations

import csv
from pathlib import Path


def build_metadata(video_dir: str, out_csv: str, max_clips: int = 10000) -> int:
    """扫描 video_dir 下 mp4，生成 (video_path, prompt, duration) metadata CSV。"""
    rows = []
    for v in sorted(Path(video_dir).rglob("*.mp4"))[:max_clips]:
        rows.append({"video_path": v.name, "prompt": "", "duration": ""})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "prompt", "duration"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)
