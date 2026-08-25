"""AnimateDiff 视频数据集（动态分辨率处理）。

视频数据宽高比多样（16:9 / 4:3 / 9:16），直接缩放到 512×512 会严重畸变。
本模块用 center-crop / pad 统一为正方形，并把原始宽高比写入 metadata CSV 备查。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2  # noqa: F401  (经 opencv-python 读取视频帧)
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def center_crop(frame, target_w: int, target_h: int):
    """把一帧中心裁剪到 target_w×target_h。"""
    h, w = frame.shape[:2]
    x0 = max(0, (w - target_w) // 2)
    y0 = max(0, (h - target_h) // 2)
    return frame[y0:y0 + target_h, x0:x0 + target_w]


def pad_to_square(frame, target_w: int, target_h: int):
    """把一帧用黑边补到 target_w×target_h（保全构图）。"""
    import numpy as np

    h, w = frame.shape[:2]
    top = max(0, (target_h - h) // 2)
    bottom = target_h - h - top
    left = max(0, (target_w - w) // 2)
    right = target_w - w - left
    return np.pad(frame, ((top, bottom), (left, right), (0, 0)))


def process_video(video_path, target_res: int = 512, crop_method: str = "center"):
    """读取视频，按 crop_method 统一为正方形帧序列；返回原始宽高比。

    metadata['original_aspect_ratio'] = original_w / original_h
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if crop_method == "center":
            frames.append(center_crop(frame, target_res, target_res))
        elif crop_method == "pad":
            frames.append(pad_to_square(frame, target_res, target_res))
    cap.release()
    if not frames:
        return None, 0.0
    original_w, original_h = frames[0].shape[1], frames[0].shape[0]
    return frames, original_w / original_h


class ComicVideoDataset(Dataset):
    """从 metadata CSV 加载 (video_path, prompt)，返回处理后的视频帧 + 提示词。"""

    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        num_frames: int = 16,
        resolution: int = 512,
        crop_method: str = "center",
        augment: dict | None = None,
    ):
        self.video_folder = Path(video_folder)
        self.num_frames = num_frames
        self.resolution = resolution
        self.crop_method = crop_method
        self.augment = augment or {}
        self.items: list[tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.items.append((row["video_path"], row.get("prompt", "")))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        video_path, prompt = self.items[idx]
        frames, ar = process_video(
            Path(self.video_folder) / video_path, self.resolution, self.crop_method
        )
        if frames is None or len(frames) < self.num_frames:
            # 帧数不足：循环补到最后帧
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        frames = frames[:self.num_frames]
        # 骨架：真实实现应由数据管线把 RGB 帧转成 (T,C,H,W) 张量并归一化到 [-1,1]
        return {
            "pixel_values": torch.zeros((self.num_frames, 3, self.resolution, self.resolution)),
            "prompt": prompt,
            "aspect_ratio": ar,
        }
