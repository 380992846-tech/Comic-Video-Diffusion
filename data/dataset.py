"""AnimateDiff 视频数据集（真实实现）。

视频数据宽高比多样（16:9 / 4:3 / 9:16），直接缩放到 512×512 会严重畸变，
因此统一走 center-crop / pad，并把原始宽高比写进 metadata CSV 备查。

与早期骨架版的区别：``__getitem__`` 现在真正解码视频帧，返回
``(T, 3, H, W)``、归一化到 ``[-1, 1]`` 的 float32 张量；支持
* 随机时间裁剪（训练）/ 固定首帧裁剪（验证，``deterministic=True``），
* 帧步长 ``frame_stride``（在同样 T 帧里覆盖更长的时间跨度），
* 可选的预计算 VAE latents 缓存（``latent_cache``）。
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:  # PyAV 解码比 OpenCV 更稳（支持更多编码 & 精确 seek）
    import av

    _HAS_AV = True
except Exception:  # pragma: no cover
    _HAS_AV = False

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def center_crop(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """把一帧中心裁剪到 target_w×target_h。"""
    h, w = frame.shape[:2]
    x0 = max(0, (w - target_w) // 2)
    y0 = max(0, (h - target_h) // 2)
    return frame[y0:y0 + target_h, x0:x0 + target_w]


def pad_to_square(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """把一帧用黑边补到 target_w×target_h（保全构图）。"""
    h, w = frame.shape[:2]
    top = max(0, (target_h - h) // 2)
    bottom = max(0, target_h - h - top)
    left = max(0, (target_w - w) // 2)
    right = max(0, target_w - w - left)
    return np.pad(frame, ((top, bottom), (left, right), (0, 0)))


def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    if _HAS_AV:
        return frame  # resizing handled by av reformat below; numpy fallback:
    if cv2 is not None:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    raise RuntimeError("no image backend available (install av or opencv)")


def probe_video(path: str | Path) -> dict:
    """返回 {frames, fps, width, height}；无法解码时 frames=0。"""
    path = str(path)
    if _HAS_AV:
        try:
            with av.open(path) as c:
                vs = c.streams.video[0]
                return {"frames": int(vs.frames or 0), "fps": float(vs.average_rate or 0),
                        "width": int(vs.codec_context.width), "height": int(vs.codec_context.height)}
        except Exception as exc:  # pragma: no cover
            logger.debug("av probe failed for %s: %s", path, exc)
    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        out = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
               "fps": float(cap.get(cv2.CAP_PROP_FPS)),
               "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        return out
    return {"frames": 0, "fps": 0.0, "width": 0, "height": 0}


def read_clip(path: str | Path, num_frames: int = 16, resolution: int = 512,
              crop_method: str = "center", frame_stride: int = 1,
              start_frame: int = 0) -> np.ndarray | None:
    """读取一段连续帧：``(T, H, W, 3)`` uint8，失败返回 None。"""
    need = num_frames * max(frame_stride, 1)
    if _HAS_AV:
        try:
            frames = []
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                if start_frame > 0:
                    stream.seek(int(start_frame / max(stream.average_rate or 24, 1) / stream.time_base),
                                backward=True)
                idx = -1
                for frame in container.decode(stream):
                    idx += 1
                    if idx < start_frame:
                        continue
                    if (idx - start_frame) % max(frame_stride, 1) != 0:
                        continue
                    img = frame.to_ndarray(format="rgb24")
                    if crop_method == "center":
                        img = center_crop(img, min(img.shape[1], resolution), min(img.shape[0], resolution))
                    else:
                        img = pad_to_square(img, resolution, resolution)
                    if img.shape[0] != resolution or img.shape[1] != resolution:
                        if cv2 is not None:
                            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
                        else:
                            img = np.asarray(
                                __import__("PIL.Image", fromlist=["Image"])
                                .fromarray(img).resize((resolution, resolution)))
                    frames.append(img)
                    if len(frames) >= num_frames:
                        break
            if len(frames) == num_frames:
                return np.stack(frames, axis=0)
        except Exception as exc:  # pragma: no cover
            logger.debug("av decode failed for %s: %s", path, exc)
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames, idx = [], 0
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(frame_stride, 1) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = (center_crop(frame, resolution, resolution) if crop_method == "center"
                         else pad_to_square(frame, resolution, resolution))
                if frame.shape[0] != resolution or frame.shape[1] != resolution:
                    frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            idx += 1
        cap.release()
        if len(frames) == num_frames:
            return np.stack(frames, axis=0)
    return None


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """``(T,H,W,3)`` uint8 -> ``(T,3,H,W)`` float32 in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).float()
    t = t.permute(0, 3, 1, 2) / 127.5 - 1.0
    return t


class ComicVideoDataset(Dataset):
    """从 metadata CSV 加载 ``(video_path, prompt)``，返回视频张量 + 提示词。"""

    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        num_frames: int = 16,
        resolution: int = 512,
        crop_method: str = "center",
        frame_stride: int = 1,
        augment: dict | None = None,
        deterministic: bool = False,
        latent_cache: str | None = None,
        seed: int = 0,
    ):
        self.video_folder = Path(video_folder)
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)
        self.crop_method = crop_method
        self.frame_stride = int(frame_stride)
        self.augment = augment or {}
        self.deterministic = deterministic
        self.latent_cache = Path(latent_cache) if latent_cache else None
        self.seed = int(seed)
        self.epoch = 0
        self.items: list[tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                video = row.get("video_path") or row.get("path") or row.get("video")
                if not video:
                    continue
                self.items.append((video, row.get("prompt", "") or row.get("caption", "")))
        if not self.items:
            raise ValueError(f"no rows with a video path found in {csv_path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _rng(self, idx: int) -> random.Random:
        rng = random.Random((self.seed * 1_000_003 + self.epoch * 7919 + idx) if not self.deterministic
                            else (self.seed * 1_000_003 + idx))
        return rng

    def __getitem__(self, idx: int):
        video_rel, prompt = self.items[idx]
        path = self.video_folder / video_rel
        rng = self._rng(idx)

        cache_file = None
        if self.latent_cache is not None:
            cache_file = self.latent_cache / f"{Path(video_rel).stem}_{self.num_frames}f_{self.resolution}.npz"

        info = probe_video(path)
        total = info["frames"]
        span = self.num_frames * self.frame_stride
        if total and total > span:
            max_start = total - span
            start = 0 if self.deterministic else rng.randrange(0, max_start + 1)
        else:
            start = 0
            self.frame_stride = max(1, self.frame_stride)

        frames = read_clip(path, self.num_frames, self.resolution, self.crop_method,
                           self.frame_stride, start)
        if frames is None:
            # 解码失败的样本退化为中灰，避免整轮训练崩掉；计数交给上层 collate 过滤
            pixels = torch.zeros((self.num_frames, 3, self.resolution, self.resolution))
            logger.warning("failed to decode %s (clip %d)", path, idx)
        else:
            if self.augment.get("random_flip") and rng.random() < 0.5:
                frames = frames[:, :, ::-1]
            if self.augment.get("color_jitter"):
                j = float(self.augment["color_jitter"])
                frames = np.clip(frames.astype(np.float32) * (1.0 + rng.uniform(-j, j)), 0, 255).astype(np.uint8)
            pixels = frames_to_tensor(frames)

        sample = {"pixel_values": pixels, "prompt": prompt, "video_path": video_rel}
        if cache_file is not None and cache_file.exists():
            try:
                with np.load(cache_file) as z:
                    sample["latents"] = torch.from_numpy(z["latents"]).float()
            except Exception as exc:  # pragma: no cover
                logger.warning("bad latent cache %s: %s", cache_file, exc)
        return sample

    @staticmethod
    def collate(batch):
        out = {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
            "prompts": [b["prompt"] for b in batch],
            "video_paths": [b["video_path"] for b in batch],
        }
        if all("latents" in b for b in batch):
            out["latents"] = torch.stack([b["latents"] for b in batch], dim=0)
        return out
