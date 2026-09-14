"""真实的视频生成评测：FVD / CLIP score / 时序一致性 / 运动量。

设计原则：**只报真实算出来的数**。任何一个指标因为权重下不下来、包缺失或
输入不满足要求而算不了时，输出 JSON 里写 ``"unavailable"`` + 原因，绝不用
占位数字糊过去。

输入
----
* ``--gen label=dir``（可重复）：待评测的生成结果目录，例如
  ``--gen baseline=runs/base/videos --gen lora=runs/lora/videos``；
* ``--real_dir``：真实视频参考目录（算 FVD 视频分布距离用，可选）。

指标
----
* **FVD** —— Kinetics-400 I3D（TorchScript，``flateon/FVD-I3D-torchscript``）
  倒数第二层 400 维 logits 作为特征，两组特征的高斯分布做 Fréchet 距离。
  FVD 需要足够多的视频才有统计意义：协方差在样本数 < 特征维数（400）时是
  奇异的，本脚本用 shrinkage 兜住数值，并在结果里带 ``reliable`` 标记
  （经验阈值：每组 ≥ 50 条视频）。
* **CLIP score** —— ``openai/clip-vit-base-patch32``，逐帧图像嵌入与提示词
  文本嵌入的平均余弦相似度（×100）。
* **时序一致性** —— 相邻帧 LPIPS（AlexNet 特征，低=更稳定）与像素级平均
  绝对差（低=更稳定）。
* **运动量** —— Farneback 稠密光流的平均幅度（像素）。

输出
----
``--output_json``：完整结果（含 unavailable 原因、逐视频明细、两两 FVD）；
``--output_csv``：逐视频行 + 汇总行；同时往 stdout 打人类可读表格。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import probe_video, read_clip  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
I3D_FEATURE_DIM = 400
I3D_MIN_FRAMES = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成视频的真实评测（FVD / CLIP / 时序 / 运动）")
    p.add_argument("--gen", action="append", default=[], metavar="LABEL=DIR",
                   help="待评测生成目录，可重复：--gen baseline=path --gen lora=path")
    p.add_argument("--real_dir", default=None, help="真实视频参考目录（可选，用于 FVD vs real）")
    p.add_argument("--output_json", default="paper/figures/eval_metrics.json")
    p.add_argument("--output_csv", default="paper/figures/eval_metrics.csv")
    p.add_argument("--metrics", default="clip,lpips,flow,fvd",
                   help="逗号分隔：clip,lpips,flow,fvd")
    p.add_argument("--resolution", type=int, default=256, help="指标统一的短边像素")
    p.add_argument("--temporal_frames", type=int, default=16, help="连续帧时序指标的帧数")
    p.add_argument("--clip_frames", type=int, default=8, help="CLIP 抽帧数")
    p.add_argument("--i3d_frames", type=int, default=10, help="FVD 每个视频抽帧数（I3D 最少 10）")
    p.add_argument("--i3d_stride", type=int, default=1)
    p.add_argument("--flow_step", type=int, default=1, help="光流相邻帧步长")
    p.add_argument("--limit", type=int, default=0, help="每个目录最多评测 N 条")
    p.add_argument("--prompts_json", default=None,
                   help="文件名/词干 -> 提示词 的 json（CLIP score 用）")
    p.add_argument("--prompt", default=None, help="所有视频共用的提示词（缺省时 CLIP 记 unavailable）")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--i3d_repo", default="flateon/FVD-I3D-torchscript")
    p.add_argument("--i3d_weights", default=None, help="本地 i3d_torchscript.pt（默认走 HF 下载）")
    p.add_argument("--i3d_batch", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 输入 / 解码
# --------------------------------------------------------------------------- #
def collect_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)


def load_frames(path: Path, num_frames: int, resolution: int, stride: int = 1) -> np.ndarray | None:
    """``(T,H,W,3)`` uint8；解码失败返回 None。"""
    return read_clip(path, num_frames=num_frames, resolution=resolution,
                     crop_method="center", frame_stride=stride, start_frame=0)


# --------------------------------------------------------------------------- #
# FVD：I3D 特征 + Fréchet 距离
# --------------------------------------------------------------------------- #
class I3DExtractor:
    """Kinetics-400 I3D（TorchScript）。权重走 HF，本地给 --i3d_weights 也行。"""

    def __init__(self, weights: str | None, device, repo: str, batch: int = 4):
        import torch

        self.torch = torch
        self.device = device
        self.batch = max(1, int(batch))
        if weights and Path(weights).is_file():
            path = Path(weights)
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=repo, filename="i3d_torchscript.pt"))
        self.weights_path = path
        self.model = torch.jit.load(str(path), map_location="cpu").eval().to(device)
        for prm in self.model.parameters():
            prm.requires_grad_(False)

    def _prep(self, clip: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(clip)).float() / 255.0   # (T,H,W,3) [0,1]
        x = x.permute(0, 3, 1, 2).unsqueeze(0)                             # (1,T,3,H,W)
        x = x * 2.0 - 1.0                                                  # 先缩放再插值（与参考实现一致）
        b, t, c, h, w = x.shape
        if h != 224 or w != 224:
            x = x.reshape(b * t, c, h, w)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                                align_corners=False)
            x = x.reshape(b, t, c, 224, 224)
        return x.permute(0, 2, 1, 3, 4).contiguous()                       # (B,3,T,224,224)

    def features(self, clips: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad(), torch.jit.fuser("none"):
            for i in range(0, len(clips), self.batch):
                chunk = clips[i:i + self.batch]
                if any(c.shape[0] < I3D_MIN_FRAMES for c in chunk):
                    raise ValueError(f"I3D 至少需要 {I3D_MIN_FRAMES} 帧")
                batch = torch.cat([self._prep(c) for c in chunk], dim=0).to(self.device)
                try:
                    feats = self.model(batch, rescale=False, resize=False, return_features=True)
                except TypeError:
                    feats = self.model(batch)
                    if isinstance(feats, (tuple, list)):
                        feats = feats[0]
                if feats.dim() == 1:
                    feats = feats.unsqueeze(0)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)


def gaussian_stats(features: np.ndarray, shrinkage: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    if features.shape[0] < 2:
        cov = np.eye(features.shape[1], dtype=np.float64)
    else:
        cov = np.cov(features, rowvar=False).astype(np.float64)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0], dtype=np.float64)
    return mu.astype(np.float64), cov


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    import scipy.linalg

    mu1, c1 = gaussian_stats(f1)
    mu2, c2 = gaussian_stats(f2)
    diff = mu1 - mu2
    # scipy >= 1.18 去掉了 sqrtm 的 disp 关键字参数，直接位置参数调用最兼容
    covmean = scipy.linalg.sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm 出现显著虚部，协方差矩阵数值不稳定")
        covmean = covmean.real
    value = float(diff @ diff + np.trace(c1 + c2 - 2.0 * covmean))
    return value


# --------------------------------------------------------------------------- #
# CLIP score
# --------------------------------------------------------------------------- #
class ClipScorer:
    def __init__(self, model_name: str, device):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception:  # 兼容老接口
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_name)
        self.projection_dim = int(getattr(self.model.config, "projection_dim", 0) or 0)

    @staticmethod
    def _embeds(out, expected_dim: int):
        """transformers 5.x 的 get_*_features 返回 BaseModelOutputWithPooling。

        投影后的嵌入在 ``pooler_output``（维度 = projection_dim），而不是
        ``last_hidden_state``（那是编码器隐状态）。这里两种都兼容，
        优先取维度对得上的那一个。
        """
        import torch

        if torch.is_tensor(out):
            return out
        candidates = []
        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            val = getattr(out, attr, None)
            if torch.is_tensor(val):
                candidates.append(val)
        if not candidates and isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            candidates.append(out[0])
        if not candidates:
            raise TypeError(f"无法从 {type(out)} 取出嵌入张量")
        if expected_dim:
            for c in candidates:
                if c.shape[-1] == expected_dim:
                    return c
            raise TypeError(f"没有维度为 {expected_dim} 的嵌入，候选维度={[tuple(c.shape) for c in candidates]}")
        return candidates[0]

    def score(self, frames: np.ndarray, prompt: str) -> float:
        torch = self.torch
        take = min(len(frames), 8)
        idx = np.linspace(0, len(frames) - 1, take).round().astype(int)
        images = [frames[i] for i in idx]
        with torch.no_grad():
            img_in = self.processor(images=images, return_tensors="pt")
            img_in = {k: v.to(self.device) for k, v in img_in.items()}
            img_emb = self._embeds(self.model.get_image_features(**img_in), self.projection_dim)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_in = self.processor(text=[prompt], return_tensors="pt", padding=True,
                                    truncation=True)
            txt_in = {k: v.to(self.device) for k, v in txt_in.items()}
            txt_emb = self._embeds(self.model.get_text_features(**txt_in), self.projection_dim)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cos = (img_emb @ txt_emb.T).squeeze(-1)
        return float(cos.mean())


# --------------------------------------------------------------------------- #
# 时序一致性 / 运动量
# --------------------------------------------------------------------------- #
class TemporalMetrics:
    def __init__(self, device):
        import torch
        import lpips

        self.torch = torch
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval().to(device)

    def lpips_consecutive(self, frames: np.ndarray) -> float:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(frames)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        vals = []
        with torch.no_grad():
            for i in range(len(x) - 1):
                a = x[i:i + 1].to(self.device)
                b = x[i + 1:i + 2].to(self.device)
                vals.append(float(self.lpips(a, b).mean()))
        return float(np.mean(vals)) if vals else float("nan")


def frame_diff_l1(frames: np.ndarray) -> float:
    x = frames.astype(np.float32) / 255.0
    if len(x) < 2:
        return float("nan")
    return float(np.abs(np.diff(x, axis=0)).mean())


def flow_magnitude(frames: np.ndarray, step: int = 1) -> float:
    import cv2

    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for i in range(0, len(gray) - step, step):
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + step], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
    return float(np.mean(mags)) if mags else float("nan")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_gen_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--gen 需要 LABEL=DIR 形式，收到: {spec}")
        label, path = spec.split("=", 1)
        path = Path(path)
        if not path.is_dir():
            raise SystemExit(f"--gen {label}: 目录不存在 {path}")
        out.append((label.strip(), path))
    return out


def per_video_metrics(video: Path, args, enabled: set[str], scorers: dict) -> dict:
    info = probe_video(video)
    row = {"video": video.name, "rel": str(video), "n_frames": info["frames"],
           "fps": round(info["fps"], 3), "width": info["width"], "height": info["height"],
           "notes": []}

    frames = load_frames(video, args.temporal_frames, args.resolution, stride=1)
    if frames is None:
        row["notes"].append("decode_failed")
        return row

    if "lpips" in enabled and scorers.get("temporal") is not None:
        try:
            row["temporal_lpips"] = round(scorers["temporal"].lpips_consecutive(frames), 4)
        except Exception as exc:
            row["notes"].append(f"lpips_failed:{type(exc).__name__}")
    row["frame_diff_l1"] = round(frame_diff_l1(frames), 5)

    if "flow" in enabled:
        try:
            mag = flow_magnitude(frames, max(1, args.flow_step))
            row["flow_mean_px"] = round(mag, 4)
            row["flow_mean_norm"] = round(mag / max(frames.shape[2], 1), 5)
        except Exception as exc:
            row["notes"].append(f"flow_failed:{type(exc).__name__}")

    if "clip" in enabled and scorers.get("clip") is not None:
        prompt = scorers["prompts"].get(video.name) or scorers["prompts"].get(video.stem) \
            or scorers["prompts"].get(str(video)) or args.prompt
        if prompt:
            try:
                clip_frames = frames[: args.clip_frames] if len(frames) >= args.clip_frames else frames
                cos = scorers["clip"].score(clip_frames, prompt)
                row["clip_score"] = round(100.0 * cos, 3)
                row["clip_score_clamped"] = round(100.0 * max(cos, 0.0), 3)
                row["prompt"] = prompt
            except Exception as exc:
                row["notes"].append(f"clip_failed:{type(exc).__name__}:{exc}")
        else:
            row["notes"].append("no_prompt")
    return row


def summarize(rows: list[dict]) -> dict:
    def agg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                and np.isfinite(r[key])]
        return round(float(np.mean(vals)), 5) if vals else None

    return {
        "n_videos": len(rows),
        "n_decoded": sum(1 for r in rows if "decode_failed" not in r.get("notes", [])),
        "clip_score": agg("clip_score"),
        "clip_score_clamped": agg("clip_score_clamped"),
        "n_clip": sum(1 for r in rows if isinstance(r.get("clip_score"), (int, float))),
        "temporal_lpips": agg("temporal_lpips"),
        "frame_diff_l1": agg("frame_diff_l1"),
        "flow_mean_px": agg("flow_mean_px"),
        "flow_mean_norm": agg("flow_mean_norm"),
    }


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    started = time.time()

    import torch
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    enabled = {m.strip() for m in args.metrics.split(",") if m.strip()}
    gen_specs = parse_gen_specs(args.gen or [])
    real_dir = Path(args.real_dir) if args.real_dir else None
    if not gen_specs and real_dir is None:
        raise SystemExit("至少要给一个 --gen LABEL=DIR 或 --real_dir")
    print(f"[eval] device={device} metrics={sorted(enabled)} "
          f"dirs={[l for l, _ in gen_specs]} real={real_dir}")

    prompts = {}
    if args.prompts_json and Path(args.prompts_json).is_file():
        prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        print(f"[eval] prompts loaded: {len(prompts)}")

    unavailable: dict[str, str | None] = {"fvd": None, "clip": None, "lpips": None, "flow": None}
    scorers: dict = {"prompts": prompts}

    if "lpips" in enabled:
        try:
            scorers["temporal"] = TemporalMetrics(device)
            print("[eval] LPIPS(AlexNet) 就绪")
        except Exception as exc:
            unavailable["lpips"] = f"{type(exc).__name__}: {exc}"
            print(f"[eval] LPIPS 不可用 -> {unavailable['lpips']}")

    if "clip" in enabled:
        if not args.prompt and not prompts:
            unavailable["clip"] = "没有提供任何提示词（--prompt / --prompts_json）"
            print(f"[eval] CLIP score 跳过 -> {unavailable['clip']}")
        else:
            try:
                scorers["clip"] = ClipScorer(args.clip_model, device)
                print(f"[eval] CLIP({args.clip_model}) 就绪")
            except Exception as exc:
                unavailable["clip"] = f"{type(exc).__name__}: {exc}"
                print(f"[eval] CLIP 不可用 -> {unavailable['clip']}")

    # ---- 收集视频 ----
    dirs: list[tuple[str, Path, list[Path]]] = []
    for label, path in gen_specs:
        vids = collect_videos(path)
        if args.limit:
            vids = vids[: args.limit]
        dirs.append((label, path, vids))
        print(f"[eval] {label}: {len(vids)} videos in {path}")
    real_vids: list[Path] = []
    if real_dir is not None:
        real_vids = collect_videos(real_dir)
        if args.limit:
            real_vids = real_vids[: args.limit]
        print(f"[eval] real: {len(real_vids)} videos in {real_dir}")

    # ---- 逐视频指标 ----
    per_video: dict[str, list[dict]] = {}
    for label, _, vids in dirs:
        rows = []
        for v in vids:
            row = per_video_metrics(v, args, enabled, scorers)
            row["label"] = label
            rows.append(row)
            print(f"[eval] {label}/{v.name}: " + " ".join(
                f"{k}={row[k]}" for k in ("clip_score", "temporal_lpips", "flow_mean_px")
                if k in row) + (f" notes={row['notes']}" if row["notes"] else ""))
        per_video[label] = rows

    # ---- FVD ----
    fvd_results: dict[str, dict] = {}
    fvd_features: dict[str, tuple[np.ndarray, list[str]]] = {}
    if "fvd" in enabled:
        need = [(l, p, v) for l, p, v in dirs] + ([("real", real_dir, real_vids)] if real_dir else [])
        clips: dict[str, list[np.ndarray]] = {}
        reason = None
        try:
            extractor = I3DExtractor(args.i3d_weights, device, args.i3d_repo, args.i3d_batch)
            print(f"[fvd] I3D 权重 {extractor.weights_path}")
        except Exception as exc:
            extractor = None
            reason = f"I3D 权重/模型不可用: {type(exc).__name__}: {exc}"
            print(f"[fvd] 不可用 -> {reason}")
        if extractor is not None:
            for label, _, vids in need:
                got = []
                for v in vids:
                    fr = load_frames(v, args.i3d_frames, args.resolution, stride=args.i3d_stride)
                    if fr is not None and len(fr) >= I3D_MIN_FRAMES:
                        got.append(fr)
                clips[label] = got
                print(f"[fvd] {label}: {len(got)}/{len(vids)} 条满足 I3D 输入要求")
                if not got:
                    reason = reason or f"{label} 没有可用于 FVD 的视频"
            try:
                for label, got in clips.items():
                    if got:
                        fvd_features[label] = (extractor.features(got), [c.shape[0] for c in got])
            except Exception as exc:
                reason = f"I3D 前向失败: {type(exc).__name__}: {exc}"
                fvd_features = {}
                print(f"[fvd] 不可用 -> {reason}")
        unavailable["fvd"] = reason
        if fvd_features and real_dir is not None and "real" in fvd_features:
            for label, _ in gen_specs:
                if label in fvd_features:
                    fvd_results[f"{label}_vs_real"] = {
                        "fvd": round(frechet_distance(fvd_features[label][0],
                                                      fvd_features["real"][0]), 4),
                        "n_a": fvd_features[label][0].shape[0],
                        "n_b": fvd_features["real"][0].shape[0],
                    }
        labels = [l for l, _ in gen_specs if l in fvd_features]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                fvd_results[f"{a}_vs_{b}"] = {
                    "fvd": round(frechet_distance(fvd_features[a][0], fvd_features[b][0]), 4),
                    "n_a": fvd_features[a][0].shape[0], "n_b": fvd_features[b][0].shape[0],
                }
        for key, res in fvd_results.items():
            res["feature_dim"] = I3D_FEATURE_DIM
            res["reliable"] = min(res["n_a"], res["n_b"]) >= 50
            res["reliability_note"] = (
                f"FVD 经验上需要每组 ≥50 条视频；当前 {res['n_a']} vs {res['n_b']}，"
                "点估计仅供方向性参考" if not res["reliable"] else "样本量足够")

    # ---- 汇总 / 输出 ----
    summary = {label: summarize(rows) for label, rows in per_video.items()}
    for label, s in summary.items():
        for key, res in fvd_results.items():
            if key.startswith(f"{label}_vs_"):
                s[key] = res["fvd"]

    result = {
        "config": vars(args),
        "device": str(device),
        "elapsed_s": round(time.time() - started, 2),
        "unavailable": unavailable,
        "dirs": {
            label: {"path": str(path), "n_videos_found": len(vids), **summary.get(label, {})}
            for label, path, vids in dirs
        },
        "real": ({"path": str(real_dir), "n_videos_found": len(real_vids)} if real_dir else None),
        "fvd": fvd_results,
        "per_video": per_video,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "label", "video", "n_videos", "n_frames", "fps", "width", "height",
            "clip_score", "clip_score_clamped", "temporal_lpips", "frame_diff_l1",
            "flow_mean_px", "flow_mean_norm", "fvd", "notes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for label, rows in per_video.items():
            for row in rows:
                w.writerow({"kind": "video", **{k: row.get(k, "") for k in cols if k != "kind"},
                            "notes": ";".join(row.get("notes", []))})
        for label, s in summary.items():
            w.writerow({"kind": "summary", "label": label, **{
                k: (s.get(k) if s.get(k) is not None else "") for k in
                ("n_videos", "clip_score", "clip_score_clamped", "temporal_lpips",
                 "frame_diff_l1", "flow_mean_px", "flow_mean_norm")}})
        for key, res in fvd_results.items():
            w.writerow({"kind": "fvd", "label": key, "fvd": res["fvd"],
                        "notes": f"n={res['n_a']}vs{res['n_b']} reliable={res['reliable']}"})

    # ---- 人类可读表格 ----
    print("\n=== 汇总 ===")
    header = f"{'label':<14}{'n':>4}{'clip':>9}{'lpips':>9}{'framediff':>11}{'flow_px':>9}"
    print(header)
    print("-" * len(header))
    for label, s in summary.items():
        def fmt(v):
            return "  n/a   " if v is None else f"{v:>8.3f} "
        print(f"{label:<14}{s['n_videos']:>4}{fmt(s['clip_score'])}{fmt(s['temporal_lpips'])}"
              f"{fmt(s['frame_diff_l1'])}{fmt(s['flow_mean_px'])}")
    if fvd_results:
        print("\n=== FVD ===")
        for key, res in fvd_results.items():
            flag = "" if res["reliable"] else "  (样本量不足，仅方向性参考)"
            print(f"{key:<28} {res['fvd']:>12.2f}  n={res['n_a']}vs{res['n_b']}{flag}")
    elif unavailable["fvd"]:
        print(f"\n=== FVD ===\nunavailable: {unavailable['fvd']}")
    for name, reason in unavailable.items():
        if reason:
            print(f"[warn] {name}: unavailable -> {reason}")
    print(f"\n[done] json -> {out_json}\n[done] csv  -> {out_csv}\n[done] {result['elapsed_s']}s")
    return 0


def hard_exit(code: int) -> None:
    """绕开解释器退出阶段的 SIGABRT（实测环境缺陷，不是本脚本的问题）。

    该实例上（diffusers/torch cu130 nightly）只要在 CUDA 上跑过 VAE 前向，
    解释器正常退出时就会 ``Aborted (core dumped)``（exit 134）；JSON/CSV 在崩溃
    前已写完，因此这里直接 ``os._exit`` 让退出码如实反映脚本结果。
    设 ``CVD_NO_HARD_EXIT=1`` 可关闭。
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
