"""用**固定的验证集 + 固定的噪声/时间步**评估每个臂的 checkpoint。

为什么需要它：训练日志里的 loss 是"每步随机抽一个 timestep、batch=1"的 MSE，
方差极大（实测同臂内 step 间在 0.05~0.09 抖动），用它比较不同臂几乎没有统计意义。
本脚本对每个臂用同一批验证 clip、同一组 timestep、同一组噪声张量算 MSE，
给出低方差的确定性指标（可复现：完全由 seed 决定）。

用法：
    python scripts/eval_val_loss.py --ckpt /root/autodl-tmp/runs/e1_lora_r32/checkpoints/final.safetensors \
      --csv_path .../video_metadata.csv --video_folder .../anime_clips_all \
      --output_json results/val_loss_e1_lora_r32.json
    # base（不加载任何 checkpoint，只跑预训练 motion adapter）
    python scripts/eval_val_loss.py --output_json results/val_loss_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import ComicVideoDataset  # noqa: E402
from models.dac_attention import wrap_motion_attention  # noqa: E402
from models.lora import inject_lora, load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="省略则评估预训练 base（不加载任何权重）")
    p.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/sd15")
    p.add_argument("--motion_adapter_id_or_path", default="/root/autodl-tmp/models/adapter-v15-2")
    p.add_argument("--csv_path", default="/root/autodl-tmp/data/anime_clips/video_metadata.csv")
    p.add_argument("--video_folder", default="/root/autodl-tmp/data/anime_clips_all")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_clips", type=int, default=20, help="验证 clip 数（固定取一段切片）")
    p.add_argument("--clips_start", type=int, default=324,
                   help="验证切片起始下标；默认取数据集末尾（对 e3 子集臂是真正 hold-out）")
    p.add_argument("--num_timesteps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from diffusers import AutoencoderKL, DDPMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id_or_path, subfolder="text_encoder",
                                                 torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.model_id_or_path, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model_id_or_path, subfolder="unet", torch_dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(args.model_id_or_path, subfolder="scheduler")
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_id_or_path, torch_dtype=dtype)
    model = UNetMotionModel.from_unet2d(unet, adapter)
    model.requires_grad_(False).to(device).eval()

    meta: dict = {}
    if args.ckpt:
        blob_meta_file = Path(args.ckpt).with_suffix(".json")
        if blob_meta_file.exists():
            meta = json.loads(blob_meta_file.read_text(encoding="utf-8"))
        if meta.get("use_dac"):
            wrap_motion_attention(model, tau_threshold=2.0, verbose=False)
        mode = meta.get("mode", "lora")
        if mode == "lora":
            inject_lora(model, "motion_modules", ("to_q", "to_k", "to_v", "to_out.0"),
                        int(meta.get("lora_rank", 32)), int(meta.get("lora_alpha", 64)))
        model.to(device)
        res = load_checkpoint(args.ckpt, model)
        print(f"[ckpt] {args.ckpt} loaded={res['loaded']} unmatched={len(res['unmatched'])} mode={mode}")
        if res["unmatched"]:
            raise RuntimeError(f"{len(res['unmatched'])} unmatched keys")

    ds = ComicVideoDataset(csv_path=args.csv_path, video_folder=args.video_folder,
                           num_frames=args.num_frames, resolution=args.resolution, seed=args.seed)
    ds.deterministic = True
    start = min(args.clips_start, max(len(ds.items) - args.num_clips, 0))
    ds.items = ds.items[start: start + args.num_clips]
    print(f"[val] clips={len(ds.items)} (slice start={start}) frames={args.num_frames} res={args.resolution}")

    # 固定时间步：均匀覆盖 0..999
    timesteps = torch.linspace(0, scheduler.config.num_train_timesteps - 1, args.num_timesteps).long()

    per_clip, per_t = [], []
    for idx in range(len(ds.items)):
        sample = ds[idx]
        pixels = sample["pixel_values"].unsqueeze(0).to(device, dtype)
        with torch.no_grad():
            flat = pixels.reshape(args.num_frames, *pixels.shape[2:]).to(dtype)
            lat = vae.encode(flat).latent_dist.sample() * vae.config.scaling_factor
            lat = lat.reshape(1, args.num_frames, *lat.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
            ids = tokenizer([sample["prompt"]], padding="max_length",
                            max_length=tokenizer.model_max_length, truncation=True,
                            return_tensors="pt").input_ids.to(device)
            emb = text_encoder(ids)[0].to(dtype).repeat_interleave(args.num_frames, dim=0)

        losses = []
        for j, t in enumerate(timesteps):
            g = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + idx * 1009 + j)
            noise = torch.randn(lat.shape, generator=g, dtype=torch.float32).to(device, dtype)
            tt = t.to(device).repeat(1)
            noisy = scheduler.add_noise(lat, noise, tt)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                pred = model(noisy, tt, encoder_hidden_states=emb, return_dict=False)[0]
            if pred.shape != noise.shape and pred.shape[1] == noise.shape[2]:
                pred = pred.permute(0, 2, 1, 3, 4)
            losses.append(float(F.mse_loss(pred.float(), noise.float())))
        per_clip.append(sum(losses) / len(losses))
        per_t.append(losses)
        if (idx + 1) % 5 == 0:
            print(f"  clip {idx + 1}/{len(ds.items)} mean={per_clip[-1]:.5f}")

    import statistics as st

    result = {
        "ckpt": args.ckpt or "base(pretrained motion adapter)",
        "mode": meta.get("mode", "base"),
        "use_dac": bool(meta.get("use_dac", False)),
        "lora_rank": meta.get("lora_rank", 0),
        "samples": meta.get("samples", 0),
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "num_clips": len(per_clip),
        "num_timesteps": args.num_timesteps,
        "seed": args.seed,
        "clips_start": start,
        "val_mse_mean": st.fmean(per_clip),
        "val_mse_std_across_clips": st.pstdev(per_clip) if len(per_clip) > 1 else 0.0,
        "val_mse_min": min(per_clip),
        "val_mse_max": max(per_clip),
        "per_clip_mse": per_clip,
        "per_timestep_mse_mean": [st.fmean([pt[j] for pt in per_t]) for j in range(args.num_timesteps)],
        "timesteps": timesteps.tolist(),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))
    print(f"[ok] wrote {out}")
    os._exit(0)  # 该实例 CUDA 之后正常退出会 core dump


if __name__ == "__main__":
    main()
