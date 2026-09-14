"""单卡 AnimateDiff 动模块 LoRA 微调（真实训练脚本）。

训练目标：把 motion adapter（``guoyww/animatediff-motion-adapter-v1-5-2``，
即旧文档里的 ``Shanghai_AI_Laboratory/...``）适配到漫画/动画视频域。
冻结 SD1.5 主干（VAE / text encoder / UNet 空间层），只训练：

* ``--train_mode lora``    —— motion module 内注意力的 LoRA（默认，E2 秩消融对象）
* ``--train_mode partial`` —— motion module 尾部 N 个 block 全参数（E1 的对照臂）
* ``--train_mode motion``  —— 整个 motion adapter 全参数（E1 的强对照臂，显存最贵）

可选 ``--use_dac`` 把 DAC 温度化时序注意力（``models/dac_attention.py``）接到
motion module 的 frame-axis self-attention 上，并追加集中度损失：

    L = L_diff + dac_loss_weight * mean_h lambda_h * (C(A_h) - mu_h)^2

所有指标以 CSV + TensorBoard 双写；CSV 是论文数据的唯一来源（禁止手改）。
"""

from __future__ import annotations

import os as _os

# 必须早于 torch 导入：8 个 DataLoader worker × 128 个 OMP 线程会造成严重超订，
# 实测把 step 时间从 0.84s 拖到 ~2s。
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import ComicVideoDataset  # noqa: E402
from models.dac_attention import collect_dac_losses, dac_head_statistics, wrap_motion_attention  # noqa: E402
from models.lora import inject_lora, save_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("animatediff_sft")

DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AnimateDiff comic-style motion-module fine-tuning")
    p.add_argument("--model_id_or_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--motion_adapter_id_or_path",
                   default="guoyww/animatediff-motion-adapter-v1-5-2")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--video_folder", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default=None, help="记录用实验名（默认取 output_dir 名）")

    # 训练目标
    p.add_argument("--sft_type", default="lora", choices=["lora", "partial", "motion", "full"])
    p.add_argument("--train_mode", default=None, help="同 --sft_type，优先使用")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--partial_last_n", type=int, default=4,
                   help="--train_mode partial 时解冻的 motion block 数（尾部）")

    # DAC
    p.add_argument("--use_dac", action="store_true")
    p.add_argument("--dac_loss_weight", type=float, default=0.01)
    p.add_argument("--dac_tau_init", type=float, default=1.0)
    p.add_argument("--dac_tau_threshold", type=float, default=2.0)
    p.add_argument("--dac_mu", type=float, default=0.5)
    p.add_argument("--dac_lambda_init", type=float, default=0.1)

    # 数据
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--crop_method", default="center", choices=["center", "pad"])
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--color_jitter", type=float, default=0.0)
    p.add_argument("--latent_cache", default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0, help="只用前 N 个样本（E3 数据效率）")

    # 优化
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler_type", default="cosine", choices=["cosine", "constant", "linear"])
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_train_steps", type=int, default=0, help=">0 时按固定步数训练（等预算对照）")
    p.add_argument("--gradient_checkpointing", type=lambda s: str(s).lower() in ("1", "true", "yes"),
                   default=True)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    # IO
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--report_to", default="csv,tensorboard")
    p.add_argument("--wandb_project", default="comic_video_diffusion")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--dry_run", action="store_true", help="只跑 3 步并打印显存，用于冒烟")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVLogger:
    """追加式指标日志：论文数据源。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys: list[str] | None = None
        self.fh = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = None

    def log(self, row: dict) -> None:
        row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
        if self.writer is None:
            self.keys = list(row.keys())
            self.writer = csv.DictWriter(self.fh, fieldnames=self.keys, extrasaction="ignore")
            if self.fh.tell() == 0:
                self.writer.writeheader()
        self.writer.writerow({k: row.get(k, "") for k in self.keys})
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def lr_lambda_factory(args, total_steps: int):
    warmup = max(args.warmup_steps, 0)
    if args.lr_scheduler_type == "constant":
        base = lambda s: 1.0  # noqa: E731
    elif args.lr_scheduler_type == "linear":
        base = lambda s: max(0.0, 1.0 - s / max(total_steps, 1))  # noqa: E731
    else:
        base = lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s / max(total_steps, 1), 1.0)))  # noqa: E731
    return lambda s: (s + 1) / max(warmup, 1) if s < warmup else base(s)


def build_models(args, device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKL, DDPMScheduler, MotionAdapter, UNet2DConditionModel, UNetMotionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    logger.info("loading sd components from %s", args.model_id_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id_or_path, subfolder="text_encoder",
                                                 torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id_or_path, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_id_or_path, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id_or_path, subfolder="scheduler")

    logger.info("loading motion adapter from %s", args.motion_adapter_id_or_path)
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_id_or_path, torch_dtype=dtype)
    motion_unet = UNetMotionModel.from_unet2d(unet, adapter)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    motion_unet.requires_grad_(False)
    for m in (vae, text_encoder, motion_unet):
        m.to(device)
    vae.eval()
    text_encoder.eval()
    return tokenizer, text_encoder, vae, motion_unet, noise_scheduler, adapter


def configure_trainable(args, motion_unet) -> dict:
    mode = args.train_mode or args.sft_type
    info: dict = {"mode": mode}
    if mode == "lora":
        targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
        wrapped = inject_lora(motion_unet, "motion_modules", targets,
                              args.lora_rank, args.lora_alpha, args.lora_dropout)
        info.update({"lora_modules": len(wrapped), "lora_targets": list(targets),
                     "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                     "first_modules": wrapped[:3]})
        logger.info("injected LoRA into %d motion-module linears (rank=%d alpha=%d)",
                    len(wrapped), args.lora_rank, args.lora_alpha)
        if not wrapped:
            raise RuntimeError("no motion-module linear matched; check --lora_targets")
    elif mode == "partial":
        # 注意：diffusers 0.40 里 motion module 的类名是 AnimateDiffTransformer3D（老版本叫 MotionModule），
        # 所以按"路径以 motion_modules.<i> 结尾"来定位，不依赖类名。
        import re as _re

        blocks = [(n, m) for n, m in motion_unet.named_modules()
                  if _re.search(r"motion_modules\.\d+$", n)]
        if not blocks:
            raise RuntimeError("mode=partial: no motion modules found (unexpected diffusers layout)")
        for _, mod in blocks[-max(args.partial_last_n, 1):]:
            for p in mod.parameters():
                p.requires_grad_(True)
        info.update({"unfrozen_blocks": min(args.partial_last_n, len(blocks)),
                     "total_blocks": len(blocks),
                     "unfrozen_names": [n for n, _ in blocks[-max(args.partial_last_n, 1):]]})
        logger.info("partial fine-tuning: unfroze last %d/%d motion blocks",
                    info["unfrozen_blocks"], len(blocks))
    elif mode == "motion":
        for name, p in motion_unet.named_parameters():
            if "motion" in name:
                p.requires_grad_(True)
        logger.info("full motion-adapter fine-tuning")
    else:
        raise NotImplementedError("'full' (whole UNet) is out of scope for a single 32GB GPU")

    if args.use_dac:
        wrap_motion_attention(motion_unet, tau_init=args.dac_tau_init,
                              tau_threshold=args.dac_tau_threshold, mu_init=args.dac_mu,
                              lambda_init=args.dac_lambda_init, verbose=True)
        info["use_dac"] = True
        info["dac_loss_weight"] = args.dac_loss_weight

    trainable = sum(p.numel() for p in motion_unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in motion_unet.parameters())
    info.update({"trainable_params": trainable, "total_unet_params": total,
                 "trainable_pct": 100.0 * trainable / total})
    logger.info("trainable params: %.2fM / %.1fM (%.3f%%)",
                trainable / 1e6, total / 1e6, 100.0 * trainable / total)
    return info


def encode_prompts(tokenizer, text_encoder, prompts, device, dtype):
    ids = tokenizer(list(prompts), padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0].to(dtype)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or out_dir.name

    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = CSVLogger(out_dir / "metrics.csv")
    tb_writer = None
    if "tensorboard" in args.report_to:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        except Exception as exc:  # pragma: no cover
            logger.warning("tensorboard unavailable: %s", exc)

    tokenizer, text_encoder, vae, motion_unet, noise_scheduler, adapter = build_models(args, device, dtype)
    ckpt_info = configure_trainable(args, motion_unet)
    # LoRA / DAC 都是 build_models 之后才注入的新模块（默认在 CPU），这里统一搬到 GPU
    motion_unet.to(device)
    if args.gradient_checkpointing:
        motion_unet.enable_gradient_checkpointing()

    dataset = ComicVideoDataset(
        csv_path=args.csv_path, video_folder=args.video_folder,
        num_frames=args.num_frames, resolution=args.resolution, crop_method=args.crop_method,
        frame_stride=args.frame_stride,
        augment={"random_flip": args.random_flip, "color_jitter": args.color_jitter},
        latent_cache=args.latent_cache, seed=args.seed,
    )
    if args.max_samples and args.max_samples < len(dataset):
        dataset.items = dataset.items[: args.max_samples]
        logger.info("truncated dataset to %d samples (E3 data-efficiency arm)", len(dataset.items))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=ComicVideoDataset.collate,
                        pin_memory=True, drop_last=True,
                        prefetch_factor=4 if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0)

    params = [p for p in motion_unet.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            f"train_mode={ckpt_info.get('mode')} produced 0 trainable parameters; "
            "check configure_trainable / the diffusers motion-module layout")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    steps_per_epoch = max(len(loader) // args.gradient_accumulation_steps, 1)
    total_steps = (args.max_train_steps if args.max_train_steps > 0
                   else int(steps_per_epoch * args.num_train_epochs))
    if args.dry_run:
        total_steps = min(total_steps, 3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(args, total_steps))

    start_step = 0
    if args.resume_from:
        from models.lora import load_checkpoint
        res = load_checkpoint(args.resume_from, motion_unet)
        logger.info("resumed %s (loaded %d tensors)", args.resume_from, res["loaded"])
        start_step = int(res["meta"].get("step", 0))

    logger.info("run=%s total_steps=%d steps/epoch=%d batch=%d grad_accum=%d res=%d frames=%d",
                run_name, total_steps, steps_per_epoch, args.batch_size,
                args.gradient_accumulation_steps, args.resolution, args.num_frames)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))
    global_step, micro = start_step, 0
    logged_shapes = False
    running: dict[str, float] = {}
    t_start = time.time()
    motion_unet.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    accum = max(args.gradient_accumulation_steps, 1)

    while not stop:
        dataset.set_epoch(global_step // max(steps_per_epoch, 1))
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            prompts = batch["prompts"]
            bsz, frames = pixels.shape[0], pixels.shape[1]

            with torch.no_grad():
                if "latents" in batch:
                    # 预计算 VAE latents：(B, C, T, h, w)，直接跳过 VAE 前向
                    lat = batch["latents"].to(device, dtype, non_blocking=True)
                    lat = lat * vae.config.scaling_factor
                else:
                    flat = pixels.reshape(bsz * frames, *pixels.shape[2:]).to(dtype)
                    lat = vae.encode(flat).latent_dist.sample() * vae.config.scaling_factor
                    # UNetMotionModel 约定：sample 为 (batch, channel, num_frames, h, w)（它自己
                    # 用 sample.shape[2] 取帧数），并且 encoder_hidden_states 的 batch 必须是
                    # batch*num_frames（AnimateDiffPipeline 也是这么 repeat 的）。
                    lat = lat.reshape(bsz, frames, *lat.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
                emb = encode_prompts(tokenizer, text_encoder, prompts, device, dtype)
                emb = emb.repeat_interleave(frames, dim=0)

            noise = torch.randn_like(lat)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(lat, noise, timesteps)

            with torch.autocast("cuda", dtype=dtype, enabled=(args.mixed_precision != "no")):
                pred = motion_unet(noisy, timesteps, encoder_hidden_states=emb, return_dict=False)[0]
                if pred.shape != noise.shape:
                    # 个别版本返回 (B, T, C, h, w)：统一回输入的 (B, C, T, h, w)
                    if pred.shape[1] == noise.shape[2] and pred.shape[2] == noise.shape[1]:
                        pred = pred.permute(0, 2, 1, 3, 4)
                    else:
                        raise RuntimeError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(noise.shape)}")
                if not logged_shapes:
                    logger.info("shape check: latents=%s pred=%s emb=%s",
                                tuple(noise.shape), tuple(pred.shape), tuple(emb.shape))
                    logged_shapes = True
                target = (noise_scheduler.get_velocity(lat, noise, timesteps)
                          if noise_scheduler.config.prediction_type == "v_prediction" else noise)
                diff_loss = F.mse_loss(pred.float(), target.float())
                loss = diff_loss
                conc_loss = None
                if args.use_dac:
                    conc_loss = collect_dac_losses(motion_unet)
                    if conc_loss is not None:
                        loss = loss + args.dac_loss_weight * conc_loss

            scaled = loss / accum
            if args.mixed_precision == "fp16":
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro += 1
            running["loss"] = running.get("loss", 0.0) + float(diff_loss.detach())
            running["count"] = running.get("count", 0.0) + 1
            if conc_loss is not None:
                running["dac_conc"] = running.get("dac_conc", 0.0) + float(conc_loss.detach())

            if micro % accum == 0:
                if args.mixed_precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                if args.mixed_precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0 or args.dry_run:
                    n = max(running.get("count", 1.0), 1.0)
                    row = {
                        "step": global_step,
                        "epoch": round(global_step / max(steps_per_epoch, 1), 4),
                        "loss": running.get("loss", 0.0) / n,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t_start, 2),
                        "vram_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                    if device.type == "cuda" else 0.0),
                        "run": run_name, "mode": ckpt_info["mode"],
                        "rank": args.lora_rank if ckpt_info["mode"] == "lora" else 0,
                        "use_dac": int(args.use_dac), "samples": len(dataset),
                        "resolution": args.resolution, "num_frames": args.num_frames,
                    }
                    if "dac_conc" in running:
                        row["dac_conc"] = running["dac_conc"] / n
                        row.update(dac_head_statistics(motion_unet))
                    metrics.log(row)
                    if tb_writer:
                        for k, v in row.items():
                            if isinstance(v, (int, float)) and k != "step":
                                tb_writer.add_scalar(k, v, global_step)
                    logger.info("step %d/%d loss=%.4f lr=%.2e elapsed=%.0fs vram=%.1fGB",
                                global_step, total_steps, row["loss"], row["lr"],
                                row["elapsed_s"], row["vram_gb"])
                    running = {}

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(out_dir / "checkpoints" / f"iter-{global_step}.safetensors",
                                    motion_unet,
                                    {**ckpt_info, **vars(args), "step": global_step, "run_name": run_name})
                    logger.info("saved checkpoint at step %d", global_step)

                if global_step >= total_steps:
                    stop = True
                    break
        if global_step >= total_steps:
            break

    save_checkpoint(out_dir / "checkpoints" / "final.safetensors", motion_unet,
                    {**ckpt_info, **vars(args), "step": global_step, "run_name": run_name})
    metrics.close()
    if tb_writer:
        tb_writer.close()
    logger.info("done: %d steps in %.1f min, checkpoints in %s",
                global_step, (time.time() - t_start) / 60, out_dir / "checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
