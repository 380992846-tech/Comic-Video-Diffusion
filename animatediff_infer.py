"""AnimateDiff 采样入口（真实实现）。

用 diffusers 原生组件组装推理管线：

    MotionAdapter  →  UNetMotionModel.from_unet2d(UNet2D, adapter)  →  AnimateDiffPipeline

然后把训练产出的 checkpoint（``models/lora.save_checkpoint`` 写的
``.safetensors``）灌回 motion UNet：

* ``--train_mode lora``    —— 先按 checkpoint meta 里的 rank/alpha/targets
  重新注入 ``LoRALinear``，再 ``models.lora.load_checkpoint`` 拷权重；
* ``--train_mode partial`` / ``motion`` —— 训练时解冻的是原始参数，直接 load；
* meta 里 ``use_dac=True`` —— 先把 ``attn1`` 换成 ``DACTemporalWrapper`` 再 load，
  否则 ``log_tau`` / ``lambda_h`` 会对不上。

**不允许静默忽略 unmatched**：``load_checkpoint`` 的返回值会被完整打印，
只要出现 unmatched 键（除非显式加 ``--allow_unmatched_ckpt``）或一条都没匹配上，
就带着键名列表报错退出——静默地用半个 LoRA 出片是论文里最贵的一类 bug。

导出走 imageio + imageio-ffmpeg（``imageio.v2.mimwrite``），返回写出的 mp4 路径。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.lora import inject_lora, load_checkpoint  # noqa: E402

DEFAULT_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnimateDiff 采样（diffusers + 训练 LoRA/DAC 权重）")
    # --- 与旧骨架版保持兼容的参数 ---
    p.add_argument("--model_id_or_path", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--motion_adapter_id_or_path",
                   default="Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2")
    p.add_argument("--ckpt_dir", default=None,
                   help="训练产出的 .safetensors（文件，或含 final.safetensors / iter-*.safetensors 的目录）；"
                        "留空 = 跑未微调的 base motion module")
    p.add_argument("--prompt", required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--output_dir", default="paper/demo_videos")
    p.add_argument("--num_samples", type=int, default=None, help="[旧参数] 等价于 --num_videos")
    # --- 新增 ---
    p.add_argument("--resolution", type=int, default=512, help="输出正方形边长（SD1.5 建议 512）")
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_videos", type=int, default=1)
    p.add_argument("--fps", type=int, default=8, help="导出 mp4 的帧率")
    p.add_argument("--negative_prompt", default="low quality, blurry, watermark, text, jpeg artifacts")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--decode_chunk_size", type=int, default=8, help="VAE 分块解码，显存不够就调小")
    p.add_argument("--scheduler", default="ddim", choices=["ddim", "euler", "pndm", "lms", "dpm"])
    p.add_argument("--cpu_offload", action="store_true", help="显存不足时把各组件轮流放到 CPU")
    p.add_argument("--prefix", default="sample", help="输出文件名前缀")
    p.add_argument("--allow_unmatched_ckpt", action="store_true",
                   help="checkpoint 有对不上的键时只告警不报错（默认报错）")
    return p.parse_args()


def resolve_checkpoint(ckpt: str | None) -> Path | None:
    """``--ckpt_dir`` 允许是文件 / 目录 / 省略 / 'none'。"""
    if not ckpt or ckpt.strip().lower() in ("none", "null", ""):
        return None
    path = Path(ckpt)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("final.safetensors",):
            if (path / name).is_file():
                return path / name
        iters = sorted(path.glob("iter-*.safetensors"),
                       key=lambda q: int("".join(ch for ch in q.stem if ch.isdigit()) or 0))
        if iters:
            return iters[-1]
        nested = path / "checkpoints"
        if nested.is_dir():
            return resolve_checkpoint(str(nested))
        safes = sorted(path.glob("*.safetensors"))
        if len(safes) == 1:
            return safes[0]
    if not path.suffix and path.with_suffix(".safetensors").is_file():
        return path.with_suffix(".safetensors")
    raise FileNotFoundError(f"--ckpt_dir 找不到可用的 checkpoint: {ckpt}")


def _loadable_path(path: Path) -> Path:
    """绕开「名字叫 .safetensors、内容是 torch.save 包」的问题。

    ``models/lora.save_checkpoint`` 用 ``torch.save`` 写文件，但命名为
    ``.safetensors``；而 torch 2.14 起 ``torch.load`` 会**按扩展名**把
    ``.safetensors`` 交给 safetensors 加载器，于是直接报
    ``SafetensorError: Error while deserializing header: header too large``。

    这里先按原路径试读；读不动就在旁边做一个同内容的 ``.cvdload.pt``
    硬链接（跨设备时退化为复制），让 torch 按 pickle/zip 路径解析。
    真正的 safetensors 文件能一次读通，不会走到这里。
    """
    import os
    import shutil

    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return path
    except Exception as exc:  # noqa: BLE001 - 任何读取失败都尝试扩展名兜底
        alt = path.with_name(path.name + ".cvdload.pt")
        if not (alt.exists() and alt.stat().st_size == path.stat().st_size):
            try:
                if alt.exists():
                    alt.unlink()
                os.link(path, alt)
            except OSError:
                shutil.copy2(path, alt)
        print(f"[ckpt] {path.name} 按原路径读取失败（{type(exc).__name__}），"
              f"改用扩展名兜底副本 {alt.name}")
        torch.load(alt, map_location="cpu", weights_only=True)  # 兜底也必须读通
        return alt


def peek_meta(path: Path) -> dict:
    """读 checkpoint 的 meta，用来决定注入 LoRA / DAC 的超参。

    三条路径都要试，因为实测踩过坑：``models/lora.save_checkpoint`` 现在写的是**真正的
    safetensors**（meta 存在文件头），而 ``torch.load`` 读 safetensors 会直接返回
    张量字典、``.get("meta")`` 恒为空 —— 于是 rank/use_dac 全部悄悄回退成默认值
    （32 / False），导致 e2_rank8/16/64、e1_motion_full、e4_dac_r32 五个臂的出片全部失败。
    所以：先读 safetensors 文件头 → 再读同名 .json sidecar → 最后才退回 torch pickle。
    """
    import json as _json

    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt") as fh:
                raw = (fh.metadata() or {}).get("meta")
            if raw:
                return dict(_json.loads(raw))
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] safetensors 头读取失败: {type(exc).__name__}: {exc}")
        side = path.with_suffix(".json")
        if side.exists():
            try:
                return dict(_json.loads(side.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] sidecar 读取失败: {type(exc).__name__}: {exc}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict):
            return dict(blob.get("meta", {}) or {})
    except Exception as exc:  # noqa: BLE001
        print(f"[ckpt] meta 读取失败: {type(exc).__name__}: {exc}")
    return {}


def infer_from_keys(ckpt_path: Path, meta: dict) -> dict:
    """不信任 meta，直接从权重键名/形状推断 mode / rank / use_dac。

    这是最可靠的一层：LoRA 训练存 ``*.lora_A.weight``（rank 就是它的第 0 维），
    全参/部分解冻存原始权重名，DAC 会额外存 ``*.log_tau``。
    """
    from safetensors.torch import load_file as _load_st

    if ckpt_path.suffix != ".safetensors":
        return {"mode": None, "rank": None, "use_dac": None, "n_keys": 0}
    try:
        sd = _load_st(str(ckpt_path))
    except Exception:  # noqa: BLE001
        return {"mode": None, "rank": None, "use_dac": None, "n_keys": 0}

    keys = list(sd.keys())
    lora_keys = [k for k in keys if k.endswith(".lora_A.weight")]
    raw_attn = [k for k in keys if k.endswith(".attn1.to_q.weight")]
    dac_keys = [k for k in keys if k.endswith(".log_tau")]
    rank = int(sd[lora_keys[0]].shape[0]) if lora_keys else None
    if lora_keys and not raw_attn:
        mode = "lora"
    elif raw_attn and not lora_keys:
        mode = "partial"
    else:
        mode = None
    return {"mode": mode, "rank": rank, "use_dac": bool(dac_keys), "n_keys": len(keys)}


def build_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from diffusers import (AnimateDiffPipeline, AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler,
                           EulerDiscreteScheduler, LMSDiscreteScheduler, MotionAdapter,
                           PNDMScheduler, UNet2DConditionModel, UNetMotionModel)
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[build] sd={args.model_id_or_path}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id_or_path, subfolder="text_encoder",
                                                 torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id_or_path, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_id_or_path, subfolder="unet",
                                                torch_dtype=dtype)
    print(f"[build] motion_adapter={args.motion_adapter_id_or_path}")
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_id_or_path, torch_dtype=dtype)
    motion_unet = UNetMotionModel.from_unet2d(unet, adapter)
    motion_unet.eval()
    vae.eval()
    text_encoder.eval()

    sched_cls = {"ddim": DDIMScheduler, "euler": EulerDiscreteScheduler,
                 "lms": LMSDiscreteScheduler, "pndm": PNDMScheduler,
                 "dpm": DPMSolverMultistepScheduler}[args.scheduler]
    try:
        scheduler = sched_cls.from_pretrained(args.model_id_or_path, subfolder="scheduler")
    except Exception as exc:  # 某些仓库没有 euler/pndm 的 config
        print(f"[build] scheduler config 回退默认值: {exc}")
        scheduler = sched_cls()

    pipe = AnimateDiffPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                               unet=motion_unet, motion_adapter=adapter, scheduler=scheduler)
    # 注意：这里**故意不**把 pipe 搬到 device / 开 offload。
    # inject_lora / wrap_motion_attention 会新建 CPU 上的 nn.Module，
    # 若先把模型搬到 cuda 再注入，新参数会留在 CPU 上，前向时报
    # "Expected all tensors to be on the same device"。
    # 正确顺序：注入并 load 完权重后，再统一 .to(device) / enable_model_cpu_offload()。
    return pipe, motion_unet, adapter


def apply_checkpoint(motion_unet, ckpt_path: Path, args: argparse.Namespace,
                     device: torch.device) -> dict:
    """把 checkpoint 灌回 motion UNet，并返回/打印完整的匹配统计。"""
    load_path = _loadable_path(ckpt_path)
    meta = peek_meta(load_path)
    probe = infer_from_keys(ckpt_path, meta)
    mode = probe["mode"] or meta.get("mode") or meta.get("train_mode") or meta.get("sft_type") or "lora"
    rank = int(probe["rank"] or meta.get("lora_rank") or 32)
    alpha = int(meta.get("lora_alpha") or (2 * rank))
    # meta["lora_targets"] 可能是字符串（trainer 里 vars(args) 覆盖了 ckpt_info 的 list），
    # 直接 tuple(str) 会得到逐字符元组 -> 只注入到 42 个模块、252 个键不匹配。
    raw_targets = meta.get("lora_targets") or DEFAULT_LORA_TARGETS
    if isinstance(raw_targets, str):
        targets = tuple(t.strip() for t in raw_targets.split(",") if t.strip())
    else:
        targets = tuple(raw_targets)
    use_dac = bool(probe["use_dac"]) if probe["use_dac"] is not None else bool(meta.get("use_dac", False))

    print(f"[ckpt] {ckpt_path}")
    print(f"[ckpt] key-probe: mode={probe['mode']} rank={probe['rank']} "
          f"use_dac={probe['use_dac']} keys={probe['n_keys']}")
    print(f"[ckpt] meta: mode={mode} rank={rank} alpha={alpha} use_dac={use_dac} "
          f"step={meta.get('step')} run={meta.get('run_name')}")

    # DAC 必须先包：wrap_motion_attention 需要 attn1.to_q 是原始 nn.Linear
    # （要读 out_features 推 dim_head）；若先注入 LoRA，to_q 会变成 LoRALinear
    # 而报 `'LoRALinear' object has no attribute 'out_features'`。
    # 两种顺序产生的参数名完全一致（都是 ...attn1.attn.to_q.lora_A.weight），
    # 所以先 DAC 后 LoRA 不会影响 checkpoint 兼容性。
    dac_wrapped = 0
    if use_dac:
        from models.dac_attention import wrap_motion_attention
        dac_wrapped = len(wrap_motion_attention(motion_unet, tau_init=1.0, tau_threshold=2.0,
                                                mu_init=0.5, lambda_init=0.1, verbose=False))
        print(f"[ckpt] 注入 DACTemporalWrapper: {dac_wrapped} 个 attn1")

    if mode == "lora":
        wrapped = inject_lora(motion_unet, "motion_modules", targets, rank, alpha, 0.0)
        print(f"[ckpt] 注入 LoRALinear: {len(wrapped)} 个模块 (targets={list(targets)})")
        if not wrapped:
            raise RuntimeError("没有任何 motion-module linear 被注入，检查注入 scope/targets")
    elif mode in ("partial", "motion", "full"):
        print(f"[ckpt] mode={mode}：训练时直接改的是原参数，不需要注入 adapter")
    else:
        raise ValueError(f"未知的 checkpoint mode: {mode}")

    stats = load_checkpoint(load_path, motion_unet)
    total_keys = stats["loaded"] + len(stats["unmatched"])
    print(f"[ckpt] loaded={stats['loaded']} unmatched={len(stats['unmatched'])} "
          f"keys_in_ckpt={total_keys}")

    # 注入出来的 LoRA/DAC 参数是在 CPU 上新建的，权重 load 完必须整体搬一次，
    # 否则前向会报 device 不匹配（参数的 device 不会跟着 copy_ 一起变）。
    cast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    motion_unet.to(device=device, dtype=cast_dtype)
    print(f"[ckpt] 注入后的 motion UNet 已搬到 {device} / {cast_dtype}")

    if stats["loaded"] == 0:
        raise RuntimeError(f"checkpoint {ckpt_path} 一个键都没匹配上，拒绝用它出片")
    if stats["unmatched"]:
        preview = stats["unmatched"][:10]
        print(f"[ckpt] !! 有 {len(stats['unmatched'])} 个键未匹配，前 10 个: {preview}")
        if not args.allow_unmatched_ckpt:
            raise RuntimeError(
                f"checkpoint 有 {len(stats['unmatched'])} 个未匹配键（上面已列出）。"
                "确认结构性差异后可用 --allow_unmatched_ckpt 显式放行。")
    return {"mode": mode, "rank": rank, "alpha": alpha, "use_dac": use_dac,
            "loaded": stats["loaded"], "unmatched": stats["unmatched"], "meta": meta}


def export_video(frames: np.ndarray, path: Path, fps: int) -> Path:
    """``(T,H,W,C)`` float[0,1] -> mp4（imageio-ffmpeg）。"""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.rint(np.asarray(frames) * 255.0), 0, 255).astype(np.uint8)
    imageio.mimwrite(str(path), list(u8), fps=int(fps), codec="libx264", quality=8,
                     pixelformat="yuv420p", macro_block_size=1)
    return path


def generate(pipe, args: argparse.Namespace, device: torch.device) -> list[Path]:
    num_videos = args.num_videos if args.num_samples is None else args.num_samples
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for i in range(max(1, num_videos)):
        seed = args.seed + i
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            num_frames=args.num_frames,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="np",
            decode_chunk_size=args.decode_chunk_size,
        )
        frames = np.asarray(result.frames[0])
        dt = time.time() - t0
        dst = out_dir / f"{args.prefix}_{i:02d}_seed{seed}.mp4"
        export_video(frames, dst, args.fps)
        size_mb = dst.stat().st_size / 2**20
        peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
        print(f"[gen] {i + 1}/{max(1, num_videos)} seed={seed} frames={frames.shape[0]} "
              f"{frames.shape[2]}x{frames.shape[1]} {dt:.1f}s peak_vram={peak:.1f}GB -> {dst} "
              f"({size_mb:.2f} MB)")
        written.append(dst)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    return written


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    ckpt_path = resolve_checkpoint(args.ckpt_dir)
    print(f"[run] device={device} dtype={args.dtype} frames={args.num_frames} "
          f"res={args.resolution} steps={args.num_inference_steps} cfg={args.guidance_scale} "
          f"ckpt={ckpt_path or '(base only)'}")

    pipe, motion_unet, _ = build_pipeline(args, device, dtype)
    ckpt_info = None
    if ckpt_path is not None:
        ckpt_info = apply_checkpoint(motion_unet, ckpt_path, args, device)
    else:
        print("[ckpt] 未提供 checkpoint：跑未微调的 base motion module")
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("[build] enable_model_cpu_offload()")
    else:
        pipe.to(device)

    written = generate(pipe, args, device)
    manifest = {
        "args": vars(args), "device": str(device),
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_stats": ({"mode": ckpt_info["mode"], "rank": ckpt_info["rank"],
                              "alpha": ckpt_info["alpha"], "use_dac": ckpt_info["use_dac"],
                              "loaded": ckpt_info["loaded"],
                              "unmatched": ckpt_info["unmatched"]} if ckpt_info else None),
        "outputs": [str(p) for p in written],
    }
    manifest_path = Path(args.output_dir) / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(written)} 个视频 -> {[str(p) for p in written]}")
    print(f"[done] manifest -> {manifest_path}")
    return 0


def hard_exit(code: int) -> None:
    """绕开解释器退出阶段的 SIGABRT（实测环境缺陷，不是本脚本的问题）。

    该实例上（diffusers 0.40.0 + torch 2.14.0.dev+cu130）只要在 CUDA 上跑过
    diffusers 的 VAE/UNet 前向，解释器正常退出时就会 ``Aborted (core dumped)``
    （exit 134）；视频文件在崩溃前已经写完，所以这里直接 ``os._exit`` 让退出码
    如实反映脚本结果。设 ``CVD_NO_HARD_EXIT=1`` 可关闭。
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
