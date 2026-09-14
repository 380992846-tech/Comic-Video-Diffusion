"""在**训练好的** DAC 模型上做时序饱和分析（论文里的核心图）。

和 `models/dac_motion_module.analyze_temporal_saturation` 的区别：后者作用于随机初始化的
单模块，本脚本作用于真实训练出来的 checkpoint 上**全部**被 DAC 包装的注意力块，
统计每头的温度 tau_h、集中度 C(A_h)、以及推理期会被跳过的头比例。

用法：
    python scripts/analyze_dac.py \
      --ckpt /root/autodl-tmp/runs/e4_dac_r32/checkpoints/final.safetensors \
      --output_json paper/figures/dac_analysis.json \
      --output_fig paper/figures/dac_temporal_saturation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dac_attention import DACTemporalWrapper, concentration_index  # noqa: E402
from models.lora import inject_lora, load_checkpoint  # noqa: E402


def build_model(args, dtype):
    from diffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel

    unet = UNet2DConditionModel.from_pretrained(args.model_id_or_path, subfolder="unet", torch_dtype=dtype)
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_id_or_path, torch_dtype=dtype)
    model = UNetMotionModel.from_unet2d(unet, adapter)
    model.requires_grad_(False)
    # 先 LoRA 再 DAC，与 animatediff_sft.py::configure_trainable 的顺序保持一致
    inject_lora(model, "motion_modules", ("to_q", "to_k", "to_v", "to_out.0"),
                args.lora_rank, args.lora_alpha)
    from models.dac_attention import wrap_motion_attention
    wrap_motion_attention(model, tau_threshold=args.tau_threshold, verbose=True)
    res = load_checkpoint(args.ckpt, model)
    print(f"[ckpt] loaded={res['loaded']} unmatched={len(res['unmatched'])}")
    if res["unmatched"]:
        raise RuntimeError(f"checkpoint has {len(res['unmatched'])} unmatched keys")
    return model


@torch.no_grad()
def head_table(model, correlations, frames, batch, n_tokens, seed=0):
    wrappers = [m for m in model.modules() if isinstance(m, DACTemporalWrapper)]
    if not wrappers:
        raise RuntimeError("no DACTemporalWrapper in model (was --use_dac used during training?)")
    g = torch.Generator().manual_seed(seed)
    results = []
    device = next(model.parameters()).device
    for corr in correlations:
        per_head_C = []
        for w in wrappers:
            dim = w.heads * w.dim_head
            shared = torch.randn(batch, n_tokens, 1, dim, generator=g).to(device)
            noise = torch.randn(batch, n_tokens, frames, dim, generator=g).to(device)
            x = corr * shared.expand(-1, -1, frames, -1) + (1 - corr) * noise
            w.eval()
            w(x)                                    # attention 权重存进 last_attn_weights
            attn = w.last_attn_weights              # (B, H, F, F)
            c = concentration_index(attn).mean(dim=0)   # (H,)
            per_head_C.append(c.cpu())
        C = torch.cat(per_head_C).double()
        taus = torch.cat([w.log_tau.detach().exp().double().cpu() for w in wrappers])
        results.append({
            "correlation": corr,
            "C_mean": float(C.mean()), "C_p90": float(C.quantile(0.9)), "C_max": float(C.max()),
            "tau_mean": float(taus.mean()), "tau_max": float(taus.max()),
            "n_heads": int(C.numel()),
        })
    return results, wrappers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/sd15")
    ap.add_argument("--motion_adapter_id_or_path", default="/root/autodl-tmp/models/adapter-v15-2")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--tau_threshold", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n_tokens", type=int, default=16)
    ap.add_argument("--output_json", default="paper/figures/dac_analysis.json")
    ap.add_argument("--output_fig", default="paper/figures/dac_temporal_saturation.png")
    args = ap.parse_args()

    model = build_model(args, torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    corr_list = [0.5, 0.8, 0.95, 0.99]
    results, wrappers = head_table(model, corr_list, args.frames, args.batch, args.n_tokens)

    taus = torch.cat([w.log_tau.detach().exp().double().cpu() for w in wrappers])
    skip_frac = float((taus > args.tau_threshold).float().mean())
    summary = {
        "ckpt": args.ckpt,
        "n_blocks": len(wrappers),
        "n_heads": int(taus.numel()),
        "tau_threshold": args.tau_threshold,
        "tau_mean": float(taus.mean()), "tau_max": float(taus.max()),
        "skip_frac": skip_frac,
        "per_correlation": results,
    }

    # 图：C(A_h) 随帧相关性的变化 + tau 分布
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [r["correlation"] for r in results]
    axes[0].plot(xs, [r["C_mean"] for r in results], "o-", label="mean C")
    axes[0].plot(xs, [r["C_p90"] for r in results], "s--", label="p90 C")
    axes[0].plot(xs, [r["C_max"] for r in results], "^:", label="max C")
    axes[0].set_xlabel("frame correlation of synthetic temporal tokens")
    axes[0].set_ylabel("C(A_h) = 1 - H/log F")
    axes[0].set_title("temporal saturation vs frame correlation")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].hist(taus.numpy(), bins=30, color="steelblue")
    axes[1].axvline(args.tau_threshold, color="crimson", linestyle="--",
                    label=f"skip threshold={args.tau_threshold}")
    axes[1].set_xlabel("learned per-head temperature tau_h")
    axes[1].set_ylabel("#heads")
    axes[1].set_title(f"tau distribution (skip_frac={skip_frac:.3f})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=160)
    plt.close(fig)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[ok] wrote {out} and {args.output_fig}")
    import os
    os._exit(0)  # 该实例上 CUDA 之后正常退出会 core dump，这里硬退


if __name__ == "__main__":
    main()
