"""实测加速对照：在当前 AnimateDiff 训练/采样负载上比较几种可落地的加速手段。

对照组（都在同一进程内**交错**测量，抵消与他任务的相互干扰）：
  A. baseline      —— 现状（SDPA 默认后端，fp16/bf16，无 compile）
  B. channels_last —— 卷积友好内存格式
  C. sdpa_backend  —— 强制 flash / mem_efficient 后端
  D. sage          —— 用 SageAttention 1.0.6 替换 SDPA（仅 attention 部分）
  E. compile       —— torch.compile(mode="reduce-overhead") 包住 motion UNet（可能不支持）

同时回答"稀疏注意力算子能不能帮上忙"：打印每一层 attention 的**序列长度**，
并给出 SDPA 在 n=16 / 1024 / 2304 / 4096 上的耗时，说明块稀疏在什么长度才可能赢。

用法：
    python scripts/bench_speed.py --resolution 384 --frames 16 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.lora import inject_lora  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id_or_path", default="/root/autodl-tmp/models/sd15")
    p.add_argument("--motion_adapter_id_or_path", default="/root/autodl-tmp/models/adapter-v15-2")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output_json", default="/root/comic_video_diffusion/results/speed_bench.json")
    return p.parse_args()


def build(args, dtype, device):
    from diffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel

    unet = UNet2DConditionModel.from_pretrained(args.model_id_or_path, subfolder="unet", torch_dtype=dtype)
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_id_or_path, torch_dtype=dtype)
    model = UNetMotionModel.from_unet2d(unet, adapter)
    model.requires_grad_(False)
    inject_lora(model, "motion_modules", ("to_q", "to_k", "to_v", "to_out.0"),
                args.lora_rank, args.lora_rank * 2)
    model.to(device)
    return model


def timeit(fn, repeats: int) -> float:
    """返回中位耗时（秒）。前向一次做 warmup。"""
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def main() -> int:
    args = parse_args()
    dtype = torch.bfloat16
    device = torch.device("cuda")
    cfg = (1, 4, args.frames, args.resolution // 8, args.resolution // 8)
    print(f"[cfg] latent={cfg} frames={args.frames} res={args.resolution} dtype=bf16")

    model = build(args, dtype, device)
    model.train()
    model.enable_gradient_checkpointing()
    lat = torch.randn(*cfg, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (1,), device=device)
    emb = torch.randn(cfg[0] * args.frames, 77, 768, device=device, dtype=dtype)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(lat, t, encoder_hidden_states=emb, return_dict=False)[0]
            loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
        loss.backward()
        opt.step()

    results = {}

    # --- A. baseline
    results["baseline_train_step_s"] = timeit(step, args.repeats)
    print(f"[A] baseline train step : {results['baseline_train_step_s']:.4f} s")

    # --- B. channels_last（视频张量是 5D，用 channels_last_3d；仅作参考，可能不适用）
    try:
        lat_cl = lat.contiguous(memory_format=torch.channels_last_3d)
        model.to(memory_format=torch.channels_last_3d)

        def step_cl():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                out = model(lat_cl, t, encoder_hidden_states=emb, return_dict=False)[0]
                loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
            loss.backward()
            opt.step()
        results["channels_last3d_train_step_s"] = timeit(step_cl, args.repeats)
        print(f"[B] channels_last_3d    : {results['channels_last3d_train_step_s']:.4f} s "
              f"({results['baseline_train_step_s']/results['channels_last3d_train_step_s']:.3f}x)")
        model.to(memory_format=torch.contiguous_format)
    except Exception as exc:
        print(f"[B] channels_last_3d: FAILED {type(exc).__name__}: {str(exc)[:120]}")
        results["channels_last3d_train_step_s"] = None

    # --- C. 强制 SDPA 后端
    for name, backend in (("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with torch.nn.attention.sdpa_kernel([backend]):
                key = f"sdpa_{name}_train_step_s"
                results[key] = timeit(step, args.repeats)
            print(f"[C] sdpa={name:14s}: {results[key]:.4f} s "
                  f"({results['baseline_train_step_s']/results[key]:.3f}x)")
        except Exception as exc:
            print(f"[C] sdpa={name}: FAILED {type(exc).__name__}: {str(exc)[:120]}")

    # --- D. SageAttention（monkeypatch F.scaled_dot_product_attention）
    try:
        from sageattention import sageattn
        orig_sdpa = F.scaled_dot_product_attention
        stats = {"calls": 0, "sage": 0}

        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
            stats["calls"] += 1
            if (attn_mask is None and dropout_p == 0.0 and query.dtype in (torch.float16, torch.bfloat16)
                    and query.shape[-2] >= 256 and query.shape[-1] % 8 == 0):
                try:
                    stats["sage"] += 1
                    q = query.transpose(1, 2).contiguous()
                    k = key.transpose(1, 2).contiguous()
                    v = value.transpose(1, 2).contiguous()
                    return sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal).transpose(1, 2)
                except Exception:
                    stats["sage"] -= 1
            return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, **kw)

        F.scaled_dot_product_attention = sage_sdpa
        results["sage_train_step_s"] = timeit(step, args.repeats)
        results["sage_calls"] = stats
        print(f"[D] sageattention       : {results['sage_train_step_s']:.4f} s "
              f"({results['baseline_train_step_s']/results['sage_train_step_s']:.3f}x) "
              f"patched_calls={stats}")
        F.scaled_dot_product_attention = orig_sdpa
    except Exception as exc:
        print(f"[D] sageattention: FAILED {type(exc).__name__}: {str(exc)[:160]}")
        results["sage_train_step_s"] = None

    # --- E. torch.compile
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        def step_compiled():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                out = compiled(lat, t, encoder_hidden_states=emb, return_dict=False)[0]
                loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
            loss.backward()
            opt.step()
        results["compile_train_step_s"] = timeit(step_compiled, max(args.repeats, 3))
        print(f"[E] torch.compile       : {results['compile_train_step_s']:.4f} s "
              f"({results['baseline_train_step_s']/results['compile_train_step_s']:.3f}x)")
    except Exception as exc:
        print(f"[E] torch.compile: FAILED {type(exc).__name__}: {str(exc)[:200]}")
        results["compile_train_step_s"] = None

    # --- 序列长度与裸 attention 耗时（回答"稀疏算子有没有用武之地"）
    print("\n[seqlen] 裸 SDPA 在不同序列长度上的耗时（SD1.5 attention 维度: 8 heads x 40 dim）")
    seq_rows = []
    for n in (16, 256, 1024, 2304, 4096, 8192):
        q = torch.randn(1, 8, n, 40, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v), 20) * 1000
        seq_rows.append({"seq_len": n, "sdpa_ms": round(ms, 3)})
        print(f"   n={n:5d}: {ms:8.3f} ms")
    results["sdpa_by_seqlen"] = seq_rows
    results["attention_seq_lens_in_this_run"] = {
        "temporal_axis": args.frames,
        "spatial_top_level": (args.resolution // 8) ** 2,
        "note": "块稀疏/窗口稀疏只有在 n 远大于窗口半径时才有收益；本任务的时序轴只有 "
                f"{args.frames} 步，窗口(64) 已经覆盖全部帧 => 稀疏化收益为 0。",
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] wrote {out}")
    if torch.cuda.is_available():
        print(f"[vram] peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    os._exit(0)


if __name__ == "__main__":
    main()
