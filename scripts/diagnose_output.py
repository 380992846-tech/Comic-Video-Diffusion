#!/usr/bin/env python3
"""诊断「出片抽象 / 糊 / 花」到底是哪一环出了问题。

三组等条件对照（同一 prompt / seed / 帧数 / 步数）：

    base_512     只用预训练 AnimateDiff（不挂 checkpoint） → 管线基准
    ckpt_512     挂你的训练 checkpoint，512                → 训练是否伤到画面
    ckpt_384     同一 checkpoint，384                      → 分辨率的影响

判读规则（写进输出的 verdict，不猜）：

    1. base_512 本身就抽象/糊           → 是管线或配置问题（调度器、CFG、dtype、attention）
    2. base_512 好、ckpt_512 明显更差   → 是训练（欠拟合 / 过拟合 / LoRA 接了不该接的层）
    3. ckpt_384 比 ckpt_512 还差很多    → 分辨率问题（AnimateDiff 原生 512）
    4. 三组都不行                         → 先别练，从 loader / 权重对齐查起

指标（全部本地、不需要联网）：

    sharpness      拉普拉斯方差（越高越锐，糊/抽象会掉）
    temporal_diff  相邻帧平均绝对差（太低=静止成图，太高=闪烁/花屏）
    flicker        相邻帧差的标准差（越高越抖）
    brightness / saturation / color_std

用法：

    python scripts/diagnose_output.py \
        --model_path /root/autodl-tmp/models/sd15 \
        --motion_adapter /root/autodl-tmp/models/adapter-v15-2 \
        --ckpt /root/autodl-tmp/runs/e1_lora_r32/checkpoints/final.safetensors \
        --prompt "a comic style character walking in the rain, 2d cel animation" \
        --resolutions 512,384 --seed 1234

产物：results/diagnose_<时间戳>/{base_512,ckpt_512,ckpt_384}.mp4 + report.json + report.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnimateDiff 出片质量三组对照诊断")
    p.add_argument("--model_path", required=True, help="SD1.5 主干目录")
    p.add_argument("--motion_adapter", required=True, help="AnimateDiff 运动适配器目录")
    p.add_argument("--ckpt", default=None, help="训练得到的 checkpoint（.safetensors / 目录）；省略则只跑 base")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="low quality, blurry, watermark, text, jpeg artifacts")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resolutions", default="512,384", help="逗号分隔，第一项作为 512 基准")
    p.add_argument("--output_dir", default="results/diagnose")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--infer_script", default="animatediff_infer.py")
    p.add_argument("--skip_run", action="store_true", help="只分析已有 mp4（目录里按名字找）")
    return p.parse_args()


# --------------------------------------------------------------------------- 跑出片
def run_infer(args: argparse.Namespace, out_dir: Path, name: str, res: int, with_ckpt: bool) -> dict:
    cmd = [
        args.python, args.infer_script,
        "--model_id_or_path", args.model_path,
        "--motion_adapter_id_or_path", args.motion_adapter,
        "--prompt", args.prompt,
        "--negative_prompt", args.negative_prompt,
        "--num_frames", str(args.num_frames),
        "--resolution", str(res),
        "--num_inference_steps", str(args.steps),
        "--seed", str(args.seed),
        "--output_dir", str(out_dir),
        "--prefix", name,
        "--num_videos", "1",
    ]
    if with_ckpt and args.ckpt:
        cmd += ["--ckpt_dir", args.ckpt]
    t0 = time.time()
    print(f"\n=== 出片 {name}（{res}px{' + checkpoint' if with_ckpt and args.ckpt else ' · 仅 base'}）===")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    tail = (proc.stdout or "").strip().splitlines()[-8:]
    err = (proc.stderr or "").strip().splitlines()[-8:] if proc.returncode else []
    for line in tail:
        print("  " + line)
    if err:
        print("  [stderr] " + " | ".join(err[-3:]))
    return {
        "name": name,
        "resolution": res,
        "exit_code": proc.returncode,
        "wall_seconds": round(elapsed, 2),
        "stdout_tail": tail,
        "stderr_tail": err,
        "cmd": " ".join(cmd),
    }


# --------------------------------------------------------------------------- 指标
def read_frames(video: Path) -> np.ndarray:
    """读 mp4 → (T,H,W,3) uint8。优先 imageio，退 cv2。"""
    try:
        import imageio.v3 as iio
        frames = iio.imread(video, plugin="pyav")
        if frames.ndim == 3:  # 单帧
            frames = frames[None]
        return np.asarray(frames[..., :3], dtype=np.uint8)
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise RuntimeError(f"读不出帧：{video}")
    return np.stack(out)


def metrics_of(video: Path) -> dict:
    frames = read_frames(video).astype(np.float32) / 255.0  # (T,H,W,3)
    gray = frames.mean(axis=-1)
    t = gray.shape[0]

    # 锐度：拉普拉斯方差（按帧平均）
    lap_var = []
    for i in range(t):
        g = gray[i]
        lap = (
            -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        )
        lap_var.append(float(lap.var()))

    # 时间动态：相邻帧差
    if t >= 2:
        diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
        tdiff, flicker = float(diffs.mean()), float(diffs.std())
    else:
        tdiff = flicker = 0.0

    sat = float((frames.max(-1) - frames.min(-1)).mean())

    return {
        "video": str(video),
        "frames": int(t),
        "sharpness": round(float(np.mean(lap_var)), 6),
        "sharpness_first_frame": round(float(lap_var[0]), 6) if lap_var else 0.0,
        "temporal_diff": round(tdiff, 6),
        "flicker": round(flicker, 6),
        "brightness": round(float(gray.mean()), 6),
        "brightness_std": round(float(gray.std()), 6),
        "saturation": round(sat, 6),
    }


# --------------------------------------------------------------------------- 判读
def verdict(cases: list[dict], reports: dict[str, dict]) -> dict:
    """按四条规定给出结论，不做没有数据的推断。"""
    ok = {n: r for n, r in reports.items() if r.get("exit_code", 1) == 0 and "video_metrics" in r}
    if not ok:
        return {"status": "inconclusive", "reason": "没有任何一组成功出片，先修运行环境", "checks": []}

    base_name = next((c["name"] for c in cases if c["name"].startswith("base")), None)
    ckpt_name = next((c["name"] for c in cases if c["name"].startswith("ckpt")), None)
    checks, out = [], {"status": "ok", "checks": out_checks(checks)}

    def g(name, key):
        return ok[name]["video_metrics"].get(key) if name in ok else None

    if base_name and ckpt_name:
        bs, cs = g(base_name, "sharpness"), g(ckpt_name, "sharpness")
        if bs and cs:
            ratio = cs / bs
            checks.append({"check": "checkpoint 是否伤画面", "base": bs, "ckpt": cs, "ratio": round(ratio, 3),
                           "pass": ratio >= 0.8,
                           "hint": "ckpt 明显比 base 糊 → 训练/加载问题" if ratio < 0.8 else "ckpt 不比 base 糊"})
            if ratio < 0.8:
                out["status"] = "checkpoint_degraded"

        bd, cd = g(base_name, "temporal_diff"), g(ckpt_name, "temporal_diff")
        if bd is not None and cd is not None:
            checks.append({"check": "运动幅度是否崩", "base": bd, "ckpt": cd,
                           "ratio": round(cd / bd, 3) if bd else None,
                           "pass": bd == 0 or 0.5 <= cd / bd <= 2.0,
                           "hint": "ckpt 几乎不动或抖得离谱 → 运动模块/帧数配置有问题"})

    ck512 = next((c["name"] for c in cases if c["name"].startswith("ckpt") and c["resolution"] == 512), None)
    ck384 = next((c["name"] for c in cases if c["name"].startswith("ckpt") and c["resolution"] == 384), None)
    if ck512 and ck384 and ck512 in ok and ck384 in ok:
        s512, s384 = g(ck512, "sharpness"), g(ck384, "sharpness")
        if s512 and s384:
            ratio = s384 / s512
            checks.append({"check": "分辨率影响", "512": s512, "384": s384, "ratio": round(ratio, 3),
                           "pass": ratio >= 0.85,
                           "hint": "384 明显更糊 → 回到 512 训练/出片" if ratio < 0.85 else "分辨率影响有限"})

    if base_name and base_name in ok:
        bs = g(base_name, "sharpness")
        bd = g(base_name, "temporal_diff")
        broken = (bs is not None and bs < 0.001) or (bd is not None and bd < 1e-4)
        if broken:
            checks.append({"check": "base 本身是否就糊/静止", "sharpness": bs, "temporal_diff": bd, "pass": False,
                           "hint": "base 几乎没有高频细节或几乎不动 → 先查调度器/CFG/dtype/帧数配置，别练了"})
            out["status"] = "pipeline_broken"

    return out


def out_checks(checks: list) -> list:
    return checks


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_dir) / f"diagnose_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    res_list = [int(x) for x in str(args.resolutions).split(",") if x.strip()]
    cases: list[dict] = []
    if not args.skip_run and args.ckpt:
        cases.append({"name": "base_512", "res": res_list[0], "ckpt": False})
        cases.append({"name": f"ckpt_{res_list[0]}", "res": res_list[0], "ckpt": True})
        for res in res_list[1:]:
            cases.append({"name": f"ckpt_{res}", "res": res, "ckpt": True})
    elif not args.skip_run:
        cases.append({"name": f"base_{res_list[0]}", "res": res_list[0], "ckpt": False})

    runs = []
    for case in cases:
        sub = out_root / case["name"]
        sub.mkdir(parents=True, exist_ok=True)
        runs.append(run_infer(args, sub, case["name"], case["res"], case["ckpt"]))
        runs[-1]["name"] = case["name"]
        runs[-1]["resolution"] = case["res"]

    reports: dict[str, dict] = {}
    for run in runs:
        name = run["name"]
        report = dict(run)
        vids = sorted((out_root / name).glob("*.mp4"))
        if vids and run.get("exit_code") == 0:
            try:
                report["video_metrics"] = metrics_of(vids[-1])
            except Exception as exc:  # noqa: BLE001
                report["video_metrics_error"] = f"{type(exc).__name__}: {exc}"
        reports[name] = report

    result = {
        "created": stamp,
        "config": {
            "model_path": args.model_path,
            "motion_adapter": args.motion_adapter,
            "ckpt": args.ckpt,
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "seed": args.seed,
            "resolutions": res_list,
        },
        "runs": reports,
        "verdict": verdict([{"name": c["name"], "resolution": c["res"]} for c in cases], reports),
    }
    (out_root / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 出片诊断 {stamp}", "", f"- prompt: `{args.prompt}`",
             f"- seed: {args.seed} · frames: {args.num_frames} · steps: {args.steps}", ""]
    for name, rep in reports.items():
        v = rep.get("video_metrics") or {}
        lines.append(f"## {name}")
        lines.append(f"- 退出码: {rep.get('exit_code')} · 耗时: {rep.get('wall_seconds')}s")
        if v:
            lines.append(f"- 锐度(拉普拉斯方差): {v['sharpness']}")
            lines.append(f"- 帧间差(运动): {v['temporal_diff']} · 抖动: {v['flicker']}")
            lines.append(f"- 亮度: {v['brightness']} · 饱和度: {v['saturation']}")
        else:
            lines.append(f"- 没有可用出片（见 report.json 的 stdout/stderr）")
        lines.append("")
    lines.append("## 判读")
    for c in result["verdict"].get("checks", []):
        lines.append(f"- [{'通过' if c.get('pass') else '不通过'}] {c.get('check')}：{c.get('hint')}")
    if not result["verdict"].get("checks"):
        lines.append(f"- 结论: {result['verdict'].get('status')} / {result['verdict'].get('reason', '')}")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n产物目录: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
