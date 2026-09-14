"""把真实实验数据渲染成 README/论文用的 Markdown 表格（不生成任何未实测的数字）。

用法：
    python scripts/write_results.py --results_dir results --out results/tables.md
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        out = []
        for row in csv.DictReader(fh):
            parsed = {}
            for k, v in row.items():
                if v in ("", None):
                    continue
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            out.append(parsed)
        return out


def f(v, nd=4, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return dash if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = Path(args.results_dir)
    rows = read_rows(res / "summary.csv")
    if not rows:
        print("[error] summary.csv empty")
        return 1

    by_name = {str(r["run"]): r for r in rows}
    sections: list[str] = ["# 实验数据表（由真实 metrics.csv 生成）", ""]

    def pick(names):
        return [by_name[n] for n in names if n in by_name]

    def row_of(r, extra_first=None):
        vals = [
            f(r.get("run")), f(r.get("mode")),
            str(int(r.get("rank", 0))) if r.get("rank") else "—",
            f(r.get("use_dac")), str(int(r.get("samples", 0))),
            f(r.get("final_step"), 0), f(r.get("loss_first")), f(r.get("loss_last")),
            f(r.get("loss_tail_mean")), f(r.get("sec_per_step"), 3),
            f(r.get("total_minutes"), 2), f(r.get("peak_vram_gb"), 2),
        ]
        return ([extra_first] if extra_first else []) + vals

    headers = ["run", "mode", "rank", "DAC", "samples", "steps", "loss(1st)", "loss(last)",
               "loss(tail mean)", "s/step", "minutes", "peak VRAM (GB)"]

    e1 = pick(["e1_lora_r32", "e1_partial_last4", "e1_motion_full"])
    if e1:
        sections += ["## E1 — 训练目标对照（等预算：400 step）", "",
                     md_table(headers, [row_of(r) for r in e1]), ""]

    e2 = pick(["e2_rank8", "e2_rank16", "e1_lora_r32", "e2_rank64"])
    if e2:
        sections += ["## E2 — LoRA 秩消融（rank 8/16/32/64）", "",
                     md_table(headers, [row_of(r) for r in e2]), ""]

    e3 = pick(["e3_data50", "e3_data150", "e3_data300", "e1_lora_r32"])
    if e3:
        sections += ["## E3 — 数据效率（固定 400 step，训练集 50/150/300/344）", "",
                     md_table(headers, [row_of(r) for r in e3]), ""]

    e4 = pick(["e1_lora_r32", "e4_dac_r32"])
    if e4:
        extra_h = ["run", "rank", "DAC", "samples", "steps", "loss(tail mean)", "C_mean", "C_max",
                   "tau_mean", "skip_frac", "n_heads", "peak VRAM (GB)"]
        e4_rows = []
        for r in e4:
            e4_rows.append([
                f(r.get("run")), str(int(r.get("rank", 0))), f(r.get("use_dac")),
                str(int(r.get("samples", 0))), f(r.get("final_step"), 0),
                f(r.get("loss_tail_mean")), f(r.get("C_mean_last")), f(r.get("C_max_last")),
                f(r.get("tau_mean_last")), f(r.get("tau_mean_last")), f(r.get("C_mean_last")),
                f(r.get("peak_vram_gb"), 2),
            ])
        sections += ["## E4 — DAC 开/关（同 rank 32、同预算）", "",
                     md_table(extra_h, e4_rows), "",
                     "> C/tau 列取该臂**最后一个记录点**；DAC 臂的 `conc_loss` 权重 0.01，"
                     "非 DAC 臂这些列为 —。", ""]

    if (res / "eval_all_arms.csv").exists():
        sections += ["## 生成质量评测（真实生成视频，eval_fvd.py 输出）", "",
                     (res / "eval_all_arms.csv").read_text(encoding="utf-8").strip(), ""]

    text = "\n".join(sections) + "\n"
    out = Path(args.out) if args.out else res / "tables.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
