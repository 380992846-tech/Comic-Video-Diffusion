"""汇总所有实验臂的 metrics.csv → 论文用的结果表（真实数据，不插值不编造）。

用法：
    python scripts/collect_results.py --runs_dir /root/autodl-tmp/runs \
        --output_dir results

产出：
    results/summary.csv      每臂一行（配置 + 关键指标）
    results/loss_curves.csv  每臂的完整 loss 曲线（长表）
    results/summary.md       Markdown 表格，可直接贴进 README / 论文
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for k, v in row.items():
                if v in ("", None):
                    continue
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def summarize(run: str, rows: list[dict]) -> dict:
    losses = [r["loss"] for r in rows if isinstance(r.get("loss"), float)]
    tail = losses[-max(1, len(losses) // 5):] if losses else []
    steps = [r["step"] for r in rows if isinstance(r.get("step"), float)]
    elapsed = [r["elapsed_s"] for r in rows if isinstance(r.get("elapsed_s"), float)]
    vram = [r["vram_gb"] for r in rows if isinstance(r.get("vram_gb"), float)]
    first, last = rows[0], rows[-1]
    total_s = (elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0
    step_span = (steps[-1] - steps[0]) if len(steps) > 1 else 0
    return {
        "run": run,
        "mode": first.get("mode", ""),
        "rank": first.get("rank", 0),
        "use_dac": first.get("use_dac", 0),
        "samples": first.get("samples", 0),
        "resolution": first.get("resolution", 0),
        "num_frames": first.get("num_frames", 0),
        "steps_logged": len(rows),
        "final_step": steps[-1] if steps else 0,
        "loss_first": losses[0] if losses else math.nan,
        "loss_last": losses[-1] if losses else math.nan,
        "loss_tail_mean": sum(tail) / len(tail) if tail else math.nan,
        "loss_min": min(losses) if losses else math.nan,
        "sec_per_step": (total_s / step_span) if step_span else math.nan,
        "total_minutes": total_s / 60.0 if total_s else math.nan,
        "peak_vram_gb": max(vram) if vram else math.nan,
        "dac_conc_last": next((r["dac_conc"] for r in reversed(rows)
                               if isinstance(r.get("dac_conc"), float)), math.nan),
        "tau_mean_last": next((r["tau_mean"] for r in reversed(rows)
                               if isinstance(r.get("tau_mean"), float)), math.nan),
        "C_mean_last": next((r["C_mean"] for r in reversed(rows)
                             if isinstance(r.get("C_mean"), float)), math.nan),
    }


def fmt(v, nd=4):
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, curves = [], []
    for metrics in sorted(runs_dir.glob("*/metrics.csv")):
        rows = read_metrics(metrics)
        if not rows:
            continue
        run = metrics.parent.name
        summaries.append(summarize(run, rows))
        for r in rows:
            curves.append({"run": run, **r})

    if not summaries:
        print(f"[error] no metrics.csv found under {runs_dir}")
        return 1

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    curve_keys = sorted({k for c in curves for k in c})
    with open(out_dir / "loss_curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=curve_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(curves)

    cols = ["run", "mode", "rank", "use_dac", "samples", "final_step", "loss_first",
            "loss_last", "loss_tail_mean", "sec_per_step", "total_minutes", "peak_vram_gb"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for s in sorted(summaries, key=lambda x: x["run"]):
        lines.append("| " + " | ".join(fmt(s.get(c, "")) for c in cols) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False,
                                                     default=str), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ok] wrote {out_dir/'summary.csv'}, {out_dir/'loss_curves.csv'}, {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
