"""把真实实验数据画成论文/README 用图（不插值、不美化、不编造）。

输入：scripts/collect_results.py 产出的 results/summary.csv + results/loss_curves.csv
输出：results/figures/*.png

用法：
    python scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results
    python scripts/make_figures.py --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
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


def smooth(values: list[float], window: int = 5) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    res = Path(args.results_dir)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(res / "summary.csv")
    curves = read_csv(res / "loss_curves.csv")
    if not summary:
        print("[error] summary.csv is empty")
        return 1

    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in curves:
        if "step" in row and "loss" in row:
            by_run[str(row["run"])].append(row)
    for rows in by_run.values():
        rows.sort(key=lambda r: r["step"])

    # 1) 所有臂的 loss 曲线（滑动平均）
    plt.figure(figsize=(9, 5.5))
    for run, rows in sorted(by_run.items()):
        steps = [r["step"] for r in rows]
        losses = smooth([r["loss"] for r in rows])
        plt.plot(steps, losses, label=run, linewidth=1.6)
    plt.xlabel("optimizer step (batch = grad_accum x 1)")
    plt.ylabel("MSE loss (smoothed)")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.title("Training loss - all arms (equal budget)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_all_loss_curves.png", dpi=160)
    plt.close()

    def scatter(xs, ys, xlabel, ylabel, title, fname, labeler=None):
        plt.figure(figsize=(7, 4.5))
        pairs = sorted(zip(xs, ys))
        plt.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-", linewidth=1.6)
        if labeler:
            for x, y in pairs:
                plt.annotate(labeler(x), (x, y), textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()

    # 2) E2 秩消融
    ranks = [(int(r["rank"]), r["loss_tail_mean"]) for r in summary
             if str(r.get("mode")) == "lora" and int(r.get("rank", 0)) > 0
             and str(r["run"]).startswith("e2_")]
    if ranks:
        scatter([r for r, _ in ranks], [v for _, v in ranks], "LoRA rank",
                "tail mean MSE loss", "E2: rank ablation", "fig_e2_rank.png",
                labeler=lambda x: f"r{int(x)}")

    # 3) E3 数据效率
    data = [(int(r["samples"]), r["loss_tail_mean"]) for r in summary
            if str(r["run"]).startswith("e3_")]
    if data:
        scatter([s for s, _ in data], [v for _, v in data], "# training samples seen",
                "tail mean MSE loss", "E3: data efficiency (fixed steps)", "fig_e3_data.png")

    # 4) E1 模式对照（loss + 可训练参数量）
    modes = [r for r in summary if str(r["run"]).startswith("e1_")]
    if modes:
        labels = [str(r["run"]) for r in modes]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].bar(labels, [r["loss_tail_mean"] for r in modes], color="steelblue")
        axes[0].set_ylabel("tail mean MSE loss")
        axes[0].set_title("E1: training objective")
        axes[1].bar(labels, [r["peak_vram_gb"] for r in modes], color="darkorange")
        axes[1].set_ylabel("peak VRAM (GB)")
        axes[1].set_title("E1: memory cost")
        for ax in axes:
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="x", labelrotation=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_e1_modes.png", dpi=160)
        plt.close()

    # 5) E4 DAC 对照：集中度/温度随时间 vs baseline
    dac_runs = [r for r in summary if int(r.get("use_dac", 0)) == 1]
    base_runs = [r for r in summary if int(r.get("use_dac", 0)) == 0 and int(r.get("rank", 0)) == 32
                 and str(r["run"]).startswith("e1_")]
    if dac_runs:
        plt.figure(figsize=(9, 5))
        run = str(dac_runs[0]["run"])
        rows = by_run.get(run, [])
        for key, style in (("dac_conc", "-"), ("C_mean", "--"), ("C_max", ":")):
            xs = [r["step"] for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            if xs:
                plt.plot(xs, ys, style, label=f"{key} (DAC arm)", linewidth=1.6)
        xs = [r["step"] for r in rows if "tau_mean" in r]
        ys = [r["tau_mean"] for r in rows if "tau_mean" in r]
        if xs:
            plt.plot(xs, ys, "-.", label="tau_mean (DAC arm)", linewidth=1.6)
        plt.xlabel("optimizer step")
        plt.ylabel("value")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.title("E4: DAC concentration / temperature during training")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_e4_dac.png", dpi=160)
        plt.close()
    if base_runs and dac_runs:
        plt.figure(figsize=(7, 4.5))
        for r in base_runs + dac_runs:
            rows = by_run.get(str(r["run"]), [])
            if not rows:
                continue
            plt.plot([x["step"] for x in rows], smooth([x["loss"] for x in rows]),
                     label=str(r["run"]), linewidth=1.8)
        plt.xlabel("optimizer step")
        plt.ylabel("MSE loss (smoothed)")
        plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=9)
        plt.title("E4: DAC on/off at equal budget (rank 32)")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_e4_dac_vs_base.png", dpi=160)
        plt.close()

    print(f"[ok] wrote figures to {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print("   ", p.name, f"{p.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
