#!/usr/bin/env bash
# 对所有臂（含 base）做固定验证集 / 固定噪声的确定性评估，并汇总成表。
#
#   bash scripts/eval_all_val_loss.sh
#
# 验证切片默认取数据集末尾 20 条（e3 的 50/150/300 子集臂完全没见过这些片段）。
set -u
V="${PY:-/root/autodl-tmp/venv-train/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${RUNS_DIR:-/root/autodl-tmp/runs}"
OUT="${OUT:-$REPO/results}"
NUM_CLIPS="${NUM_CLIPS:-20}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-10}"
ALL_ARMS="${ALL_ARMS:-base e1_lora_r32 e1_partial_last4 e1_motion_full e2_rank8 e2_rank16 e2_rank64 e3_data50 e3_data150 e3_data300 e4_dac_r32}"

cd "$REPO"
mkdir -p "$OUT"
for arm in $ALL_ARMS; do
  json="$OUT/val_loss_$arm.json"
  if [ -f "$json" ]; then echo "[skip] $arm"; continue; fi
  ckpt="$RUNS/$arm/checkpoints/final.safetensors"
  if [ "$arm" = "base" ]; then
    echo "[eval] base (no checkpoint)"
    $V scripts/eval_val_loss.py --num_clips "$NUM_CLIPS" --num_timesteps "$NUM_TIMESTEPS" \
      --output_json "$json" 2>&1 | tail -12
  elif [ -f "$ckpt" ]; then
    echo "[eval] $arm"
    $V scripts/eval_val_loss.py --ckpt "$ckpt" --num_clips "$NUM_CLIPS" \
      --num_timesteps "$NUM_TIMESTEPS" --output_json "$json" 2>&1 | tail -12
  else
    echo "[warn] $arm: missing $ckpt"
  fi
done

echo "[aggregate]"
$V - <<'PY'
import json, csv, glob, os
rows = []
for path in sorted(glob.glob("/root/comic_video_diffusion/results/val_loss_*.json")) or sorted(glob.glob("results/val_loss_*.json")):
    d = json.load(open(path, encoding="utf-8"))
    rows.append({
        "arm": os.path.basename(path)[len("val_loss_"):-len(".json")],
        "mode": d.get("mode"), "rank": d.get("lora_rank"), "use_dac": int(bool(d.get("use_dac"))),
        "samples": d.get("samples"), "val_mse_mean": round(d["val_mse_mean"], 6),
        "val_mse_std": round(d["val_mse_std_across_clips"], 6),
        "val_mse_min": round(d["val_mse_min"], 6), "val_mse_max": round(d["val_mse_max"], 6),
        "num_clips": d.get("num_clips"), "num_timesteps": d.get("num_timesteps"),
        "clips_start": d.get("clips_start"),
    })
if not rows:
    print("[error] no val_loss json found"); raise SystemExit(1)
out_csv = "results/val_loss_summary.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
hdr = list(rows[0].keys())
lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
for r in sorted(rows, key=lambda x: x["val_mse_mean"]):
    lines.append("| " + " | ".join(str(r[h]) for h in hdr) + " |")
text = "\n".join(lines) + "\n"
open("results/val_loss_table.md", "w", encoding="utf-8").write(text)
print(text)
print(f"[ok] {out_csv} + results/val_loss_table.md")
PY
echo "[done]"
