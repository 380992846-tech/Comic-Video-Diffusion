# 实验清单（Anchor-LoRA · A800 80GB · 总计 3–4 小时）

> 所有时间/显存都是**估算**，基准来自 README §5 在 4080S 的实测（384/16f：0.85 s/sample），
> A800 按 1.3–1.4× 更快折算。**每个臂跑完第一件事是用实测值替换估算**，并写进
> `runs/<arm>/metrics.csv` 的 `sec_per_step` / `peak_vram_gb`。

## 0. 统一设置（A–F 全部相同）

| 项 | 值 |
|---|---|
| 分辨率 / 帧数 | **384×384 / 12 帧** |
| 目标 | LoRA r8（zero-init，不动主干） |
| 步数 | **600 optimizer steps**（有效 batch = 1×4） |
| 精度 | bf16 + 梯度检查点 + `OMP_NUM_THREADS=1` |
| 数据 | `docs/DATA_PUBLIC.md`：Genshin 33 + Landscape 65 = 98 条 |
| 评测指标 | DINO 主体相似度、edge-SSIM、HSV 距离、光流、FVD-lite |

## 1. 主矩阵

| 臂 | 组件 | 命令要点 | 估算时长 | 估算显存 |
|---|---|---|---|---|
| **A** | 纯 LoRA 基线 | `--use_ref_tokens` 关，`λ_edge=λ_color=0` | **25–30 min** | ~11 GB |
| **B** | + 参考帧 token | `--use_ref_tokens --ref_token_count 4` | 28–32 min | ~11.5 GB |
| **C** | + 线稿损失 | `--lambda_edge 0.05` | 28–32 min | ~11 GB |
| **D** | + 颜色损失 | `--lambda_color 0.02` | 28–32 min | ~11 GB |
| **E** | 全组合 | B+C+D | **30–35 min** | ~11.5 GB |
| **F** | 权重消融 | `λ_edge ∈ {0.01, 0.1}`（其余同 E） | 2 × 30 min | ~11.5 GB |

合计 **≈ 3–4 小时**（含评测）。

## 2. 每臂命令模板

```bash
export PYTHONPATH=/root/autodl-tmp/shadow      # 影子包（见 docs/AUTODL_5090.md）
PY=/root/miniconda3/bin/python

COMMON="--model_id_or_path /root/autodl-tmp/models/sd15 \
  --motion_adapter_id_or_path /root/autodl-tmp/models/adapter-v15-2 \
  --csv_path /root/autodl-tmp/data/index/video_metadata.csv \
  --video_folder /root/autodl-tmp/data/clips \
  --resolution 384 --num_frames 12 --lora_rank 8 \
  --max_train_steps 600 --gradient_accumulation_steps 4 \
  --mixed_precision bf16 --logging_steps 20 --save_steps 200"

# A
$PY animatediff_sft.py $COMMON --output_dir /root/autodl-tmp/runs/A_baseline
# B
$PY animatediff_sft.py $COMMON --output_dir /root/autodl-tmp/runs/B_ref --use_ref_tokens --ref_token_count 4
# C
$PY animatediff_sft.py $COMMON --output_dir /root/autodl-tmp/runs/C_edge --lambda_edge 0.05
# D
$PY animatediff_sft.py $COMMON --output_dir /root/autodl-tmp/runs/D_color --lambda_color 0.02
# E
$PY animatediff_sft.py $COMMON --output_dir /root/autodl-tmp/runs/E_full \
   --use_ref_tokens --ref_token_count 4 --lambda_edge 0.05 --lambda_color 0.02
```

## 3. 评测（每臂 100 条 384×12f 出片）

```bash
# 出片（同一 prompt 集、同一 seed 集）
$PY animatediff_infer.py --ckpt_dir /root/autodl-tmp/runs/E_full/checkpoints/final.safetensors \
  --prompt "<来自评测 prompt 集>" --num_frames 12 --resolution 384 \
  --output_dir samples/E_full

# 轻量指标（DINO 主体相似度 / edge-SSIM / HSV 距离 / 光流）—— 不装 detectron2
$PY scripts/eval_consistency.py --pred_dir samples/E_full \
  --ref_dir /root/autodl-tmp/data/refs --output_json results/eval_E_full.json

# FVD-lite（仓库已有）
$PY scripts/eval_fvd.py --pred_dir samples/E_full --real_dir <held-out clips> \
  --output_json results/eval_E_full_fvd.json
```

## 4. 判定标准（写进论文前先自查）

| 断言 | 通过条件 | 不通过怎么办 |
|---|---|---|
| 主体一致性 | B/E 的 DINO 相似度 > A 且差异 > 2 个种子标准差 | 加大 ref tokens 或训练步数 |
| 线稿保持 | C/E 的 edge-SSIM > A 且 FVD 不显著变差 | 降 λ_edge 到 0.02 |
| 颜色一致 | D/E 的 HSV 距离 < A | 降 λ_color 到 0.01 |
| 运动未退化 | 光流幅度与 A 差异 < 20% | 检查损失权重是否过大 |
| 综合 | E > max(A,B,C,D) 或至少 ≥ 各单项 | 说明组件不叠加，如实写 |

## 5. 记录规范

1. 每臂目录：`metrics.csv`（`step, loss, lr, sec_per_step, peak_vram_gb`）
2. 每臂必须存 `checkpoints/step-0200/0400/final.safetensors`，便于选最优而非取最后
3. 任何进论文的数字都要能指到文件与行号；负结果不藏（如实写进 Limitations）
4. 数据划分：训练/评测**按 clip 切**，锚帧与监督窗口不重叠
