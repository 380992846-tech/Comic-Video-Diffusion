# scripts/ — 脚本说明（全部为可运行实现）

## 训练 / 实验

| 脚本 | 用途 | 状态 |
|---|---|---|
| `train_lora.sh` | 单臂训练入口（LoRA / partial / 全 motion adapter，可选 DAC） | ✅ 实测跑通 |
| `run_experiments.sh` | E1–E4 等预算逐臂串行训练（10 臂） | ✅ 实测跑通 |
| `after_sweep.sh` | 训练结束后自动：汇总 → 出图 → 逐臂出片 → 评测 | ✅ 实测跑通 |
| `rerun_partial.sh` | 补齐失败臂并重新汇总 | ✅ 实测跑通 |
| `final_report.sh` | DAC 时序饱和分析 + 定性对比图 + 结果表 | ✅ 实测跑通 |

```bash
# 全部消融（等预算：400 step × grad_accum 4，384×384×16 帧）
STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 \
DATA_DIR=/root/autodl-tmp/data/anime_clips \
VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
bash scripts/run_experiments.sh
```

## 数据 / 特征

| 脚本 | 用途 | 状态 |
|---|---|---|
| `precompute_vae.py` | 预计算 VAE latents（训练循环可直接消费 `batch["latents"]`） | ✅ 实测跑通（11 条 10.6s，重跑 0 todo） |
| `test_vae_cache.py` | 兼容性自检 | ⚠️ 功能已被 `precompute_vae.py --verify` 覆盖，保留占位 |

```bash
python scripts/precompute_vae.py \
  --csv_path <data_dir>/video_metadata.csv --video_folder <视频根目录> \
  --cache_dir <缓存目录> --model_id_or_path <sd15 diffusers 目录> \
  --num_frames 16 --resolution 384 --num_workers 8 --batch_size 4 --encode_chunk 4
```

* 缓存文件名 `<stem>_<num_frames>f_<resolution>.npz`，键名 `latents`，形状 `(4, T, H/8, W/8)`，
  与 `data/dataset.py` 的查找规则严格一致（已实测 dataset 能读到）。
* 已存在默认跳过（断点续跑）；`--overwrite` 重算，`--verify` 写回读校验。
* `animatediff_sft.py` 在 `batch` 里出现 `latents` 时**跳过 VAE 前向**（显存与时间都省）。

## 评测 / 出图

| 脚本 | 用途 | 状态 |
|---|---|---|
| `animatediff_infer.py`（仓库根） | 出片：base 或带 checkpoint，导出 mp4 + manifest | ✅ 实测跑通 |
| `eval_fvd.py` | FVD(I3D) + CLIP score + LPIPS + 光流幅度 | ✅ 实测跑通 |
| `collect_results.py` | 汇总各臂 `metrics.csv` → `summary.csv` / `loss_curves.csv` | ✅ 实测跑通 |
| `make_figures.py` | 论文图：loss 曲线 / rank / 数据效率 / 模式对照 / DAC | ✅ 实测跑通 |
| `write_results.py` | 生成 README/论文用的 Markdown 数据表 | ✅ 实测跑通 |
| `make_qualitative_grid.py` | 各臂同 prompt/seed 的等间隔帧网格图 | ✅ 实测跑通 |
| `analyze_dac.py` | 训练好的 DAC 模型上的时序饱和分析（C(A_h)、tau 分布、跳头比例） | ✅ 实测跑通 |
| `run_inference_all_arms.sh` | 逐臂统一 prompt/seed 出片（可选关键臂 52 条给 FVD） | ✅ 实测跑通 |

```bash
python scripts/eval_fvd.py \
  --gen base=/root/autodl-tmp/samples/base \
  --gen e1_lora_r32=/root/autodl-tmp/samples/e1_lora_r32 \
  --real_dir /root/autodl-tmp/reference_clips \
  --output_json results/eval_all_arms.json --output_csv results/eval_all_arms.csv
```

* **FVD**：Kinetics-400 I3D（`flateon/FVD-I3D-torchscript`）+ 400 维特征的高斯 Fréchet 距离；
  每组 < 50 条视频时标 `reliable: false`，只作方向性参考。
* **CLIP score**：`openai/clip-vit-base-patch32`；**时序一致性**：相邻帧 LPIPS(AlexNet) + 像素差；
  **运动量**：Farneback 光流平均幅度。
* 指标算不出来时 JSON/CSV 写 `unavailable` + 原因，**绝不填占位数字**。

## 环境注记（实测于 AutoDL RTX 4080 SUPER 32G / torch 2.14+cu130 / diffusers 0.40）

* 解释器**正常退出**时会 `Aborted (core dumped)`（只要在 CUDA 上跑过 diffusers 前向）。
  推理/评测脚本在收尾处用 `os._exit` 绕开，让退出码如实反映结果；`CVD_NO_HARD_EXIT=1` 可关闭。
* `models/lora.py` 现在写的是**真正的 safetensors**（`safetensors.torch.save_file`），
  meta 存在文件头——早先用 `torch.save` 写 `.safetensors` 后缀会被 safetensors 加载器劫持。
* `rc`/SFTP 上传：单个文件上传曾因 PowerShell `ConvertTo-Json` 把单元素数组退化成对象而失败，已修。
