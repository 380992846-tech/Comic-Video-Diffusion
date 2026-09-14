# Comic Video Diffusion

Single-GPU fine-tuning for comic/anime-style video generation with AnimateDiff —
plus **DAC temporal attention** (Differentiable Attention Concentration).

![python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14cu130-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

---

## TL;DR

在**单张 32GB 消费级显卡**（RTX 4080 SUPER，实测 11.6GB 峰值显存）上，用 LoRA 微调
AnimateDiff 的 motion adapter，把 SD1.5 主干完全冻结。附带一套把
**温度化时序注意力（DAC）** 真正接进 motion module 的实现，以及 E1–E4 四组等预算消融实验。

本仓库**不再是骨架**：所有脚本都是可运行的真实实现，README 里的每个数字都来自
实例上真实跑出来的 `metrics.csv`（原始日志在 `results/`，见 [结果](#结果) 一节）。

> 与本 README 配套的两份材料：
> - `paper/main.md` —— 论文稿（方法、实验、局限性）
> - `docs/H3_LORA_FEASIBILITY.md` —— MiniMax H3 LoRA 的可行性实测评估（结论：单卡 32G 不可行）

---

## 从骨架到可运行：改了什么

| 文件 | 之前（骨架） | 现在（真实实现） |
|---|---|---|
| `animatediff_sft.py` | 只解析参数并打印，训练"委托给 submodule" | 真实训练器：SD1.5 + `UNetMotionModel` + LoRA/DAC，bf16 + 梯度检查点 + CSV/TensorBoard 双写 |
| `data/dataset.py` | `__getitem__` 返回 `torch.zeros` 占位张量 | 真实解码（PyAV→OpenCV 兜底）、随机时间裁剪、`(T,3,H,W)` 归一化到 `[-1,1]`、可选 latent 缓存 |
| `data/prepare_comic_dataset.py` | 把 clip 落成 16 张 PNG（数据集放大 20–50×） | 索引 + 运动强度过滤 + 写 CSV，**零额外磁盘占用**（可选 `--materialize`） |
| `animatediff_infer.py` | 打印参数后退出 | 真实出片：`AnimateDiffPipeline` + 加载训练 checkpoint → mp4，并打印 `loaded/unmatched` 统计 |
| `scripts/precompute_vae.py` | 不存在（README 提到但文件缺失） | 真实 VAE latent 预计算，多进程、断点续跑、与 dataset 命名对齐 |
| `scripts/eval_fvd.py` | 不存在 | 真实 FVD(I3D) + CLIP score + LPIPS + 光流幅度，算不出就写 `unavailable` 并给原因 |
| `models/lora.py` | 不存在 | 自研 LoRA（无 peft 依赖）+ safetensors 存取 |
| `models/dac_attention.py` | 不存在 | 把 `models/dac_motion_module.py` 的 DAC 真正接进 motion module 的 `attn1` |
| `scripts/run_experiments.sh` | 不存在 | E1–E4 等预算逐臂串行训练驱动 |
| `scripts/collect_results.py` / `make_figures.py` | 不存在 | 汇总真实 `metrics.csv` → 结果表 + 论文图 |

---

## 环境（实例实测版本）

```bash
python -m venv --system-site-packages /root/autodl-tmp/venv-train
/root/autodl-tmp/venv-train/bin/pip install diffusers peft accelerate opencv-python-headless imageio-ffmpeg
```

| 组件 | 实测版本 |
|---|---|
| GPU | NVIDIA RTX 4080 SUPER 32GB（Ada，sm_89，**支持 FP8，不支持 NVFP4**） |
| torch | 2.14.0.dev+cu130 |
| diffusers | 0.40.0 |
| transformers | 5.14.1 |
| 其它 | peft 0.20.0、accelerate 1.15.0、sageattention 1.0.6、imageio-ffmpeg |

**基础模型**（都在 HF 上，非 gated）：

| 用途 | 仓库 | 本地路径 |
|---|---|---|
| SD1.5 主干 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `/root/autodl-tmp/models/sd15` |
| Motion adapter | `guoyww/animatediff-motion-adapter-v1-5-2` | `/root/autodl-tmp/models/adapter-v15-2` |

> ⚠️ 下载踩坑记录：
> 1. 本文档早期引用的 `Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2` **不存在**，正确的是 `guoyww/...`。
> 2. HF 的 Xet 存储在本实例上会 `401 Unauthorized`（`cas-server.xethub.hf.co`），**必须 `export HF_HUB_DISABLE_XET=1`**。
> 3. `stable-diffusion-v1-5/stable-diffusion-v1-5` 匿名可下（`gated=False`），但要显式列 `allow_patterns`。

---

## 数据准备

```bash
python data/prepare_comic_dataset.py \
  --input_dir /root/autodl-tmp/data/anime_clips_all \
  --output_dir /root/autodl-tmp/data/anime_clips \
  --num_frames 16 --resolution 512 --num_workers 24 --min_motion 0.004
```

产出 `video_metadata.csv`（`video_path,prompt,num_frames,fps,width,height,aspect_ratio,motion`）
与 `prepare_stats.json`。

**关于 AnimeShooter 的坑**：官方 `Anime-Shooter/AnimeShooter` 是 **gated** 的，而公开镜像
`qiulu66/AnimeShooter` 里**只有 29427 个 JSON 标注（story script / shot / 视觉描述），没有视频**——
视频需要按 YouTube ID 自行抓取。因此本仓库改用真实存在的 2D 动画片段
（`svjack/Anime_Landscape_Videos*`，517 条 mp4），索引后按运动强度过滤：

| 统计 | 数值 |
|---|---|
| 输入视频 | 517 |
| 解码失败剔除 | 12 |
| 静止镜头剔除（`motion < 0.004`） | 161 |
| **保留** | **344** |
| motion 均值 / 中位 / P90 | 0.0306 / 0.0170 / 0.0651 |

---

## 训练

单臂：

```bash
RUN_NAME=lora_baseline LORA_RANK=32 STEPS=400 bash scripts/train_lora.sh
```

全部 E1–E4：

```bash
STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 \
DATA_DIR=/root/autodl-tmp/data/anime_clips \
VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
bash scripts/run_experiments.sh
```

### 三种训练目标

| `--train_mode` | 训练什么 | 可训练参数（实测） | 用途 |
|---|---|---|---|
| `lora`（默认） | motion module 内注意力的 `to_q/to_k/to_v/to_out.0` 低秩旁路 | r8: 2.4M / r32: **9.67M（0.73%）** / r64: 19.3M | E2 秩消融主臂 |
| `partial` | motion module 尾部 N 个 block 全参数 | 按 block 计 | E1 对照臂 |
| `motion` | 整个 motion adapter 全参数 | ~50M | E1 强对照臂 |

### 实测速度与显存（RTX 4080 SUPER 32G）

| 配置 | 每样本耗时 | 峰值显存 |
|---|---|---|
| 512×512×16f，LoRA r32 | 2.4 s | 18.2 GB |
| **384×384×16f，LoRA r32（本实验采用）** | **0.85 s** | **11.6 GB** |
| 384×384×16f，LoRA r32 + DAC | 0.87 s | 11.1 GB |

> 两个真实的性能坑：
> 1. `UNetMotionModel` 要求输入 `(B, C, T, H, W)`（内部用 `sample.shape[2]` 取帧数），
>    而 `encoder_hidden_states` 的 batch **必须是 `B*T`**（`AnimateDiffPipeline` 也是这么 `repeat_interleave` 的）。
> 2. DataLoader worker 会各自继承 128 个 OMP 线程，线程超订把 step 时间从 0.85s 拖到 ~2s；
>    `animatediff_sft.py` 顶部已 `OMP_NUM_THREADS=1`。

---

## 实验协议（E1–E4，等预算）

**所有臂共用**：384×384、16 帧、batch 1 × grad_accum 4、LR 1e-4（cosine，warmup 50）、
**400 个 optimizer step = 1600 个样本**、bf16、梯度检查点、同一份 344 条数据。

| 臂 | 问题 | 变量 |
|---|---|---|
| `e1_lora_r32` | 基准 | LoRA r32（锚点，同时是 E2/E3/E4 的对照） |
| `e1_partial_last4` | 参数高效 vs 部分全参 | 解冻尾部 4 个 motion block |
| `e1_motion_full` | 上界（全 motion adapter） | 全参数 |
| `e2_rank8/16/64` | 秩-容量权衡 | rank 8 / 16 / 64 |
| `e3_data50/150/300` | 数据效率 | 训练集 50 / 150 / 300 条 |
| `e4_dac_r32` | **DAC 是否有效** | 同一 LoRA r32 + DAC 温度化时序注意力与集中度损失 |

原始指标：`/root/autodl-tmp/runs/<arm>/metrics.csv`（每臂一份，逐 step）。
汇总：

```bash
python scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results
python scripts/make_figures.py --results_dir results
```

---

## DAC 时序注意力

`models/dac_motion_module.py` 是自洽的 DAC 模块（可学习每头温度 `tau_h`、固定目标集中度损失、
推理期近似均匀头跳过），`models/dac_attention.py` 把它**真正插进** pretrained motion module 的
frame-axis self-attention（`AnimateDiffTransformer3D.transformer_blocks[*].attn1`）：

* 保留预训练 `to_q/to_k/to_v/to_out` 权重，**只新增每头** `log_tau`、`lambda_h` 与 `mu`；
* 训练：`logits / tau_h` + `lambda_h * (C(A_h) - mu_h)^2`，`C(A) = 1 - H(A)/log F`；
* 推理：`tau_h > threshold` 的头用**帧轴平均**代替 `Q K^T`。

实测（384×384×16f，rank 32）：包装 **21 个** attention 块、共 **168 个头**；
训练中 `C_mean≈0.22–0.28`、`C_max≈0.87`、`tau_mean≈1.0`（每头温度确实在被学习）。

---

## 推理与评测

```bash
# 出片（--ckpt_dir 可以是 .safetensors / 目录 / 省略=只用 base）
python animatediff_infer.py \
  --model_id_or_path /root/autodl-tmp/models/sd15 \
  --motion_adapter_id_or_path /root/autodl-tmp/models/adapter-v15-2 \
  --ckpt_dir /root/autodl-tmp/runs/e1_lora_r32/checkpoints/final.safetensors \
  --prompt "a comic style character walking in the rain, 2d cel animation" \
  --num_frames 16 --resolution 512 --num_inference_steps 25 \
  --output_dir samples/e1_lora_r32

# VAE latent 预计算（可断点续跑）
python scripts/precompute_vae.py --csv_path .../video_metadata.csv --video_folder ... \
  --cache_dir /root/autodl-tmp/data/latent_cache --resolution 384 --num_frames 16

# 评测两个方向（FVD / CLIP / LPIPS / 光流）
python scripts/eval_fvd.py --pred_dir samples/e1_lora_r32 --real_dir reference_clips \
  --output_json results/eval_e1_lora_r32.json
```

实测：base 出片 16 帧 512×512 10 步 = 7.3 s、峰值 7.8 GB；带 LoRA checkpoint
`loaded=336 unmatched=0`。评测脚本在样本不足（<50 条/组）时会**显式标注 `reliable:false`**，
指标算不出来时写 `unavailable` + 原因，绝不填占位数字。

---

## 局限性（必须如实说明）

1. **数据是动画风景/空镜切片**，不是角色演出：`svjack/Anime_Landscape_Videos*` 是真实 2D 动画
   视频，但内容以风景为主，因此本仓库的"漫画风格"结论只对**画风与镜头运动**成立，
   对角色一致性、口型、复杂动作不成立。
2. **AnimeShooter 未能使用**：官方 gated，公开镜像只有标注没有视频。
3. **算力预算受限**：每臂 1600 个样本、384×384、16 帧，远低于论文级训练量
   （原 README 设想的 2 万 step × 5 组消融在 A800 上也要 5–7 天）。
   所有结论都是"同等预算下的相对比较"，不是收敛后的绝对质量排名。
4. 评测的 FVD 依赖 I3D 权重与样本量；样本不足时以 CLIP/LPIPS/光流为准并明确标注。

---

## 复现清单

```bash
# 0) 环境
python -m venv --system-site-packages /root/autodl-tmp/venv-train
/root/autodl-tmp/venv-train/bin/pip install diffusers peft accelerate opencv-python-headless imageio-ffmpeg
export HF_HUB_DISABLE_XET=1 && source /etc/network_turbo

# 1) 权重
#    stable-diffusion-v1-5/stable-diffusion-v1-5  -> models/sd15
#    guoyww/animatediff-motion-adapter-v1-5-2     -> models/adapter-v15-2

# 2) 数据
#    svjack/Anime_Landscape_Videos{,_Captioned,_Splited} -> data/anime_clips_all
python data/prepare_comic_dataset.py --input_dir data/anime_clips_all \
  --output_dir data/anime_clips --num_frames 16 --resolution 512 --num_workers 24 --min_motion 0.004

# 3) 训练 + 汇总
STEPS=400 GRAD_ACCUM=4 RESOLUTION=384 NUM_FRAMES=16 bash scripts/run_experiments.sh
python scripts/collect_results.py --runs_dir /root/autodl-tmp/runs --output_dir results
python scripts/make_figures.py --results_dir results
```

---

## References

- AnimateDiff (ICLR 2024) — motion module architecture
- MiniMax H3 — 开放的 768p 视频 + 立体声模型（本仓库的可行性评估见 `docs/`）
- diffusers `AnimateDiffPipeline` / `UNetMotionModel`
- **License**: Apache 2.0
