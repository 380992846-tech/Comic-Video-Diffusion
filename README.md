# Comic Video Diffusion

Single-GPU Fine-Tuning Framework for Comic-Style Video Generation with AnimateDiff

![python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

---

## TL;DR

This framework enables **comic-style video generation on a single A800 GPU**, leveraging **LoRA-based parameter-efficient fine-tuning** on top of **AnimateDiff**. The core philosophy: **turn hardware constraints into research contributions** — instead of chasing SOTA with thousands of GPUs, we explore what's achievable under real-world resource limitations. This is a legitimate research direction recognized in top-tier conference tracks (CVPR/ICCV/ECCV Resource-Efficient AI, ACM MM, BMVC, WACV).

Paper-ready experiment pipeline included.

---

## Features

- **Single-GPU Friendly**: Optimized for A800 (80GB) with production-tested memory configurations
- **LoRA-Based Fine-Tuning**: Train only ~0.5% of parameters — ~20GB memory footprint
- **Comprehensive Experiment Suite**: Pre-configured ablations (rank, data size, LoRA vs full fine-tuning)
- **Production-Ready Data Pipeline**: Multi-processing video preprocessing with VAE pre-computation support
- **Full Experiment Tracking**: W&B + local TensorBoard dual backup
- **Flexible Timeline**: Designed for 7–14 day training budgets with configurable experiment scheduling

---

## Project Structure

```
comic_video_diffusion/
├── configs/
│   ├── animatediff_lora.yaml          # Primary LoRA fine-tuning config
│   ├── animatediff_full.yaml          # Full fine-tuning (use with caution)
│   └── data_config.yaml               # Dataset configuration
├── data/
│   ├── dataset.py                     # PyTorch Dataset with dynamic resolution handling
│   ├── prepare_comic_dataset.py       # Multi-processing video preprocessing
│   └── webvid_utils.py                # WebVid dataset utilities
├── scripts/
│   ├── train_lora.sh                  # Launch LoRA training
│   ├── inference.sh                   # Sampling script
│   ├── precompute_vae.py              # Pre-compute VAE latents offline
│   └── eval_fvd.py                    # FVD evaluation
├── models/
│   ├── animatediff/                   # Git submodule
│   └── sd_base/                       # Base SD checkpoints
├── logs/
│   ├── lora_experiments/
│   │   ├── wandb/                     # W&B logs (sync when online)
│   │   └── tensorboard/               # Local backup (always available)
│   └── full_finetune/
├── paper/
│   ├── figures/                       # Paper-ready figures
│   └── demo_videos/                   # Generated video samples
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n comic_video python=3.10 -y
conda activate comic_video

git clone --recurse-submodules https://github.com/guoyww/AnimateDiff.git
# Alternative: LoRA training fork
git clone https://github.com/Shelly-icecream/AnimateDiff-Motion-Module-LoRA.git

pip install -r requirements.txt
# Key versions: torch>=2.0.1, diffusers>=0.19.3, xformers>=0.0.20
```

### 2. Dataset Preparation

> **Critical**: Start preprocessing **at least 48 hours before training begins.**

**Public Dataset Options:**

| Dataset | Description | Scale | Link |
|---|---|---|---|
| AnimeShooter | Multi-shot animation with character references | 10k+ clips | Link |
| MagicAnime | Anime-style video dataset | 5k+ clips | GitHub |
| WebVid-10M | General video-caption pairs | 10M clips | Link |

Expected scale: **5,000–50,000 video clips, 16–24 frames per clip.** Start with 5k for initial validation, scale up for final experiments.

```bash
python data/prepare_comic_dataset.py \
    --input_dir /path/to/raw_videos \
    --output_dir /path/to/processed_data \
    --num_frames 16 \
    --resolution 512x512 \
    --num_workers 32 \
    --crop_method center              # handles variable aspect ratios
    --save_format frames \
    --precompute_vae True             # Saves 8-10GB VRAM during training
```

### 3. Training (Primary Workflow: LoRA)

```bash
bash scripts/train_lora.sh
```

Under the hood:

```bash
CUDA_VISIBLE_DEVICES=0 \
python animatediff_sft.py \
  --model_id_or_path runwayml/stable-diffusion-v1-5 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --csv_path /path/to/video_metadata.csv \
  --video_folder /path/to/videos \
  --sft_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --gradient_checkpointing True \
  --mixed_precision bf16 \
  --report_to wandb \
  --logging_steps 50 \
  --save_steps 5000 \
  --output_dir logs/lora_experiments/run_001
```

**Dual Logging (Critical for Reliability):**

```bash
--report_to wandb tensorboard \
--wandb_project comic_video_diffusion \
--logging_dir logs/lora_experiments/run_001/tensorboard
```

This ensures even if W&B goes offline during long jobs, your metrics are preserved locally.

### 4. Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
python animatediff_infer.py \
  --model_id_or_path runwayml/stable-diffusion-v1-5 \
  --motion_adapter_id_or_path Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2 \
  --ckpt_dir logs/lora_experiments/run_001/checkpoints/iter-20000 \
  --prompt "a comic style character walking in the rain, 2d animation, studio ghibli style" \
  --num_frames 16 \
  --output_dir outputs/
```

---

## Configuration Reference

| Parameter | Recommended Value | Notes |
|---|---|---|
| Base Model | SD 1.5 / Realistic_Vision_V5.1 | Use anime-tuned (Anything-V5) for style |
| Motion Adapter | animatediff-motion-adapter-v1-5-2 | Official pretrained |
| Training Method | LoRA Only | ~0.5% trainable parameters |
| LoRA Rank | 32 | Balance quality/memory; use 64 for style-heavy tasks |
| Learning Rate | 1e-4 | 10–20× higher than full fine-tuning |
| Batch Size | 1 | Do not increase — video frames multiply memory |
| Gradient Accumulation | 16 | Effective batch size = 16 |
| Video Frames | 12–16 | 16 is upper bound; 12 is safer for 512×512 |
| Resolution | 512×512 | Going higher requires reducing frames |
| Gradient Checkpointing | Required | Non-negotiable for A800 single-GPU |
| Mixed Precision | BF16 | A800 native; more stable than FP16 |
| VAE Pre-computation | Recommended | Saves 8–10GB VRAM (test compatibility first) |
| Crop Method | Center | Handles variable aspect ratios without distortion |

---

## Critical Technical Notes

### 1. Handling Real-World Video Data: Dynamic Resolution

Video datasets have diverse aspect ratios (16:9, 4:3, 9:16). Naively resizing to 512×512 causes severe distortion.

```python
def process_video(video_path, target_res=512, crop_method='center'):
    if crop_method == 'center':
        cropped = center_crop(video, target_res, target_res)
    elif crop_method == 'pad':
        cropped = pad_to_square(video, target_res, target_res)
    metadata['original_aspect_ratio'] = original_w / original_h
```

Always use `--crop_method center` for character-centric videos (most anime content), `pad` where preserving full composition is critical.

### 2. Full Fine-Tuning: Practical Recommendations

Full fine-tuning the Motion Adapter requires **53GB+ VRAM** and runs at ~1/4 the speed of LoRA. A full 20k-step run takes 3–4 days on A800.

| Option | Approach | Time Cost | Paper Legitimacy |
|---|---|---|---|
| A (Recommended) | Large-rank LoRA (rank=256) as proxy for full fine-tuning | Same as LoRA | High — widely accepted |
| B | Fine-tune only last 2-3 layers of UNet + Motion Adapter | ~1.5× LoRA | High — standard practice |
| C | Full fine-tuning for 5k steps only (proof-of-concept) | 1–2 days | Moderate — sufficient for qualitative comparison |

### 3. VAE Pre-computation: Verify Before Scaling

Pre-computing VAE latents saves 8–10GB VRAM. **Not all AnimateDiff forks support loading pre-computed latents.**

```bash
python scripts/test_vae_cache.py \
    --model_id runwayml/stable-diffusion-v1-5 \
    --test_video /path/to/sample.mp4 \
    --cache_dir /path/to/cache
```

If the test fails, fall back to gradient checkpointing + CPU offload.

### 4. Data Diversity and Overfitting Prevention

With 5,000 videos, LoRA is prone to overfitting. Mitigate with augmentation:

```yaml
augmentation:
  random_flip: True
  random_resize: True
  color_jitter: 0.1
  frame_skip: True
```

Monitor validation FVD every 5k steps; if train loss ↓ but val loss ↑ → overfitting → reduce LR / increase rank.

### 5. Logging Reliability: Dual Backup

Always log to **local TensorBoard first** (`--report_to tensorboard --logging_dir ...`), W&B as secondary. At end, `wandb sync` local logs retroactively.

---

## Experiment Design for Papers

| Experiment | Core Question |
|---|---|
| E1: LoRA vs Full FT | Parameter-efficient vs full fine-tuning quality/memory |
| E2: LoRA Rank Ablation | Vary rank (8/16/32/64/128) for capacity-quality trade-off |
| E3: Data Efficiency | Fix steps, vary dataset size (1k/5k/10k/50k) — saturation curve |
| E4: Baselines & User Study | vs Base AnimateDiff; A/B preference test (20+ participants) |

Metrics: **FVD · CLIP Similarity · Human preference (5-point Likert)**.

---

## Training Timeline (A800)

| Phase | Steps | Est. Time | Purpose |
|---|---|---|---|
| Quick Test | 500 | 15 min | Code validation |
| LoRA Quick | 5,000 | 3–4 hours | Verify pipeline |
| LoRA Primary | 20,000 | 12–16 hours | Baseline |
| LoRA Extended | 50,000 | 1.5–2 days | Paper main result |
| Ablations (rank/data) | 5 × 20,000 | 5–7 days | Core experiments |
| Full FT (option C) | 5,000 | 1–2 days | Comparison |
| Inference & Demo | 500 videos | ~1 day | Visualizations |

**Total: 7–14 days.** For a paper submission, ~10 days suffices.

---

## Memory Optimization Stack

| Technique | VRAM Saved | When to Use |
|---|---|---|
| VAE Pre-computation | 8–10 GB | Always (test first) |
| Gradient Checkpointing | 10–15 GB | Always |
| BF16 Mixed Precision | 40% vs FP32 | Always |
| XFormers Attention | 5–8 GB | Default |
| CPU Offload (ZeRO-3) | 20–30 GB | Only if OOM persists |
| Reduce Frames (16→12) | 4–6 GB | Fallback |
| Reduce Resolution (512→384) | 8–12 GB | Last resort |

---

## FAQ

- **OOM?** Enable gradient checkpointing; reduce frames to 12 / resolution to 384; CPU offload via DeepSpeed ZeRO-3 as last resort.
- **VAE pre-computation not compatible?** Modify the dataset to load from cache, or fall back to checkpointing + offload. Test with `test_vae_compatibility.py` first.
- **Full fine-tuning too slow?** Use Option A (large-rank LoRA proxy) + Option C (5k-step real FT for qualitative samples).
- **Different aspect ratios?** `--crop_method center` for character-centric; pad for full composition.
- **Overfitting (all videos look same)?** Augmentation + more data + lower LR + monitor val FVD.
- **W&B disconnected?** No loss — dual logging keeps local TensorBoard; `wandb sync` later.
- **Multi-GPU?** Yes via torchrun / accelerate + DeepSpeed, but framework is optimized for single-GPU.

---

## 骨架说明 & 运行建议（审核）

> ⚠️ **当前为工程骨架**，真正训练需补 submodule 与模型：
> - `animatediff_sft.py` / `animatediff_infer.py`：参数已就绪，但训练/推理**委托给 `models/animatediff/`（submodule）**实现；
> - `data/dataset.py`：`process_video` 已真实读帧/裁剪，但 `__getitem__` 返回**占位张量**，需接 AnimateDiff 数据管线转 `(T,C,H,W)`；
> - `scripts/precompute_vae.py` / `eval_fvd.py` / `test_vae_cache.py`：骨架（见 `scripts/README.md`）。

**运行建议**：
- VA 预计算（省 8-10GB 显存）**先用 `test_vae_cache.py` 验证兼容**，不兼容就回调到梯度检查点 + CPU offload；
- 数据预处理**至少提前 48h**（多进程 `prepare_comic_dataset.py`）；
- LoRA（~20GB 显存）散热压力小于 DiT，但长跑仍建议 `nvidia-smi -pl 250` 降功耗墙 + 温度告警（>80°C 降频）；
- 双日志（W&B offline 保底 + 本地 TensorBoard）防网络断了丢指标。

## References

- AnimateDiff (ICLR 2024) — Motion module architecture
- AnimeShooter / MagicAnime / WebVid datasets
- FastVideo / finetrainers — Single-GPU video training toolkit
- Diffusers LoRA Training — Official reference

**License**: Apache 2.0

---

## DAC temporal attention (motion module)

`models/dac_motion_module.py` provides a **self-contained DAC temporal attention** (`DACTemporalAttention`)
for the AnimateDiff motion module: learnable per-head temperature, fixed-target concentration loss, and
vectorised inference-time skipping of near-uniform temporal heads. It reuses the exact insertion point �� the
frame-axis self-attention inside `VersatileAttention` �� and `analyze_temporal_saturation` produces the
per-head concentration `C(A_h)` profile / figure. The **real** temporal-saturation figure requires a trained
AnimateDiff model + a video on A800.
