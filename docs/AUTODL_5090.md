# AutoDL · RTX 5090 跑这本仓库

> 旧实例已经不在了，这是从零重建的步骤。5090 是 Blackwell（**sm_120**），最容易踩的坑是
> **torch 必须 CUDA 12.8+ 的构建**——老轮子（cu118/cu121）根本认不出这块卡。

## 1. 选实例

| 项 | 选择 |
|---|---|
| GPU | RTX 5090（32GB） |
| 镜像 | 官方镜像里带 **CUDA 12.8+** 的基础镜像（AutoDL「PyTorch 2.7/2.8 + CUDA 12.8」这类） |
| 数据盘 | 100GB 起（模型 ~8GB + 数据集 + checkpoint 每份 ~几十 MB，但 FVD 评测要存样本） |

开机后先确认：

```bash
nvidia-smi                      # 应该看到 RTX 5090 / 32607MiB
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

如果 `torch` 看不到 CUDA，或 CUDA < 12.8，直接跑 `bash scripts/autodl_5090_setup.sh`，它会自动换成 cu128 轮子。

## 2. 拉代码与数据

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> comic-video && cd comic-video
# 数据集：本地没有 clips，需要从旧机器/网盘/公开集恢复
mkdir -p /root/autodl-tmp/data/anime_clips_all
#   video_metadata.csv 的约定见 data/prepare_comic_dataset.py
```

## 3. 环境自举

```bash
WORK=/root/autodl-tmp bash scripts/autodl_5090_setup.sh
```

脚本做四件事：查 sm_120 → 装 cu128 torch（必要时）→ 装 diffusers/accelerate 等 → 拉 SD1.5 + AnimateDiff v1_5_2 → 跑一次 8 帧 256px 冒烟测试。

## 4. 先做诊断，别急着练

```bash
python scripts/diagnose_output.py \
  --model_path /root/autodl-tmp/models/sd15 \
  --motion_adapter /root/autodl-tmp/models/adapter-v15-2 \
  --ckpt /root/autodl-tmp/runs/e1_lora_r32/checkpoints/final.safetensors \
  --prompt "a comic style character walking in the rain, 2d cel animation" \
  --resolutions 512,384 --seed 1234
```

产物在 `results/diagnose_<时间戳>/`：三组 mp4 + `report.json` + `report.md`。判读规则写在文件头：

- base 就糊 → 管线/配置问题（调度器、CFG、dtype）
- base 好、ckpt 糊 → 训练或加载问题（先看 `peek_meta`/`infer_from_keys` 是否把 rank/DAC 认对）
- 384 比 512 糊很多 → 回 512

> 上一次"出片抽象"最可能的原因是 checkpoint 元信息读不到（safetensors 的 meta 用 `torch.load` 读为空，
> rank/use_dac 静默回退成 32/False），那五个臂出片全废。`animatediff_infer.py` 里已经加了
> `infer_from_keys` 兜底——**先重跑一条确认，再谈训练**。

## 5. 重训（512 / r16 / 2000 步）

```bash
STEPS=2000 RESOLUTION=512 LORA_RANK=16 GRAD_ACCUM=4 \
  DATA_DIR=/root/autodl-tmp/data/anime_clips \
  VIDEO_FOLDER=/root/autodl-tmp/data/anime_clips_all \
  bash scripts/train_lora.sh
```

等价的手工命令（配置见 `configs/animatediff_lora_5090.yaml`）：

```bash
python animatediff_sft.py \
  --model_id_or_path /root/autodl-tmp/models/sd15 \
  --motion_adapter_id_or_path /root/autodl-tmp/models/adapter-v15-2 \
  --csv_path /root/autodl-tmp/data/anime_clips/video_metadata.csv \
  --video_folder /root/autodl-tmp/data/anime_clips_all \
  --output_dir /root/autodl-tmp/runs/lora_r16_512_2000 \
  --sft_type lora --lora_rank 16 --lora_alpha 32 \
  --resolution 512 --num_frames 16 \
  --batch_size 1 --gradient_accumulation_steps 4 --learning_rate 1e-4 \
  --save_steps 200 --logging_steps 20
```

每 200 步存一份 `checkpoints/step-XXXX.safetensors`，配 `scripts/eval_val_loss.py` 挑最优，**别用最后一份**。

## 6. 评测

```bash
python scripts/eval_fvd.py \
  --pred_dir samples/lora_r16_512_2000 \
  --real_dir /root/autodl-tmp/data/anime_clips_all \
  --output_json results/eval_lora_r16_512_2000.json
```

样本 < 50 时脚本会标 `reliable:false`——这是诚实的标记，不是失败。README 里"没有任何 eval_*.json"的
空白就靠这一步补上。

## 7. 5090 上的预期与坑

| 项 | 预期 | 备注 |
|---|---|---|
| 512×512×16f LoRA r16 每步 | 约 **0.5–0.8 s**（旧 4080S 是 2.4 s） | 实测为准，日志里看 `sec_per_step` |
| 峰值显存 | 约 18–22 GB | 32GB 够用；不够就开 `--decode_chunk_size 4` 或 gradient checkpointing |
| 冒烟测试 | 256px / 8 帧 / 8 步 十几秒 | `samples/smoke` |
| `sageattention` | **不要装** | 对 sm_120 没有稳定轮子，用默认 SDPA |
| OMP 线程 | 脚本已设 `OMP_NUM_THREADS=1` | 不设会把 step 时间从 0.85s 拖到 2s |
| HF 下载 | `export HF_ENDPOINT=https://hf-mirror.com` | 国内直连很慢 |
| fp16 溢出 | 用 `bf16` | Blackwell 上 bf16 更稳 |
