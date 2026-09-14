# scripts/ — 辅助脚本说明

| 脚本 | 用途 | 状态 |
|---|---|---|
| `train_lora.sh` | 启动 LoRA 微调 | ✅ 可运行（委托 `animatediff_sft.py`） |
| `inference.sh` | 采样生成视频 | ✅ 可运行（委托 `animatediff_infer.py`，已实测出片） |
| `precompute_vae.py` | 预计算 VAE latents（省 8-10GB 显存） | ✅ 可运行（多进程 / 断点续跑，已实测 11 条视频 10.6s） |
| `eval_fvd.py` | FVD + CLIP score + 时序一致性 + 运动量 | ✅ 可运行（I3D/LPIPS/CLIP 全部实测出数） |
| `test_vae_cache.py` | 验证 VAE 预计算兼容性 | ⚠️ 骨架（其功能已由 `precompute_vae.py --verify` 覆盖） |

## precompute_vae.py

```bash
python scripts/precompute_vae.py \
  --csv_path <data_dir>/video_metadata.csv --video_folder <视频根目录> \
  --cache_dir <缓存目录> --model_id_or_path <sd15 diffusers 目录> \
  --num_frames 16 --resolution 512 --num_workers 8 --batch_size 4 --encode_chunk 4
```

* 缓存文件名 `<stem>_<num_frames>f_<resolution>.npz`，键名 `latents`，
  与 `data/dataset.py` 的查找规则严格一致；形状 `(4, T, H/8, W/8)`。
* 已存在的缓存默认跳过（断点续跑）；`--overwrite` 强制重算，`--verify` 写回读校验。
* `--encode_chunk` 控制单次 VAE 前向的帧数：512px fp32 下 64 帧一次会 OOM，默认 4 安全。
* `--latent_mode mean|sample`：默认后验均值（确定性）；`sample` 与在线训练的
  `.latent_dist.sample()` 一致但不可复现。

> ⚠️ 目前 `animatediff_sft.py` 的训练循环仍然**在线跑 VAE**，没有消费
> `batch["latents"]`；要用上缓存需要改训练循环（见 README 的骨架说明）。

## eval_fvd.py

```bash
python scripts/eval_fvd.py \
  --gen baseline=runs/base/videos --gen lora=runs/lora/videos \
  --real_dir data/real_clips --prompt "a comic style 2d cel animation" \
  --output_json paper/figures/eval_metrics.json --output_csv paper/figures/eval_metrics.csv
```

* **FVD**：Kinetics-400 I3D（`flateon/FVD-I3D-torchscript` 的 TorchScript 权重，
  首次运行自动下到 HF 缓存）+ 400 维特征的高斯 Fréchet 距离。每组 < 50 条视频时
  结果标 `reliable: false`，只做方向性参考。
* **CLIP score**：`openai/clip-vit-base-patch32`，逐帧图像嵌入与文本嵌入的平均余弦（×100）。
* **时序一致性**：相邻帧 LPIPS（AlexNet）+ 像素平均绝对差；**运动量**：Farneback 光流平均幅度。
* 任一指标算不出来（权重下不动、缺提示词、包缺失）时，JSON/CSV 里写
  `unavailable` + 原因，**绝不填占位数字**。

## 环境注记（实测于 AutoDL RTX 4080 SUPER 32G / diffusers 0.40 / torch 2.14+cu130）

* 解释器**正常退出**时会 `Aborted (core dumped)`（只要在 CUDA 上跑过 diffusers 前向）。
  三个脚本都在收尾处用 `os._exit` 绕开，让退出码如实反映结果；
  设 `CVD_NO_HARD_EXIT=1` 可恢复默认行为。
* `models/lora.save_checkpoint` 用 `torch.save` 写 `.safetensors` 名字的文件，
  而 torch ≥2.14 会按扩展名把它交给 safetensors 加载器。`animatediff_infer.py`
  里有扩展名兜底（同目录 `.cvdload.pt` 硬链接）；根治办法是让 `models/lora.py`
  用文件句柄 `torch.load(open(path,'rb'))` 或改用 `.pt` 后缀。
