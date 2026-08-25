# scripts/ — 辅助脚本说明

| 脚本 | 用途 | 状态 |
|---|---|---|
| `train_lora.sh` | 启动 LoRA 微调 | ✅ 可运行（委托 `animatediff_sft.py`） |
| `inference.sh` | 采样生成视频 | ✅ 可运行（委托 `animatediff_infer.py`） |
| `precompute_vae.py` | 预计算 VAE latents（省 8-10GB 显存） | ⚠️ 骨架，需按 models/animatediff 的缓存格式实现 |
| `eval_fvd.py` | FVD 评估 | ⚠️ 骨架，需接入真实 FVD 计算（如 `clean-fid` 视频扩展） |
| `test_vae_cache.py` | 验证 VAE 预计算兼容性 | ⚠️ 骨架，需按当前 fork 的 dataset 读取逻辑实现 |

> 建议在**大规模预处理前**先跑 `test_vae_cache.py`（3-5 条视频）确认：
> 1. 预计算 latents 正确落盘；
> 2. 训练 dataloader 能加载；
> 3. 显存实际省下 8-10GB。
> 若 fork 不支持加载缓存 latents，改为**梯度检查点 + CPU offload** 兜底。
