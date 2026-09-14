# MiniMax H3 LoRA 微调可行性评估（A 目标）

> 结论先给：**在这台租用实例（1 × RTX 4080 SUPER 32G）上不可行**，不是在"慢一点"的意义上不可行，
> 而是权重放不下、框架形态不匹配（唯一公开的训练配方是 2 卡同置）、显存低于官方实测的下限。
> 本文所有数字都是**实测或官方文档/权重仓库给出的原始值**，没有估算替代实测。

评估时间：2026-09-14（实例会话内实测）

---

## 1. 目标

在漫画/角色风格数据上微调 MiniMax H3（Hailuo 3.0，768p 视频 + 原生立体声），
产出可加载的 LoRA（对应 ComfyUI 侧 `--lora-path` / `--ckpt_dir` 的推理用法）。

## 2. 官方训练路径的硬性要求

| 项目 | 要求 | 来源 |
|---|---|---|
| 训练框架 | `miles-diffusion`（SFT / Flow-GRPO），rollout 走 SGLang-diffusion | [miles-diffusion](https://github.com/radixark/miles_diffusion)、[H3 recipe](https://miles.radixark.com/docs/diffusion/models/h3/h3) |
| H3 训练状态 | **未合并进 main**，只在 PR [radixark/miles_diffusion#154](https://github.com/radixark/miles_diffusion/pull/154) + [sgl-project/sglang#34365](https://github.com/sgl-project/sglang/pull/34365) | 同上 |
| 唯一公开配方 | `scripts/run_diffusion_grpo_h3_t2va_2gpu.py`：**2 张卡同置（colocate）**，reward 同置 | 同上 |
| 支持的任务 | 仅 `t2va`（视频分支训练，音频只参与 rollout）；`fl2va` / `ref2va` **尚未支持训练** | 同上 |
| 参考硬件（LoRA SFT） | 单节点 **8 × H200** | [MiniMax 本地部署文档](https://platform.minimaxi.com/docs/guides/local-deploy-h3) |
| LoRA SFT 默认超参 | rank 64 / alpha 128、LR 3e-5、`--sft-offload-encoder`（约 70GB encoder 常驻主机内存） | 同上 |
| 训练数据约束 | 必须落在服务网格：短边 768、24 FPS（±0.01）、帧数 `17n+5`、最小 107 帧 | 同上 |

### 权重体积（HF API 实测，`MiniMaxAI/MiniMax-H3`）

| 分区 | 体积 |
|---|---|
| `FL2VA/` | **134.16 GB** |
| `Ref2VA/` | 134.16 GB |
| `transformer/` | 61.73 GB |
| `text_encoder/` | 62.14 GB |
| `vae/` + `audio_vae/` | 9.70 + 0.56 GB |
| **仓库合计** | **464.24 GB**（280 个文件） |

社区量化版（`Comfy-Org/MiniMax-H3`，ComfyUI 可直接用）实测文件大小：

| 文件 | 大小 |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.28 GB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.85 GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB |
| 合计（推理最小集） | **50.22 GB** |

### 官方公布的显存实测（SGLang，1344×768 / 124 帧 / 50 步）

| 精度 | encoder 放置 | 单卡峰值显存 | 推理延迟 |
|---|---|---|---|
| BF16 | `auto`/`fold` | **83,578 MB** | 19.04 s |
| FP8 | `auto`/`fold` | **51,926 MB** | 18.03 s |
| Ref2VA BF16 | `auto` | 83,968 MB | 29.12 s |
| Ref2VA FP8 | `auto` | 52,816 MB | 27.12 s |

消费级参考：2 × RTX 5090 32GB（TP2 + layerwise 卸载）跑一次 T2VA 1344×768/124 帧/50 步 = **8 分 38 秒**。

## 3. 本机实测

| 项目 | 实测值 | 命令/证据 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER，**32,760 MiB**，compute capability **(8, 9)** | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv` |
| CPU / 内存 | 128 vCPU / 503 GB | `nproc`、`free -g` |
| 数据盘可用 | **约 34–36 GB**（`/root/autodl-tmp` 50GB，系统盘 / 剩 19GB） | `df -h` |
| 公共模型盘 | **未挂载**：`/.autodl-model/data` 不存在，ComfyUI 里所有 H3 权重软链均为断链 | `ls -la /.autodl-model/`、`[ -r $f ]` 判定 |

### 三个独立的否决条件

1. **权重放不下。** 官方 FL2VA 分区 134 GB，数据盘只有 50 GB；
   即便是社区量化最小推理集 50.22 GB 也超过可用空间（34 GB）。
2. **框架形态不匹配。** 唯一公开的 H3 训练配方是 **2 卡同置**（rollout 引擎与训练器共卡），
   单卡 32 GB 连 1 卡都占不满这个拓扑；且该配方未合并进 main，需要 Docker 镜像 + 两个 PR。
3. **显存低于官方实测下限。** H3 在原生画布上的单卡采样峰值：
   BF16 **83.6 GB**、FP8 **51.9 GB**，都远大于 32 GB。
   训练还要在此之上叠加激活值、梯度与优化器状态（LoRA rank 64 时 rank 参数本身很小，
   但**冻结的 30B 主干前反向激活**仍必须驻留或换出）。

补充：本机是 Ada（sm_89），能跑 FP8，但**不能跑 NVFP4**（需要 Blackwell sm_100+），
所以官方 `qwen3vl_32b_minimax_h3_nvfp4_awq` 那条最省显存的路径在这里也用不了。

## 4. 可行的替代方案

| 方案 | 硬件 | 磁盘 | 说明 |
|---|---|---|---|
| **A1（推荐）** | 2 × RTX PRO 6000 Blackwell 96GB（或同档 2 卡 ≥80GB） | ≥ 150 GB（FL2VA only）/ ≥ 300 GB（含 Ref2VA） | 满足 2 卡同置配方；Blackwell 可走 NVFP4 路径，显存余量最大 |
| A2 | 单节点 8 × H200 | ≥ 150 GB | 官方 LoRA SFT 参考硬件，最接近论文可复现基线 |
| A3（降级） | 2 × RTX 5090 32GB | ≥ 120 GB | 有实测数据支撑可跑 T2VA 推理（8m38s/条），训练需自行调低分辨率/关闭音频分支 |

**A1 的启动顺序**（照抄官方 recipe，把两个 PR 拉下来）：

```bash
# 1) Docker 镜像内：miles-diffusion → PR 154
cd /root/miles_diffusion
git fetch https://github.com/radixark/miles_diffusion.git pull/154/head:pr-154 && git checkout pr-154
pip install --force-reinstall -r requirements.txt && pip install -e . --no-deps

# 2) sglang-diffusion → PR 34365
cd /sgl-workspace/sglang
git fetch https://github.com/sgl-project/sglang.git pull/34365/head:pr-34365 && git checkout pr-34365
pip install -e "python[all]"

# 3) 先跑对齐检查（优化器冻结），再短训，再长跑
python3 scripts/run_diffusion_grpo_h3_t2va_2gpu.py --cuda-visible-devices 0,1 \
  --num-rollout 2 --eval-interval 0 --debug-alignment
```

## 5. 本机可以立刻做的 H3 工作（推理侧）

如果只是想在本机验证 H3 的**推理参数**（4 步 LoRA 跑 6 步、分辨率、文本编码器对照），
需要先下载约 30 GB 的最小集，且必须腾出磁盘：

| 组件 | 文件 | 大小 |
|---|---|---|
| DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB |
| 加速 LoRA | `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 1.82 GB |

文本编码器是这里的瓶颈：官方 32B INT8 版 25.28 GB（放不下），
需要改用 4B 路线（`qwen3vl_4b_fp8_scaled` + ClipProj）或 GGUF 量化编码器。
本机 ComfyUI 已就绪（`/root/ComfyUI`，v0.30.0，端口 6006，已确认
`MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` / `MiniMaxH3SigmaShift` 节点存在）。

---

**一句话**：H3 的训练门槛是"2 卡 + 130GB 权重 + 单卡 50–80GB 显存"的量级，
不是 32GB 单卡调参能覆盖的；本机适合做 H3 的**推理侧**实验，训练要在 A1/A2 规格上重开机器。
