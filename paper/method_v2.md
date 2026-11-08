# 新方向（v2）：Anchor-LoRA —— 角色/线稿锚定的动画片段适配

> 定稿取向：**只做正结果**。不碰注意力门、不写负结果；与 `CGA`（文本注意力门）、
> `TAG`（视频语言模型时序感知）、`QUBO-VAE-HMMP`（关键帧选择）三个项目**无重叠**。
> 一句话：**给定一张角色参考图，用同样的算力把 AnimateDiff 训成「认得住人和线」的版本。**

## 1. 问题

AnimateDiff 微调（LoRA）能学画风，但**不保证**：

- 同一个角色在不同片段里脸/发型/服装漂移（subject drift）
- 线稿风格在运动中被磨平（lineart wash-out）
- 配色随镜头变化（color flicker）

这些是观众第一眼就能看出来的问题，而且**不需要任何“门/稀疏化”就能度量**。

## 2. 方法

### 2.1 参考锚定（Anchor-FiLM）

把参考图编码成向量 $e = \mathrm{CLIP}(I_{\text{ref}})$，在 motion module 的每个 temporal block
输出上加一个零初始化的 **FiLM** 调制：

$$
h' = (1 + \gamma(e)) \odot h + \beta(e),\qquad \gamma,\beta\ \text{由两层 MLP 产生，零初始化}
$$

- 参数量：每个 block 约 $2d$，全模型约 **1–2 M**，可以忽略
- 零初始化 → 训练开始时与原模型**完全等价**，不会把已训好的画风带崩
- 推理时可选：不给参考图就退化成普通 LoRA（兼容性零成本）

### 2.2 线稿一致性损失

对生成帧与参考帧（或首帧）各取 Sobel/Canny 边缘图 $E(\cdot)$：

$$
\mathcal{L}_{\text{edge}} = 1 - \mathrm{SSIM}\big(E(\hat{x}_t), E(x_{\text{ref}})\big)
$$

按帧平均。它约束**结构**，不约束像素颜色，所以不会把画面拉成静图。

### 2.3 颜色一致性损失

$$
\mathcal{L}_{\text{color}} = \big\|\mu_{\text{HSV}}(\hat{x}_t) - \mu_{\text{HSV}}(x_{\text{ref}})\big\|_2^2
\;+\;\big\|\sigma_{\text{HSV}}(\hat{x}_t) - \sigma_{\text{HSV}}(x_{\text{ref}})\big\|_2^2
$$

### 2.4 总目标

$$
\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_e \mathcal{L}_{\text{edge}} + \lambda_c \mathcal{L}_{\text{color}}
$$

默认 $\lambda_e = 0.05,\ \lambda_c = 0.02$（消融里扫 0.01/0.05/0.1）。

## 3. 为什么它“一定有正结果”

- 三个目标都**直接对应可测指标**：DINO/CLIP 相似度、edge-SSIM、色相距离；改进了指标就是改进
- 零初始化 FiLM **不会让基线变差**（最坏情况退化为原 LoRA）→ 对比安全
- 不依赖任何“门是否触发”的前提，避免 CGA 那个负结果

## 4. 实验矩阵（短预算，总计约 3–4 小时）

统一设置：**384×384、12 帧、LoRA r8、600 步、有效 batch 4、bf16**（A800 上每步约 1.2–1.6 s）

| 臂 | 内容 | 估时 |
|---|---|---|
| A | LoRA 基线 | 25–30 min |
| B | + Anchor-FiLM | 28–32 min |
| C | + 线稿损失（无 FiLM） | 28–32 min |
| D | + 颜色损失（无 FiLM） | 28–32 min |
| E | 全组合（FiLM + edge + color） | 30–35 min |
| F | 消融：$\lambda_e$ × {0.01, 0.1} | 2 × 30 min |

**核心 A–E ≈ 2.5 小时；加 F 和评测 ≈ 3.5–4 小时**，与你批准的预算一致。

评测（138 条公开片段 + 各自参考帧）：

| 指标 | 工具 | 预期方向 |
|---|---|---|
| Subject consistency | DINO ViT-B/16 余弦相似度（帧 vs 参考） | ↑ |
| Lineart preservation | edge-SSIM（Sobel） | ↑ |
| Color fidelity | HSV 均值/方差距离 | ↑（距离↓） |
| Motion | 光流幅度（RAFT 或 Farneback） | 保持 |
| FVD-lite | 仓库已有脚本（138 条自参考） | 不显著变差 |

## 5. 数据（公开，不用合成 IP）

| 来源 | 用途 | 量 |
|---|---|---|
| `svjack/video-dataset-genshin-impact-anime-organized` | 角色演出片段 | 71 条 |
| `svjack/Anime_Landscape_Videos`（+Captioned） | 动画风格/运动多样性 | 67 条 |
| `Fitsd/anime_video_and_anime_screenshots`（选 1 部，~8 GB） | 角色截图 → 参考图池（可选，后期加） | 数千张 |
| 申请中 | Sakuga-42M / AnimeShooter | 论文级扩量 |

每条的参考帧 = 首帧（训练时随机取一帧做锚，防泄漏：锚帧与监督帧不重合）。

## 6. 与三个已有项目的关系

| 项目 | 它做什么 | 本项目不碰它 |
|---|---|---|
| CGA（文本注意力门） | 注意力温度/稀疏 | 不引入温度、不做门的分析 |
| TAG | 视频语言模型的帧序感知 | 不做 VLM 探测 |
| QUBO-VAE-HMMP | 关键帧选择/重建 | 不做帧选择预算 |

## 7. 待定

1. $\lambda_e/\lambda_c$ 的初值按上面默认，若 C/D 单臂无提升，直接进 E
2. 参考图取首帧还是“最能代表角色的帧” —— 先用首帧，若 subject 指标不稳再换
3. 论文题目候选：
   *Anchor-LoRA: Character- and Lineart-Consistent Anime Clip Adaptation at Fixed Compute*
