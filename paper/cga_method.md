# 方法稿草稿：Concentration-Gated Temporal Attention (CGA) for Budgeted AnimateDiff Adaptation

> 状态：草稿 v0.1（2026-11-08）。名称统一为 **CGA**（旧名 DAC 不再使用，见 §8）。
> 本稿的原则与仓库其余部分一致：**只写有产物支持的东西**；没有产物的数字标 `TBD`，
> 并明确它是「待测」而不是「已测」。

## 1. 一句话

在**单卡、数百条 clip** 的预算下，把 AnimateDiff 的运动适配器适配到动画/漫画域；
同时把**时序注意力集中度**（concentration）当作可优化的量，并**量化它的收益边界**——
包括一个诚实的负结果：**温度门在本设置下不触发，因此不带来加速**。

## 2. 问题设定

- 主干：SD1.5（冻结）+ AnimateDiff v1_5_2 motion adapter（可训练子集）
- 数据：$N$ 条 clip，每条 16–24 帧，512 分辨率（原生），$N \in [300, 800]$
- 预算：单卡 80GB，训练 $\le 2000$ optimizer step，LoRA rank $\le 32$

## 3. 方法

### 3.1 时序注意力与每头温度

motion module 的 temporal self-attention 在帧轴上做注意力
（`TransformerTemporalModel` 把 `(B,C,T,H,W)` reshape 成 `(B·H·W, T, C)`）。
对第 $h$ 个头：

$$
A_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\tau_h \sqrt{d_k}}\right),\qquad
\tau_h = \mathrm{softplus}(w_h),\ w_h \in \mathbb{R}\ \text{可学习}.
$$

初始化 $\tau_h = 1$（即与原模型等价），$\beta$ 项把它压向 1，防止无界的温度漂移。

### 3.2 集中度指标与损失

$$
C(A_h) = 1 - \frac{H(A_h)}{\log T},\qquad
H(A_h) = -\sum_{t} A_{h,t}\log A_{h,t}
$$

$C \in [0,1]$：$C=0$ 表示完全均匀（注意力平摊到所有帧），$C=1$ 表示完全集中。

训练目标：

$$
\mathcal{L} = \mathcal{L}_{\text{diff}}
 + \lambda \cdot \frac{1}{H}\sum_h \left(C(A_h) - \mu\right)^2
 + \beta \cdot \frac{1}{H}\sum_h \tau_h^2
$$

即：**把每个头的注意力集中度拉向目标 $\mu$**（默认 0.5），同时约束温度不漂移。

### 3.3 推理期门（gate）

若 $\tau_h > \tau_{\text{thr}}$（默认 2.0），则该头的 $QK^\top$ 被替换为其 value 的帧均值
（“跳过注意力”）。设计意图：集中度高的头可以安全跳过、省算力。

### 3.4 我们测什么（本稿的核心）

1. **$\tau$ 会不会动**：每 step 记录 $\bar\tau$、$\sigma_\tau$
2. **$C$ 会不会被损失推动**：记录 $\bar C$、$\max C$
3. **门会不会触发**：记录 `skip%` = $\Pr[\tau_h > \tau_{\text{thr}}]$
4. **动的部分值不值**：FVD / VBench(motion, subject consistency) / 光流 / 人评

## 4. 已测与待测

| 声明 | 状态 | 证据 / 产物 |
|---|---|---|
| 384×16f、LoRA r32、400 step：0.85 s/step、11.6 GB | ✅ 已测 | `results/summary.md`（`bench_omp`） |
| 秩消融：r8 尾部 loss 最低（0.0709）、r64 最差（0.0762） | ✅ 已测 | `results/summary.json` |
| 全参 motion adapter 不优于 LoRA r32（15.28 vs 11.63 GB） | ✅ 已测 | `results/summary.md` |
| CGA 臂的集中度在学（$\bar C_{\text{last}}=0.2045$） | ⚠️ 部分 | `results/loss_curves.csv` 的 `e4_dac_r32` 行 |
| **$\tau$ 在学**（$\bar\tau_{\text{last}}=0.9983$） | ❌ **否** | 同一行：$\tau$ 基本停在 1.0 |
| 门触发 / 加速 | ❌ **未触发**（推论） | $\tau\approx1 \ll 2.0$，`skip%` 预期 0 |
| 生成质量收益（FVD/VBench） | TBD | 评测脚本与产物待补（见 `docs/EXPERIMENTS.md`） |

> **这条负结果很重要**：文本域的同名机制（CGA 库）在 20 个 epoch 上也测得
> $\tau: 1.0000 \to 1.0030$、`skip% = 0.00`。两域一致：**损失只推动 $C$，不推动 $\tau$**。
> 因此**不能**把「可学习温度带来加速/稀疏」写成贡献；能站住的是：
> ① 集中度损失确实改变了时序注意力的形状；② 这个改变是否改善视频质量；③ 一个可复用的诊断协议。

## 5. 正向贡献（重写版）

1. **C1（测量）**：给出「集中度损失 → 时序注意力形状」的定量关系（$C$ 分布、$\mu$ 追踪、
   与运动指标的相关性）。这是**可复现、可检验**的贡献，不依赖门触发。
2. **C2（负结果 + 边界）**：证明**温度门在文本与视频两个域都不触发**，
   并给出加速上界（按可跳过注意力占比计算）。这类“负结果 + 边界”在复现危机背景下有价值。
3. **C3（预算内配方）**：等预算下参数化/秩/数据/分辨率的对照，给出单卡可执行的配方
   （r16、512、~2000 step、bf16+梯度检查点），包含反直觉结论（r64 更差、全参不赢 LoRA）。
4. **C4（诊断协议）**：$\tau$/$C$/$skip\%$ 三条曲线 + 门检查脚本，让后续“注意力稀疏化”类
   工作必须报告“门到底有没有开”。

> 如果要把结论从“负”翻成“正”，唯一的路是**让门真的开**：
> 见 §6 的 CGA-H 扩展，必须用实验证明，不能只改故事。

## 6. 扩展方向：CGA-H（让门真的开）

两条可实现的改法（任一）：

- **H1 · 门代理损失**：当头已经集中（$C_h > \mu_{\text{high}}$）时，奖励 $\tau_h$ 上升：
  $\mathcal{L}_{\text{gate}} = -\gamma \cdot \mathbb{1}[C_h > \mu_{\text{high}}]\cdot \log \tau_h$；
  离开门阈值越近奖励越大（可用 $\min(\tau_h, \tau_{\text{thr}})$ 的 hinge 形式更稳）。
- **H2 · 直通硬门**：训练期直接用硬门（$C_h$ 高则均值替代），梯度经 straight-through 回传；
  统计 `skip%` 作为约束（目标区间 5–20%）。

预期：H1/H2 能把 `skip%` 从 0 拉到非零，但**必须同时报告质量下降幅度**——
这是诚实的做法，也可能是论文最有趣的部分（“稀疏化的价格”）。

## 7. 与相关工作

- AnimateDiff / MotionLoRA / DomainLoRA：只调权重，不动注意力形状
- DynamicViT / A-ViT / SpAtten：token/头的稀疏化，但目标域是分类/检测，时序注意力未覆盖
- 温度缩放与熵正则：常见于对比学习/校准，但**“集中度目标 + 门”** 在视频时序注意力上的测量是空白

## 8. 命名说明

本机制原称 **DAC**，现更名 **CGA**（Concentration-Gated Attention），与
`D:\my_projects\CGA` 库一致；旧名仅作为历史别名保留在代码注释与迁移脚本里。
与其他使用 DAC 缩写的动态注意力校准工作无关。

## 9. 局限

1. 单卡 80GB、$N\le800$、16–24 帧、512 分辨率：规模远低于论文级
2. 数据为动画/漫画域切片，结论不自动外推到实拍
3. FVD 依赖参考域；若参考集与训练分布重叠，FVD 绝对值不可跨论文比较
4. 门不触发 → 本文**不声称**任何推理加速
