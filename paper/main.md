# Resource-Constrained Comic-Style Video Generation: Motion-Module LoRA and Temperature-Scaled Temporal Attention

**状态**：草稿（v0.1）。所有数字来自 `results/summary.csv` 与各臂 `metrics.csv` 的真实运行；
标注 `⟨待填⟩` 的位置将在实验跑完后用真实数据替换，**不允许填写任何未实测的数字**。

---

## Abstract

We study comic/anime-style video generation under a hard single-GPU budget
(one 32 GB consumer card, 11.6 GB peak training memory). Instead of scaling data or
parameters, we ask what a **motion-module LoRA** on top of a frozen
Stable Diffusion 1.5 + AnimateDiff stack can achieve, and whether a
**temperature-scaled temporal attention** (DAC) improves the temporal
objective at equal compute. We fine-tune only 9.67 M parameters (0.73 % of the
motion UNet) and run four equal-budget ablations — training objective
(LoRA / partial unfreeze / full motion adapter), LoRA rank (8/16/32/64),
dataset size (50/150/300 clips), and DAC on/off — each 400 optimizer steps
(1 600 samples) at 384×384, 16 frames. ⟨待填：主要结论⟩

---

## 1. Introduction

Comic-style video generation is usually approached with large video diffusion
models trained on tens of thousands of clips. Under a single-GPU budget this is
out of reach: the field's own reporting puts a 20 k-step AnimateDiff LoRA run at
12–16 h on an A800, and full fine-tuning of the motion adapter at 53 GB+ of VRAM.
We take the opposite route and treat the constraint as the experimental setting:

1. **What is actually trainable on one 32 GB card?** We compare three targets
   with identical budgets: LoRA on the motion module's attention projections,
   partial unfreezing of the last motion blocks, and full motion-adapter
   fine-tuning.
2. **Does the rank–capacity trade-off behave as expected at small scale?**
   A 4-point rank sweep at fixed steps.
3. **How fast does quality saturate with data when the step budget is fixed?**
4. **Can a temporal-attention prior help?** We implement DAC — a learnable
   per-head temperature on the frame-axis attention plus a fixed-target
   concentration loss — and test whether it changes the temporal objective at
   equal budget, and whether it induces head-level structure (near-uniform heads
   that can be skipped at inference).

Two engineering findings are, we think, independently useful: the exact tensor
contract of diffusers' `UNetMotionModel` in current versions, and the fact that
DataLoader workers inherit the process' OpenMP thread pool, which in our setup
more than doubled step time.

---

## 2. Related Work

* **AnimateDiff** (ICLR 2024) inserts temporal self-attention ("motion modules")
  into a frozen image diffusion UNet; the motion module is the only trained part
  and the natural target for parameter-efficient adaptation.
* **LoRA** (Hu et al., 2021) adapts frozen weights with low-rank updates; we apply
  it exclusively inside motion modules.
* **Temporal attention regularisation.** Prior work observes that temporal
  attention can saturate (all frames attending near-uniformly) on low-motion
  content; DAC makes that a differentiable, per-head quantity.
* **Resource-efficient fine-tuning** for video diffusion is the setting of this
  work rather than a contribution.

---

## 3. Method

### 3.1 Base stack

SD1.5 UNet + `guoyww/animatediff-motion-adapter-v1-5-2`, assembled with
`MotionAdapter` + `UNetMotionModel.from_unet2d`. VAE, text encoder and all
spatial UNet layers are frozen; only motion-module parameters receive gradients.

**Tensor contract (verified on diffusers 0.40.0).** `UNetMotionModel.forward`
expects `sample` of shape `(B, C, T, H, W)` — it reads `num_frames = sample.shape[2]`
— and **`encoder_hidden_states` must have batch `B·T`**
(`AnimateDiffPipeline` obtains this with `prompt_embeds.repeat_interleave(num_frames, dim=0)`).
Passing `(B, 77, 768)` text embeddings with a `B`-sized latent raises
`The size of tensor a (8192) must match the size of tensor b (1024)` deep inside
`BasicTransformerBlock`. We document this because it is not stated in the docstring.

### 3.2 LoRA on the motion module

`models/lora.py` wraps every `nn.Linear` whose name ends in
`to_q/to_k/to_v/to_out.0` **and whose path contains `motion_modules`**:

```
y = W x + (alpha / r) · B(A x),   A ~ Kaiming, B = 0
```

with `alpha = 2r` throughout. At rank 32 this yields 168 wrapped linears and
**9.67 M trainable parameters out of 1 322 M (0.73 %)**.

### 3.3 DAC temporal attention

`models/dac_attention.py` replaces the motion module's
`AnimateDiffTransformer3D.transformer_blocks[*].attn1` with `DACTemporalWrapper`,
which **keeps the pretrained projections** and adds only per-head scalars:

* learnable per-head temperature `tau_h = exp(log_tau_h)`:
  `A_h = softmax(Q_h K_hᵀ / (sqrt(d) · tau_h))`;
* fixed-target concentration loss with `C(A_h) = 1 - H(A_h)/log F`:
  `L_conc = mean_h lambda_h · (C(A_h) - mu_h)²`, `lambda_h = softplus(.)`,
  `mu_h = 0.5` fixed;
* total objective `L = L_diff + w · L_conc`, `w = 0.01`;
* **inference-time head skipping**: heads with `tau_h > 2.0` replace the frame-axis
  attention with a mean over frames.

Empirically 21 attention blocks / 168 heads are wrapped for SD1.5-v1.5.

---

## 4. Experimental Setup

| 项目 | 值 |
|---|---|
| Hardware | 1 × RTX 4080 SUPER 32 GB (Ada, sm_89), 128 vCPU, 503 GB RAM |
| Base | SD1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) + AnimateDiff v1.5 motion adapter |
| Resolution / frames | 384 × 384, T = 16 |
| Batch | 1 × grad-accum 4 |
| Optimizer | AdamW, LR 1e-4, cosine, warmup 50, weight decay 1e-2 |
| Budget per arm | **400 optimizer steps = 1 600 samples** |
| Precision | bf16 + gradient checkpointing |
| Peak VRAM | 11.6 GB (LoRA r32) |
| Step time | 0.85 s / sample |

**Dataset.** 517 real 2D-animation clips (`svjack/Anime_Landscape_Videos*`,
captioned variants included), indexed with a motion filter
(mean absolute inter-frame difference ≥ 0.004) leaving **344 clips**; 12 clips
were undecodable. The official `Anime-Shooter/AnimeShooter` release is gated and
its public mirrors contain **annotations only** (`29427` JSON story scripts, no
video), so it could not be used. This is a real limitation: the corpus is
anime scenery/background footage rather than character performance.

**Metrics.** Training MSE (per-step, raw CSV), peak VRAM, wall-clock; for
generated samples: FVD (I3D), CLIP score, LPIPS and Farneback optical-flow
magnitude, with an explicit `reliable:false` flag whenever a group has < 50 videos.

---

## 5. Results

### 5.1 E1 — what to train

⟨待填：E1 表格与图 `fig_e1_modes.png`⟩

### 5.2 E2 — LoRA rank

⟨待填：E2 表格与图 `fig_e2_rank.png`⟩

### 5.3 E3 — data efficiency at fixed steps

⟨待填：E3 表格与图 `fig_e3_data.png`⟩

### 5.4 E4 — DAC

⟨待填：E4 表格与图 `fig_e4_dac.png`、`fig_e4_dac_vs_base.png`；
包括训练期 C_mean / C_max / tau_mean 的演化与推理期跳头比例⟩

### 5.5 Qualitative samples

⟨待填：各臂同 prompt、同 seed 的出片帧网格图 + `paper/demo_videos/`⟩

---

## 6. Limitations

1. **Corpus mismatch.** Anime scenery clips, not character animation; claims are
   about style and camera motion, not identity or complex action.
2. **Budget.** 1 600 samples per arm at 384×384 is far below the 12–16 h / 20 k-step
   regime usually reported; all comparisons are *relative at equal budget*, not
   converged absolute quality.
3. **Single seed per arm** for the training curves (seed 42); no error bars over seeds.
4. **Evaluation sample size.** FVD requires ≥ 50 videos per group to be meaningful;
   below that we report CLIP/LPIPS/flow only.
5. Negative results in E1/E2 may be budget artefacts rather than properties of the method.

---

## 7. Conclusion

⟨待填⟩

---

## Appendix A — Reproducibility

* Exact commands: see README "复现清单".
* PyTorch 2.14.0.dev+cu130, diffusers 0.40.0, transformers 5.14.1.
* All raw per-step metrics are collected in `results/loss_curves.csv`; per-run
  CSVs are the append-only source of truth.
* Known environment quirks: `HF_HUB_DISABLE_XET=1` is required on this host;
  the interpreter aborts on exit after CUDA work unless the process exits via
  `os._exit` (worked around in the inference/eval scripts).

## Appendix B — MiniMax H3 feasibility

See `docs/H3_LORA_FEASIBILITY.md`: H3 LoRA fine-tuning is **not** feasible on a
single 32 GB card — the official FL2VA partition alone is 134.16 GB, the only
published training recipe is 2-GPU colocated (unmerged PRs), and published
single-card sampling peaks are 83.6 GB (BF16) / 51.9 GB (FP8) at the native canvas.
