"""Anchor-LoRA 的两个一致性损失（都在 latent 空间算，便宜、可微、不用 VAE 解码）。

- 线稿一致性：对预测出的 x0 与参考帧 x0 的灰度边缘图求 L1 + 余弦不相似
- 颜色一致性：逐通道均值/标准差距离

为什么在 latent 空间：训练步里每步都解码 VAE 太贵（384 下约 0.1–0.3 s），
而边缘/统计量在 latent 上仍然单调相关；论文里说明即可，推理侧再在像素空间复核。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gray(x: torch.Tensor) -> torch.Tensor:
    """`(B, C, H, W)` 或 `(B, C, T, H, W)` → 单通道均值图。"""
    return x.mean(dim=1, keepdim=True)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Sobel 边缘幅度；输入 `(B,1,H,W)` 或 `(B,1,T,H,W)`，输出同形状。"""
    if x.dim() == 5:
        b, _, t, h, w = x.shape
        flat = x.reshape(b, 1, t * h, w)
        out = sobel_edges(flat.reshape(b * t, 1, h, w))
        return out.reshape(b, 1, t, h, w)
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def edge_consistency_loss(pred_x0: torch.Tensor, ref_x0: torch.Tensor) -> torch.Tensor:
    """pred_x0 `(B,C,T,h,w)`；ref_x0 `(B,C,h,w)` 或 `(B,C,1,h,w)`。"""
    if ref_x0.dim() == 4:
        ref_x0 = ref_x0.unsqueeze(2)
    pred_e = sobel_edges(_gray(pred_x0))
    ref_e = sobel_edges(_gray(ref_x0))
    ref_e = ref_e.expand_as(pred_e)
    l1 = (pred_e - ref_e).abs().mean()
    cos = F.cosine_similarity(pred_e.flatten(1), ref_e.flatten(1), dim=1).mean()
    return l1 + (1.0 - cos)


def color_stats_loss(pred_x0: torch.Tensor, ref_x0: torch.Tensor) -> torch.Tensor:
    if ref_x0.dim() == 4:
        ref_x0 = ref_x0.unsqueeze(2)
    pm = pred_x0.mean(dim=(-1, -2))
    ps = pred_x0.std(dim=(-1, -2), unbiased=False)
    rm = ref_x0.mean(dim=(-1, -2)).expand_as(pm)
    rs = ref_x0.std(dim=(-1, -2), unbiased=False).expand_as(ps)
    return (pm - rm).abs().mean() + (ps - rs).abs().mean()


def predict_x0(noise_scheduler, noisy: torch.Tensor, pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """由模型输出还原 x0，兼容 eps / v_prediction / sample 三种预测类型。"""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy.device)
    abar = alphas_cumprod[timesteps].float()
    while abar.dim() < noisy.dim():
        abar = abar.unsqueeze(-1)
    abar = abar.to(noisy.dtype)
    ptype = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if ptype == "v_prediction":
        return abar.sqrt() * noisy - (1 - abar).sqrt() * pred
    if ptype == "sample":
        return pred
    return (noisy - (1 - abar).sqrt() * pred) / abar.clamp(min=1e-6).sqrt()
