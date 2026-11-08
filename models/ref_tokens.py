"""参考帧 token：把一段片段的参考帧（这里用首帧）编码成若干个 cross-attention token。

为什么用 token 而不是改网络结构：
- AnimateDiff 的 UNet 交叉注意力本来就接受任意长度的 `encoder_hidden_states`，
  把参考 token 拼到文本 embedding 后面即可 —— **不需要改 diffusers 内部模块**，风险最低；
- 参考 token 与 LoRA 可以一起训，参数量只有几十万级；
- 推理时不给参考图就不拼 token，退化成普通 LoRA（零成本兼容）。

用法（训练器里）：

    enc = RefTokenEncoder(in_channels=4, num_tokens=4, out_dim=text_encoder.config.hidden_size)
    ref_tokens = enc(ref_latent)                    # (B, K, D)
    emb = torch.cat([emb, ref_tokens.repeat_interleave(frames, 0)], dim=1)
"""

from __future__ import annotations

import torch
from torch import nn


class RefTokenEncoder(nn.Module):
    """把参考帧 latent 压成 K 个 token。

    输入：`(B, C, h, w)`（VAE latent，已乘 scaling factor）
    输出：`(B, K, D)`
    """

    def __init__(self, in_channels: int = 4, num_tokens: int = 4, out_dim: int = 768, hidden: int = 512):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        # 均值/标准差池化 → 2*C 维；再从它生成 K*D 个参数
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_tokens * out_dim),
        )
        # 零初始化：训练开始时 token 全 0，注意力等价于原模型，不会破坏已训画风
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ref_latent: torch.Tensor) -> torch.Tensor:
        if ref_latent.dim() == 5:  # (B, C, T, h, w)：只取第一帧
            ref_latent = ref_latent[:, :, 0]
        b = ref_latent.shape[0]
        mean = ref_latent.mean(dim=(-1, -2))
        std = ref_latent.std(dim=(-1, -2), unbiased=False)
        stats = torch.cat([mean, std], dim=-1)
        tokens = self.mlp(stats).view(b, self.num_tokens, self.out_dim)
        return self.norm(tokens)


def add_ref_tokens(emb: torch.Tensor, tokens: torch.Tensor, frames: int) -> torch.Tensor:
    """把 `(B,K,D)` 的参考 token 拼到 `(B*T, L, D)` 的文本 embedding 后面。"""
    if emb.shape[0] != tokens.shape[0] * frames:
        raise ValueError(f"batch mismatch: emb {tuple(emb.shape)} tokens {tuple(tokens.shape)} frames={frames}")
    repeated = tokens.repeat_interleave(frames, dim=0)
    return torch.cat([emb, repeated.to(emb.dtype)], dim=1)
