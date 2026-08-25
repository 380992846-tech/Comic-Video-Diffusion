"""DAC temporal attention for video diffusion motion modules.

Self-contained (pure torch, no diffusers dependency).  This is the DAC hook
that would sit inside AnimateDiff's ``VersatileAttention`` — the temporal
self-attention over ``video_length`` frames.  It applies:

* a learnable per-head temperature ``tau_h = exp(log_tau)`` to the temporal
  attention logits (frames that are highly correlated become near-uniform),
* a fixed-target concentration loss (non-negative softplus weights),
* vectorised inference-time skipping of near-uniform heads (average pooling
  over the frame axis instead of the ``Q K^T`` matmul).

``analyze_temporal_saturation`` demonstrates the *temporal saturation*
phenomenon: as frame correlation rises (adjacent frames become more alike),
the temporal attention moves toward the uniform distribution, so
``C(A_h) = 1 - H(A_h)/log(F)`` increases and the head becomes a skip candidate.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def concentration_index(attn: torch.Tensor) -> torch.Tensor:
    """C(A) = 1 - H(A)/log(F).  attn: (B, H, F, F) -> (B, H)."""
    eps = 1e-8
    f = attn.size(-1)
    ent = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1)
    return 1.0 - ent / math.log(f)


class DACTemporalAttention(nn.Module):
    """Temperature-scaled temporal self-attention (frame axis), DAC version.

    ``latents`` shape ``(B, C, F, H, W)`` is rearranged to temporal tokens
    ``(B, N, F, C)`` where ``N = H*W`` (per spatial location), then the temporal
    axis ``F`` is attended over.  Q/K/V are projected and split into heads; the
    softmax logits are divided by ``tau_h``.
    """

    def __init__(self, dim, heads, temperature_init=1.0, tau_threshold=2.0,
                 mu_init=0.5, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.dim, self.heads, self.d_k = dim, heads, dim // heads
        self.tau_threshold = float(tau_threshold)
        self.log_tau = nn.Parameter(torch.full((heads,), math.log(temperature_init)))
        self.register_buffer("mu", torch.full((heads,), mu_init))
        self.lambda_h = nn.Parameter(torch.full((heads,), softplus_inverse(0.1)))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, x, return_losses=False):
        """x: (B, N, nframes, C) temporal tokens -> (B, N, nframes, C)."""
        B, N, nframes, _ = x.shape
        Q = self.q(x).view(B, N, nframes, self.heads, self.d_k).permute(0, 1, 3, 2, 4)  # (B,N,H,fr,d)
        K = self.k(x).view(B, N, nframes, self.heads, self.d_k).permute(0, 1, 3, 2, 4)
        V = self.v(x).view(B, N, nframes, self.heads, self.d_k).permute(0, 1, 3, 2, 4)
        scale = 1.0 / math.sqrt(self.d_k)
        tau = torch.exp(self.log_tau)

        if self.training:
            scores = (Q @ K.transpose(-1, -2)) * scale / tau.view(1, 1, -1, 1, 1)
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = attn @ V  # (B,N,H,fr,d)
            weights = attn
            skip = None
        else:
            skip = tau > self.tau_threshold
            head_out = x.new_zeros(B, N, self.heads, nframes, self.d_k)
            if skip.any():
                active_indices = torch.nonzero(~skip).squeeze(-1)
                skip_indices = torch.nonzero(skip).squeeze(-1)
                if active_indices.numel() > 0:
                    Qa = Q[:, :, active_indices]
                    Ka = K[:, :, active_indices]
                    Va = V[:, :, active_indices]
                    sa = (Qa @ Ka.transpose(-1, -2)) * scale / tau[active_indices].view(1, 1, -1, 1, 1)
                    attn_a = self.dropout(F.softmax(sa, dim=-1))
                    head_out[:, :, active_indices] = attn_a @ Va
                    weights = attn_a
                else:
                    weights = x.new_zeros(B, N, 0, nframes, nframes)
                Vs = V[:, :, skip_indices]  # (B,N,skip,fr,d)
                pooled = Vs.mean(dim=-2, keepdim=True)  # (B,N,skip,1,d)
                head_out[:, :, skip_indices] = pooled.expand(-1, -1, -1, nframes, -1)
            else:
                scores = (Q @ K.transpose(-1, -2)) * scale / tau.view(1, 1, -1, 1, 1)
                attn = self.dropout(F.softmax(scores, dim=-1))
                head_out = attn @ V
                weights = attn
            out = head_out

        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, nframes, self.dim)
        out = self.proj(out)
        self.last_attn_weights = weights.detach() if weights is not None else None

        conc_loss = torch.tensor(0.0, device=x.device)
        if weights is not None and weights.shape[2] > 0:
            # weights: (B, N, h_valid, fr, fr) -> concentration over frame axis
            conc = concentration_index(weights.mean(dim=1))  # (B, h_valid)
            lam = F.softplus(self.lambda_h)
            if skip is not None and skip.any():
                valid = ~skip
                mu_v, lam_v = self.mu[valid], lam[valid]
            else:
                mu_v, lam_v = self.mu, lam
            conc_loss = (lam_v.unsqueeze(0).expand(B, -1) * (conc - mu_v.unsqueeze(0).expand(B, -1)) ** 2).mean()

        if return_losses:
            return out, conc_loss
        return out


def analyze_temporal_saturation(model: DACTemporalAttention, correlation=0.95, save_path=None):
    """Feed synthetic latents with tunable frame correlation and report C(A_h).

    Frames are sampled as ``correlation``-blended shared content + noise; higher
    correlation => frames more alike => temporal attention nearer uniform =>
    higher C.  Returns a dict {correlation, mean_C, max_C, skip_frac}.

    ``tau_threshold`` is temporarily set to ``inf`` so every head computes full
    attention and its concentration can be measured (skip is a separate decision).
    """
    model.eval()
    old_threshold = model.tau_threshold
    model.tau_threshold = float("inf")

    B, N, F, dim = 2, 8, 12, model.dim
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(B, N, 1, dim, generator=g)                 # common frame content
    noise = torch.randn(B, N, F, dim, generator=g)
    x = correlation * shared.expand(-1, -1, F, -1) + (1 - correlation) * noise  # (B,N,F,dim)

    with torch.no_grad():
        model(x)

    w = model.last_attn_weights  # (B, N, H, F, F)
    tau = model.log_tau.exp().detach()
    conc = concentration_index(w.mean(dim=0).mean(dim=0))  # (H,)
    skip_frac = float((tau > old_threshold).float().mean())
    model.tau_threshold = old_threshold

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        im0 = axes[0].imshow(conc.unsqueeze(0), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axes[0].set_title(f"C(A_h) @ corr={correlation}")
        axes[0].set_xlabel("head"); axes[0].set_yticks([])
        axes[1].bar(range(model.heads), conc.numpy(), color="steelblue")
        axes[1].set_title("concentration per temporal head")
        axes[1].set_xlabel("head"); axes[1].set_ylabel("C")
        fig.colorbar(im0, ax=axes[0])
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    return {"correlation": correlation, "mean_C": float(conc.mean()),
            "max_C": float(conc.max()), "skip_frac": skip_frac}


def main():
    model = DACTemporalAttention(dim=64, heads=4, temperature_init=1.0, tau_threshold=2.0)
    for corr in (0.5, 0.9, 0.99):
        res = analyze_temporal_saturation(model, correlation=corr, save_path=f"temporal_sat_{corr}.png")
        print(f"corr={corr}  mean_C={res['mean_C']:.3f}  skip_frac={res['skip_frac']:.2f}")
    # Force heads past the threshold to show skip kicks in.
    with torch.no_grad():
        model.log_tau.fill_(math.log(3.0))
    res = analyze_temporal_saturation(model, correlation=0.99, save_path="temporal_sat_skip.png")
    print(f"after raising tau: skip_frac={res['skip_frac']:.2f}")


if __name__ == "__main__":
    main()
