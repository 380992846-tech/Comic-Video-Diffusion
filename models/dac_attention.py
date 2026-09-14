"""DAC (Differentiable Attention Concentration) temporal attention.

This is the *training-time* integration of ``models/dac_motion_module.py`` into a
real AnimateDiff motion module.

The pretrained motion module's temporal self-attention is a standard
multi-head attention over the frame axis (``TransformerTemporalModel`` reshapes
``(B, C, T, H, W) -> (B*H*W, T, C)``).  ``DACTemporalWrapper`` keeps the
pretrained ``to_q/to_k/to_v/to_out`` projections and adds only two per-head
quantities:

* ``log_tau``  — a learnable per-head temperature, so ``logits / tau_h``;
* ``mu``/``lambda_h`` — the fixed-target concentration loss
  ``lambda_h * (C(A_h) - mu_h)^2`` with ``C(A) = 1 - H(A)/log(F)``.

At inference a head with ``tau_h > tau_threshold`` is *skipped*: its output is
the average over frames instead of a ``Q K^T`` matmul (the temporal-saturation
regime the module predicts).  ``analyze_temporal_saturation`` in
``dac_motion_module.py`` measures exactly this behaviour on the trained model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def concentration_index(attn: torch.Tensor) -> torch.Tensor:
    """C(A) = 1 - H(A)/log(F); attn ``(..., F, F)`` -> ``(...)`` (query 维取平均)."""
    eps = 1e-8
    f = attn.size(-1)
    ent = (-(attn * (attn + eps).log()).sum(dim=-1)).mean(dim=-1)
    return 1.0 - ent / math.log(max(f, 2))


class DACTemporalWrapper(nn.Module):
    """Wraps a diffusers ``Attention`` module with per-head temperature."""

    def __init__(self, attn: nn.Module, tau_init: float = 1.0, tau_threshold: float = 2.0,
                 mu_init: float = 0.5, lambda_init: float = 0.1):
        super().__init__()
        self.attn = attn
        for p in self.attn.parameters():
            p.requires_grad_(False)
        heads = int(getattr(attn, "heads", 8))
        inner = getattr(attn, "to_q", None)
        if inner is None:
            raise ValueError("DACTemporalWrapper expects a diffusers Attention with to_q")
        self.heads = heads
        self.dim_head = inner.out_features // heads
        self.tau_threshold = float(tau_threshold)
        self.log_tau = nn.Parameter(torch.full((heads,), math.log(max(tau_init, 1e-3))))
        self.register_buffer("mu", torch.full((heads,), float(mu_init)))
        self.lambda_h = nn.Parameter(torch.full((heads,), softplus_inverse(lambda_init)))
        self.last_attn_weights: torch.Tensor | None = None
        self.last_conc_loss: torch.Tensor | None = None

    # -- helpers ---------------------------------------------------------
    def _project(self, x):
        a = self.attn
        q = a.to_q(x)
        k = a.to_k(x)
        v = a.to_v(x)
        if getattr(a, "norm_q", None) is not None:
            q = a.norm_q(q)
        if getattr(a, "norm_k", None) is not None:
            k = a.norm_k(k)
        return q, k, v

    def _to_out(self, x):
        out = self.attn.to_out
        if isinstance(out, (nn.ModuleList, list, tuple)):
            for module in out:
                x = module(x)
            return x
        return out(x)

    # -- forward ---------------------------------------------------------
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None,
                return_losses: bool = False, **kwargs):
        # Only the frame-axis self-attention is replaced.
        if encoder_hidden_states is not None or hidden_states.dim() != 3:
            return self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states,
                             attention_mask=attention_mask, **kwargs)

        B, T, _ = hidden_states.shape
        if T < 2:  # single frame: nothing temporal to do
            return self.attn(hidden_states, **kwargs)

        q, k, v = self._project(hidden_states)
        q = q.view(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.dim_head)
        tau = self.log_tau.exp()

        if self.training:
            scores = (q @ k.transpose(-1, -2)) * scale / tau.view(1, -1, 1, 1)
            probs = F.softmax(scores, dim=-1)
            out = probs @ v
            skip = None
            weights = probs
        else:
            skip = tau > self.tau_threshold
            out = hidden_states.new_zeros(B, self.heads, T, self.dim_head)
            active = torch.nonzero(~skip).squeeze(-1)
            skipped = torch.nonzero(skip).squeeze(-1)
            if active.numel() > 0:
                qa, ka, va = q[:, active], k[:, active], v[:, active]
                sa = (qa @ ka.transpose(-1, -2)) * scale / tau[active].view(1, -1, 1, 1)
                pa = F.softmax(sa, dim=-1)
                out[:, active] = pa @ va
                weights = pa
            else:
                weights = hidden_states.new_zeros(B, 0, T, T)
            if skipped.numel() > 0:
                out[:, skipped] = v[:, skipped].mean(dim=2, keepdim=True).expand(-1, -1, T, -1)

        out = out.permute(0, 2, 1, 3).reshape(B, T, self.heads * self.dim_head)
        out = self._to_out(out)
        self.last_attn_weights = weights.detach()

        conc_loss = hidden_states.new_zeros(())
        if weights.shape[1] > 0:
            conc = concentration_index(weights)                       # (B, H_valid)
            lam = F.softplus(self.lambda_h)
            if skip is not None and bool(skip.any()):
                mu_v, lam_v = self.mu[~skip], lam[~skip]
            else:
                mu_v, lam_v = self.mu, lam
            conc_loss = (lam_v.view(1, -1) * (conc - mu_v.view(1, -1)) ** 2).mean()
        self.last_conc_loss = conc_loss

        if return_losses:
            return out, conc_loss
        return out


def wrap_motion_attention(model: nn.Module, tau_init: float = 1.0,
                          tau_threshold: float = 2.0, mu_init: float = 0.5,
                          lambda_init: float = 0.1, verbose: bool = False) -> list[str]:
    """Replace ``attn1`` of every motion module with ``DACTemporalWrapper``."""
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        if "motion_modules" not in name:
            continue
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}"
            if child_name == "attn1" and hasattr(child, "to_q") and hasattr(child, "to_v"):
                setattr(module, child_name, DACTemporalWrapper(
                    child, tau_init, tau_threshold, mu_init, lambda_init))
                wrapped.append(full)
    if verbose:
        print(f"[dac] wrapped {len(wrapped)} temporal attention blocks; first: {wrapped[:2]}")
    return wrapped


def collect_dac_losses(model: nn.Module) -> torch.Tensor | None:
    losses = [m.last_conc_loss for m in model.modules()
              if isinstance(m, DACTemporalWrapper) and m.last_conc_loss is not None]
    if not losses:
        return None
    return torch.stack([l for l in losses]).mean()


def dac_head_statistics(model: nn.Module) -> dict:
    """Per-head tau / C(A_h) snapshot for logging and paper figures."""
    taus, concs, threshold = [], [], 2.0
    for m in model.modules():
        if isinstance(m, DACTemporalWrapper):
            threshold = m.tau_threshold
            taus.append(m.log_tau.detach().exp().double().cpu())
            if m.last_attn_weights is not None and m.last_attn_weights.numel():
                w = m.last_attn_weights
                concs.append(concentration_index(w).mean(dim=0).double().cpu())
    if not taus:
        return {}
    tau_all = torch.cat(taus)
    out = {
        "tau_mean": float(tau_all.mean()),
        "tau_max": float(tau_all.max()),
        "skip_frac": float((tau_all > threshold).float().mean()),
        "n_heads": int(tau_all.numel()),
    }
    if concs:
        c = torch.cat(concs)
        out["C_mean"] = float(c.mean())
        out["C_max"] = float(c.max())
    return out
