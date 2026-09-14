"""Minimal LoRA utilities for the AnimateDiff motion module.

Deliberately dependency-free (no peft): the motion module is a small, known
sub-tree, so we inject ``LoRALinear`` wrappers ourselves and keep an explicit
list of adapters.  That makes the checkpoint format trivial (trainable tensors
plus a JSON meta file) and lets E2 change ``rank`` without touching the model
definition.

Naming convention for saved keys: ``<module path>`` as produced by
``named_modules()`` at injection time, e.g.
``down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_q``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / rank) * B(A(x)) with A ~ N(0, 1/r), B = 0."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # 让包装后的模块在接口上与 nn.Linear 保持一致（DAC 包装器等下游代码依赖这些属性）
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling
        return out + delta.to(out.dtype)


def _match_target(name: str, targets: tuple[str, ...]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def inject_lora(root: nn.Module, scope: str, targets: tuple[str, ...],
                rank: int, alpha: int, dropout: float = 0.0) -> list[str]:
    """Wrap matching ``nn.Linear`` modules under ``scope`` with LoRA.

    ``scope`` restricts the search to a sub-tree (``"motion_modules"``); a module
    path qualifies when it *contains* the scope token, so every motion module in
    the UNet is caught regardless of depth.
    """
    wrapped: list[str] = []
    for mod_name, module in root.named_modules():
        if scope and scope not in mod_name:
            continue
        for child_name, child in list(module.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and _match_target(full, targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(full)
    return wrapped


def lora_parameters(root: nn.Module):
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def count_lora_parameters(root: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in lora_parameters(root))
    total = sum(p.numel() for p in root.parameters())
    return trainable, total


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict) -> None:
    """把全部可训练张量（LoRA + DAC 参数）存成**真正的 safetensors**，meta 存进文件头。

    注意：不能对 ``.safetensors`` 后缀用 ``torch.save`` —— torch ≥2.14 会按扩展名
    选加载器，读的时候直接抛 ``SafetensorError``。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().to("cpu").contiguous()
             for n, p in model.named_parameters() if p.requires_grad}
    if not state:
        raise RuntimeError("no trainable parameters to save (did configure_trainable run?)")
    from safetensors.torch import save_file

    save_file(state, str(path), metadata={"meta": json.dumps(meta, ensure_ascii=False, default=str)})
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def _read_meta(path: Path) -> dict:
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as fh:
            raw = (fh.metadata() or {}).get("meta")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def load_checkpoint(path: str | Path, model: nn.Module, strict: bool = False) -> dict:
    """载入 safetensors（或旧的 torch-pickle）checkpoint，返回匹配统计。"""
    path = Path(path)
    meta: dict = {}
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
        meta = _read_meta(path)
    else:
        with open(path, "rb") as fh:
            blob = torch.load(fh, map_location="cpu", weights_only=True)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        meta = blob.get("meta", {}) if isinstance(blob, dict) else {}

    target = dict(model.named_parameters())
    loaded, missing = 0, []
    for name, tensor in state.items():
        if name in target and target[name].shape == tensor.shape:
            with torch.no_grad():
                target[name].copy_(tensor.to(target[name].dtype))
            loaded += 1
        else:
            missing.append(name)
    if missing and strict:
        raise KeyError(f"unmatched checkpoint keys ({len(missing)}): {missing[:8]}")
    return {"loaded": loaded, "unmatched": missing, "meta": meta}
