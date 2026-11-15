"""DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al., ICML 2024 Oral) vs. LoRA Engine.

Decomposes fine-tuning updates into directional low-rank matrices (B @ A) and a learnable
magnitude vector `m`, preserving zero-overhead inference merging:
  W' = diag(m) @ ((W_0 + (alpha / r) * B @ A) / ||W_0 + (alpha / r) * B @ A||_row)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from veritas_rl.types import DoRACompressionReport


class LoRALinear(nn.Module):
    """Standard Hu et al. (2022) Low-Rank Adaptation baseline."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / float(rank)

        self.weight_frozen = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias_frozen: nn.Parameter | None = None
        if base.bias is not None:
            self.bias_frozen = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def effective_weight(self) -> torch.Tensor:
        out: torch.Tensor = self.weight_frozen + self.scale * (self.lora_b @ self.lora_a)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.linear(x, self.effective_weight(), self.bias_frozen)
        return out


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024 Oral)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / float(rank)

        w0 = base.weight.detach().clone()
        self.weight_frozen = nn.Parameter(w0, requires_grad=False)
        self.bias_frozen: nn.Parameter | None = None
        if base.bias is not None:
            self.bias_frozen = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        # Initialize magnitude vector m to exact row-norms of pretrained weight W_0
        init_mag = w0.norm(p=2, dim=1)
        self.magnitude = nn.Parameter(init_mag)

        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_weight(self) -> torch.Tensor:
        directional = self.weight_frozen + self.scale * (self.lora_b @ self.lora_a)
        row_norms = directional.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        unit_dir = directional / row_norms
        out: torch.Tensor = self.magnitude.unsqueeze(1) * unit_dir
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.linear(x, self.effective_weight(), self.bias_frozen)
        return out

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        """Fold DoRA magnitude + directional update into a standard zero-overhead `nn.Linear`."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias_frozen is not None,
        )
        merged.weight.copy_(self.effective_weight())
        if self.bias_frozen is not None and merged.bias is not None:
            merged.bias.copy_(self.bias_frozen)
        return merged


def run_dora_vs_lora_benchmark(
    *,
    in_features: int = 128,
    out_features: int = 128,
    rank: int = 8,
    steps: int = 70,
    seed: int = 23,
) -> DoRACompressionReport:
    """Benchmark DoRA vs LoRA convergence on directional + magnitude shift adaptation,
    and prove bit-exact zero-latency merging equivalence.
    """
    torch.manual_seed(seed)
    base = nn.Linear(in_features, out_features, bias=False)

    # Pretrained weight W_0 has O(1) row norms; target task applies BOTH a full-rank neuron
    # magnitude rescaling (which rank-r LoRA cannot span due to Eckart-Young rank bound)
    # AND a genuine rank-r directional rotation (which DoRA isolates into B @ A).
    with torch.no_grad():
        base.weight.copy_(torch.randn(out_features, in_features) * 0.35)

    u_true = torch.randn(out_features, rank) * 0.08
    v_true = torch.randn(rank, in_features) * 0.08
    mag_scale = torch.linspace(0.45, 1.85, steps=out_features).unsqueeze(1)
    w_target = mag_scale * (base.weight.detach() + (u_true @ v_true))

    x_train = torch.randn(256, in_features)
    y_train = x_train @ w_target.T

    dora = DoRALinear(base, rank=rank, alpha=16.0)
    lora = LoRALinear(base, rank=rank, alpha=16.0)
    with torch.no_grad():
        lora.lora_a.copy_(dora.lora_a)

    opt_dora = torch.optim.AdamW(
        [
            {"params": [dora.magnitude], "lr": 8e-2},
            {"params": [dora.lora_a, dora.lora_b], "lr": 2.5e-2},
        ],
        weight_decay=0.0,
    )
    opt_lora = torch.optim.AdamW(
        [p for p in lora.parameters() if p.requires_grad],
        lr=2.5e-2,
        weight_decay=0.0,
    )

    dora_loss_val = 0.0
    lora_loss_val = 0.0
    for _ in range(steps):
        opt_dora.zero_grad()
        loss_d = F.mse_loss(dora(x_train), y_train)
        torch.autograd.backward(loss_d)
        opt_dora.step()
        dora_loss_val = float(loss_d.item())

        opt_lora.zero_grad()
        loss_l = F.mse_loss(lora(x_train), y_train)
        torch.autograd.backward(loss_l)
        opt_lora.step()
        lora_loss_val = float(loss_l.item())

    merged_linear = dora.merge_to_linear()
    with torch.no_grad():
        x_probe = torch.randn(32, in_features)
        max_diff = float((dora(x_probe) - merged_linear(x_probe)).abs().max().item())

    dense_params = in_features * out_features
    trainable = dora.trainable_parameter_count()
    reduction_pct = (1.0 - (trainable / float(dense_params))) * 100.0

    return DoRACompressionReport(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        dense_parameters=dense_params,
        dora_trainable_parameters=trainable,
        parameter_reduction_pct=round(reduction_pct, 2),
        merged_vs_unmerged_max_abs_diff=round(max_diff, 8),
        dora_final_loss=round(dora_loss_val, 5),
        lora_final_loss=round(lora_loss_val, 5),
    )
