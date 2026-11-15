"""Tests for DoRA weight decomposition, zero-overhead merge, Ring-Attention, and Megatron 2D TP."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from veritas_rl.adapters.dora import DoRALinear, LoRALinear, run_dora_vs_lora_benchmark
from veritas_rl.distributed.parallel_ops import (
    MegatronMLP2D,
    ring_attention_forward,
    verify_distributed_parallel_equivalence,
)


def test_dora_with_bias_and_validation_guards() -> None:
    base_with_bias = nn.Linear(32, 24, bias=True)
    with pytest.raises(ValueError, match="rank must be >= 1"):
        LoRALinear(base_with_bias, rank=0)
    with pytest.raises(ValueError, match="rank must be >= 1"):
        DoRALinear(base_with_bias, rank=0)

    dora = DoRALinear(base_with_bias, rank=4, alpha=8.0)
    lora = LoRALinear(base_with_bias, rank=4, alpha=8.0)
    x = torch.randn(10, 32)
    # At initialization (lora_b == 0, m == ||W_0||_row), both match base linear exactly!
    with torch.no_grad():
        ref = base_with_bias(x)
        assert float((dora(x) - ref).abs().max().item()) < 1e-5
        assert float((lora(x) - ref).abs().max().item()) < 1e-5

        dora.lora_b.normal_(0.0, 0.2)
        dora.magnitude.mul_(1.35)
        merged = dora.merge_to_linear()
        assert float((dora(x) - merged(x)).abs().max().item()) < 1e-5


def test_dora_beats_lora_on_magnitude_plus_direction_benchmark() -> None:
    report = run_dora_vs_lora_benchmark(in_features=96, out_features=96, rank=6, steps=55, seed=23)
    assert report.dora_final_loss < report.lora_final_loss * 0.1
    assert report.merged_vs_unmerged_max_abs_diff < 1e-5
    assert report.parameter_reduction_pct > 80.0


def test_ring_attention_and_megatron_validation_and_noncausal() -> None:
    q = torch.randn(2, 18, 16)
    with pytest.raises(ValueError, match="divisible"):
        ring_attention_forward(q, q, q, num_ring_shards=4)

    q_even = torch.randn(2, 24, 16)
    k_even = torch.randn(2, 24, 16)
    v_even = torch.randn(2, 24, 16)
    out_ring_nc = ring_attention_forward(q_even, k_even, v_even, num_ring_shards=3, causal=False)
    out_ref_nc = F.scaled_dot_product_attention(q_even, k_even, v_even, is_causal=False)
    assert float((out_ring_nc - out_ref_nc).abs().max().item()) < 1e-5

    with pytest.raises(ValueError, match="divisible"):
        MegatronMLP2D(torch.randn(30, 16), torch.randn(16, 30), world_size=4)


@settings(max_examples=15, deadline=None)
@given(
    shards=st.sampled_from([2, 4]),
    mult=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=1, max_value=5000),
)
def test_hypothesis_causal_ring_attention_matches_monolithic_sdpa(
    shards: int, mult: int, seed: int
) -> None:
    torch.manual_seed(seed)
    seq_len = shards * mult
    q = torch.randn(2, seq_len, 16)
    k = torch.randn(2, seq_len, 16)
    v = torch.randn(2, seq_len, 16)
    out_ring = ring_attention_forward(q, k, v, num_ring_shards=shards, causal=True)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert float((out_ring - out_ref).abs().max().item()) < 1e-5


def test_distributed_equivalence_report_verified() -> None:
    rep = verify_distributed_parallel_equivalence(world_size=4, seq_len=48, d_model=24, seed=31)
    assert rep.mathematically_verified is True
    assert rep.ring_attention_max_forward_diff < 1e-5
    assert rep.ring_attention_max_grad_diff < 1e-5
    assert rep.megatron_tp_max_forward_diff < 1e-5
    assert rep.megatron_tp_max_grad_diff < 1e-5
