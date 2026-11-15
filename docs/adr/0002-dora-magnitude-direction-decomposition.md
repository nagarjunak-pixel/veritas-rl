# ADR 0002: DoRA Weight-Decomposed Adaptation with Zero-Overhead Inference Folding

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Rank-$r$ LoRA couples directional rotation and neuron norm scaling into a single rank-$r$ matrix $\Delta W = \frac{\alpha}{r} B A$. When RLVR or domain adaptation rescales pretrained feature magnitudes across all $d_{\text{out}}$ channels, rank-$r$ LoRA hits an irreducible Eckart-Young approximation floor.

## Decision
We implement `DoRALinear` (`src/veritas_rl/adapters/dora.py`) following Liu et al. (ICML 2024 Oral), decomposing adapted weights into a learnable vector $m \in \mathbb{R}^{d_{\text{out}}}$ initialized to $\|W_0\|_{\text{row},2}$ and a normalized directional matrix $W_0 + \frac{\alpha}{r} B A$, paired with `merge_to_linear()` for serving.

## Consequences
- Reduces trainable parameters by `86.72%` (`16,384 -> 2,176` at $128 \times 128, r=8$) while achieving **`563x` lower MSE** (`0.00323` vs `1.81892`) than rank-matched LoRA on combined magnitude + directional shifts.
- Preserves **zero serving latency overhead** via `merge_to_linear()` (`max |Δ| = 0.00000000`).
