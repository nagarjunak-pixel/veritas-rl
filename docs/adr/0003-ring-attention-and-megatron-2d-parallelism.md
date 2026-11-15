# ADR 0003: Causal Online-Softmax Ring-Attention & Megatron 2D Tensor Parallelism

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Long-horizon chain-of-thought RLVR rollouts suffer quadratic $O(S^2)$ attention activation growth along sequence dimension $S$ and weight memory bottlenecks in MLP blocks ($4 d_{\text{model}}^2$).

## Decision
We implement a mathematically verified 2D distributed parallelism subsystem (`src/veritas_rl/distributed/parallel_ops.py`):
1. **Sequence Parallelism (`ring_attention_forward`):** Circulates $(K_p, V_p)$ blocks across $P$ ring ranks using online log-sum-exp rescaling $(O, m, \ell)$ and skips strictly future shards ($p > r$) under causal masking.
2. **Tensor Parallelism (`MegatronMLP2D`):** Splits $W_1$ column-parallel and $W_2$ row-parallel across $P$ ranks with zero communication at the $\operatorname{GeLU}$ nonlinearity and a single `All-Reduce(SUM)` at layer exit.

## Consequences
- Verified forward AND PyTorch autograd backward equivalence (`< 1.5e-6` max absolute difference) against monolithic `F.scaled_dot_product_attention` and dense MLPs across `Hypothesis` randomized sequence lengths and shard counts.
