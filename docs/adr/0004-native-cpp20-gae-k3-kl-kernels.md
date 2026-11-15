# ADR 0004: SHA-256 Cached Native C++20 Kernels with Bit-Compatible NumPy Fallback

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Token-level Generalized Advantage Estimation (`GAE-λ`), Schulman $k_3$ KL penalties, group advantage normalization, and online-softmax block reductions sit on the inner loop of RLVR rollout post-processing. Requiring PyBind11/CMake wheel compilation complicates lightweight container and CI portability, while pure Python loops incur interpreter overhead.

## Decision
We ship `csrc/veritas_kernels.cpp` with `extern "C"` ABI entrypoints compiled on demand via `clang++`/`g++ -O3 -std=c++20 -shared -fPIC`, keyed by the SHA-256 digest of `veritas_kernels.cpp` inside `.veritas_cache/`, and paired with an exact NumPy fallback in `VeritasKernelBridge`.

## Consequences
- Zero build-time fragility, instant `<1ms` cached dynamic library loading after first compilation, and `Hypothesis`-verified numerical parity (`< 1e-5`) between native C++20 and NumPy fallback paths.
