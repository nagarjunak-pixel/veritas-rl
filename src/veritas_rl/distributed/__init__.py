"""2D Distributed Parallelism: Online-Softmax Ring-Attention & Megatron Tensor Parallelism."""

from veritas_rl.distributed.parallel_ops import (
    MegatronMLP2D,
    ring_attention_forward,
    verify_distributed_parallel_equivalence,
)

__all__ = [
    "MegatronMLP2D",
    "ring_attention_forward",
    "verify_distributed_parallel_equivalence",
]
