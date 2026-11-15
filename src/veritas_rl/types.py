"""Typed domain models for VeritasRL post-training, DoRA, 2D parallelism, and test-time compute."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GRPOEpochMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: int
    mean_verifiable_reward: float
    pass_at_1_pct: float
    mean_k3_kl: float = Field(ge=0.0)
    clipped_token_fraction: float = Field(ge=0.0, le=1.0)
    policy_loss: float


class GRPOTrainingReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_backend: str
    group_size_g: int
    clip_eps_low: float
    clip_eps_high: float
    kl_beta: float
    pre_rl_pass_at_1_pct: float
    post_rl_pass_at_1_pct: float
    absolute_accuracy_lift_pct: float
    final_mean_k3_kl: float
    zero_critic_memory_saved_pct: float
    history: list[GRPOEpochMetrics]
    sample_solved_trajectories: list[str]


class DoRACompressionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    in_features: int
    out_features: int
    rank: int
    dense_parameters: int
    dora_trainable_parameters: int
    parameter_reduction_pct: float
    merged_vs_unmerged_max_abs_diff: float
    dora_final_loss: float
    lora_final_loss: float


class DistributedEquivalenceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    world_size_shards: int
    sequence_length: int
    d_model: int
    ring_attention_max_forward_diff: float
    ring_attention_max_grad_diff: float
    megatron_tp_max_forward_diff: float
    megatron_tp_max_grad_diff: float
    peak_attention_matrix_memory_reduction_x: float
    mathematically_verified: bool


class RealTTCTaskOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    category: str
    ground_truth: int
    pass_at_1_answer: int | None
    pass_at_1_correct: bool
    best_of_n_answer: int | None
    best_of_n_correct: bool
    prm_top_score: float
    samples_drawn: int
    latency_ms: float


class RealTTCBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    num_tasks: int
    parallel_samples_n: int
    pass_at_1_count: int
    pass_at_1_pct: float
    prm_best_of_n_count: int
    prm_best_of_n_pct: float
    accuracy_lift_pct: float
    mean_task_latency_ms: float
    outcomes: list[RealTTCTaskOutcome]
