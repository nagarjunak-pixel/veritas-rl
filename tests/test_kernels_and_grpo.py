"""Unit and Hypothesis property tests for C++20 native kernels, NumPy fallback, and GRPO/DAPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from veritas_rl.kernels.native_bridge import VeritasKernelBridge, get_veritas_bridge
from veritas_rl.rlvr.grpo_trainer import (
    MOD_BASE,
    TOK_ADD,
    TOK_MUL2,
    TOK_S1,
    TOK_S2,
    TOK_SUB,
    SymbolicEquationTask,
    build_verifiable_math_curriculum,
    run_grpo_verifiable_training,
    verify_symbolic_rollout,
)


def test_native_cpp20_vs_numpy_fallback_exact_parity() -> None:
    cpp_bridge = get_veritas_bridge()
    py_bridge = VeritasKernelBridge(csrc_path=Path("/tmp/nonexistent_veritas_kernel.cpp"))
    assert cpp_bridge.is_native_cpp is True
    assert "c++20-native" in cpp_bridge.backend
    assert py_bridge.is_native_cpp is False
    assert py_bridge.backend == "numpy-fallback"

    rng = np.random.default_rng(7)
    rewards_2d = rng.uniform(0.0, 1.0, size=(6, 8)).astype(np.float32)
    adv_cpp = cpp_bridge.group_relative_advantages(rewards_2d)
    adv_py = py_bridge.group_relative_advantages(rewards_2d)
    np.testing.assert_allclose(adv_cpp, adv_py, rtol=1e-5, atol=1e-5)

    rewards = rng.normal(0.0, 1.0, size=16).astype(np.float32)
    values = rng.normal(0.0, 0.5, size=16).astype(np.float32)
    lp_new = rng.uniform(-3.0, -0.1, size=16).astype(np.float32)
    lp_old = lp_new + rng.normal(0.0, 0.25, size=16).astype(np.float32)
    lp_ref = lp_new + rng.normal(0.0, 0.35, size=16).astype(np.float32)

    gae_c, kl_c, surr_c, clip_c = cpp_bridge.gae_and_k3_kl(rewards, values, lp_new, lp_old, lp_ref)
    gae_p, kl_p, surr_p, clip_p = py_bridge.gae_and_k3_kl(rewards, values, lp_new, lp_old, lp_ref)
    np.testing.assert_allclose(gae_c, gae_p, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(kl_c, kl_p, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(surr_c, surr_p, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(clip_c, clip_p)

    # Online-softmax merge parity across sequential blocks + empty block edge case
    acc_c = np.zeros(8, dtype=np.float32)
    acc_p = np.zeros(8, dtype=np.float32)
    m_c, l_c = -1e30, 0.0
    m_p, l_p = -1e30, 0.0
    for _ in range(3):
        sc = rng.normal(0.0, 1.5, size=5).astype(np.float32)
        vb = rng.normal(0.0, 1.0, size=(5, 8)).astype(np.float32)
        acc_c, m_c, l_c = cpp_bridge.online_softmax_merge(sc, vb, acc_c, m_c, l_c)
        acc_p, m_p, l_p = py_bridge.online_softmax_merge(sc, vb, acc_p, m_p, l_p)
    np.testing.assert_allclose(acc_c / l_c, acc_p / l_p, rtol=1e-5, atol=1e-5)

    empty_sc = np.zeros(0, dtype=np.float32)
    empty_vb = np.zeros((0, 8), dtype=np.float32)
    acc_empty, m_empty, l_empty = cpp_bridge.online_softmax_merge(
        empty_sc, empty_vb, acc_c, m_c, l_c
    )
    np.testing.assert_allclose(acc_empty, acc_c)
    assert m_empty == m_c and l_empty == l_c


@settings(max_examples=40, deadline=None)
@given(
    vals=st.lists(
        st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
        min_size=12,
        max_size=12,
    )
)
def test_hypothesis_group_advantage_zero_mean_and_k3_kl_nonnegative(vals: list[float]) -> None:
    bridge = get_veritas_bridge()
    mat = np.array(vals, dtype=np.float32).reshape(3, 4)
    adv = bridge.group_relative_advantages(mat, eps=1e-5)
    row_means = np.abs(np.mean(adv, axis=1))
    assert float(np.max(row_means)) < 1e-4

    arr = np.array(vals, dtype=np.float32)
    zeros = np.zeros_like(arr)
    lp_ref = arr * 0.5 - 1.5
    lp_new = -arr * 0.4 - 1.2
    _, kl, _, _ = bridge.gae_and_k3_kl(zeros, zeros, lp_new, lp_new, lp_ref)
    assert float(np.min(kl)) >= 0.0


def test_symbolic_verifier_and_grpo_post_training_lift() -> None:
    tasks = build_verifiable_math_curriculum(seed=17)
    assert len(tasks) == 48
    t = SymbolicEquationTask(
        a=3,
        op1=TOK_MUL2,
        b=1,
        op2=TOK_SUB,
        c=2,
        expected_step1=(3 * 2 + 1) % MOD_BASE,
        expected_final=((3 * 2 + 1) - 2) % MOD_BASE,
    )
    assert verify_symbolic_rollout(t, [1, 2]) == (0.0, False)
    assert verify_symbolic_rollout(t, [TOK_ADD, t.expected_step1, TOK_S2, t.expected_final]) == (
        0.0,
        False,
    )
    assert verify_symbolic_rollout(t, [TOK_S1, (t.expected_step1 + 1) % MOD_BASE, TOK_S2, 0]) == (
        0.10,
        False,
    )
    r_mid, ok_mid = verify_symbolic_rollout(
        t, [TOK_S1, t.expected_step1, TOK_S2, (t.expected_final + 1) % MOD_BASE]
    )
    assert abs(r_mid - 0.40) < 1e-6 and ok_mid is False
    r_full, ok_full = verify_symbolic_rollout(
        t, [TOK_S1, t.expected_step1, TOK_S2, t.expected_final]
    )
    assert abs(r_full - 1.00) < 1e-6 and ok_full is True

    report = run_grpo_verifiable_training(group_size_g=10, grpo_epochs=18, seed=17)
    assert report.post_rl_pass_at_1_pct > report.pre_rl_pass_at_1_pct
    assert report.final_mean_k3_kl >= 0.0
    assert len(report.history) == 18
    assert len(report.sample_solved_trajectories) >= 1
