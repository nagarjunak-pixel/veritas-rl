"""Zero-Mock Test-Time Compute Scaling & Step-Level Process Reward Model (PRM) on Live Ollama."""

from veritas_rl.ttc.prm_ollama import (
    MathOlympiadTask,
    build_olympiad_ttc_suite,
    run_real_ttc_benchmark,
    score_process_steps,
)

__all__ = [
    "MathOlympiadTask",
    "build_olympiad_ttc_suite",
    "run_real_ttc_benchmark",
    "score_process_steps",
]
