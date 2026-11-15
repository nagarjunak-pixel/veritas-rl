"""Rich & JSON command-line interface for VeritasRL (`veritas`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from veritas_rl.adapters.dora import run_dora_vs_lora_benchmark
from veritas_rl.api.app import create_app
from veritas_rl.distributed.parallel_ops import verify_distributed_parallel_equivalence
from veritas_rl.rlvr.grpo_trainer import run_grpo_verifiable_training
from veritas_rl.ttc.prm_ollama import run_real_ttc_benchmark

app = typer.Typer(
    name="veritas",
    help="VeritasRL: Frontier RLVR Post-Training, DoRA, 2D Parallelism & Step-Level PRM Engine",
    add_completion=False,
)
console = Console()


@app.command("train-grpo")
def train_grpo_cmd(
    group_size: int = typer.Option(
        12, help="Rollouts G per prompt for critic-free group advantage"
    ),
    epochs: int = typer.Option(32, help="GRPO/DAPO policy optimization epochs"),
    clip_high: float = typer.Option(0.28, help="DAPO asymmetric upper clip epsilon_high"),
    seed: int = typer.Option(17, help="Deterministic RNG seed"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Run real-gradient GRPO + DAPO post-training with C++20 advantages & Schulman k3 KL."""
    report = run_grpo_verifiable_training(
        group_size_g=group_size,
        grpo_epochs=epochs,
        clip_eps_high=clip_high,
        seed=seed,
    )
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    tbl = Table(title=f"VeritasRL GRPO / DAPO Verifiable-Reward Training ({report.kernel_backend})")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")
    tbl.add_row(
        "Group Size (G) / Critic Memory Saved",
        f"G={report.group_size_g} ({report.zero_critic_memory_saved_pct:.0f}% saved)",
    )
    tbl.add_row(
        "DAPO Clip Range [1 - eps_low, 1 + eps_high]",
        f"[{1.0 - report.clip_eps_low:.2f}, {1.0 + report.clip_eps_high:.2f}]",
    )
    tbl.add_row("Pre-RL SFT Baseline Pass@1", f"{report.pre_rl_pass_at_1_pct:.2f}%")
    tbl.add_row("Post-GRPO Policy Pass@1", f"{report.post_rl_pass_at_1_pct:.2f}%")
    tbl.add_row("Absolute Verifiable Accuracy Lift", f"+{report.absolute_accuracy_lift_pct:.2f}%")
    tbl.add_row("Final Schulman Non-Negative k3 KL", f"{report.final_mean_k3_kl:.5f}")
    console.print(tbl)
    for sample in report.sample_solved_trajectories[:3]:
        console.print(f"  [bold emerald]✓[/] {sample}")


@app.command("benchmark-dora")
def benchmark_dora_cmd(
    rank: int = typer.Option(8, help="Low-rank decomposition rank r"),
    steps: int = typer.Option(70, help="AdamW adaptation steps"),
    seed: int = typer.Option(23, help="Deterministic RNG seed"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Benchmark DoRA (ICML 2024 Oral) vs LoRA and verify zero-overhead inference weight merging."""
    report = run_dora_vs_lora_benchmark(rank=rank, steps=steps, seed=seed)
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    tbl = Table(title="VeritasRL DoRA Weight-Decomposed Adaptation & Zero-Latency Merge")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="magenta")
    tbl.add_row(
        "Dense vs DoRA Trainable Parameters",
        f"{report.dense_parameters:,} -> {report.dora_trainable_parameters:,} (-{report.parameter_reduction_pct:.2f}%)",
    )
    tbl.add_row("LoRA Final MSE Loss", f"{report.lora_final_loss:.5f}")
    tbl.add_row("DoRA Final MSE Loss", f"{report.dora_final_loss:.5f}")
    tbl.add_row("Merged vs Unmerged Max |Delta|", f"{report.merged_vs_unmerged_max_abs_diff:.8f}")
    console.print(tbl)


@app.command("verify-distributed")
def verify_distributed_cmd(
    world_size: int = typer.Option(4, help="Virtual Ring / Tensor-Parallel rank count P"),
    seq_len: int = typer.Option(64, help="Sequence length S divisible by P"),
    seed: int = typer.Option(31, help="Deterministic RNG seed"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Verify exact forward + backward gradient parity for Ring-Attention & Megatron 2D TP."""
    report = verify_distributed_parallel_equivalence(
        world_size=world_size,
        seq_len=seq_len,
        seed=seed,
    )
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    tbl = Table(
        title=f"VeritasRL 2D Distributed Parallelism Equivalence (P={report.world_size_shards})"
    )
    tbl.add_column("Subsystem", style="cyan")
    tbl.add_column("Max |Delta|", justify="right", style="green")
    tbl.add_row(
        "Causal Online-Softmax Ring-Attention (Forward)",
        f"{report.ring_attention_max_forward_diff:.8f}",
    )
    tbl.add_row(
        "Causal Online-Softmax Ring-Attention (Autograd dQ/dK)",
        f"{report.ring_attention_max_grad_diff:.8f}",
    )
    tbl.add_row(
        "Megatron Column+Row 2D Tensor-Parallel MLP (Forward)",
        f"{report.megatron_tp_max_forward_diff:.8f}",
    )
    tbl.add_row(
        "Megatron Column+Row 2D Tensor-Parallel MLP (Autograd dX/dW)",
        f"{report.megatron_tp_max_grad_diff:.8f}",
    )
    tbl.add_row("Mathematically Verified (< 1e-4)", str(report.mathematically_verified))
    console.print(tbl)


@app.command("eval-ttc")
def eval_ttc_cmd(
    model: str = typer.Option("qwen3.5:4b", help="Ollama model tag"),
    samples: int = typer.Option(6, help="Parallel rollouts N per Olympiad task"),
    live_only: bool = typer.Option(False, "--live-only", help="Require live Ollama inference"),
    save_path: Path | None = typer.Option(None, help="Optional custom output JSON path"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Evaluate single-sample Pass@1 vs Step-Verified PRM Best-of-N on Olympiad reasoning."""
    report = run_real_ttc_benchmark(
        model=model,
        parallel_samples_n=samples,
        save_path=save_path,
        allow_offline_replay=not live_only,
    )
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    tbl = Table(
        title=f"Zero-Mock Test-Time Compute Scaling ({report.model_name}, N={report.parallel_samples_n})"
    )
    tbl.add_column("Task ID", style="cyan")
    tbl.add_column("Truth", justify="right")
    tbl.add_column("Pass@1", justify="right")
    tbl.add_column("PRM Best-of-N", justify="right")
    tbl.add_column("Top PRM", justify="right")
    for o in report.outcomes:
        p1_str = (
            f"[green]{o.pass_at_1_answer} ✓[/]"
            if o.pass_at_1_correct
            else f"[red]{o.pass_at_1_answer} ✗[/]"
        )
        bon_str = (
            f"[green]{o.best_of_n_answer} ✓[/]"
            if o.best_of_n_correct
            else f"[red]{o.best_of_n_answer} ✗[/]"
        )
        tbl.add_row(o.task_id, str(o.ground_truth), p1_str, bon_str, f"{o.prm_top_score:.2f}")
    console.print(tbl)
    console.print(
        f"[bold]Summary:[/] Pass@1 = {report.pass_at_1_count}/{report.num_tasks} ({report.pass_at_1_pct:.2f}%) "
        f"-> PRM Best-of-{report.parallel_samples_n} = {report.prm_best_of_n_count}/{report.num_tasks} "
        f"({report.prm_best_of_n_pct:.2f}%, Lift: +{report.accuracy_lift_pct:.2f}%)"
    )


@app.command("demo")
def demo_cmd(
    as_json: bool = typer.Option(False, "--json", help="Emit unified JSON report"),
) -> None:
    """Run the complete 4-pillar VeritasRL verification suite end-to-end."""
    grpo = run_grpo_verifiable_training()
    dora = run_dora_vs_lora_benchmark()
    dist = verify_distributed_parallel_equivalence()
    payload = {
        "grpo": grpo.model_dump(),
        "dora": dora.model_dump(),
        "distributed": dist.model_dump(),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(
        f"[bold cyan]1. GRPO/DAPO ({grpo.kernel_backend}):[/] Pass@1 {grpo.pre_rl_pass_at_1_pct:.2f}% -> {grpo.post_rl_pass_at_1_pct:.2f}% (+{grpo.absolute_accuracy_lift_pct:.2f}%)"
    )
    console.print(
        f"[bold magenta]2. DoRA vs LoRA:[/] MSE {dora.lora_final_loss:.5f} -> {dora.dora_final_loss:.5f} | Merge |Delta| = {dora.merged_vs_unmerged_max_abs_diff:.8f}"
    )
    console.print(
        f"[bold green]3. 2D Parallelism (P={dist.world_size_shards}):[/] RingFwd={dist.ring_attention_max_forward_diff:.8f}, TPFwd={dist.megatron_tp_max_forward_diff:.8f} (Verified={dist.mathematically_verified})"
    )


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8040, help="Bind port"),
) -> None:
    """Launch the VeritasRL FastAPI & Frontier RLVR Workbench server."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
