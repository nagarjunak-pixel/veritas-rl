"""FastAPI control plane and Frontier RLVR & Parallelism Workbench backend for VeritasRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from veritas_rl.adapters.dora import run_dora_vs_lora_benchmark
from veritas_rl.distributed.parallel_ops import verify_distributed_parallel_equivalence
from veritas_rl.kernels.native_bridge import get_veritas_bridge
from veritas_rl.rlvr.grpo_trainer import run_grpo_verifiable_training
from veritas_rl.ttc.prm_ollama import default_ttc_artifact_path
from veritas_rl.types import (
    DistributedEquivalenceReport,
    DoRACompressionReport,
    GRPOTrainingReport,
    RealTTCBenchmarkReport,
)


class GRPORequest(BaseModel):
    group_size_g: int = Field(default=12, ge=2, le=24)
    grpo_epochs: int = Field(default=32, ge=4, le=60)
    clip_eps_low: float = Field(default=0.20, ge=0.05, le=0.50)
    clip_eps_high: float = Field(default=0.28, ge=0.05, le=0.60)
    kl_beta: float = Field(default=0.01, ge=0.0, le=0.25)
    seed: int = Field(default=17)


class DoRARequest(BaseModel):
    in_features: int = Field(default=128, ge=16, le=512)
    out_features: int = Field(default=128, ge=16, le=512)
    rank: int = Field(default=8, ge=1, le=64)
    steps: int = Field(default=70, ge=10, le=200)
    seed: int = Field(default=23)


class DistributedRequest(BaseModel):
    world_size: int = Field(default=4, ge=2, le=8)
    seq_len: int = Field(default=64, ge=16, le=256)
    d_model: int = Field(default=32, ge=8, le=128)
    seed: int = Field(default=31)


_WORKBENCH_CACHE: dict[str, Any] | None = None


def _load_ttc_snapshot() -> dict[str, Any] | None:
    artifact = default_ttc_artifact_path()
    if not artifact.exists():
        return None
    report = RealTTCBenchmarkReport.model_validate(json.loads(artifact.read_text(encoding="utf-8")))
    return report.model_dump()


def _get_workbench_cache() -> dict[str, Any]:
    global _WORKBENCH_CACHE
    if _WORKBENCH_CACHE is None:
        bridge = get_veritas_bridge()
        grpo = run_grpo_verifiable_training(
            group_size_g=12,
            grpo_epochs=32,
            clip_eps_low=0.20,
            clip_eps_high=0.28,
            kl_beta=0.01,
            seed=17,
        )
        dora = run_dora_vs_lora_benchmark(
            in_features=128,
            out_features=128,
            rank=8,
            steps=70,
            seed=23,
        )
        dist = verify_distributed_parallel_equivalence(
            world_size=4,
            seq_len=64,
            d_model=32,
            seed=31,
        )
        _WORKBENCH_CACHE = {
            "service": "veritas-rl",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
            "grpo": grpo.model_dump(),
            "dora": dora.model_dump(),
            "distributed": dist.model_dump(),
            "real_qwen_ttc": _load_ttc_snapshot(),
        }
    else:
        _WORKBENCH_CACHE["real_qwen_ttc"] = _load_ttc_snapshot()
    return _WORKBENCH_CACHE


def create_app(ui_dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="VeritasRL Frontier Post-Training & Parallelism Engine",
        version="1.0.0",
        description=(
            "GRPO/DAPO Verifiable-Reward RL, DoRA Weight Decomposition, "
            "Online-Softmax Ring-Attention + Megatron 2D TP & Step-Level PRM Test-Time Compute"
        ),
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        bridge = get_veritas_bridge()
        return {
            "status": "ok",
            "service": "veritas-rl",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
        }

    @app.get("/api/v1/overview")
    async def overview() -> dict[str, Any]:
        return _get_workbench_cache()

    @app.post("/api/v1/grpo/train", response_model=GRPOTrainingReport)
    async def grpo_train(req: GRPORequest) -> GRPOTrainingReport:
        return run_grpo_verifiable_training(
            group_size_g=req.group_size_g,
            grpo_epochs=req.grpo_epochs,
            clip_eps_low=req.clip_eps_low,
            clip_eps_high=req.clip_eps_high,
            kl_beta=req.kl_beta,
            seed=req.seed,
        )

    @app.post("/api/v1/dora/benchmark", response_model=DoRACompressionReport)
    async def dora_benchmark(req: DoRARequest) -> DoRACompressionReport:
        return run_dora_vs_lora_benchmark(
            in_features=req.in_features,
            out_features=req.out_features,
            rank=req.rank,
            steps=req.steps,
            seed=req.seed,
        )

    @app.post("/api/v1/distributed/verify", response_model=DistributedEquivalenceReport)
    async def distributed_verify(req: DistributedRequest) -> DistributedEquivalenceReport:
        return verify_distributed_parallel_equivalence(
            world_size=req.world_size,
            seq_len=req.seq_len,
            d_model=req.d_model,
            seed=req.seed,
        )

    resolved_dist = ui_dist_dir or (Path(__file__).resolve().parents[3] / "ui" / "dist")
    assets_dir = resolved_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_model=None)
    async def root_page() -> FileResponse | JSONResponse:
        index_html = resolved_dist / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse(
            {
                "service": "veritas-rl",
                "mode": "headless-api",
                "docs": "/docs",
                "overview": "/api/v1/overview",
            }
        )

    return app


app = create_app()
