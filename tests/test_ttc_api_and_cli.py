"""Tests for Step-Level PRM, Ollama TTC benchmark, FastAPI endpoints, and Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from veritas_rl.api.app import create_app
from veritas_rl.cli import app as cli_app
from veritas_rl.ttc.prm_ollama import (
    build_olympiad_ttc_suite,
    run_real_ttc_benchmark,
    score_process_steps,
    select_prm_best_of_n,
)


def test_step_level_prm_catches_arithmetic_hallucinations() -> None:
    assert score_process_steps([], None) == 0.05
    valid_steps = [
        "Step 1: -5 + 23 = 18",
        "Step 2: 18 * 3 = 54",
        "Step 3: 54 + 6 = 60, final answer is 60",
    ]
    broken_steps = [
        "Step 1: 17 * 6 = 92",
        "Step 2: 92 - 32 = 60, final answer is 60",
    ]
    # Also exercise unsupported division / malformed equation graceful skip
    skip_steps = [
        "Step 1: 20 / 4 = 5 and 10 + 15 = 25",
        "Therefore 25",
    ]
    valid_score = score_process_steps(valid_steps, 60)
    broken_score = score_process_steps(broken_steps, 60)
    skip_score = score_process_steps(skip_steps, 25)
    assert valid_score > 0.75
    assert broken_score <= 0.15
    assert skip_score > 0.40

    best_ans, top_prm = select_prm_best_of_n(
        [(broken_steps, 99), (broken_steps, 99), (valid_steps, 60)]
    )
    assert best_ans == 60
    assert top_prm == valid_score

    none_ans, _ = select_prm_best_of_n([([], None)])
    assert none_ans is None


def test_ttc_benchmark_mocked_live_and_offline_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = build_olympiad_ttc_suite()
    assert len(tasks) == 8
    out_file = tmp_path / "ttc_eval.json"

    call_counter = {"n": 0}

    def fake_get(url: str, timeout: float = 2.0) -> httpx.Response:
        req = httpx.Request("GET", url)
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:4b"}]}, request=req)

    def fake_post(url: str, json: dict[str, Any], timeout: float = 60.0) -> httpx.Response:
        call_counter["n"] += 1
        idx = call_counter["n"]
        req = httpx.Request("POST", url)
        # Inject edge cases on first sample of certain calls so Best-of-N beats Pass@1
        if idx % 3 == 1:
            return httpx.Response(
                200,
                json={"response": '{"steps": ["bad: 4 * 4 = 19"], "final_answer": false}'},
                request=req,
            )
        if idx == 2:
            return httpx.Response(200, json={"response": "[1, 2, 3]"}, request=req)
        if idx == 5:
            return httpx.Response(
                200,
                json={"response": '{"steps": ["x"], "final_answer": "not-a-num"}'},
                request=req,
            )
        # Extract problem match to return exact verified ground truth on good rollouts
        prompt_str = str(json.get("prompt", ""))
        truth = 18
        for t in tasks:
            if t.problem_text in prompt_str:
                truth = t.ground_truth
                break
        payload = f'{{"steps": ["verified: {truth} + 0 = {truth}"], "final_answer": {truth}}}'
        return httpx.Response(200, json={"response": payload}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    live_rep = run_real_ttc_benchmark(
        model="qwen3.5:4b",
        parallel_samples_n=3,
        save_path=out_file,
        allow_offline_replay=False,
    )
    assert out_file.exists()
    assert live_rep.prm_best_of_n_pct == 100.0
    assert live_rep.prm_best_of_n_pct > live_rep.pass_at_1_pct

    # Now simulate Ollama offline -> verify deterministic replay and live_only RuntimeError
    def offline_get(url: str, timeout: float = 2.0) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    replayed = run_real_ttc_benchmark(save_path=out_file, allow_offline_replay=True)
    assert replayed.prm_best_of_n_count == live_rep.prm_best_of_n_count

    with pytest.raises(RuntimeError, match="unreachable"):
        run_real_ttc_benchmark(
            save_path=tmp_path / "missing.json",
            allow_offline_replay=False,
        )


async def test_fastapi_endpoints_and_static_serving(tmp_path: Path) -> None:
    app_default = create_app()
    transport = httpx.ASGITransport(app=app_default)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_health = await client.get("/healthz")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_over = await client.get("/api/v1/overview")
        assert r_over.status_code == 200
        assert r_over.json()["distributed"]["mathematically_verified"] is True

        r_grpo = await client.post(
            "/api/v1/grpo/train",
            json={"group_size_g": 6, "grpo_epochs": 6, "seed": 17},
        )
        assert r_grpo.status_code == 200
        assert len(r_grpo.json()["history"]) == 6

        r_dora = await client.post(
            "/api/v1/dora/benchmark",
            json={"in_features": 64, "out_features": 64, "rank": 4, "steps": 25, "seed": 23},
        )
        assert r_dora.status_code == 200
        assert r_dora.json()["merged_vs_unmerged_max_abs_diff"] < 1e-5

        r_dist = await client.post(
            "/api/v1/distributed/verify",
            json={"world_size": 2, "seq_len": 32, "d_model": 16, "seed": 31},
        )
        assert r_dist.status_code == 200
        assert r_dist.json()["mathematically_verified"] is True

        r_root = await client.get("/")
        assert r_root.status_code == 200

    # Also verify headless fallback when ui_dist_dir has no index.html
    app_headless = create_app(ui_dist_dir=tmp_path / "empty_ui")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_headless),
        base_url="http://testserver",
    ) as h_client:
        r_head = await h_client.get("/")
        assert r_head.status_code == 200
        assert r_head.json()["mode"] == "headless-api"


def test_cli_commands_rich_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    res_grpo = runner.invoke(cli_app, ["train-grpo", "--group-size", "6", "--epochs", "6"])
    assert res_grpo.exit_code == 0
    res_grpo_j = runner.invoke(
        cli_app, ["train-grpo", "--group-size", "6", "--epochs", "6", "--json"]
    )
    assert res_grpo_j.exit_code == 0
    assert json.loads(res_grpo_j.stdout)["group_size_g"] == 6

    res_dora = runner.invoke(cli_app, ["benchmark-dora", "--steps", "20"])
    assert res_dora.exit_code == 0
    res_dora_j = runner.invoke(cli_app, ["benchmark-dora", "--steps", "20", "--json"])
    assert res_dora_j.exit_code == 0
    assert "dora_final_loss" in json.loads(res_dora_j.stdout)

    res_dist = runner.invoke(
        cli_app, ["verify-distributed", "--world-size", "2", "--seq-len", "32"]
    )
    assert res_dist.exit_code == 0
    res_dist_j = runner.invoke(
        cli_app, ["verify-distributed", "--world-size", "2", "--seq-len", "32", "--json"]
    )
    assert res_dist_j.exit_code == 0
    assert json.loads(res_dist_j.stdout)["mathematically_verified"] is True

    res_demo = runner.invoke(cli_app, ["demo"])
    assert res_demo.exit_code == 0
    res_demo_j = runner.invoke(cli_app, ["demo", "--json"])
    assert res_demo_j.exit_code == 0
    assert "grpo" in json.loads(res_demo_j.stdout)

    # Create a synthetic saved TTC artifact and force offline replay via CLI
    fake_ttc = tmp_path / "saved_ttc.json"
    fake_ttc.write_text(
        json.dumps(
            {
                "model_name": "qwen3.5:4b",
                "num_tasks": 1,
                "parallel_samples_n": 3,
                "pass_at_1_count": 0,
                "pass_at_1_pct": 0.0,
                "prm_best_of_n_count": 1,
                "prm_best_of_n_pct": 100.0,
                "accuracy_lift_pct": 100.0,
                "mean_task_latency_ms": 120.0,
                "outcomes": [
                    {
                        "task_id": "TTC-01",
                        "category": "Test",
                        "ground_truth": 18,
                        "pass_at_1_answer": 12,
                        "pass_at_1_correct": False,
                        "best_of_n_answer": 18,
                        "best_of_n_correct": True,
                        "prm_top_score": 0.95,
                        "samples_drawn": 3,
                        "latency_ms": 120.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def offline_get(url: str, timeout: float = 2.0) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    res_ttc = runner.invoke(cli_app, ["eval-ttc", "--save-path", str(fake_ttc)])
    assert res_ttc.exit_code == 0
    res_ttc_j = runner.invoke(cli_app, ["eval-ttc", "--save-path", str(fake_ttc), "--json"])
    assert res_ttc_j.exit_code == 0
    assert json.loads(res_ttc_j.stdout)["prm_best_of_n_pct"] == 100.0
