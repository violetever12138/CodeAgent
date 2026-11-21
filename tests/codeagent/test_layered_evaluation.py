import json
import sys
from dataclasses import replace

from codeagent.evaluation.cli import main
from codeagent.evaluation.layered import (
    ValidationLevel,
    ValidationStatus,
    evaluate_layer,
)
from codeagent.evaluation.manifest import EvaluationManifest
from codeagent.evaluation.metrics import (
    EvaluationResult,
    aggregate,
    compare_paired,
    result_to_dict,
)
from codeagent.evaluation.report import write_validation_report


def _row(
    index: int,
    *,
    resolved: bool = True,
    claimed: bool = True,
    tokens: int = 100,
    wall: float = 2.0,
    infra: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        instance_id=f"task-{index}",
        agent_claimed_success=claimed,
        patch_generated=True,
        patch_applied=True,
        harness_resolved=resolved,
        termination_reason="completed",
        verification_decision="accept",
        input_tokens=tokens,
        steps=3,
        tool_calls=4,
        wall_seconds=wall,
        cost=0.01,
        infra_error=infra,
        task_type="bugfix" if index % 2 else "feature",
    )


def _manifest() -> EvaluationManifest:
    return EvaluationManifest(
        run_id="layered-run",
        git_commit="abc",
        agent_variant="multi",
        task_set_version="python-repo-v1",
        model_provider="provider",
        model_name="model",
        model_revision="revision",
        temperature=0.0,
        seed=1,
        budget={"tokens": 200, "wall_seconds": 5, "tool_calls": 10},
        sandbox_profile="repo_task_v1",
        prompt_version="p1",
        policy_version="policy-v1",
        harness_version="harness-v1",
    )


def test_aggregate_reports_resource_distributions_and_task_buckets():
    metrics = aggregate((_row(1), _row(2, resolved=False)))

    assert metrics["infrastructure_error_rate"] == 0.0
    assert metrics["patch_generation_rate"] == 1.0
    assert metrics["step_distribution"]["mean"] == 3.0
    assert metrics["tool_call_distribution"]["p95"] == 4.0
    assert metrics["cost_coverage"] == 1.0
    assert set(metrics["by_task_type"]) == {"bugfix", "feature"}


def test_aggregate_exposes_context_index_multi_agent_and_safety_metrics():
    row = replace(_row(1), telemetry={
        "context.recall_at_12": 0.8, "context.mrr_at_12": 1.0,
        "context.ndcg_at_12": 0.9, "context.file_diversity": 0.5,
        "context.stale_hit_rate": 0.0, "context.rerank_ms": 12.0,
        "context.reranker_fallback": 0, "index.scanned_files": 2,
        "index.rechunked_files": 1, "multi_agent.effective_work_units": 3,
        "safety.timeout_process_reclaims": 1,
    })
    metrics = aggregate((row,))

    assert metrics["context"]["recall_at_12"] == 0.8
    assert metrics["index"]["rechunked_files"] == 1
    assert metrics["multi_agent"]["effective_work_units"] == 3
    assert metrics["safety"]["timeout_process_reclaims"] == 1


def test_l0_contract_gate_passes_only_with_complete_contract_evidence():
    rows = tuple(_row(index) for index in range(8))

    result = evaluate_layer(ValidationLevel.L0, rows)

    assert result.status == ValidationStatus.PASS
    assert result.profile_name == "l0-contract-v1"


def test_empty_layer_is_not_run_and_small_l1_is_inconclusive():
    assert evaluate_layer("l0", ()).status == ValidationStatus.NOT_RUN

    result = evaluate_layer("l1", (_row(1),))

    assert result.status == ValidationStatus.INCONCLUSIVE
    assert any(check.metric == "baseline" for check in result.checks)
    assert any(check.metric == "manifest" for check in result.checks)


def test_l1_gate_checks_quality_budget_and_paired_performance(tmp_path):
    baseline = tuple(_row(index) for index in range(20))
    candidate = tuple(
        _row(index, tokens=120, wall=2.5) for index in range(20)
    )

    result = evaluate_layer(
        "l1",
        candidate,
        baseline=baseline,
        manifest=_manifest(),
        baseline_manifest=_manifest(),
    )
    write_validation_report(result, tmp_path)

    assert result.status == ValidationStatus.PASS
    assert result.metrics["budget"]["token_compliance"] == 1.0
    assert result.metrics["paired"]["resolved_delta"] == 0.0
    assert result.metrics["paired"]["mean_token_ratio"] == 1.2
    saved = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert saved["status"] == "pass"
    assert "L1" in (tmp_path / "validation.md").read_text(encoding="utf-8")


def test_l1_gate_fails_on_false_completion_and_quality_regression():
    baseline = tuple(_row(index) for index in range(20))
    candidate = tuple(
        _row(index, resolved=index < 10) for index in range(20)
    )

    result = evaluate_layer(
        "l1",
        candidate,
        baseline=baseline,
        manifest=_manifest(),
        baseline_manifest=_manifest(),
    )

    assert result.status == ValidationStatus.FAIL
    failed = {check.metric for check in result.checks if check.status == ValidationStatus.FAIL}
    assert "false_completion_rate" in failed
    assert "paired.resolved_delta" in failed


def test_l1_fails_when_experiment_controls_do_not_match():
    baseline = tuple(_row(index) for index in range(20))
    candidate = tuple(_row(index) for index in range(20))
    changed = _manifest()
    changed = EvaluationManifest(
        **{
            **changed.__dict__,
            "budget": {"tokens": 400, "wall_seconds": 5, "tool_calls": 10},
        }
    )

    result = evaluate_layer(
        "l1",
        candidate,
        baseline=baseline,
        manifest=_manifest(),
        baseline_manifest=changed,
    )

    assert result.status == ValidationStatus.FAIL
    assert result.metrics["experiment"]["control_match"] == 0.0


def test_manifest_v1_and_v2_are_not_compared_as_an_improvement():
    rows = tuple(_row(index) for index in range(20))
    result = evaluate_layer(
        "l1", rows, baseline=rows,
        manifest=_manifest(), baseline_manifest=replace(_manifest(), schema_version=1),
    )

    assert result.status == ValidationStatus.FAIL
    assert result.metrics["experiment"]["schema_match"] == 0.0


def test_l2_is_not_run_without_inspected_docker_isolation():
    rows = tuple(_row(index) for index in range(50))
    result = evaluate_layer(
        "l2", rows, baseline=rows,
        manifest=replace(
            _manifest(), execution_backend="windows-guarded",
            execution_strength="process_guarded_trusted",
        ),
        baseline_manifest=_manifest(),
    )

    assert result.status == ValidationStatus.NOT_RUN
    assert result.checks[0].metric == "safety.docker_isolation"


def test_paired_metrics_expose_coverage_quality_and_resource_ratios():
    baseline = (_row(1, tokens=100, wall=2), _row(2, resolved=False, tokens=100, wall=4))
    candidate = (_row(1, tokens=150, wall=3), _row(2, tokens=150, wall=6))

    paired = compare_paired(baseline, candidate)

    assert paired["task_coverage"] == 1.0
    assert paired["paired_eligible_coverage"] == 1.0
    assert paired["resolved_delta"] == 0.5
    assert paired["mean_token_ratio"] == 1.5
    assert paired["p95_wall_time_ratio"] == 1.5


def test_eval_cli_writes_and_enforces_l0_validation(tmp_path, monkeypatch):
    rows = tuple(_row(index) for index in range(8))
    source = tmp_path / "contracts.jsonl"
    source.write_text(
        "\n".join(json.dumps(result_to_dict(row)) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codeswarm-eval",
            str(source),
            "--output",
            str(output),
            "--level",
            "l0",
            "--enforce",
        ],
    )

    main()

    validation = json.loads(
        (output / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "pass"
