"""Persist raw rows and machine/human-readable aggregate reports."""

from __future__ import annotations

import json
from pathlib import Path

from codeswarm_agent.evaluation.layered import LayerValidationResult
from codeswarm_agent.evaluation.metrics import EvaluationResult, aggregate, result_to_dict


def write_report(results: tuple[EvaluationResult, ...], output_directory: str | Path) -> dict:
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result_to_dict(result), ensure_ascii=False) + "\n")
    metrics = aggregate(results)
    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def value(path: str) -> object:
        current: object = metrics
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    lines = [
        "# CodeSwarm Evaluation Report", "",
        f"- All tasks: {metrics['all_tasks']}",
        f"- Eligible tasks: {metrics['eligible_tasks']}",
        f"- Infrastructure failures: {metrics['infra_failed']}",
        f"- Infrastructure error rate: {metrics['infrastructure_error_rate']}",
        f"- Resolved: {metrics['resolved']}",
        f"- Resolved@1: {metrics['resolved_at_1']}",
        f"- Claimed success: {metrics['claimed_success']}",
        f"- False completions: {metrics['false_completed']}",
        f"- False completion rate: {metrics['false_completion_rate']}",
        f"- Patch apply rate: {metrics['patch_apply_rate']}",
        f"- Mean/P95 tokens: {value('token_distribution.mean')} / {value('token_distribution.p95')}",
        f"- Mean/P95 wall seconds: {value('wall_time_distribution.mean')} / {value('wall_time_distribution.p95')}",
        f"- Mean steps: {value('step_distribution.mean')}",
        f"- Mean tool calls: {value('tool_call_distribution.mean')}",
        f"- Actual mode distribution: {metrics['actual_mode_distribution']}",
        f"- Cost coverage: {metrics['cost_coverage']}",
        f"- Cost per resolved task: {metrics['cost_per_resolved']}",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def write_validation_report(
    result: LayerValidationResult, output_directory: str | Path
) -> None:
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    (root / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Layered Validation Report",
        "",
        f"- Level: {result.level.value.upper()}",
        f"- Profile: {result.profile_name}",
        f"- Status: {result.status.value.upper()}",
        "",
        "| Metric | Observed | Rule | Threshold | Status |",
        "|---|---:|---|---:|---|",
    ]
    for check in result.checks:
        observed = "N/A" if check.observed is None else f"{check.observed:.6g}"
        threshold = "N/A" if check.threshold is None else f"{check.threshold:.6g}"
        lines.append(
            f"| {check.metric} | {observed} | {check.comparison} | "
            f"{threshold} | {check.status.value.upper()} |"
        )
    (root / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
