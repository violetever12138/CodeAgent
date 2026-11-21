"""Task-level metrics with explicit denominators and paired comparison."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class EvaluationResult:
    instance_id: str
    agent_claimed_success: bool
    patch_generated: bool
    patch_applied: bool
    harness_resolved: bool
    termination_reason: str
    verification_decision: str
    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0
    cost: float | None = None
    infra_error: str | None = None
    task_type: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id cannot be empty")
        if min(
            self.input_tokens, self.output_tokens, self.steps,
            self.tool_calls, self.wall_seconds,
        ) < 0:
            raise ValueError("usage values cannot be negative")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[index],
    }


def _bucket_metrics(rows: tuple[EvaluationResult, ...]) -> dict[str, object]:
    resolved = tuple(row for row in rows if row.harness_resolved)
    claimed = tuple(row for row in rows if row.agent_claimed_success)
    false_completed = tuple(row for row in claimed if not row.harness_resolved)
    return {
        "eligible_tasks": len(rows),
        "resolved": len(resolved),
        "resolved_at_1": len(resolved) / len(rows) if rows else None,
        "false_completion_rate": (
            len(false_completed) / len(claimed) if claimed else None
        ),
        "token_distribution": _distribution(
            [float(row.total_tokens) for row in rows]
        ),
        "wall_time_distribution": _distribution(
            [row.wall_seconds for row in rows]
        ),
    }


def aggregate(results: Iterable[EvaluationResult]) -> dict[str, object]:
    rows = tuple(results)
    identifiers = [row.instance_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate instance_id in evaluation results")
    eligible = tuple(row for row in rows if row.infra_error is None)
    resolved = tuple(row for row in eligible if row.harness_resolved)
    claimed = tuple(row for row in eligible if row.agent_claimed_success)
    false_completed = tuple(row for row in claimed if not row.harness_resolved)
    unresolved = tuple(row for row in eligible if not row.harness_resolved)
    known_costs = [row.cost for row in eligible if row.cost is not None]
    task_types = sorted({row.task_type or "unclassified" for row in eligible})
    actual_modes = Counter(
        row.artifacts.get("actual_mode", "unrecorded") for row in eligible
    )
    routing_pairs = Counter(
        (
            row.artifacts.get("expected_mode", "unrecorded"),
            row.artifacts.get("actual_mode", "unrecorded"),
        )
        for row in eligible
    )
    metrics = {
        "all_tasks": len(rows),
        "eligible_tasks": len(eligible),
        "infra_failed": len(rows) - len(eligible),
        "infrastructure_error_rate": (
            (len(rows) - len(eligible)) / len(rows) if rows else None
        ),
        "resolved": len(resolved),
        "resolved_at_1": len(resolved) / len(eligible) if eligible else None,
        "tokens_per_resolved": (
            sum(row.total_tokens for row in eligible) / len(resolved)
            if resolved else None
        ),
        "patch_generation_rate": (
            sum(row.patch_generated for row in eligible) / len(eligible)
            if eligible else None
        ),
        "claimed_success": len(claimed),
        "false_completed": len(false_completed),
        "false_completion_rate": (
            len(false_completed) / len(claimed) if claimed else None
        ),
        "patch_apply_rate": (
            sum(row.patch_applied for row in eligible) / len(eligible) if eligible else None
        ),
        "token_distribution": _distribution(
            [float(row.total_tokens) for row in eligible]
        ),
        "wall_time_distribution": _distribution(
            [row.wall_seconds for row in eligible]
        ),
        "step_distribution": _distribution([float(row.steps) for row in eligible]),
        "tool_call_distribution": _distribution(
            [float(row.tool_calls) for row in eligible]
        ),
        "cost_coverage": len(known_costs) / len(eligible) if eligible else None,
        "cost_distribution": _distribution([float(value) for value in known_costs]),
        "cost_per_resolved": (
            sum(known_costs) / len(resolved)
            if resolved and len(known_costs) == len(eligible) else None
        ),
        "resolved_token_distribution": _distribution(
            [float(row.total_tokens) for row in resolved]
        ),
        "unresolved_token_distribution": _distribution(
            [float(row.total_tokens) for row in unresolved]
        ),
        "resolved_wall_time_distribution": _distribution(
            [row.wall_seconds for row in resolved]
        ),
        "unresolved_wall_time_distribution": _distribution(
            [row.wall_seconds for row in unresolved]
        ),
        "by_task_type": {
            task_type: _bucket_metrics(
                tuple(
                    row for row in eligible
                    if (row.task_type or "unclassified") == task_type
                )
            )
            for task_type in task_types
        },
        "actual_mode_distribution": dict(sorted(actual_modes.items())),
        "expected_to_actual_modes": {
            expected: {
                actual: routing_pairs[(expected, actual)]
                for actual in sorted(
                    {pair[1] for pair in routing_pairs if pair[0] == expected}
                )
            }
            for expected in sorted({pair[0] for pair in routing_pairs})
        },
        "infra_errors": {
            row.instance_id: row.infra_error for row in rows if row.infra_error is not None
        },
    }
    metrics.update(_aggregate_telemetry(eligible))
    return metrics


def _aggregate_telemetry(rows: tuple[EvaluationResult, ...]) -> dict[str, object]:
    def values(name: str) -> list[float]:
        return [float(row.telemetry[name]) for row in rows if name in row.telemetry]

    def mean(name: str) -> float | None:
        observed = values(name)
        return statistics.fmean(observed) if observed else None

    def total(name: str) -> float:
        return sum(values(name))

    rerank = values("context.rerank_ms")
    fallbacks = values("context.reranker_fallback")
    return {
        "context": {
            "recall_at_12": mean("context.recall_at_12"),
            "mrr_at_12": mean("context.mrr_at_12"),
            "ndcg_at_12": mean("context.ndcg_at_12"),
            "stale_hit_rate": mean("context.stale_hit_rate"),
            "file_diversity": mean("context.file_diversity"),
            "rerank_latency": _distribution(rerank),
            "fallback_rate": statistics.fmean(fallbacks) if fallbacks else None,
        },
        "index": {
            "scanned_files": total("index.scanned_files"),
            "rechunked_files": total("index.rechunked_files"),
            "updated_rows": total("index.updated_rows"),
            "full_rebuilds": total("index.full_rebuilds"),
            "artifact_insert_latency_ms": _distribution(
                values("index.artifact_insert_latency_ms")
            ),
        },
        "multi_agent": {
            "effective_work_units": mean("multi_agent.effective_work_units"),
            "parallelism": mean("multi_agent.parallelism"),
            "integration_success_rate": mean("multi_agent.integration_success"),
            "patch_conflict_rate": mean("multi_agent.patch_conflict"),
            "security_block_rate": mean("multi_agent.security_block"),
            "replan_count": mean("multi_agent.replan_count"),
        },
        "safety": {
            "policy_refusals": total("safety.policy_refusals"),
            "timeout_process_reclaims": total("safety.timeout_process_reclaims"),
            "environment_leaks": total("safety.environment_leaks"),
            "workspace_escapes": total("safety.workspace_escapes"),
            "docker_inspect_verified_rate": mean("safety.docker_inspect_verified"),
        },
    }


def compare_paired(
    baseline: Iterable[EvaluationResult], treatment: Iterable[EvaluationResult]
) -> dict[str, object]:
    left = {row.instance_id: row for row in baseline}
    right = {row.instance_id: row for row in treatment}
    common = sorted(left.keys() & right.keys())
    usable = [
        key for key in common
        if left[key].infra_error is None and right[key].infra_error is None
    ]
    wins = [key for key in usable if right[key].harness_resolved and not left[key].harness_resolved]
    losses = [key for key in usable if left[key].harness_resolved and not right[key].harness_resolved]
    ties = [key for key in usable if left[key].harness_resolved == right[key].harness_resolved]
    union = left.keys() | right.keys()
    baseline_resolved = sum(left[key].harness_resolved for key in usable)
    treatment_resolved = sum(right[key].harness_resolved for key in usable)

    def ratio(candidate_values: list[float], baseline_values: list[float]) -> float | None:
        if not candidate_values or not baseline_values:
            return None
        baseline_mean = statistics.fmean(baseline_values)
        candidate_mean = statistics.fmean(candidate_values)
        if baseline_mean == 0:
            return 1.0 if candidate_mean == 0 else None
        return candidate_mean / baseline_mean

    def p95(values: list[float]) -> float | None:
        return _distribution(values)["p95"]

    baseline_wall_p95 = p95([left[key].wall_seconds for key in usable])
    candidate_wall_p95 = p95([right[key].wall_seconds for key in usable])
    p95_wall_ratio = (
        candidate_wall_p95 / baseline_wall_p95
        if baseline_wall_p95 not in {None, 0} and candidate_wall_p95 is not None
        else 1.0 if baseline_wall_p95 == candidate_wall_p95 == 0 else None
    )
    return {
        "common_tasks": len(common),
        "paired_eligible": len(usable),
        "task_coverage": len(common) / len(union) if union else None,
        "paired_eligible_coverage": len(usable) / len(common) if common else None,
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(ties),
        "win_ids": wins,
        "loss_ids": losses,
        "excluded_ids": sorted(set(common) - set(usable)),
        "baseline_resolved_at_1": (
            baseline_resolved / len(usable) if usable else None
        ),
        "treatment_resolved_at_1": (
            treatment_resolved / len(usable) if usable else None
        ),
        # Compatibility key for historical report readers.
        "candidate_resolved_at_1": (
            treatment_resolved / len(usable) if usable else None
        ),
        "resolved_delta": (
            (treatment_resolved - baseline_resolved) / len(usable)
            if usable else None
        ),
        "mean_token_ratio": ratio(
            [float(right[key].total_tokens) for key in usable],
            [float(left[key].total_tokens) for key in usable],
        ),
        "mean_wall_time_ratio": ratio(
            [right[key].wall_seconds for key in usable],
            [left[key].wall_seconds for key in usable],
        ),
        "p95_wall_time_ratio": p95_wall_ratio,
    }


def result_to_dict(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)
