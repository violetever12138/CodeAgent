"""Versioned quality gates for L0 contracts, L1 repo tasks, and L2 benchmarks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from codeswarm_agent.evaluation.manifest import EvaluationManifest
from codeswarm_agent.evaluation.metrics import EvaluationResult, aggregate, compare_paired


class ValidationLevel(str, Enum):
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


class Comparison(str, Enum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"


@dataclass(frozen=True)
class MetricRule:
    metric: str
    comparison: Comparison
    threshold: float
    description: str

    def __post_init__(self) -> None:
        if not self.metric.strip() or not self.description.strip():
            raise ValueError("metric rule identity cannot be empty")
        if not math.isfinite(self.threshold):
            raise ValueError("metric threshold must be finite")


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    level: ValidationLevel
    minimum_eligible: int
    baseline_required: bool
    manifest_required: bool
    rules: tuple[MetricRule, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("validation profile name cannot be empty")
        if self.minimum_eligible <= 0:
            raise ValueError("minimum_eligible must be positive")
        names = [rule.metric for rule in self.rules]
        if not names or len(names) != len(set(names)):
            raise ValueError("validation profile rules must be non-empty and unique")


@dataclass(frozen=True)
class GateCheck:
    metric: str
    observed: float | None
    comparison: str
    threshold: float | None
    status: ValidationStatus
    description: str


@dataclass(frozen=True)
class LayerValidationResult:
    level: ValidationLevel
    profile_name: str
    status: ValidationStatus
    checks: tuple[GateCheck, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROFILES: dict[ValidationLevel, ValidationProfile] = {
    ValidationLevel.L0: ValidationProfile(
        name="l0-contract-v1",
        level=ValidationLevel.L0,
        minimum_eligible=8,
        baseline_required=False,
        manifest_required=False,
        rules=(
            MetricRule("resolved_at_1", Comparison.EQ, 1.0, "all mechanism contracts pass"),
            MetricRule(
                "infrastructure_error_rate", Comparison.EQ, 0.0,
                "contract runner has no infrastructure failures",
            ),
            MetricRule(
                "false_completion_rate", Comparison.EQ, 0.0,
                "no contract is reported complete while unresolved",
            ),
        ),
    ),
    ValidationLevel.L1: ValidationProfile(
        name="l1-python-repo-v1",
        level=ValidationLevel.L1,
        minimum_eligible=20,
        baseline_required=True,
        manifest_required=True,
        rules=(
            MetricRule("infrastructure_error_rate", Comparison.LTE, 0.10, "infra failures <= 10%"),
            MetricRule("patch_apply_rate", Comparison.GTE, 0.90, "at least 90% of patches apply"),
            MetricRule("false_completion_rate", Comparison.LTE, 0.20, "false completion <= 20%"),
            MetricRule("budget.token_compliance", Comparison.EQ, 1.0, "all tasks respect token budget"),
            MetricRule("budget.wall_compliance", Comparison.EQ, 1.0, "all tasks respect wall budget"),
            MetricRule("experiment.control_match", Comparison.EQ, 1.0, "baseline controls match"),
            MetricRule("paired.task_coverage", Comparison.EQ, 1.0, "treatment and baseline task sets match"),
            MetricRule("paired.paired_eligible_coverage", Comparison.GTE, 0.90, "paired coverage >= 90%"),
            MetricRule("paired.resolved_delta", Comparison.GTE, 0.0, "no Resolved@1 regression"),
            MetricRule("paired.mean_token_ratio", Comparison.LTE, 1.50, "mean tokens <= 1.5x baseline"),
            MetricRule("paired.p95_wall_time_ratio", Comparison.LTE, 2.0, "P95 latency <= 2x baseline"),
        ),
    ),
    ValidationLevel.L2: ValidationProfile(
        name="l2-external-benchmark-v1",
        level=ValidationLevel.L2,
        minimum_eligible=50,
        baseline_required=True,
        manifest_required=True,
        rules=(
            MetricRule("infrastructure_error_rate", Comparison.LTE, 0.05, "infra failures <= 5%"),
            MetricRule("patch_apply_rate", Comparison.GTE, 0.95, "at least 95% of patches apply"),
            MetricRule("false_completion_rate", Comparison.LTE, 0.10, "false completion <= 10%"),
            MetricRule("budget.token_compliance", Comparison.EQ, 1.0, "all tasks respect token budget"),
            MetricRule("budget.wall_compliance", Comparison.EQ, 1.0, "all tasks respect wall budget"),
            MetricRule("experiment.control_match", Comparison.EQ, 1.0, "baseline controls match"),
            MetricRule("paired.task_coverage", Comparison.EQ, 1.0, "treatment and baseline task sets match"),
            MetricRule("paired.paired_eligible_coverage", Comparison.GTE, 0.95, "paired coverage >= 95%"),
            MetricRule("paired.resolved_delta", Comparison.GTE, 0.0, "no Resolved@1 regression"),
            MetricRule("paired.mean_token_ratio", Comparison.LTE, 1.25, "mean tokens <= 1.25x baseline"),
            MetricRule("paired.p95_wall_time_ratio", Comparison.LTE, 1.50, "P95 latency <= 1.5x baseline"),
        ),
    ),
}


def profile_from_dict(raw: dict[str, Any]) -> ValidationProfile:
    return ValidationProfile(
        name=str(raw["name"]),
        level=ValidationLevel(str(raw["level"]).lower()),
        minimum_eligible=int(raw["minimum_eligible"]),
        baseline_required=bool(raw.get("baseline_required", False)),
        manifest_required=bool(raw.get("manifest_required", False)),
        rules=tuple(
            MetricRule(
                metric=str(rule["metric"]),
                comparison=Comparison(str(rule["comparison"]).lower()),
                threshold=float(rule["threshold"]),
                description=str(rule.get("description", rule["metric"])),
            )
            for rule in raw["rules"]
        ),
    )


def _get_metric(metrics: dict[str, Any], dotted_name: str) -> float | None:
    value: Any = metrics
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _budget_limit(budget: dict[str, int | float | None], *names: str) -> float | None:
    for name in names:
        value = budget.get(name)
        if value is not None:
            return float(value)
    return None


def _budget_metrics(
    rows: tuple[EvaluationResult, ...], manifest: EvaluationManifest | None
) -> dict[str, float | None]:
    eligible = tuple(row for row in rows if row.infra_error is None)
    if manifest is None or not eligible:
        return {"token_compliance": None, "wall_compliance": None, "tool_compliance": None}
    token_limit = _budget_limit(
        manifest.budget, "tokens", "total_tokens", "max_total_tokens"
    )
    wall_limit = _budget_limit(
        manifest.budget, "wall_seconds", "max_wall_time_seconds"
    )
    tool_limit = _budget_limit(manifest.budget, "tool_calls", "max_tool_calls")

    def compliance(limit: float | None, values: Iterable[float]) -> float | None:
        observed = tuple(values)
        if limit is None:
            return None
        return sum(value <= limit for value in observed) / len(observed)

    return {
        "token_compliance": compliance(
            token_limit, (float(row.total_tokens) for row in eligible)
        ),
        "wall_compliance": compliance(
            wall_limit, (row.wall_seconds for row in eligible)
        ),
        "tool_compliance": compliance(
            tool_limit, (float(row.tool_calls) for row in eligible)
        ),
    }


def _control_metrics(
    treatment: EvaluationManifest | None,
    baseline: EvaluationManifest | None,
) -> dict[str, float | None]:
    if treatment is None or baseline is None:
        return {"control_match": None}
    if treatment.schema_version != baseline.schema_version:
        return {"control_match": 0.0, "schema_match": 0.0}
    fields = [
        "git_commit", "task_set_version", "task_set_checksum", "model_provider",
        "model_name", "model_revision", "temperature", "seed", "budget",
        "sandbox_profile", "harness_version", "dependency_lock_hash",
        "execution_backend", "execution_strength", "docker_inspect_verified",
        "architecture", "orchestration_backend", "langgraph_version",
        "checkpoint_backend", "orchestration_state_schema",
        "context_profile", "reranker", "index_schema",
        "artifact_schema", "environment", "prompt_version", "policy_version",
    ]
    # v1 comparisons preserve the original, smaller control surface.
    if treatment.schema_version == 1:
        fields = [
            "task_set_version", "model_provider", "model_name", "model_revision",
            "temperature", "seed", "budget", "sandbox_profile", "harness_version",
        ]
    allowed = set(treatment.experiment_factors)
    mismatches = [
        name for name in fields
        if name not in allowed and getattr(treatment, name) != getattr(baseline, name)
    ]
    return {
        "control_match": float(not mismatches),
        "schema_match": 1.0,
        "mismatch_count": float(len(mismatches)),
    }


def _check_rule(rule: MetricRule, metrics: dict[str, Any]) -> GateCheck:
    observed = _get_metric(metrics, rule.metric)
    if observed is None:
        return GateCheck(
            rule.metric, None, rule.comparison.value, rule.threshold,
            ValidationStatus.INCONCLUSIVE, rule.description,
        )
    passed = {
        Comparison.GTE: observed >= rule.threshold,
        Comparison.LTE: observed <= rule.threshold,
        Comparison.EQ: abs(observed - rule.threshold) <= 1e-12,
    }[rule.comparison]
    return GateCheck(
        rule.metric, observed, rule.comparison.value, rule.threshold,
        ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        rule.description,
    )


def evaluate_layer(
    level: ValidationLevel | str,
    candidate: Iterable[EvaluationResult],
    *,
    baseline: Iterable[EvaluationResult] | None = None,
    manifest: EvaluationManifest | None = None,
    baseline_manifest: EvaluationManifest | None = None,
    profile: ValidationProfile | None = None,
) -> LayerValidationResult:
    level = ValidationLevel(str(level).lower()) if not isinstance(level, ValidationLevel) else level
    selected = profile or DEFAULT_PROFILES[level]
    if selected.level != level:
        raise ValueError("validation profile level does not match requested level")
    rows = tuple(candidate)
    metrics = aggregate(rows)
    metrics["budget"] = _budget_metrics(rows, manifest)
    metrics["experiment"] = _control_metrics(manifest, baseline_manifest)
    baseline_rows = tuple(baseline) if baseline is not None else None
    metrics["paired"] = (
        compare_paired(baseline_rows, rows) if baseline_rows is not None else {}
    )
    if not rows:
        return LayerValidationResult(
            level, selected.name, ValidationStatus.NOT_RUN, (), metrics
        )

    if level == ValidationLevel.L2 and (
        manifest is None
        or manifest.execution_backend != "docker"
        or manifest.execution_strength != "container_inspected"
        or not manifest.docker_inspect_verified
    ):
        backend_check = GateCheck(
            "safety.docker_isolation", 0.0, "eq", 1.0,
            ValidationStatus.NOT_RUN,
            "L2 requires Docker isolation confirmed by container Inspect",
        )
        return LayerValidationResult(
            level, selected.name, ValidationStatus.NOT_RUN, (backend_check,), metrics,
        )

    checks: list[GateCheck] = []
    eligible = int(metrics["eligible_tasks"])
    checks.append(
        GateCheck(
            "eligible_tasks", float(eligible), "gte", float(selected.minimum_eligible),
            ValidationStatus.PASS if eligible >= selected.minimum_eligible
            else ValidationStatus.INCONCLUSIVE,
            "minimum sample size",
        )
    )
    if selected.baseline_required and baseline_rows is None:
        checks.append(
            GateCheck(
                "baseline", None, "required", None,
                ValidationStatus.INCONCLUSIVE, "paired baseline is required",
            )
        )
    if selected.manifest_required and manifest is None:
        checks.append(
            GateCheck(
                "manifest", None, "required", None,
                ValidationStatus.INCONCLUSIVE, "checksummed manifest is required",
            )
        )
    if selected.manifest_required and selected.baseline_required and baseline_manifest is None:
        checks.append(
            GateCheck(
                "baseline_manifest", None, "required", None,
                ValidationStatus.INCONCLUSIVE,
                "checksummed baseline manifest is required",
            )
        )
    checks.extend(_check_rule(rule, metrics) for rule in selected.rules)
    statuses = {check.status for check in checks}
    status = (
        ValidationStatus.INCONCLUSIVE
        if ValidationStatus.INCONCLUSIVE in statuses
        else ValidationStatus.FAIL
        if ValidationStatus.FAIL in statuses
        else ValidationStatus.PASS
    )
    return LayerValidationResult(level, selected.name, status, tuple(checks), metrics)
