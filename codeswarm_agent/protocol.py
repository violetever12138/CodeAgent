"""Shared protocols for task-level multi-agent orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class WorkerRole(str, Enum):
    MANAGER = "manager"
    WORKER = "worker"
    AUDITOR = "auditor"
    EXPLORER = "explorer"
    DEVELOPER = "developer"
    SECURITY = "security"
    INTEGRATOR = "integrator"
    TESTER = "tester"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class DependencyMode(str, Enum):
    """Whether dependencies must succeed or merely reach a terminal state."""

    ALL_SUCCEEDED = "all_succeeded"
    ALL_TERMINAL = "all_terminal"


class ReviewVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REPLAN = "replan"


class ApprovalAction(str, Enum):
    ACCEPT = "accept"
    REPLAN = "replan"
    TERMINATE = "terminate"
    ROUTE_SINGLE = "route_single"


TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    wall_time_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model_calls=self.model_calls + other.model_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            wall_time_seconds=self.wall_time_seconds + other.wall_time_seconds,
        )


@dataclass(frozen=True)
class Budget:
    max_parallel: int = 3
    max_workers: int = 20
    max_retries_per_task: int = 1
    max_replans: int = 1
    # Deprecated evaluation-schema field. Cooperative planning uses max_work_units;
    # this remains readable so historical manifests can still be loaded.
    max_candidates: int = 2
    max_work_units: int = 3
    max_wall_time_seconds: float = 1800.0
    max_total_tokens: int | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        positive_fields = {
            "max_parallel": self.max_parallel,
            "max_workers": self.max_workers,
            "max_wall_time_seconds": self.max_wall_time_seconds,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_retries_per_task < 0 or self.max_replans < 0:
            raise ValueError("retry and replan limits cannot be negative")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.max_work_units <= 0:
            raise ValueError("max_work_units must be positive")


@dataclass(frozen=True)
class TaskRequest:
    issue: str
    repository: str
    base_commit: str | None = None
    acceptance_criteria: tuple[str, ...] = ()
    test_commands: tuple[str, ...] = ()
    force_multi_agent: bool | None = None
    # Kept in the request schema for compatibility with saved request files.
    candidate_count: int | None = None
    max_work_units: int | None = None
    budget: Budget = field(default_factory=Budget)

    def __post_init__(self) -> None:
        if not self.issue.strip():
            raise ValueError("issue cannot be empty")
        if not self.repository.strip():
            raise ValueError("repository cannot be empty")
        if self.candidate_count is not None and self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.max_work_units is not None and self.max_work_units <= 0:
            raise ValueError("max_work_units must be positive")


@dataclass
class TaskNode:
    id: str
    role: WorkerRole
    goal: str
    depends_on: tuple[str, ...] = ()
    input_artifacts: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    dependency_mode: DependencyMode = DependencyMode.ALL_SUCCEEDED
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.goal.strip():
            raise ValueError("task id and goal cannot be empty")
        if self.id in self.depends_on:
            raise ValueError(f"task {self.id} cannot depend on itself")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


@dataclass(frozen=True)
class Evidence:
    id: str
    producer: str
    task_id: str
    kind: str
    claim: str
    artifact_id: str | None = None
    source_path: str | None = None
    line_range: tuple[int, int] | None = None
    base_commit: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class VerificationRecord:
    command: str
    passed: bool
    artifact_id: str | None = None
    exit_code: int | None = None
    workspace_version: str | None = None
    patch_sha256: str | None = None


@dataclass(frozen=True)
class ReviewDecision:
    target_task_id: str
    verdict: ReviewVerdict
    reasons: tuple[str, ...] = ()
    confidence: float = 0.0
    missing_tests: tuple[str, ...] = ()
    rework_task_ids: tuple[str, ...] = ()

    @property
    def candidate_task_id(self) -> str:
        """Backward-compatible alias for persisted pre-cooperation traces."""
        return self.target_task_id


@dataclass(frozen=True)
class ApprovalRequest:
    reason: str
    allowed_actions: tuple[ApprovalAction, ...]
    hard_gates_passed: bool
    replan_round: int


@dataclass(frozen=True)
class WorkerResult:
    task_id: str
    attempt: int
    role: WorkerRole
    status: WorkerStatus
    summary: str
    evidence: tuple[Evidence, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)
    verifications: tuple[VerificationRecord, ...] = ()
    review_decisions: tuple[ReviewDecision, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutcome:
    run_id: str
    status: RunStatus
    mode: str
    run_directory: str | None = None
    selected_integrated_task_id: str | None = None
    selected_patch_artifact_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    review_decisions: tuple[ReviewDecision, ...] = ()
    usage: Usage = field(default_factory=Usage)
    failure_reason: str | None = None
    pending_approval: ApprovalRequest | None = None
    checkpoint_thread_id: str | None = None

    @property
    def selected_candidate_task_id(self) -> str | None:
        """Backward-compatible name for pre-cooperation evaluation callers."""
        return self.selected_integrated_task_id


def to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses and enums to JSON-safe primitive values."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def usage_from_json(raw: dict[str, Any] | None) -> Usage:
    return Usage(**(raw or {}))


def evidence_from_json(raw: dict[str, Any]) -> Evidence:
    payload = dict(raw)
    if payload.get("line_range") is not None:
        payload["line_range"] = tuple(payload["line_range"])
    return Evidence(**payload)


def verification_from_json(raw: dict[str, Any]) -> VerificationRecord:
    return VerificationRecord(**raw)


def review_decision_from_json(raw: dict[str, Any]) -> ReviewDecision:
    payload = dict(raw)
    payload["verdict"] = ReviewVerdict(payload["verdict"])
    for name in ("reasons", "missing_tests", "rework_task_ids"):
        payload[name] = tuple(payload.get(name, ()))
    return ReviewDecision(**payload)


def worker_result_from_json(raw: dict[str, Any]) -> WorkerResult:
    payload = dict(raw)
    payload["role"] = WorkerRole(payload["role"])
    payload["status"] = WorkerStatus(payload["status"])
    payload["evidence"] = tuple(evidence_from_json(item) for item in payload.get("evidence", ()))
    payload["artifact_ids"] = tuple(payload.get("artifact_ids", ()))
    payload["changed_files"] = tuple(payload.get("changed_files", ()))
    payload["unresolved"] = tuple(payload.get("unresolved", ()))
    payload["usage"] = usage_from_json(payload.get("usage"))
    payload["verifications"] = tuple(
        verification_from_json(item) for item in payload.get("verifications", ())
    )
    payload["review_decisions"] = tuple(
        review_decision_from_json(item) for item in payload.get("review_decisions", ())
    )
    payload.pop("work_units", None)
    return WorkerResult(**payload)


def budget_from_json(raw: dict[str, Any]) -> Budget:
    return Budget(**raw)


def task_request_from_json(raw: dict[str, Any]) -> TaskRequest:
    payload = dict(raw)
    payload["acceptance_criteria"] = tuple(payload.get("acceptance_criteria", ()))
    payload["test_commands"] = tuple(payload.get("test_commands", ()))
    payload["budget"] = budget_from_json(payload.get("budget", {}))
    return TaskRequest(**payload)


def task_node_from_json(raw: dict[str, Any]) -> TaskNode:
    payload = dict(raw)
    payload["role"] = WorkerRole(payload["role"])
    payload["dependency_mode"] = DependencyMode(
        payload.get("dependency_mode", DependencyMode.ALL_SUCCEEDED.value)
    )
    payload["status"] = TaskStatus(payload.get("status", TaskStatus.PENDING.value))
    for name in ("depends_on", "input_artifacts", "write_scope"):
        payload[name] = tuple(payload.get(name, ()))
    return TaskNode(**payload)
