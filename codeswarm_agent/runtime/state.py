"""Worker-local phase machine, deadline, budget, and auditable stop reasons."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from codeswarm_agent.telemetry import event as observe_event


class Phase(str, Enum):
    CREATED = "created"
    THINKING = "thinking"
    ACTING = "acting"
    RECOVERING = "recovering"
    VALIDATING = "validating"
    STOPPED = "stopped"


class StopReason(str, Enum):
    HANDOFF = "handoff"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    BUDGET = "budget_exceeded"
    RESOURCE_CLEANUP_ERROR = "resource_cleanup_error"


class IllegalTransition(RuntimeError):
    pass


class RuntimeDeadlineExceeded(TimeoutError):
    pass


class RuntimeBudgetExceeded(RuntimeError):
    pass


_ALLOWED = {
    Phase.CREATED: {Phase.THINKING, Phase.STOPPED},
    Phase.THINKING: {Phase.THINKING, Phase.ACTING, Phase.RECOVERING,
                     Phase.VALIDATING, Phase.STOPPED},
    Phase.ACTING: {Phase.THINKING, Phase.RECOVERING, Phase.STOPPED},
    Phase.RECOVERING: {Phase.THINKING, Phase.STOPPED},
    Phase.VALIDATING: {Phase.THINKING, Phase.STOPPED},
    Phase.STOPPED: set(),
}


@dataclass(frozen=True)
class RuntimeLimits:
    deadline_at: float
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    model_attempt_timeout: float = 120.0
    tool_attempt_timeout: float = 120.0


class RuntimeController:
    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        worker_id: str,
        trace_path: Path,
        limits: RuntimeLimits,
    ) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.worker_id = worker_id
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.phase = Phase.CREATED
        self.stop_reason: StopReason | None = None
        self.failure_reason: StopReason | None = None
        self.model_attempts = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.emit("runtime_created", deadline_at=limits.deadline_at)

    def transition(self, target: Phase) -> None:
        if target not in _ALLOWED[self.phase]:
            raise IllegalTransition(f"{self.phase.value} -> {target.value}")
        previous = self.phase
        self.phase = target
        self.emit("state_transition", previous=previous.value, target=target.value)

    def remaining_seconds(self, attempt_limit: float | None = None) -> float:
        remaining = self.limits.deadline_at - time.monotonic()
        if remaining <= 0:
            self.failure_reason = StopReason.TIMEOUT
            raise RuntimeDeadlineExceeded("worker deadline exceeded")
        return min(remaining, attempt_limit) if attempt_limit is not None else remaining

    def before_model(self) -> float:
        self.transition(Phase.THINKING)
        self._check_tokens()
        timeout = self.remaining_seconds(self.limits.model_attempt_timeout)
        self.model_attempts += 1
        self.emit("model_attempt_started", attempt=self.model_attempts, timeout=timeout)
        return timeout

    def after_model(self, response) -> None:
        self.model_calls += 1
        usage = getattr(response, "usage", None)
        if usage:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
        self.emit(
            "model_response", model=getattr(response, "model", None),
            finish_reason=getattr(response, "finish_reason", None),
            tool_call_count=len(getattr(response, "tool_calls", None) or []),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )

    def before_tool(self, tool_name: str, call_id: str) -> float:
        if self.phase != Phase.ACTING:
            self.transition(Phase.ACTING)
        self._check_tokens()
        if self.limits.max_tool_calls is not None and self.tool_calls >= self.limits.max_tool_calls:
            self.failure_reason = StopReason.BUDGET
            raise RuntimeBudgetExceeded("worker tool-call budget exhausted")
        timeout = self.remaining_seconds(self.limits.tool_attempt_timeout)
        self.tool_calls += 1
        self.emit("tool_call_started", tool=tool_name, call_id=call_id, timeout=timeout)
        return timeout

    def after_tool(self, tool_name: str, call_id: str, success: bool, error: str | None) -> None:
        self.emit("tool_call_finished", tool=tool_name, call_id=call_id,
                  success=success, error=error)

    def after_tool_batch(self, has_failure: bool) -> None:
        self.transition(Phase.RECOVERING if has_failure else Phase.THINKING)

    def before_completion(self) -> None:
        self.transition(Phase.VALIDATING)
        self.remaining_seconds()

    def completion_rejected(self, reasons: tuple[str, ...]) -> None:
        self.emit("completion_rejected", reasons=reasons)
        self.transition(Phase.THINKING)

    def finish(self, reason: StopReason) -> None:
        if self.phase == Phase.STOPPED:
            return
        self.transition(Phase.STOPPED)
        self.stop_reason = reason
        self.emit("runtime_stopped", reason=reason.value,
                  model_calls=self.model_calls, tool_calls=self.tool_calls,
                  input_tokens=self.input_tokens, output_tokens=self.output_tokens)

    def emit(self, event: str, **payload) -> None:
        row = {
            "event": event, "at": time.time(), "run_id": self.run_id,
            "task_id": self.task_id, "worker_id": self.worker_id,
            "phase": self.phase.value, **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        observe_event(event, **payload)

    def _check_tokens(self) -> None:
        if (
            self.limits.max_tokens is not None
            and self.input_tokens + self.output_tokens >= self.limits.max_tokens
        ):
            self.failure_reason = StopReason.BUDGET
            raise RuntimeBudgetExceeded("worker token budget exhausted")
