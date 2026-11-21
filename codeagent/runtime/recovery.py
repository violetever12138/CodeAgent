"""Classified model retry and deterministic feedback for failed tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import uuid
from enum import Enum

from codeagent.runtime.state import Phase, RuntimeController, StopReason


class FailureKind(str, Enum):
    INVALID_ARGUMENT = "invalid_argument"
    TOOL_NOT_FOUND = "tool_not_found"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    EXECUTION_FAILED = "execution_failed"
    INTERNAL_ERROR = "internal_error"


def classify_tool_failure(result) -> FailureKind:
    error = (result.error or "").upper()
    for code in ("INVALID_ARGUMENT", "TOOL_NOT_FOUND", "TIMEOUT", "PERMISSION_DENIED"):
        if error.startswith(f"POLICY/{code}") or error.startswith(f"TOOL/{code}"):
            return FailureKind(code.lower())
    if "TIMED OUT" in error or "TIMEOUT" in error:
        return FailureKind.TIMEOUT
    if "NOT FOUND" in error:
        return FailureKind.TOOL_NOT_FOUND
    return FailureKind.EXECUTION_FAILED


def tool_feedback(results: list) -> str | None:
    failures = [result for result in results if not result.success]
    if not failures:
        return None
    lines = ["Tool failure observations; do not claim these actions succeeded:"]
    for result in failures:
        kind = classify_tool_failure(result)
        action = {
            FailureKind.INVALID_ARGUMENT: "correct the arguments; do not retry unchanged",
            FailureKind.TOOL_NOT_FOUND: "select an available tool",
            FailureKind.PERMISSION_DENIED: "stay within role/workspace policy; do not retry",
            FailureKind.TIMEOUT: "check state before any retry, especially after writes",
            FailureKind.EXECUTION_FAILED: "inspect the output and revise the approach",
        }.get(kind, "inspect the error")
        lines.append(f"- {result.name} ({result.call_id}): {kind.value}; {action}")
    return "\n".join(lines)


def retryable_model_error(error: Exception) -> bool:
    # Tool arguments are decoded inside the provider adapter, before any tool is
    # dispatched. A malformed model response can be requested again safely.
    if isinstance(error, json.JSONDecodeError):
        return True
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 425, 429} or 500 <= status <= 599
    # OpenAI-compatible clients wrap httpx/socket failures in SDK exceptions that
    # are not subclasses of Python's ConnectionError.  Keep this dependency-light
    # and also inspect the exception chain so provider adapters remain swappable.
    retryable_names = {"APIConnectionError", "APITimeoutError", "ConnectError"}
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if type(current).__name__ in retryable_names or isinstance(
            current, (ConnectionError, ConnectionResetError, TimeoutError)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class ModelRecovery:
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.25) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def call(self, operation, controller: RuntimeController):
        from trae_agent.utils.request_observer import logical_call
        token = logical_call.set(uuid.uuid4().hex)
        try:
            return await self._call(operation, controller)
        finally:
            logical_call.reset(token)

    async def _call(self, operation, controller: RuntimeController):
        for attempt in range(1, self.max_attempts + 1):
            timeout = controller.before_model()
            try:
                produced = operation()
                if inspect.isawaitable(produced):
                    response = await asyncio.wait_for(produced, timeout)
                else:
                    response = produced
                controller.after_model(response)
                return response
            except asyncio.CancelledError:
                controller.failure_reason = StopReason.CANCELLED
                controller.emit("model_attempt_cancelled", attempt=attempt)
                raise
            except asyncio.TimeoutError:
                controller.failure_reason = StopReason.TIMEOUT
                controller.emit("model_attempt_timed_out", attempt=attempt)
                # The private provider thread may still finish; never dispatch tools from it.
                raise
            except Exception as error:
                transient = retryable_model_error(error)
                controller.emit(
                    "model_attempt_failed", attempt=attempt,
                    error_type=type(error).__name__, retryable=transient,
                )
                if not transient or attempt == self.max_attempts:
                    raise
                delay = min(self.base_delay * (2 ** (attempt - 1)), 5.0)
                delay *= random.uniform(0.8, 1.2)
                if controller.remaining_seconds() <= delay:
                    controller.failure_reason = StopReason.TIMEOUT
                    raise
                controller.transition(Phase.RECOVERING)
                controller.emit("model_retry_scheduled", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable model retry state")
