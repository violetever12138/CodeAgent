"""Run-wide budget accounting shared by every worker."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from codeswarm_agent.protocol import Budget, Usage, to_jsonable


@dataclass(frozen=True)
class BudgetReservation:
    id: str
    owner: str
    estimate: Usage


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.started_at = time.monotonic()
        self.usage = Usage()
        self.workers_started = 0
        self._lock = asyncio.Lock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._reserved = Usage()
        self._claimed_operations: set[str] = set()
        self._recorded_operations: set[str] = set()

    async def can_start(self) -> bool:
        async with self._lock:
            return not self._exhausted_unlocked() and self.workers_started < self.budget.max_workers

    async def claim_worker(self, operation_id: str | None = None) -> bool:
        async with self._lock:
            if operation_id and operation_id in self._claimed_operations:
                return True
            if self._exhausted_unlocked() or self.workers_started >= self.budget.max_workers:
                return False
            self.workers_started += 1
            if operation_id:
                self._claimed_operations.add(operation_id)
            return True

    async def record(self, usage: Usage, operation_id: str | None = None) -> None:
        async with self._lock:
            if operation_id and operation_id in self._recorded_operations:
                return
            self.usage = self.usage + usage
            if operation_id:
                self._recorded_operations.add(operation_id)

    async def reconcile_committed(self, operation_id: str, usage: Usage) -> None:
        """Restore a durable result that may be newer than the graph checkpoint."""

        async with self._lock:
            if operation_id not in self._claimed_operations:
                self.workers_started += 1
                self._claimed_operations.add(operation_id)
            if operation_id not in self._recorded_operations:
                self.usage = self.usage + usage
                self._recorded_operations.add(operation_id)

    async def reserve(self, owner: str, estimate: Usage) -> BudgetReservation | None:
        """Atomically reserve bounded usage so concurrent owners cannot over-admit."""
        if min(estimate.input_tokens, estimate.output_tokens, estimate.tool_calls) < 0:
            raise ValueError("reservation estimates cannot be negative")
        async with self._lock:
            projected = self.usage + self._reserved + estimate
            if (
                self.budget.max_total_tokens is not None
                and projected.total_tokens > self.budget.max_total_tokens
            ):
                return None
            if (
                self.budget.max_tool_calls is not None
                and projected.tool_calls > self.budget.max_tool_calls
            ):
                return None
            if (
                time.monotonic() - self.started_at
                + self._reserved.wall_time_seconds
                + estimate.wall_time_seconds
                > self.budget.max_wall_time_seconds
            ):
                return None
            reservation = BudgetReservation(uuid.uuid4().hex, owner, estimate)
            self._reservations[reservation.id] = reservation
            self._reserved = self._reserved + estimate
            return reservation

    async def commit(self, reservation_id: str, actual: Usage) -> None:
        async with self._lock:
            reservation = self._reservations.pop(reservation_id)
            self._reserved = self._subtract(self._reserved, reservation.estimate)
            self.usage = self.usage + actual

    async def release(self, reservation_id: str) -> None:
        async with self._lock:
            reservation = self._reservations.pop(reservation_id)
            self._reserved = self._subtract(self._reserved, reservation.estimate)

    async def exhausted(self) -> bool:
        async with self._lock:
            return self._exhausted_unlocked() or self.workers_started >= self.budget.max_workers

    async def remaining_wall_time(self) -> float:
        async with self._lock:
            elapsed = time.monotonic() - self.started_at
            return max(0.0, self.budget.max_wall_time_seconds - elapsed)

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            elapsed = time.monotonic() - self.started_at
            return {
                "limits": to_jsonable(self.budget),
                "usage": to_jsonable(self.usage),
                "workers_started": self.workers_started,
                "reserved": to_jsonable(self._reserved),
                "reservation_count": len(self._reservations),
                "claimed_operations": sorted(self._claimed_operations),
                "recorded_operations": sorted(self._recorded_operations),
                "elapsed_seconds": elapsed,
                "remaining_wall_time_seconds": max(
                    0.0, self.budget.max_wall_time_seconds - elapsed
                ),
            }

    @classmethod
    def restore(cls, budget: Budget, snapshot: dict[str, object] | None) -> BudgetTracker:
        tracker = cls(budget)
        if not snapshot:
            return tracker
        tracker.usage = Usage(**dict(snapshot.get("usage", {})))
        tracker.workers_started = int(snapshot.get("workers_started", 0))
        tracker._claimed_operations = set(snapshot.get("claimed_operations", ()))
        tracker._recorded_operations = set(snapshot.get("recorded_operations", ()))
        # A suspended run does not consume wall-clock budget while waiting for a human.
        tracker.started_at = time.monotonic() - float(snapshot.get("elapsed_seconds", 0.0))
        return tracker

    def _exhausted_unlocked(self) -> bool:
        projected = self.usage + self._reserved
        if (
            time.monotonic() - self.started_at + self._reserved.wall_time_seconds
            >= self.budget.max_wall_time_seconds
        ):
            return True
        if (
            self.budget.max_total_tokens is not None
            and projected.total_tokens >= self.budget.max_total_tokens
        ):
            return True
        return (
            self.budget.max_tool_calls is not None
            and projected.tool_calls >= self.budget.max_tool_calls
        )

    @staticmethod
    def _subtract(left: Usage, right: Usage) -> Usage:
        return Usage(
            input_tokens=left.input_tokens - right.input_tokens,
            output_tokens=left.output_tokens - right.output_tokens,
            model_calls=left.model_calls - right.model_calls,
            tool_calls=left.tool_calls - right.tool_calls,
            wall_time_seconds=left.wall_time_seconds - right.wall_time_seconds,
        )
