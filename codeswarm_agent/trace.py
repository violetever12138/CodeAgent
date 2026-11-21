"""Parent/child event tracing for multi-agent runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codeswarm_agent.evidence.events import EventStore
from codeswarm_agent.telemetry import event as observe_event


class EventTracer:
    def __init__(self, run_id: str, path: str | Path) -> None:
        self.run_id = run_id
        self.path = Path(path).resolve()
        self.store = EventStore(run_id, self.path)

    def emit(
        self,
        event: str,
        *,
        task_id: str | None = None,
        worker_id: str | None = None,
        parent_task_ids: tuple[str, ...] = (),
        **data: Any,
    ) -> None:
        entity_type = "worker" if worker_id else "task" if task_id else "run"
        entity_id = worker_id or task_id
        self.store.append(
            event,
            entity_type=entity_type,
            entity_id=entity_id,
            payload={
                "event": event,
                "task_id": task_id,
                "worker_id": worker_id,
                "parent_task_ids": list(parent_task_ids),
                "data": data,
            },
        )
        observe_event(event, task_id=task_id, worker_id=worker_id, **data)

    def emit_once(
        self,
        operation_id: str,
        event: str,
        *,
        task_id: str | None = None,
        worker_id: str | None = None,
        parent_task_ids: tuple[str, ...] = (),
        **data: Any,
    ) -> None:
        entity_type = "worker" if worker_id else "task" if task_id else "run"
        entity_id = worker_id or task_id
        self.store.append_once(
            operation_id,
            event,
            entity_type=entity_type,
            entity_id=entity_id,
            payload={
                "event": event,
                "task_id": task_id,
                "worker_id": worker_id,
                "parent_task_ids": list(parent_task_ids),
                "data": data,
            },
        )
        observe_event(event, task_id=task_id, worker_id=worker_id, **data)
