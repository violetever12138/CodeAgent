"""Append-only hash-chained event storage with deterministic offline replay."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class EventIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    sequence: int
    run_id: str
    event_type: str
    entity_type: str
    entity_id: str | None
    parent_event_id: str | None
    state_version: int
    timestamp: str
    payload: dict[str, Any]
    previous_checksum: str | None
    checksum: str


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _json_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Freeze values exactly as they will be represented in the JSONL log."""
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


class EventStore:
    def __init__(self, run_id: str, path: str | Path) -> None:
        self.run_id = run_id
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        existing = self.read_all() if self.path.exists() else ()
        if existing and existing[-1].run_id != run_id:
            raise EventIntegrityError("event log belongs to another run")
        self._sequence = existing[-1].sequence if existing else 0
        self._last_event_id = existing[-1].event_id if existing else None
        self._last_checksum = existing[-1].checksum if existing else None
        self._operations = {
            str(record.payload["operation_id"]): record
            for record in existing
            if record.payload.get("operation_id")
        }

    def append(
        self,
        event_type: str,
        *,
        entity_type: str = "run",
        entity_id: str | None = None,
        state_version: int = 1,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        with self._lock:
            body = {
                "event_id": uuid.uuid4().hex,
                "sequence": self._sequence + 1,
                "run_id": self.run_id,
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "parent_event_id": self._last_event_id,
                "state_version": state_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload or {},
                "previous_checksum": self._last_checksum,
            }
            body = _json_values(body)
            checksum = hashlib.sha256(_canonical(body)).hexdigest()
            record = EventRecord(**body, checksum=checksum)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._sequence = record.sequence
            self._last_event_id = record.event_id
            self._last_checksum = record.checksum
            return record

    def append_once(
        self,
        operation_id: str,
        event_type: str,
        *,
        entity_type: str = "run",
        entity_id: str | None = None,
        state_version: int = 1,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Append an audit event once, even when a graph node is replayed."""

        if not operation_id.strip():
            raise ValueError("operation_id cannot be empty")
        with self._lock:
            operation_payload = {**(payload or {}), "operation_id": operation_id}
            existing = self._operations.get(operation_id)
            if existing is not None:
                if (
                    existing.event_type != event_type
                    or existing.entity_type != entity_type
                    or existing.entity_id != entity_id
                    or existing.payload != _json_values(operation_payload)
                ):
                    raise EventIntegrityError(
                        f"operation {operation_id} was replayed with different content"
                    )
                return existing
            record = self.append(
                event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                state_version=state_version,
                payload=operation_payload,
            )
            self._operations[operation_id] = record
            return record

    def validate_cursor(self, sequence: int, checksum: str | None) -> None:
        """Verify that a persisted graph cursor is a valid prefix of this log."""

        records = self.read_all()
        if sequence < 0 or sequence > len(records):
            raise EventIntegrityError("checkpoint event cursor is ahead of the event log")
        observed = records[sequence - 1].checksum if sequence else None
        if observed != checksum:
            raise EventIntegrityError("checkpoint event cursor checksum mismatch")

    def read_all(self) -> tuple[EventRecord, ...]:
        if not self.path.exists():
            return ()
        records: list[EventRecord] = []
        previous_checksum = None
        previous_event_id = None
        for expected_sequence, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                raw = json.loads(line)
                record = EventRecord(**raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise EventIntegrityError(f"invalid event at line {expected_sequence}") from exc
            body = asdict(record)
            observed_checksum = body.pop("checksum")
            expected_checksum = hashlib.sha256(_canonical(body)).hexdigest()
            if record.sequence != expected_sequence:
                raise EventIntegrityError("event sequence is not contiguous")
            if record.previous_checksum != previous_checksum:
                raise EventIntegrityError("event checksum chain is broken")
            if record.parent_event_id != previous_event_id:
                raise EventIntegrityError("event parent chain is broken")
            if observed_checksum != expected_checksum:
                raise EventIntegrityError("event checksum mismatch")
            records.append(record)
            previous_checksum = record.checksum
            previous_event_id = record.event_id
        return tuple(records)

    def replay(self, reducer: Callable[[Any, EventRecord], Any], initial: Any) -> Any:
        state = initial
        for event in self.read_all():
            state = reducer(state, event)
        return state

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def last_checksum(self) -> str | None:
        return self._last_checksum
