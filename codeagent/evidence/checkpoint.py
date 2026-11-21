"""Atomic, checksummed checkpoints created only at side-effect-safe points."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CheckpointIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class Checkpoint:
    schema_version: int
    run_id: str
    event_sequence: int
    event_checksum: str | None
    repository_base: str
    graph: dict[str, Any]
    budget: dict[str, Any]
    artifact_ids: tuple[str, ...]
    inflight_write_operations: tuple[str, ...] = ()

    @property
    def is_safe_point(self) -> bool:
        return not self.inflight_write_operations


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> None:
        if not checkpoint.is_safe_point:
            raise ValueError("checkpoint contains in-flight write operations")
        body = asdict(checkpoint)
        digest = hashlib.sha256(self._canonical(body)).hexdigest()
        payload = {**body, "checksum": digest}
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )
        temporary.replace(self.path)

    def load(
        self, *, run_id: str | None = None, repository_base: str | None = None,
        event_sequence: int | None = None, event_checksum: str | None = None,
    ) -> Checkpoint:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        observed = raw.pop("checksum", None)
        expected = hashlib.sha256(self._canonical(raw)).hexdigest()
        if observed != expected:
            raise CheckpointIntegrityError("checkpoint checksum mismatch")
        raw["artifact_ids"] = tuple(raw.get("artifact_ids", ()))
        raw["inflight_write_operations"] = tuple(
            raw.get("inflight_write_operations", ())
        )
        checkpoint = Checkpoint(**raw)
        if not checkpoint.is_safe_point:
            raise CheckpointIntegrityError("checkpoint is not a safe point")
        if run_id is not None and checkpoint.run_id != run_id:
            raise CheckpointIntegrityError("checkpoint run id mismatch")
        if repository_base is not None and checkpoint.repository_base != repository_base:
            raise CheckpointIntegrityError("repository base mismatch")
        if event_sequence is not None and checkpoint.event_sequence > event_sequence:
            raise CheckpointIntegrityError("checkpoint is ahead of the event log")
        if (
            event_checksum is not None
            and checkpoint.event_sequence == event_sequence
            and checkpoint.event_checksum != event_checksum
        ):
            raise CheckpointIntegrityError("checkpoint event checksum mismatch")
        return checkpoint

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
