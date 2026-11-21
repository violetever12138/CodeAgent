"""DTO exposed to reviewer/selector code; hidden labels never cross this boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceCandidate:
    candidate_id: str
    patch_artifact_id: str
    changed_files: tuple[str, ...]
    public_test_artifact_ids: tuple[str, ...]
    summary: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "InferenceCandidate":
        """Deliberately copy only inference-safe fields, ignoring gold/hidden labels."""
        return cls(
            candidate_id=str(record["candidate_id"]),
            patch_artifact_id=str(record["patch_artifact_id"]),
            changed_files=tuple(map(str, record.get("changed_files", ()))),
            public_test_artifact_ids=tuple(
                map(str, record.get("public_test_artifact_ids", ()))
            ),
            summary=str(record.get("summary", "")),
        )
