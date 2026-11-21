"""Keep complete tool evidence outside the prompt and return bounded observations."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from codeagent.artifacts import ArtifactStore
from trae_agent.utils.redaction import redact_text

if TYPE_CHECKING:
    from trae_agent.tools.base import ToolCall, ToolResult


_FAILURE = re.compile(r"(failed|failure|error|exception|traceback|assert|fatal)", re.I)


@dataclass(frozen=True)
class Observation:
    artifact_id: str
    preview: str
    raw_chars: int
    truncated_upstream: bool


class ObservationReducer:
    def __init__(self, artifacts: ArtifactStore, max_preview_chars: int = 2800) -> None:
        self.artifacts = artifacts
        self.max_preview_chars = max_preview_chars
        self.by_call_id: dict[str, Observation] = {}

    def reduce(self, call: ToolCall, result: ToolResult) -> ToolResult:
        raw = (result.result or "") + (f"\nERROR: {result.error}" if result.error else "")
        if call.name.lower().startswith("mcp__"):
            raw = redact_text(raw)
        record = self.artifacts.put_text(f"tool-{call.name}", raw or "<empty>", ".log")
        lines = raw.splitlines()
        failures = [line[:300] for line in lines if _FAILURE.search(line)][:8]
        if len(raw) <= self.max_preview_chars:
            preview = raw
        else:
            head = raw[: self.max_preview_chars // 3]
            tail = raw[-self.max_preview_chars // 3 :]
            middle = "\n".join(failures)[: self.max_preview_chars // 3]
            preview = f"{head}\n[...middle omitted...]\n{middle}\n[...tail...]\n{tail}"
        upstream_clipped = "<response clipped>" in raw or "<artifact clipped>" in raw
        prefix = (
            f"[tool artifact={record.id} success={result.success} "
            f"raw_chars={len(raw)} upstream_clipped={upstream_clipped}]\n"
        )
        preview = prefix + preview
        self.by_call_id[call.call_id] = Observation(
            artifact_id=record.id, preview=preview, raw_chars=len(raw),
            truncated_upstream=upstream_clipped,
        )
        return replace(result, result=preview, error=None if result.success else result.error)
