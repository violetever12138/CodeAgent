"""Typed, auditable units used by retrieval and context packing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeChunk:
    id: str
    path: str
    language: str
    symbol: str | None
    start_line: int
    end_line: int
    text: str
    file_hash: str
    repository_version: str
    kind: str = "file"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: CodeChunk
    score: float
    channels: tuple[str, ...]
    ranks: dict[str, int]


@dataclass(frozen=True)
class ContextItem:
    id: str
    kind: str
    text: str
    source: str
    repository_version: str
    priority: float
    required: bool = False
    file_hash: str | None = None
    artifact_id: str | None = None
    group: str | None = None


@dataclass(frozen=True)
class ExcludedItem:
    id: str
    reason: str


@dataclass(frozen=True)
class ContextPackage:
    repository_version: str
    role: str
    items: tuple[ContextItem, ...]
    excluded: tuple[ExcludedItem, ...]
    input_tokens: int
    budget_tokens: int
    tokenizer: str
    compacted_steps: int = 0
    estimated_request_tokens: int = 0

    def render(self) -> str:
        sections = []
        for item in self.items:
            # Retrieved repository content is data, never a higher-priority instruction.
            sections.append(
                f"[context kind={item.kind} source={item.source} "
                f"version={item.repository_version} id={item.id}]\n"
                f"data_json={json.dumps(item.text, ensure_ascii=False)}\n[/context]"
            )
        return "\n\n".join(sections)
