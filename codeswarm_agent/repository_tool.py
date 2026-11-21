"""A bounded read-only repository tool for Explorer and Reviewer workers."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from typing_extensions import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class RepositoryReadTool(Tool):
    def __init__(
        self,
        project_root: str | Path,
        model_provider: str | None = None,
        max_file_bytes: int = 1_000_000,
        query=None,
    ) -> None:
        super().__init__(model_provider)
        self.project_root = Path(project_root).resolve()
        self.max_file_bytes = max_file_bytes
        self.query = query

    @override
    def get_name(self) -> str:
        return "repository_read"

    @override
    def get_description(self) -> str:
        return (
            "Read-only repository access. List files, read a line range, or search text. "
            + ("Use query with pattern=current information gap for routed code retrieval. " if self.query else "")
            + "Every path is constrained to the assigned workspace and the tool cannot edit files."
        )

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation to perform: list, read, search, or query (if available).",
                enum=["list", "read", "search", *(("query",) if self.query else ())],
            ),
            ToolParameter(
                name="path",
                type=["string", "null"],
                description="Absolute or repository-relative path; null means repository root.",
                required=False,
            ),
            ToolParameter(
                name="pattern",
                type=["string", "null"],
                description="Regular expression for search; bounded natural-language question for query.",
                required=False,
            ),
            ToolParameter(
                name="start_line",
                type=["integer", "null"],
                description="First 1-based line for read; null means line 1.",
                required=False,
            ),
            ToolParameter(
                name="end_line",
                type=["integer", "null"],
                description="Last 1-based line for read; null means end of file.",
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            operation = str(arguments.get("operation", ""))
            path = self._resolve(arguments.get("path"))
            if operation == "query" and self.query:
                question = arguments.get("pattern")
                if not isinstance(question, str) or not question.strip() or len(question) > 3000:
                    return ToolExecResult(error="query requires a nonempty pattern of at most 3000 characters", error_code=-1)
                scope = "" if path == self.project_root else path.relative_to(self.project_root).as_posix()
                return ToolExecResult(output=await asyncio.to_thread(self.query, question, scope))
            if operation == "list":
                return ToolExecResult(output=self._list(path))
            if operation == "read":
                return ToolExecResult(
                    output=self._read(
                        path,
                        self._optional_int(arguments.get("start_line")),
                        self._optional_int(arguments.get("end_line")),
                    )
                )
            if operation == "search":
                pattern = arguments.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    return ToolExecResult(error="pattern is required for search", error_code=-1)
                return ToolExecResult(output=self._search(path, pattern))
            return ToolExecResult(error=f"unsupported operation: {operation}", error_code=-1)
        except (OSError, UnicodeError, ValueError, re.error) as exc:
            return ToolExecResult(error=str(exc), error_code=-1)

    def _resolve(self, value: object) -> Path:
        if value is None or (isinstance(value, str) and value in {"", "null"}):
            candidate = self.project_root
        else:
            raw_path = Path(str(value))
            candidate = raw_path if raw_path.is_absolute() else self.project_root / raw_path
            candidate = candidate.resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise ValueError(f"path escapes assigned workspace: {candidate}")
        if not candidate.exists():
            raise ValueError(f"path does not exist: {candidate}")
        return candidate

    def _list(self, path: Path) -> str:
        if path.is_file():
            return path.relative_to(self.project_root).as_posix()
        entries: list[str] = []
        for item in sorted(path.rglob("*")):
            relative = item.relative_to(self.project_root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if len(relative.parts) > len(path.relative_to(self.project_root).parts) + 2:
                continue
            entries.append(relative.as_posix() + ("/" if item.is_dir() else ""))
            if len(entries) >= 500:
                entries.append("<listing clipped>")
                break
        return "\n".join(entries)

    def _read(self, path: Path, start: int | None, end: int | None) -> str:
        if not path.is_file():
            raise ValueError("read operation requires a file")
        if path.stat().st_size > self.max_file_bytes:
            raise ValueError(f"file exceeds {self.max_file_bytes} byte read limit")
        lines = path.read_text(encoding="utf-8").splitlines()
        first = start or 1
        last = end or len(lines)
        if first < 1 or last < first:
            raise ValueError("invalid line range")
        selected = [
            f"{number:6}\t{lines[number - 1]}"
            for number in range(first, min(last, len(lines)) + 1)
        ]
        return "\n".join(selected)

    def _search(self, path: Path, pattern: str) -> str:
        matcher = re.compile(pattern)
        files = [path] if path.is_file() else path.rglob("*")
        matches: list[str] = []
        for file_path in files:
            if not file_path.is_file() or file_path.stat().st_size > self.max_file_bytes:
                continue
            relative = file_path.relative_to(self.project_root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, 1):
                if matcher.search(line):
                    matches.append(f"{relative.as_posix()}:{line_number}:{line}")
                    if len(matches) >= 300:
                        matches.append("<search results clipped>")
                        return "\n".join(matches)
        return "\n".join(matches)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) else None
