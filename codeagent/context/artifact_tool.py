"""Read bounded line ranges from an explicitly referenced run artifact."""

from __future__ import annotations

from typing_extensions import override

from codeagent.artifacts import ArtifactStore
from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class ArtifactReadTool(Tool):
    def __init__(self, artifacts: ArtifactStore, model_provider: str | None = None):
        super().__init__(model_provider)
        self.artifacts = artifacts

    @override
    def get_name(self) -> str:
        return "artifact_read"

    @override
    def get_description(self) -> str:
        return "Read up to 201 numbered lines of a referenced CodeSwarm artifact by ID."

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter("artifact_id", "string", "Artifact ID from context or tool output."),
            ToolParameter("start_line", "integer", "First line, 1-based."),
            ToolParameter("end_line", "integer", "Last line, inclusive; at most 200 lines later."),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            artifact_id = str(arguments["artifact_id"])
            start = int(str(arguments["start_line"]))
            end = int(str(arguments["end_line"]))
            return ToolExecResult(output=self.artifacts.read_lines(artifact_id, start, end))
        except (KeyError, ValueError, OSError) as exc:
            return ToolExecResult(error=str(exc), error_code=-1)
