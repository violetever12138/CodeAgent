"""A narrow integration tool that applies only coordinator-authorized patch artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

from typing_extensions import override

from codeagent.artifacts import ArtifactStore
from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class PatchApplyTool(Tool):
    def __init__(
        self,
        workspace: str | Path,
        artifacts: ArtifactStore,
        allowed_artifacts: dict[str, str],
        model_provider: str | None = None,
    ) -> None:
        super().__init__(model_provider)
        self.workspace = Path(workspace).resolve()
        self.artifacts = artifacts
        self.allowed_artifacts = dict(allowed_artifacts)

    @override
    def get_name(self) -> str:
        return "patch_apply"

    @override
    def get_description(self) -> str:
        return (
            "Apply one coordinator-authorized component patch to the integration workspace. "
            "Call once per component in the supplied topological order."
        )

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                "task_id",
                "string",
                "Developer task whose patch must be applied.",
                enum=list(self.allowed_artifacts),
            )
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.allowed_artifacts:
            return ToolExecResult(error="patch is not authorized for integration", error_code=-1)
        artifact_id = self.allowed_artifacts[task_id]
        try:
            record = self.artifacts.get(artifact_id)
            if record.kind != "patch":
                return ToolExecResult(
                    error="authorized artifact is not a patch", error_code=-1
                )
            patch = self.artifacts.read_text(artifact_id).encode("utf-8")
        except (KeyError, OSError, UnicodeError, ValueError) as exc:
            return ToolExecResult(error=str(exc), error_code=-1)
        checked = subprocess.run(
            ["git", "apply", "--check", "--binary", "--whitespace=nowarn", "-"],
            cwd=self.workspace,
            input=patch,
            capture_output=True,
            timeout=30,
            check=False,
        )
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", "--binary", "--whitespace=nowarn", "-"],
            cwd=self.workspace, input=patch, capture_output=True, timeout=30, check=False,
        )
        if reverse.returncode == 0:
            if checked.returncode == 0:
                return ToolExecResult(error="patch state is ambiguous; rebuild a clean workspace before retrying", error_code=-1)
            return ToolExecResult(output=f"already applied task={task_id} artifact={artifact_id}")
        if checked.returncode != 0:
            return ToolExecResult(
                error=checked.stderr.decode("utf-8", "replace").strip()
                or f"component patch from {task_id} conflicts",
                error_code=checked.returncode,
            )
        applied = subprocess.run(
            ["git", "apply", "--binary", "--whitespace=nowarn", "-"],
            cwd=self.workspace,
            input=patch,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if applied.returncode != 0:
            return ToolExecResult(
                error=applied.stderr.decode("utf-8", "replace").strip()
                or f"component patch from {task_id} could not be applied",
                error_code=applied.returncode,
            )
        return ToolExecResult(
            output=f"applied task={task_id} artifact={artifact_id}"
        )
