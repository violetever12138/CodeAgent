"""Public entry point for the Manager V2 orchestration backend."""

from pathlib import Path

from codeagent.cancellation import request_run_cancellation
from codeagent.langgraph_backend import LangGraphOrchestrationBackend
from codeagent.protocol import TaskOutcome, TaskRequest
from codeagent.worker import WorkerFactory


class CodeSwarmOrchestrator:
    def __init__(
        self, worker_factory: WorkerFactory, *, run_root: str | Path | None = None,
        keep_workspaces: bool = False, workspace_manager_factory=None,
        architecture_mode: str = "manager_v2",
    ) -> None:
        if architecture_mode != "manager_v2":
            raise ValueError("this release requires the manager_v2 architecture")
        self.backend = LangGraphOrchestrationBackend(
            worker_factory, run_root=run_root, keep_workspaces=keep_workspaces,
            workspace_manager_factory=workspace_manager_factory,
        )

    async def run(self, request: TaskRequest) -> TaskOutcome:
        return await self.backend.run(request)

    async def resume(
        self, run_directory: str | Path, *, approval_decision: str | None = None,
        approval_comment: str | None = None,
    ) -> TaskOutcome:
        return await self.backend.resume(
            run_directory, approval_decision=approval_decision,
            approval_comment=approval_comment,
        )

    @staticmethod
    def cancel(run_directory: str | Path, reason: str = "cancelled by user") -> Path:
        return request_run_cancellation(run_directory, reason)
