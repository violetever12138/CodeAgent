"""Isolated workspaces for writing and test workers."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from codeagent.protocol import TaskNode, WorkerRole
from codeagent.runtime.completion import PATCH_ADD_PATHSPEC


class WorkspaceError(RuntimeError):
    pass


class WorkspaceManager:
    isolated_writes = True

    async def acquire(self, node: TaskNode) -> Path:
        raise NotImplementedError

    async def cleanup(self) -> None:
        raise NotImplementedError

    async def reset(self, task_id: str) -> None:
        """Discard an uncommitted task workspace before replaying a worker."""


class GitWorkspaceManager(WorkspaceManager):
    """Create isolated component worktrees and rebuild clean derived workspaces."""

    def __init__(
        self,
        repository: str | Path,
        worktree_root: str | Path,
        base_commit: str | None = None,
        keep: bool = False,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.keep = keep
        self._workspaces: dict[str, Path] = {}
        self._created: list[Path] = []
        self._lock = asyncio.Lock()
        if not self.repository.is_dir():
            raise WorkspaceError(f"repository is not a directory: {self.repository}")
        self._is_git = self._git_check()
        if not self._is_git:
            raise WorkspaceError("CodeSwarm write isolation currently requires a Git repository")
        self.base_commit = self._git_rev_parse(base_commit or "HEAD")
        if self.worktree_root.exists():
            for path in self.worktree_root.iterdir():
                if path.is_dir():
                    self._workspaces[path.name] = path
                    self._created.append(path)

    async def acquire(self, node: TaskNode) -> Path:
        async with self._lock:
            workspace_from = node.metadata.get("workspace_from") or node.metadata.get(
                "candidate_task_id"
            )
            needs_derived_workspace = (
                node.role == WorkerRole.TESTER and node.metadata.get("phase") == "verify"
            ) or (
                node.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER} and isinstance(workspace_from, str)
            )
            if (
                needs_derived_workspace
                and isinstance(workspace_from, str)
                and workspace_from in self._workspaces
            ):
                if node.id in self._workspaces:
                    return self._workspaces[node.id]
                path = self.worktree_root / node.id
                self.worktree_root.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    self._create_workspace_from_patch,
                    self._workspaces[workspace_from],
                    path,
                )
                self._workspaces[node.id] = path
                self._created.append(path)
                return path
            if isinstance(workspace_from, str) and workspace_from in self._workspaces:
                self._workspaces[node.id] = self._workspaces[workspace_from]
                return self._workspaces[node.id]
            needs_isolation = node.role in {
                WorkerRole.MANAGER,
                WorkerRole.DEVELOPER,
                WorkerRole.WORKER,
                WorkerRole.INTEGRATOR,
            } or (
                node.role == WorkerRole.TESTER and node.metadata.get("phase") == "reproduce"
            )
            if not needs_isolation:
                self._workspaces[node.id] = self.repository
                return self.repository
            if node.id in self._workspaces:
                return self._workspaces[node.id]
            path = self.worktree_root / node.id
            self.worktree_root.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._create_workspace, path)
            self._workspaces[node.id] = path
            self._created.append(path)
            return path

    async def cleanup(self) -> None:
        if self.keep:
            return
        for path in reversed(self._created):
            if path != self.worktree_root and self.worktree_root not in path.parents:
                raise WorkspaceError(f"refusing to clean unexpected path: {path}")
            await asyncio.to_thread(self._remove_workspace, path)
        self._created.clear()

    async def reset(self, task_id: str) -> None:
        async with self._lock:
            path = self._workspaces.pop(task_id, None)
            if path is None:
                candidate = self.worktree_root / task_id
                path = candidate if candidate.exists() else None
            if path is None or path == self.repository:
                return
            if path in self._workspaces.values():
                return  # A read-only alias must never delete its source delivery.
            if path != self.worktree_root and self.worktree_root not in path.parents:
                raise WorkspaceError(f"refusing to reset unexpected path: {path}")
            await asyncio.to_thread(self._remove_workspace, path)
            self._created = [item for item in self._created if item != path]

    def _create_workspace(self, path: Path) -> None:
        if path.exists():
            raise WorkspaceError(f"workspace already exists: {path}")
        completed = subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), self.base_commit],
            cwd=self.repository,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip() or "git worktree add failed")

    def _create_workspace_from_patch(self, source: Path, path: Path) -> None:
        """Rebuild a source workspace from its patch in a fresh worktree."""
        self._create_workspace(path)
        subprocess.run(
            ["git", "add", "-N", "--", *PATCH_ADD_PATHSPEC], cwd=source,
            capture_output=True, timeout=30, check=False,
        )
        patch = subprocess.run(
            ["git", "--no-pager", "diff", "HEAD", "--binary"], cwd=source,
            capture_output=True, timeout=30, check=False,
        )
        if patch.returncode != 0:
            self._remove_workspace(path)
            raise WorkspaceError(
                patch.stderr.decode("utf-8", "replace").strip()
                or "cannot export source patch"
            )
        applied = subprocess.run(
            ["git", "apply", "--binary", "--whitespace=nowarn", "-"], cwd=path,
            input=patch.stdout, capture_output=True, timeout=30, check=False,
        )
        if applied.returncode != 0:
            self._remove_workspace(path)
            raise WorkspaceError(
                applied.stderr.decode("utf-8", "replace").strip()
                or "source patch does not apply cleanly"
            )

    def _remove_workspace(self, path: Path) -> None:
        if self._is_git:
            completed = subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=self.repository,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if completed.returncode == 0 or not path.exists():
                return
        if path.exists():
            shutil.rmtree(path)

    def _git_check(self) -> bool:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=self.repository,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def _git_rev_parse(self, revision: str) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", revision],
            cwd=self.repository,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip() or f"cannot resolve {revision}")
        return completed.stdout.strip()
