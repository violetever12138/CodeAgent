"""Execution backends for trusted Windows tasks and isolated Docker tasks."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


@dataclass(frozen=True)
class ExecutionResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_limited: bool = False
    process_tree_reclaimed: bool = False
    backend_strength: str = "process_guarded_trusted"


class ExecutionBackend(Protocol):
    name: str
    strength: str

    def run(self, command: str, *, cwd: Path, temp_dir: Path | None = None) -> ExecutionResult: ...


class WindowsGuardedBackend:
    """One-shot argv execution for trusted repositories; this is not an OS sandbox."""

    name = "windows-guarded"
    strength = "process_guarded_trusted"
    _SAFE_ENV = {
        "SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE", "PYTHONIOENCODING", "PYTHONUTF8",
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        memory_mb: int = 4096,
        process_limit: int = 64,
        output_limit_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if min(timeout_seconds, memory_mb, process_limit, output_limit_bytes) <= 0:
            raise ValueError("execution limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.process_limit = process_limit
        self.output_limit_bytes = output_limit_bytes
        self._active_lock = threading.Lock()
        self._active: dict[int, tuple[object, subprocess.Popen]] = {}

    def run(self, command: str, *, cwd: Path, temp_dir: Path | None = None) -> ExecutionResult:
        if os.name != "nt":
            raise RuntimeError("WindowsGuardedBackend is available only on Windows")
        workspace = cwd.resolve()
        argv = self.normalize_argv(command)
        private_temp = temp_dir or Path(tempfile.mkdtemp(prefix="codeswarm-guarded-"))
        private_temp.mkdir(parents=True, exist_ok=True)
        environment = {key: value for key, value in os.environ.items() if key.upper() in self._SAFE_ENV}
        environment.update({
            "TMP": str(private_temp), "TEMP": str(private_temp), "TMPDIR": str(private_temp),
            "PYTEST_DEBUG_TEMPROOT": str(private_temp),
            "PYTEST_ADDOPTS": "--rootdir=. -p no:cacheprovider",
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        })
        process = subprocess.Popen(
            argv, cwd=workspace, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        job = self._assign_job(process)
        with self._active_lock:
            self._active[process.pid] = (job, process)
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        total = 0
        total_lock = threading.Lock()
        output_limited = threading.Event()

        def drain(stream, target: list[bytes]) -> None:
            nonlocal total
            while True:
                block = stream.read(65_536)
                if not block:
                    return
                with total_lock:
                    remaining = self.output_limit_bytes - total
                    if remaining > 0:
                        target.append(block[:remaining])
                        total += min(len(block), remaining)
                    if len(block) > remaining:
                        output_limited.set()
                        self._terminate_job(job, process)

        readers = [
            threading.Thread(target=drain, args=(process.stdout, stdout_parts), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr_parts), daemon=True),
        ]
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + self.timeout_seconds
        timed_out = False
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                self._terminate_job(job, process)
                break
            time.sleep(0.02)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._terminate_job(job, process)
            process.wait(timeout=2)
        for reader in readers:
            reader.join(timeout=2)
        reclaimed = timed_out or output_limited.is_set()
        with self._active_lock:
            self._active.pop(process.pid, None)
        self._close_job(job)
        suffix = b"\n<output limit exceeded>" if output_limited.is_set() else b""
        return ExecutionResult(
            argv=tuple(argv), returncode=process.returncode if process.returncode is not None else -1,
            stdout=(b"".join(stdout_parts) + suffix).decode("utf-8", "replace"),
            stderr=b"".join(stderr_parts).decode("utf-8", "replace"),
            timed_out=timed_out, output_limited=output_limited.is_set(),
            process_tree_reclaimed=reclaimed,
        )

    def cancel_all(self) -> None:
        """Terminate every active process tree owned by this backend instance."""

        with self._active_lock:
            active = tuple(self._active.values())
        for job, process in active:
            self._terminate_job(job, process)

    @staticmethod
    def normalize_argv(command: str) -> list[str]:
        tokens = shlex.split(command, posix=True)
        if not tokens:
            raise ValueError("empty command")
        executable = Path(tokens[0]).name.lower().removesuffix(".exe")
        if executable in {"pytest", "ruff", "mypy"}:
            return [sys.executable, "-m", executable, *tokens[1:]]
        if executable in {"python", "python3", "py"}:
            return [sys.executable, *tokens[1:]]
        return tokens

    def _assign_job(self, process: subprocess.Popen):
        try:
            import win32api
            import win32con
            import win32job
        except ImportError as exc:
            process.terminate()
            raise RuntimeError("pywin32 is required for the guarded Windows backend") from exc
        job = win32job.CreateJobObject(None, "")
        information = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation,
        )
        information["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        )
        information["BasicLimitInformation"]["ActiveProcessLimit"] = self.process_limit
        information["ProcessMemoryLimit"] = self.memory_mb * 1024 * 1024
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, information,
        )
        handle = win32api.OpenProcess(
            win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA,
            False, process.pid,
        )
        try:
            win32job.AssignProcessToJobObject(job, handle)
        except Exception:
            process.terminate()
            self._close_job(job)
            raise
        finally:
            handle.Close()
        return job

    @staticmethod
    def _terminate_job(job, process: subprocess.Popen) -> None:
        try:
            import win32job

            win32job.TerminateJobObject(job, 1)
        except Exception:
            process.kill()

    @staticmethod
    def _close_job(job) -> None:
        with contextlib.suppress(Exception):
            job.Close()


class GuardedBashTool(Tool):
    """Bash-compatible tool schema backed by independent guarded processes."""

    def __init__(
        self,
        backend: WindowsGuardedBackend,
        *,
        cwd: str | Path,
        temp_dir: str | Path,
        model_provider: str | None = None,
    ) -> None:
        super().__init__(model_provider)
        self.backend = backend
        self.cwd = Path(cwd).resolve()
        self.temp_dir = Path(temp_dir).resolve()

    @override
    def get_name(self) -> str:
        return "bash"

    @override
    def get_description(self) -> str:
        return (
            "Run one allowlisted command in a fresh process. State is not persistent; "
            "the process tree, memory, wall time, output, environment and temp root are bounded."
        )

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="command", type="string", description="Command to execute."),
            ToolParameter(
                name="restart", type="boolean", description="Compatibility flag; no session exists.",
                required=self.model_provider == "openai",
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if arguments.get("restart"):
            return ToolExecResult(output="stateless guarded backend is ready")
        command = arguments.get("command")
        if not isinstance(command, str):
            return ToolExecResult(error="command is required", error_code=-1)
        try:
            result = await asyncio.to_thread(
                self.backend.run, command, cwd=self.cwd, temp_dir=self.temp_dir,
            )
        except Exception as exc:
            return ToolExecResult(error=f"guarded execution failed: {exc}", error_code=-1)
        error = result.stderr
        if result.timed_out:
            error = f"timed out after {self.backend.timeout_seconds}s; process tree reclaimed\n{error}"
        return ToolExecResult(output=result.stdout, error=error, error_code=result.returncode)

    @override
    async def close(self) -> None:
        self.backend.cancel_all()


@dataclass(frozen=True)
class DockerBackend:
    image: str
    profile: object | None = None
    name: str = "docker"
    strength: str = "container_inspected"

    def __post_init__(self) -> None:
        from codeswarm_agent.safety.sandbox_profile import SandboxProfile

        profile = self.profile or SandboxProfile()
        profile.validate_image(self.image)
        object.__setattr__(self, "profile", profile)

    def agent_config(self, workspace: Path) -> dict[str, object]:
        """Configuration consumed by Trae's per-worker DockerManager."""

        return {
            "image": self.image,
            "workspace_dir": str(workspace.resolve()),
            "security_options": self.profile.docker_options(),
        }

    def run(self, command: str, *, cwd: Path, temp_dir: Path | None = None) -> ExecutionResult:
        raise RuntimeError(
            "DockerBackend is session based; execute through the worker DockerToolExecutor"
        )
