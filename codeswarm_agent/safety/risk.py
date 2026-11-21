"""Fail-closed command classification for the intentionally small Bash surface."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum


class CommandRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class CommandAssessment:
    risk: CommandRisk
    rule_id: str
    reason: str
    executable: str | None = None

    @property
    def allowed_without_approval(self) -> bool:
        return self.risk in {CommandRisk.LOW, CommandRisk.MEDIUM}


_CONTROL = re.compile(r"(?:\r|\n|&&|\|\||[;|`<>]|\$\(|(?<!\^)&)")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_FORBIDDEN = {
    "bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh",
    "sudo", "su", "runas", "rm", "del", "erase", "rmdir", "format",
    "diskpart", "shutdown", "reboot", "taskkill", "kill", "chmod", "chown",
    "curl", "wget", "ssh", "scp", "sftp", "nc", "ncat", "netcat",
    "docker", "podman", "kubectl", "reg", "regedit",
}
_READ_ONLY = {
    "pwd", "ls", "dir", "rg", "grep", "findstr", "cat", "type", "head",
    "tail", "sed", "wc", "where", "which", "tree",
}
_TESTS = {"pytest", "tox", "ruff", "mypy", "pyright"}


def _tokens(command: str) -> list[str]:
    # posix=True is deterministic for the command grammar accepted by the Agent even on Windows.
    return shlex.split(command, posix=True)


def assess_command(command: str) -> CommandAssessment:
    """Classify a single command. Unknown or compound shell programs fail closed."""
    value = command.strip()
    if not value:
        return CommandAssessment(CommandRisk.FORBIDDEN, "CMD-000", "empty command")
    if len(value) > 8192:
        return CommandAssessment(CommandRisk.FORBIDDEN, "CMD-001", "command is too long")
    if _CONTROL.search(value):
        return CommandAssessment(
            CommandRisk.FORBIDDEN, "CMD-002",
            "compound shell syntax, redirection, substitution, and background jobs are disabled",
        )
    try:
        tokens = _tokens(value)
    except ValueError as exc:
        return CommandAssessment(CommandRisk.FORBIDDEN, "CMD-003", f"invalid quoting: {exc}")
    if not tokens or _ENV_ASSIGNMENT.match(tokens[0]):
        return CommandAssessment(
            CommandRisk.FORBIDDEN, "CMD-004", "environment assignment is not an executable",
        )
    executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    executable = executable.removesuffix(".exe")
    lowered = [token.lower() for token in tokens]
    if executable in _FORBIDDEN:
        return CommandAssessment(
            CommandRisk.FORBIDDEN, "CMD-010", f"{executable} is outside the command capability",
            executable,
        )
    if executable == "git":
        subcommand = lowered[1] if len(lowered) > 1 else ""
        if subcommand in {"status", "diff", "log", "show", "rev-parse", "ls-files"}:
            return CommandAssessment(CommandRisk.LOW, "CMD-100", "read-only git command", executable)
        return CommandAssessment(
            CommandRisk.FORBIDDEN, "CMD-011", "state-changing git commands are disabled",
            executable,
        )
    if executable in _READ_ONLY:
        return CommandAssessment(CommandRisk.LOW, "CMD-101", "read-only inspection command", executable)
    if executable in _TESTS:
        return CommandAssessment(CommandRisk.MEDIUM, "CMD-200", "bounded test/static-check command", executable)
    if executable in {"python", "python3", "py"}:
        if len(lowered) >= 3 and lowered[1] == "-m" and lowered[2] in {
            "pytest", "unittest", "compileall", "ruff", "mypy",
        }:
            return CommandAssessment(
                CommandRisk.MEDIUM, "CMD-201", "allowlisted Python module", executable,
            )
        return CommandAssessment(
            CommandRisk.HIGH, "CMD-300", "arbitrary Python execution requires approval", executable,
        )
    return CommandAssessment(
        CommandRisk.HIGH, "CMD-301", "command is not in the allowlist", executable,
    )
