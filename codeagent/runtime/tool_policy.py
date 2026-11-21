"""Role, shape, path, and conflict checks before tool side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from codeagent.safety.risk import CommandRisk, assess_command


class ToolEffect(str, Enum):
    READ = "read"
    STATEFUL = "stateful"
    WRITE = "write"
    EXTERNAL_WRITE = "external_write"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    reason: str
    effect: ToolEffect
    resource_key: str | None = None
    rule_id: str = "TOOL-ALLOW"
    risk: str | None = None


_READ_TOOLS = {"repositoryread", "artifactread"}
_WRITE_TOOLS = {"strreplacebasededittool", "jsonedittool", "patchapply"}
_PATH_KEYS = {"path", "file_path", "working_dir", "workspace_dir", "output_path"}


def normalize_name(name: str) -> str:
    return name.lower().replace("_", "")


def read_tool_name(name: str) -> str:
    """Keep the native read guard and consumption rules for MCP read aliases."""
    if name.lower().startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3 and normalize_name(parts[2]) in _READ_TOOLS:
            return normalize_name(parts[2])
    return normalize_name(name)


def within_write_scope(path: str, scopes: tuple[str, ...]) -> bool:
    """Match a repository-relative path against an explicit work-unit scope."""
    normalized = path.replace("\\", "/").lstrip("./")
    for raw_scope in scopes:
        scope = raw_scope.replace("\\", "/").lstrip("./")
        if scope == "**" or normalized == scope:
            return True
        if scope.endswith("/**") and normalized.startswith(scope[:-2]):
            return True
        if PurePosixPath(normalized).match(scope):
            return True
    return False


class ToolPolicy:
    def __init__(
        self,
        role,
        workspace: Path,
        tools: list | None = None,
        write_scope: tuple[str, ...] = (),
        additional_allowed: tuple[str, ...] = (),
    ) -> None:
        self.role = role
        self.workspace = workspace.resolve()
        self.allowed = {
            normalize_name(name) for name in (*role.allowed_tools, *additional_allowed)
        }
        self.write_scope = write_scope
        self.tools = {normalize_name(tool.name): tool for tool in tools or []}
        if len(self.tools) != len(tools or []):
            raise ValueError("normalized tool names collide")

    def effect(self, name: str) -> ToolEffect:
        normalized = normalize_name(name)
        tool = self.tools.get(normalized)
        declared = getattr(tool, "policy_effect", None)
        if declared == "read":
            return ToolEffect.READ
        if declared == "workspace_write":
            return ToolEffect.WRITE
        if declared == "external_write":
            return ToolEffect.EXTERNAL_WRITE
        if normalized in _READ_TOOLS:
            return ToolEffect.READ
        return ToolEffect.WRITE if normalized in _WRITE_TOOLS else ToolEffect.STATEFUL

    def validate(self, call) -> PolicyDecision:
        normalized = normalize_name(call.name)
        effect = self.effect(call.name)
        if normalized not in self.allowed:
            return PolicyDecision(False, "PERMISSION_DENIED", "tool is not allowed for role",
                                  effect)
        if effect == ToolEffect.EXTERNAL_WRITE:
            return PolicyDecision(
                False, "PERMISSION_DENIED", "external-write MCP tools are disabled",
                effect, rule_id="MCP-EXTERNAL-WRITE-DENY",
            )
        if self.tools and normalized not in self.tools:
            return PolicyDecision(False, "TOOL_NOT_FOUND", "allowed tool is unavailable", effect)
        if not isinstance(call.arguments, dict):
            return PolicyDecision(False, "INVALID_ARGUMENT", "arguments must be an object",
                                  effect)
        if normalized == "bash":
            command = call.arguments.get("command")
            if not isinstance(command, str):
                return PolicyDecision(
                    False, "INVALID_ARGUMENT", "command must be a string", effect,
                    rule_id="TOOL-BASH-SHAPE",
                )
            assessment = assess_command(command)
            if assessment.risk == CommandRisk.FORBIDDEN:
                return PolicyDecision(
                    False, "PERMISSION_DENIED", assessment.reason, effect,
                    rule_id=assessment.rule_id, risk=assessment.risk.value,
                )
            if assessment.risk == CommandRisk.HIGH:
                return PolicyDecision(
                    False, "APPROVAL_REQUIRED", assessment.reason, effect,
                    rule_id=assessment.rule_id, risk=assessment.risk.value,
                )
        tool = self.tools.get(normalized)
        if tool is not None:
            error = self._validate_parameters(tool, call.arguments)
            if error:
                return PolicyDecision(False, "INVALID_ARGUMENT", error, effect)
        for key, value in call.arguments.items():
            if key not in _PATH_KEYS or value is None:
                continue
            if not isinstance(value, str):
                return PolicyDecision(False, "INVALID_ARGUMENT", f"{key} must be a path",
                                      effect)
            if value == "null":
                continue
            raw = Path(value)
            candidate = (raw if raw.is_absolute() else self.workspace / raw).resolve()
            if candidate != self.workspace and self.workspace not in candidate.parents:
                return PolicyDecision(False, "PERMISSION_DENIED",
                                      f"{key} escapes worker workspace", effect)
            if effect == ToolEffect.WRITE and self.write_scope:
                relative = candidate.relative_to(self.workspace).as_posix()
                if not within_write_scope(relative, self.write_scope):
                    return PolicyDecision(
                        False,
                        "PERMISSION_DENIED",
                        f"{key} is outside assigned write scope",
                        effect,
                        rule_id="TOOL-WRITE-SCOPE",
                    )
        resource = "shell:session" if normalized == "bash" else None
        for key in _PATH_KEYS:
            if isinstance(call.arguments.get(key), str):
                resource = str(call.arguments[key])
                break
        return PolicyDecision(
            True, "ALLOW", "validated", effect, resource,
            rule_id=(assessment.rule_id if normalized == "bash" else "TOOL-ALLOW"),
            risk=(assessment.risk.value if normalized == "bash" else None),
        )

    def partition(self, calls: list) -> list[list]:
        """Only adjacent independent reads share a parallel batch; order is preserved."""
        batches: list[list] = []
        reads: list = []
        for call in calls:
            if self.effect(call.name) == ToolEffect.READ:
                reads.append(call)
                continue
            if reads:
                batches.append(reads)
                reads = []
            batches.append([call])
        if reads:
            batches.append(reads)
        return batches

    @staticmethod
    def _validate_parameters(tool, arguments: dict) -> str | None:
        validate = getattr(tool, "validate_arguments", None)
        if validate:
            return validate(arguments)
        parameters = {parameter.name: parameter for parameter in tool.parameters}
        unknown = set(arguments) - set(parameters)
        if unknown:
            return f"unexpected argument(s): {', '.join(sorted(unknown))}"
        for name, parameter in parameters.items():
            if name not in arguments:
                if parameter.required:
                    return f"missing required argument: {name}"
                continue
            value = arguments[name]
            allowed_types = (
                [parameter.type] if isinstance(parameter.type, str) else parameter.type
            )
            if value is None and ("null" in allowed_types or not parameter.required):
                continue
            if not any(ToolPolicy._is_type(value, name) for name in allowed_types):
                return f"invalid type for {parameter.name}: expected {allowed_types}"
            if parameter.enum and value not in parameter.enum:
                return f"invalid value for {parameter.name}: {value}"
        return None

    @staticmethod
    def _is_type(value, schema_type: str) -> bool:
        return {
            "string": lambda: isinstance(value, str),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": lambda: isinstance(value, bool),
            "object": lambda: isinstance(value, dict),
            "array": lambda: isinstance(value, list),
            "null": lambda: value is None,
        }.get(schema_type, lambda: False)()
