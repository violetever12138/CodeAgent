"""Conservative initial routing and cooperative, tool-boundary handoff signals."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from codeagent.protocol import TaskRequest
from trae_agent.tools.base import Tool, ToolExecResult, ToolParameter

EVENT_KINDS = (
    "independent_task", "new_dependency", "needs_help", "scope_violation",
    "remaining_shrank", "coordination_cost", "context_isolation", "no_progress", "tool_failure",
)


_RISK = ("auth", "security", "secret", "credential", "permission", "sandbox",
         "execute", "shell", "network", "安全", "权限", "密钥", "凭据", "并发")


def sensitive_issue(issue: str) -> bool:
    lowered = issue.casefold()
    return any(term in lowered for term in _RISK)


def single_agent_fast_path(request: TaskRequest) -> dict | None:
    """Only sufficient evidence can release a task; failure means undecided."""
    if (request.force_multi_agent is True or not request.test_commands
            or len(request.acceptance_criteria) > 1 or sensitive_issue(request.issue)
            or any(word in request.issue.casefold() for word in (
                "across", "refactor", "multiple", "migration", "跨", "重构", "迁移", "多个"))):
        return None
    paths = set(re.findall(r"[\w./\\-]+\.py\b", request.issue))
    if len(paths) != 1:
        return None
    relative = next(iter(paths)).replace("\\", "/")
    root = Path(request.repository).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or sensitive_issue(relative) or any(p.startswith(".") for p in Path(relative).parts):
        return None
    try:
        if path.stat().st_size > 12_000:
            return None
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) > 100:
            return None
        tree = ast.parse(text)
    except (OSError, UnicodeError, SyntaxError):
        return None
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        return None
    behaviors = request.acceptance_criteria or (request.issue,)
    plan = {"criteria": [{"id": f"user-{i}", "behavior": behavior,
                          "basis": f"user goal, explicit file:{relative}, supplied test commands",
                          "required": True, "verification": "test", "commands": list(request.test_commands),
                          "scope": [relative], "assumptions": []}
                         for i, behavior in enumerate(behaviors)],
            "revision_reason": "conservative single-Manager rule"}
    return {"action": "execute", "reason": f"bounded single-file task: {relative}",
            "acceptance_plan": plan,
            "tasks": [{"goal": request.issue, "mode": "modify", "write_scope": [relative],
                       "read_scope": [relative], "criteria_ids": [x["id"] for x in plan["criteria"]]}]}


def initial_probe(repository: str, issue: str, *, seconds: float = 0.25) -> dict:
    """Bound directory inspection and text reads; do not investigate the whole task."""
    root = Path(repository).resolve()
    started = time.monotonic()
    files = []
    inspected = 0
    skip = {"node_modules", "__pycache__", "venv", "dist", "build"}
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in skip)
        for name in sorted(names):
            inspected += 1
            if inspected > 200 or time.monotonic() - started >= seconds:
                break
            path = Path(directory) / name
            if name.startswith(".") or path.suffix not in {".py", ".toml", ".json", ".md", ".js", ".ts"}:
                continue
            files.append(path)
        if inspected > 200 or len(files) >= 80 or time.monotonic() - started >= seconds:
            break
    files = files[:80]
    files.sort(key=lambda p: (p.name not in issue, not (p.name.startswith("test_")
                             or p.name in {"pyproject.toml", "package.json"}), p.as_posix()))
    excerpts = []
    for path in files[:6]:
        if time.monotonic() - started >= seconds:
            break
        try:
            with path.open(encoding="utf-8") as stream:
                excerpts.append(f"file:{path.relative_to(root).as_posix()}\n{stream.read(2000)}")
        except (OSError, UnicodeError):
            continue
    return {"files": [p.relative_to(root).as_posix() for p in files],
            "evidence": "\n\n".join(excerpts)[:12_000], "inspected": min(inspected, 200),
            "elapsed_seconds": time.monotonic() - started}


def assignment_signature(version: str, tasks: list[dict]) -> str:
    normalized = [{"goal": " ".join(t["goal"].casefold().split()), "mode": t["mode"],
                   "read_scope": sorted(t["read_scope"]), "write_scope": sorted(t["write_scope"])}
                  for t in tasks]
    return hashlib.sha256(json.dumps([version, sorted(normalized, key=lambda t: json.dumps(t, sort_keys=True))],
                                    sort_keys=True).encode()).hexdigest()


@dataclass
class RoutingControl:
    """Ephemeral coordination only. Durable facts live in results and graph state."""

    seen: set[str] = field(default_factory=set)
    events: list[dict] = field(default_factory=list)
    pause_requested: bool = False

    def request(self, task_id: str, version: str, kind: str, reason: str, evidence: list[str]) -> bool:
        if kind not in EVENT_KINDS or not isinstance(reason, str) or not reason.strip():
            raise ValueError("routing event needs a supported kind and reason")
        if (not isinstance(evidence, list) or not evidence or len(evidence) > 8
                or any(not isinstance(x, str) or not x.strip() or len(x) > 500 for x in evidence)):
            raise ValueError("routing event needs 1-8 bounded evidence references")
        if len(reason) > 1000:
            raise ValueError("routing reason is too long")
        identity = hashlib.sha256(json.dumps([version, kind, reason.strip(), sorted(evidence)],
                                            sort_keys=True).encode()).hexdigest()
        if identity in self.seen:
            return False
        self.seen.add(identity)
        self.events.append({"id": identity, "task_id": task_id, "input_version": version,
                            "kind": kind, "reason": reason.strip(), "evidence": evidence})
        self.pause_requested = True
        return True

    def restore(self, events: list[dict]) -> None:
        for event in events:
            if event["id"] not in self.seen:
                self.seen.add(event["id"])
                self.events.append(event)
                self.pause_requested = True


class RoutingRequestTool(Tool):
    """Host-only control: request a decision, never change modes or cancel tools."""

    def __init__(self, control: RoutingControl, task_id: str, version: str, model_provider=None, *,
                 summary_completed: bool = False):
        super().__init__(model_provider)
        self.control, self.task_id, self.version = control, task_id, version
        self.summary_completed = summary_completed

    def get_name(self):
        return "routing_request"

    def get_description(self):
        return ("Report new evidence requiring Manager reassignment. Current tool operations finish "
                "before handoff. For context_isolation, evidence must describe why completed "
                "summary/cleanup still leaves independent context needs. Does not complete the task.")

    def get_parameters(self):
        return [ToolParameter("kind", "string", "Routing event", enum=list(EVENT_KINDS[:-1])),
                ToolParameter("reason", "string", "Why the current assignment needs reconsideration"),
                ToolParameter("evidence", "array", "1-8 file, error or artifact references",
                              items={"type": "string"})]

    async def execute(self, arguments):
        try:
            if arguments.get("kind") == "context_isolation" and not self.summary_completed:
                raise ValueError("context_isolation requires completed context cleanup and summary first")
            fresh = self.control.request(self.task_id, self.version, arguments.get("kind"),
                                         arguments.get("reason"), arguments.get("evidence"))
        except (ValueError, TypeError) as exc:
            return ToolExecResult(error=str(exc), error_code=-1)
        return ToolExecResult(output="handoff requested at the next safe boundary" if fresh
                              else "same information gap already considered; no new handoff")
