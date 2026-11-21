"""Evidence-based completion checks before a worker accepts task_done."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codeagent.protocol import ReviewDecision, ReviewVerdict, WorkerRole
from codeagent.runtime.tool_policy import within_write_scope

_TEST = re.compile(
    r"(^|\s)(pytest|unittest|tox|jest|vitest|mocha|cargo\s+test|go\s+test|"
    r"npm\s+test|pnpm\s+test|mvn\s+test|gradle\s+test)(\s|$)", re.I,
)
_FILE_REFERENCE = re.compile(
    r"(?:^|[\s(`])(?:[\w.-]+[/\\])*[\w.-]+\.[A-Za-z0-9]{1,12}"
    r"(?::\d+(?:-\d+)?)?",
    re.MULTILINE,
)

# Runtime-generated files must not become component code changes merely because a
# test command created them in an otherwise clean repository.
PATCH_ADD_PATHSPEC = (
    ".",
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.py[co]",
    ":(exclude,glob)**/.pytest_cache/**",
)


def is_test_command(command: str) -> bool:
    return bool(_TEST.search(command))


def normalized_command(command: str) -> str:
    return " ".join(command.split())


def _git(workspace: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=workspace, capture_output=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or "git failed")
    return result.stdout


def workspace_version(workspace: Path) -> str:
    """Hash working-tree contents independently of index intent-to-add state."""
    digest = hashlib.sha256()
    digest.update(_git(workspace, "rev-parse", "HEAD"))
    changed = _git(workspace, "diff", "HEAD", "--name-only", "-z")
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard", "-z")
    names = set(filter(None, changed.split(b"\0")))
    names.update(filter(None, untracked.split(b"\0")))
    for name in sorted(names):
        path = (workspace / name.decode("utf-8", "surrogateescape")).resolve()
        digest.update(name)
        if workspace.resolve() not in path.parents or not path.is_file():
            digest.update(b"<deleted-or-unreadable>")
            continue
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def workspace_patch(workspace: Path) -> str:
    _git(workspace, "add", "-N", "--", *PATCH_ADD_PATHSPEC)
    return _git(workspace, "--no-pager", "diff", "HEAD", "--binary").decode(
        "utf-8", "replace"
    )


def changed_paths(patch: str) -> tuple[str, ...]:
    files = re.findall(r"^diff --git a/(.+?) b/(.+?)$", patch, re.MULTILINE)
    return tuple(dict.fromkeys(right for _, right in files))


def parse_review_decisions(content: str) -> tuple[ReviewDecision, ...]:
    decisions, _ = review_decisions_with_errors(content)
    return decisions


def review_decisions_with_errors(
    content: str,
) -> tuple[tuple[ReviewDecision, ...], tuple[str, ...]]:
    if not content:
        return (), ("missing decisions JSON",)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidate = fenced.group(1) if fenced else content[content.find("{"):content.rfind("}") + 1]
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return (), ("decisions must be valid JSON",)
    if not isinstance(payload, dict):
        return (), ("decisions JSON root must be an object",)
    if not isinstance(payload.get("decisions"), list):
        return (), ("decisions must be an array",)
    decisions = []
    errors = []
    for index, item in enumerate(payload["decisions"]):
        if not isinstance(item, dict):
            errors.append(f"decisions[{index}] must be an object")
            continue
        try:
            target_task_id = item.get("target_task_id") or item.get("candidate_task_id")
            if not isinstance(target_task_id, str) or not target_task_id:
                raise ValueError("target_task_id must be a non-empty string")
            confidence = item.get("confidence", 0.0)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("confidence must be a number in [0, 1]")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be a finite number in [0, 1]")
            decisions.append(ReviewDecision(
                target_task_id=target_task_id,
                verdict=ReviewVerdict(str(item["verdict"]).lower()),
                reasons=tuple(str(reason) for reason in item.get("reasons", [])),
                confidence=float(confidence),
                missing_tests=tuple(str(test) for test in item.get("missing_tests", [])),
                rework_task_ids=tuple(
                    str(task_id) for task_id in item.get("rework_task_ids", [])
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"decisions[{index}]: {exc}")
    return tuple(decisions), tuple(errors)


@dataclass(frozen=True)
class ToolEvidence:
    call_id: str
    name: str
    command: str
    success: bool
    workspace_version: str | None
    artifact_id: str | None


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: tuple[str, ...]
    workspace_version: str | None = None


class CompletionGate:
    def __init__(
        self,
        *,
        role: WorkerRole,
        workspace: Path,
        metadata: dict,
        test_commands: tuple[str, ...],
    ) -> None:
        self.role = role
        self.workspace = workspace.resolve()
        self.metadata = metadata
        self.test_commands = test_commands
        self.evidence: list[ToolEvidence] = []
        self.last_decision = GateDecision(False, ("task_done has not been evaluated",))

    def record(self, call, result, artifact_id: str | None = None) -> None:
        normalized = call.name.lower().replace("_", "")
        command = str(call.arguments.get("command", "")) if isinstance(call.arguments, dict) else ""
        if normalized == "patchapply" and isinstance(call.arguments, dict):
            command = str(call.arguments.get("task_id", ""))
        version = None
        if normalized == "bash" and is_test_command(command):
            version = workspace_version(self.workspace)
        self.evidence.append(ToolEvidence(
            call_id=call.call_id, name=call.name, command=command,
            success=result.success, workspace_version=version,
            artifact_id=artifact_id,
        ))

    def evaluate(self, final_content: str) -> GateDecision:
        reasons: list[str] = []
        version = workspace_version(self.workspace)
        if self.role == WorkerRole.EXPLORER or (
            self.role == WorkerRole.MANAGER and self.metadata.get("task_mode") == "read"
        ):
            if not any(
                item.success and item.name.lower().replace("_", "") in
                {"repositoryread", "ckg"} for item in self.evidence
            ):
                reasons.append("no successful repository observation")
        elif self.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER}:
            patch = workspace_patch(self.workspace)
            version = workspace_version(self.workspace)
            paths = changed_paths(patch)
            business_paths = [
                path for path in paths
                if not (path.startswith("tests/") or Path(path).name.startswith("test_"))
            ]
            if not paths:
                reasons.append("no patch in assigned workspace")
            elif not business_paths and not self.metadata.get("allow_test_only"):
                reasons.append("no non-test patch in assigned workspace")
            scopes = tuple(self.metadata.get("write_scope", ()))
            outside_scope = [
                path for path in paths if scopes and not within_write_scope(path, scopes)
                and not (self.metadata.get("manager_action")
                         and path in self.metadata.get("inherited_paths", ()))
            ]
            if outside_scope:
                reasons.append(f"patch modifies files outside write scope: {outside_scope}")
            if not (self.metadata.get("single_agent") or self.metadata.get("sole_work_unit")):
                missing_paths = sorted(set(self.metadata.get("required_paths", ())) - set(paths))
                if missing_paths:
                    reasons.append(f"assigned deliverable files not changed: {missing_paths}")
            if (self.metadata.get("single_agent") or self.metadata.get("sole_work_unit")) and not self.metadata.get("manager_action"):
                reasons.extend(self._test_reasons(version))
        elif self.role == WorkerRole.SECURITY:
            decisions, schema_errors = review_decisions_with_errors(final_content)
            reasons.extend(schema_errors)
            expected = set(self.metadata.get("security_target_task_ids", []))
            actual = {item.target_task_id for item in decisions}
            if not decisions or actual != expected or len(decisions) != len(expected):
                reasons.append("missing structured security decisions for component tasks")
        elif self.role == WorkerRole.INTEGRATOR:
            patch = workspace_patch(self.workspace)
            version = workspace_version(self.workspace)
            if not changed_paths(patch):
                reasons.append("integrated workspace has no patch")
            applied = {
                item.command
                for item in self.evidence
                if item.name.lower().replace("_", "") == "patchapply" and item.success
            }
            required = set(self.metadata.get("component_task_ids", []))
            missing = required - applied
            if missing:
                reasons.append(f"component patches not integrated: {sorted(missing)}")
        elif self.role == WorkerRole.TESTER:
            if self.metadata.get("phase") == "verify":
                reasons.extend(self._test_reasons(version))
            elif not any(
                item.name.lower().replace("_", "") == "bash"
                and is_test_command(item.command)
                and not item.success
                for item in self.evidence
            ):
                reasons.append("reproduction has no observed failing test command")
        elif self.role in {WorkerRole.REVIEWER, WorkerRole.AUDITOR}:
            decisions, schema_errors = review_decisions_with_errors(final_content)
            reasons.extend(schema_errors)
            expected = set(
                self.metadata.get(
                    "review_target_task_ids",
                    self.metadata.get("candidate_task_ids", []),  # legacy trace fallback
                )
            )
            actual = {item.target_task_id for item in decisions}
            if not decisions or actual != expected or len(decisions) != len(expected):
                reasons.append("missing structured decisions for integrated result")
        self.last_decision = GateDecision(not reasons, tuple(reasons), version)
        return self.last_decision

    def accepts_implicit_explorer_completion(self, final_content: str | None) -> bool:
        """Accept a concrete Explorer report even if the model omitted ``task_done``.

        This is deliberately narrow: only Explorer workers qualify, at least one real
        repository observation must already have succeeded, and the response must be a
        substantial report containing a concrete file reference. The normal evidence
        gate remains authoritative; this only removes dependence on provider-specific
        tool-call compliance after the deliverable has already been produced.
        """
        content = (final_content or "").strip()
        if self.role != WorkerRole.EXPLORER or len(content) < 200:
            return False
        if not _FILE_REFERENCE.search(content):
            return False
        return self.evaluate(content).accepted

    def accepts_evidence_backed_completion(self, final_content: str | None) -> bool:
        """Allow a role deliverable to finish without relying on ``task_done``.

        Each role still uses its deterministic evidence contract. Explorer reports have
        additional content checks; Developer requires a business patch; Tester requires
        phase-appropriate test evidence; Reviewer requires structured decisions. Planner
        and Decomposer can finish when their work-unit JSON passes the same DAG gate.
        """
        if self.role == WorkerRole.EXPLORER:
            return self.accepts_implicit_explorer_completion(final_content)
        if self.role in {
            WorkerRole.MANAGER,
            WorkerRole.DEVELOPER,
            WorkerRole.WORKER,
            WorkerRole.AUDITOR,
            WorkerRole.SECURITY,
            WorkerRole.INTEGRATOR,
            WorkerRole.TESTER,
            WorkerRole.REVIEWER,
        }:
            return self.evaluate(final_content or "").accepted
        return False

    def accepts_terminal_evidence_completion(self, final_content: str | None) -> bool:
        """Use the role gate at a hard step boundary without weakening normal turns."""
        if self.role == WorkerRole.EXPLORER:
            return self.evaluate(final_content or "").accepted
        return self.accepts_evidence_backed_completion(final_content)

    def accepts_post_tool_completion(self) -> bool:
        """Stop a developer as soon as its evidence contract is met.

        A developer has no useful work left after it has produced a business-code patch
        and run every required test successfully against the current workspace version.
        Ending at that boundary prevents a model from invalidating fresh verification by
        continuing with speculative edits or ad-hoc checks.
        """
        return (
            self.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER}
            and self.metadata.get("task_mode") != "read"
            and bool(self.metadata.get("single_agent") or self.metadata.get("sole_work_unit"))
            and self.evaluate("").accepted
        )

    def _test_reasons(self, current_version: str) -> list[str]:
        tests = [
            item for item in self.evidence
            if item.name.lower().replace("_", "") == "bash"
            and is_test_command(item.command)
        ]
        if not tests:
            return ["no recorded test command"]
        latest = {normalized_command(item.command): item for item in tests}
        required = tuple(map(normalized_command, self.test_commands))
        if not required:
            required = tuple(latest)
        reasons = []
        for command in required:
            record = latest.get(command)
            if record is None:
                reasons.append(f"required test was not run: {command}")
            elif not record.success:
                reasons.append(f"required test failed: {command}")
            elif record.workspace_version != current_version:
                reasons.append(f"test evidence is stale after workspace change: {command}")
        return reasons
