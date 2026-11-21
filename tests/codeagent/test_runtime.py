import asyncio
import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from codeagent.protocol import ReviewDecision, ReviewVerdict, WorkerRole
from codeagent.roles import DEFAULT_ROLES
from codeagent.runtime.completion import CompletionGate, workspace_patch, workspace_version
from codeagent.runtime.recovery import ModelRecovery, retryable_model_error
from codeagent.runtime.state import (
    IllegalTransition,
    Phase,
    RuntimeController,
    RuntimeLimits,
    StopReason,
)
from codeagent.runtime.tool_policy import ToolEffect, ToolPolicy
from codeagent.worker import DeveloperReadGuard, security_rejection_reasons


def test_developer_read_guard_stops_duplicate_reads_until_an_edit():
    guard = DeveloperReadGuard(limit=2)
    assert guard.claim({"operation": "read", "path": "app.py"}) is None
    assert "already observed" in guard.claim({"path": "app.py", "operation": "read"})
    assert guard.claim({"operation": "read", "path": "registry.py"}) is None
    assert "limit" in guard.claim({"operation": "read", "path": "cache.py"})
    guard.after_tool("str_replace_based_edit_tool", True)
    assert guard.claim({"operation": "read", "path": "app.py"}) is None
    guard.after_tool("repository_read", False, {"operation": "read", "path": "app.py"})
    assert guard.claim({"operation": "read", "path": "app.py"}) is None


def _controller(tmp_path):
    return RuntimeController(
        run_id="run", task_id="task", worker_id="worker",
        trace_path=tmp_path / "runtime.jsonl",
        limits=RuntimeLimits(deadline_at=time.monotonic() + 5),
    )


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "app.py").write_text("value = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "initial")
    return root


def test_phase_machine_rejects_illegal_transition_and_terminal_reentry(tmp_path):
    controller = _controller(tmp_path)
    with pytest.raises(IllegalTransition):
        controller.transition(Phase.ACTING)
    controller.before_model()
    controller.finish(StopReason.COMPLETED)
    with pytest.raises(IllegalTransition):
        controller.before_model()


def test_model_retries_only_transient_errors_without_blocking_loop(tmp_path):
    controller = _controller(tmp_path)
    attempts = []

    class HttpError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise HttpError(429)
        return SimpleNamespace(usage=None, model="fake", finish_reason="stop", tool_calls=[])

    async def scenario():
        heartbeat = []

        async def tick():
            await asyncio.sleep(0.005)
            heartbeat.append(True)

        response, _ = await asyncio.gather(
            ModelRecovery(max_attempts=2, base_delay=0.01).call(operation, controller),
            tick(),
        )
        return response, heartbeat

    response, heartbeat = asyncio.run(scenario())
    assert response.model == "fake"
    assert heartbeat
    assert len(attempts) == 2
    assert controller.model_attempts == 2

    forbidden = _controller(tmp_path / "forbidden")

    def unauthorized():
        raise HttpError(401)

    with pytest.raises(HttpError):
        asyncio.run(ModelRecovery(max_attempts=3, base_delay=0).call(unauthorized, forbidden))
    assert forbidden.model_attempts == 1


def test_openai_compatible_connection_errors_are_retryable():
    APIConnectionError = type("APIConnectionError", (Exception,), {})
    assert retryable_model_error(APIConnectionError("connection error"))

    class WrappedError(Exception):
        pass

    wrapped = WrappedError("provider wrapper")
    wrapped.__cause__ = ConnectionResetError("peer reset")
    assert retryable_model_error(wrapped)


def test_malformed_model_tool_arguments_retry_before_dispatch(tmp_path):
    controller = _controller(tmp_path)
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise json.JSONDecodeError("Unterminated string", '{"command":"', 11)
        return SimpleNamespace(usage=None, model="fake", finish_reason="stop", tool_calls=[])

    response = asyncio.run(
        ModelRecovery(max_attempts=2, base_delay=0).call(operation, controller)
    )
    assert response.model == "fake"
    assert len(attempts) == 2


def test_tool_policy_validates_role_shape_path_and_order(tmp_path):
    tool = SimpleNamespace(
        name="repository_read",
        parameters=[SimpleNamespace(name="path", type="string", required=True, enum=None)],
    )
    policy = ToolPolicy(DEFAULT_ROLES[WorkerRole.EXPLORER], tmp_path, [tool])
    def call(name, args):
        return SimpleNamespace(name=name, arguments=args)
    assert policy.validate(call("repository_read", {"path": "app.py"})).allowed
    assert policy.validate(call("repository_read", {})).code == "INVALID_ARGUMENT"
    assert policy.validate(call("repository_read", {"path": "../escape.py"})).code == "PERMISSION_DENIED"
    assert policy.validate(call("bash", {"command": "pwd"})).code == "PERMISSION_DENIED"
    assert policy.effect("ckg") == ToolEffect.STATEFUL
    calls = [call(name, {}) for name in (
        "repository_read", "artifact_read", "bash", "repository_read",
    )]
    assert [len(batch) for batch in policy.partition(calls)] == [2, 1, 1]


def test_developer_write_scope_is_enforced_before_tool_use_and_at_completion(tmp_path):
    repo = _repository(tmp_path)
    edit = SimpleNamespace(
        name="str_replace_based_edit_tool",
        parameters=[SimpleNamespace(name="path", type="string", required=True, enum=None)],
    )
    policy = ToolPolicy(
        DEFAULT_ROLES[WorkerRole.DEVELOPER],
        repo,
        [edit],
        write_scope=("src/**",),
    )

    def call(path):
        return SimpleNamespace(
            name="str_replace_based_edit_tool", arguments={"path": path}
        )

    assert policy.validate(call("src/core.py")).allowed
    denied = policy.validate(call("tests/test_core.py"))
    assert not denied.allowed
    assert denied.rule_id == "TOOL-WRITE-SCOPE"

    gate = CompletionGate(
        role=WorkerRole.DEVELOPER,
        workspace=repo,
        metadata={"write_scope": ("app.py",)},
        test_commands=(),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    assert gate.evaluate("").accepted
    (repo / "outside.py").write_text("changed = True\n", encoding="utf-8")
    decision = gate.evaluate("")
    assert not decision.accepted
    assert "outside write scope" in decision.reasons[-1]


def test_completion_requires_current_workspace_test_evidence(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.TESTER, workspace=repo, metadata={"phase": "verify"},
        test_commands=("pytest -q",),
    )
    call = SimpleNamespace(name="bash", arguments={"command": "pytest -q"}, call_id="test-1")
    gate.record(call, SimpleNamespace(success=True))
    assert gate.evaluate("done").accepted

    (repo / "new.py").write_text("value = 2\n", encoding="utf-8")
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "new.cpython-312.pyc").write_bytes(b"generated")
    before = workspace_version(repo)
    patch = workspace_patch(repo)
    assert "new.py" in patch
    assert "__pycache__" not in patch
    assert ".pyc" not in patch
    assert workspace_version(repo) == before
    decision = gate.evaluate("done")
    assert not decision.accepted
    assert "stale" in decision.reasons[0]
    gate.record(SimpleNamespace(name="bash", arguments={"command": "pytest -q"},
                                call_id="test-2"), SimpleNamespace(success=True))
    assert gate.evaluate("done").accepted


def test_reproduction_requires_a_failing_test_not_an_arbitrary_command(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.TESTER, workspace=repo, metadata={"phase": "reproduce"},
        test_commands=("pytest -q",),
    )
    gate.record(
        SimpleNamespace(name="bash", arguments={"command": "ls -la"}, call_id="shell-1"),
        SimpleNamespace(success=False),
    )
    assert not gate.accepts_evidence_backed_completion("")

    gate.record(
        SimpleNamespace(name="bash", arguments={"command": "pytest -q"}, call_id="test-1"),
        SimpleNamespace(success=False),
    )
    assert gate.accepts_evidence_backed_completion("")


def test_explorer_can_complete_with_substantive_report_when_task_done_is_omitted(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.EXPLORER, workspace=repo, metadata={}, test_commands=(),
    )
    report = (
        "The root cause is verified in `app.py:1`: the stored value violates the "
        "documented invariant. The only caller reads that value directly, so the defect "
        "should be corrected at its source. No other writer was found in the inspected "
        "repository path; this is the complete read-only exploration report."
    )

    assert not gate.accepts_implicit_explorer_completion(report)
    gate.record(
        SimpleNamespace(name="repository_read", arguments={"path": "app.py"}, call_id="read-1"),
        SimpleNamespace(success=True),
    )
    assert not gate.accepts_implicit_explorer_completion("See app.py:1.")
    assert gate.accepts_implicit_explorer_completion(report)


def test_explorer_terminal_boundary_can_use_recorded_repository_evidence(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.EXPLORER, workspace=repo, metadata={}, test_commands=(),
    )
    gate.record(
        SimpleNamespace(name="repository_read", arguments={"path": "app.py"}, call_id="read-1"),
        SimpleNamespace(success=True),
    )

    assert not gate.accepts_implicit_explorer_completion("")
    assert gate.accepts_terminal_evidence_completion("")


def test_implicit_completion_remains_explorer_only(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.REVIEWER, workspace=repo, metadata={}, test_commands=(),
    )
    gate.record(
        SimpleNamespace(name="repository_read", arguments={"path": "app.py"}, call_id="read-1"),
        SimpleNamespace(success=True),
    )
    report = "`app.py:1` contains verified evidence. " * 20

    assert not gate.accepts_implicit_explorer_completion(report)


def test_single_agent_developer_completes_immediately_after_fresh_required_test(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.DEVELOPER,
        workspace=repo,
        metadata={"single_agent": True},
        test_commands=("pytest -q",),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    call = SimpleNamespace(
        name="bash", arguments={"command": "pytest -q"}, call_id="test-1"
    )
    gate.record(call, SimpleNamespace(success=True))

    assert gate.accepts_post_tool_completion()

    (repo / "notes.py").write_text("note = 'late edit'\n", encoding="utf-8")
    assert not gate.accepts_post_tool_completion()


def test_multi_component_developer_needs_explicit_completion_even_with_fresh_tests(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.DEVELOPER,
        workspace=repo,
        metadata={"single_agent": False},
        test_commands=("pytest -q",),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    gate.record(
        SimpleNamespace(name="bash", arguments={"command": "pytest -q"}, call_id="test-1"),
        SimpleNamespace(success=True),
    )

    assert not gate.accepts_post_tool_completion()
    assert gate.evaluate("implemented the component").accepted


def test_multi_component_gate_requires_all_named_deliverable_files(tmp_path):
    repo = _repository(tmp_path)
    (repo / "registry.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "registry.py")
    _git(repo, "commit", "-qm", "add registry")
    gate = CompletionGate(
        role=WorkerRole.DEVELOPER, workspace=repo,
        metadata={"required_paths": ["app.py", "registry.py"],
                  "write_scope": ("app.py", "registry.py")},
        test_commands=("pytest -q",),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    decision = gate.evaluate("done")
    assert not decision.accepted
    assert "registry.py" in decision.reasons[0]
    (repo / "registry.py").write_text("value = 2\n", encoding="utf-8")
    assert gate.evaluate("done").accepted


def test_security_gate_explains_invalid_confidence_instead_of_silent_discard(tmp_path):
    gate = CompletionGate(
        role=WorkerRole.SECURITY, workspace=_repository(tmp_path),
        metadata={"security_target_task_ids": ["develop-core"]}, test_commands=(),
    )
    decision = gate.evaluate(
        '{"decisions":[{"target_task_id":"develop-core",'
        '"verdict":"accept","confidence":"high"}]}'
    )
    assert not decision.accepted
    assert any("confidence must be a number" in reason for reason in decision.reasons)


def test_sole_work_unit_requires_passing_fresh_tests_before_completion(tmp_path):
    repo = _repository(tmp_path)
    gate = CompletionGate(
        role=WorkerRole.DEVELOPER,
        workspace=repo,
        metadata={"sole_work_unit": True},
        test_commands=("pytest -q",),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    assert not gate.accepts_post_tool_completion()
    gate.record(
        SimpleNamespace(name="bash", arguments={"command": "pytest -q"}, call_id="test-1"),
        SimpleNamespace(success=False),
    )
    assert not gate.accepts_post_tool_completion()
    gate.record(
        SimpleNamespace(name="bash", arguments={"command": "pytest -q"}, call_id="test-2"),
        SimpleNamespace(success=True),
    )
    assert gate.accepts_post_tool_completion()


def test_cooperative_role_gates_require_structured_outputs_and_all_component_patches(tmp_path):
    repo = _repository(tmp_path)
    security = CompletionGate(
        role=WorkerRole.SECURITY,
        workspace=repo,
        metadata={"security_target_task_ids": ["develop-core", "develop-tests"]},
        test_commands=(),
    )
    one_decision = (
        '{"decisions":[{"target_task_id":"develop-core","verdict":"accept"}]}'
    )
    assert not security.evaluate(one_decision).accepted
    all_decisions = (
        '{"decisions":['
        '{"target_task_id":"develop-core","verdict":"accept"},'
        '{"target_task_id":"develop-tests","verdict":"accept"}]}'
    )
    assert security.evaluate(all_decisions).accepted

    integrator = CompletionGate(
        role=WorkerRole.INTEGRATOR,
        workspace=repo,
        metadata={"component_task_ids": ["develop-core", "develop-tests"]},
        test_commands=(),
    )
    (repo / "app.py").write_text("value = 2\n", encoding="utf-8")
    integrator.record(
        SimpleNamespace(
            name="patch_apply", arguments={"task_id": "develop-core"}, call_id="apply-1"
        ),
        SimpleNamespace(success=True),
    )
    assert not integrator.evaluate("").accepted
    integrator.record(
        SimpleNamespace(
            name="patch_apply", arguments={"task_id": "develop-tests"}, call_id="apply-2"
        ),
        SimpleNamespace(success=True),
    )
    assert integrator.evaluate("").accepted


def test_security_non_accept_verdict_is_fail_closed():
    decisions = (
        ReviewDecision("develop-core", ReviewVerdict.ACCEPT),
        ReviewDecision(
            "develop-interface",
            ReviewVerdict.REPLAN,
            reasons=("permission broadening",),
        ),
    )

    assert security_rejection_reasons(decisions) == ("permission broadening",)
