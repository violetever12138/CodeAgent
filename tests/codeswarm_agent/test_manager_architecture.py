"""Manager contracts and a local Git/action-loop smoke; no external model calls."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from codeswarm_agent.integration_tool import PatchApplyTool
from codeswarm_agent.manager import ManagerError, acceptance_plan, required_commands
from codeswarm_agent.orchestrator import CodeSwarmOrchestrator
from codeswarm_agent.protocol import (
    Budget,
    ReviewDecision,
    ReviewVerdict,
    RunStatus,
    TaskRequest,
    Usage,
    VerificationRecord,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)
from codeswarm_agent.runtime.completion import changed_paths, workspace_patch, workspace_version
from codeswarm_agent.worker import TraeWorkerFactory
from trae_agent.utils.config import CodeSwarmConfig, Config

COMMAND = "python -m unittest -q"


def _saved_results(directory, channel="results"):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async def read():
        directory_path = Path(directory)
        async with AsyncSqliteSaver.from_conn_string(str(directory_path / "checkpoints.sqlite")) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": directory_path.name}})
            return saved.checkpoint["channel_values"][channel]

    return asyncio.run(read())


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "manager@example.invalid")
    _git(root, "config", "user.name", "Manager Test")
    (root / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "api.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (root / "test_core.py").write_text(
        "import unittest\nimport core\nclass TestCore(unittest.TestCase):\n"
        "    def test_value(self): self.assertEqual(core.VALUE, 2)\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    return root


def _plan(*, commands=None, verification="test"):
    return {"criteria": [{"id": "value", "behavior": "Return value 2", "basis": "user goal and test_core.py",
                          "required": True, "verification": verification,
                          "commands": [COMMAND] if commands is None else commands,
                          "assumptions": [], "scope": ["core.py"]}]}


def _task(path="core.py", *, mode="modify", goal="Set VALUE = 2"):
    return {"goal": goal, "mode": mode, "write_scope": [path], "criteria_ids": ["value"]}


def _action(kind, **kwargs):
    return {"action": kind, "reason": f"evidence calls for {kind}", **kwargs}


class ScriptedManager:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.states = []

    def __call__(self, prompt):
        self.states.append(json.loads(prompt.split("\nTask state:\n", 1)[1]))
        return json.dumps(next(self.decisions))


class LocalFactory:
    """Real patches/worktrees/tests; scripted model roles, optionally real services."""

    def __init__(self, *, stale=False, outside=False, fail_once=False, real_services=False,
                 create_parallelism=0, read_parallelism=0):
        self.created = []
        self.active = {"read": 0, "write": 0}
        self.max_active = {"read": 0, "write": 0}
        self.stale = stale
        self.outside = outside
        self.fail_once = fail_once
        self.writes = 0
        self.create_parallelism = create_parallelism
        self.create_waiters = 0
        self.create_barrier = None
        self.read_parallelism = read_parallelism
        self.read_barrier = None
        self.real_services = real_services
        self.config = Config.create(config_string="""
model_providers:
  local: {provider: openai, api_key: unused-test-key}
models:
  agent_model:
    model_provider: local
    model: test
    temperature: 0
    top_p: 1
    top_k: 0
    parallel_tool_calls: false
    max_retries: 0
agents:
  trae_agent: {model: agent_model, max_steps: 10, enable_lakeview: false}
""")

    def create(self, node, role):
        self.created.append(node.role)
        if self.real_services and node.role in {WorkerRole.TESTER, WorkerRole.INTEGRATOR}:
            from codeswarm_agent.safety.execution import WindowsGuardedBackend
            return TraeWorkerFactory(self.config, execution_backend=WindowsGuardedBackend()).create(node, role)
        owner = self

        class LocalWorker:
            worker_id = f"local-{node.id}"

            async def run(self, context):
                verifications, decisions, artifacts = (), (), []
                summary = "observed " + node.goal
                status = WorkerStatus.SUCCEEDED
                if node.role in {WorkerRole.MANAGER, WorkerRole.WORKER, WorkerRole.EXPLORER}:
                    label = "read" if node.metadata.get("task_mode") == "read" or node.role == WorkerRole.EXPLORER else "write"
                    owner.active[label] += 1
                    owner.max_active[label] = max(owner.max_active[label], owner.active[label])
                    if label == "read" and owner.read_parallelism:
                        if owner.read_barrier is None:
                            owner.read_barrier = asyncio.Event()
                        if owner.active["read"] == owner.read_parallelism:
                            owner.read_barrier.set()
                        await asyncio.wait_for(owner.read_barrier.wait(), 10)
                    if node.metadata.get("task_mode") == "create" and owner.create_parallelism:
                        if owner.create_barrier is None:
                            owner.create_barrier = asyncio.Event()
                        owner.create_waiters += 1
                        if owner.create_waiters == owner.create_parallelism:
                            owner.create_barrier.set()
                        await asyncio.wait_for(owner.create_barrier.wait(), 10)
                    await asyncio.sleep(0.015)
                    if label == "write":
                        owner.writes += 1
                        path = "outside.py" if owner.outside else node.write_scope[0].replace("/**", "/module.py")
                        target = context.workspace / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        value = 3 if owner.fail_once and owner.writes == 1 else 2
                        target.write_text(f"VALUE = {value}\n", encoding="utf-8")
                    owner.active[label] -= 1
                elif node.role == WorkerRole.INTEGRATOR:
                    allowed = {r.task_id: r.artifact_ids[0] for r in context.dependency_results}
                    tool = PatchApplyTool(context.workspace, context.artifacts, allowed)
                    for task_id in node.metadata["component_task_ids"]:
                        result = await tool.execute({"task_id": task_id})
                        if result.error:
                            summary = result.error
                            status = WorkerStatus.FAILED
                            break
                elif node.role == WorkerRole.TESTER:
                    integrated = next(r for r in context.dependency_results
                                      if r.task_id == node.metadata["integrated_task_id"])
                    sha = context.artifacts.get(integrated.artifact_ids[0]).sha256
                    executed = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q"],
                                              cwd=context.workspace, capture_output=True, timeout=20)
                    status = WorkerStatus.SUCCEEDED if executed.returncode == 0 else WorkerStatus.FAILED
                    verifications = tuple(VerificationRecord(
                        command=c, passed=executed.returncode == 0, exit_code=executed.returncode,
                        patch_sha256="stale" if owner.stale else sha,
                        workspace_version=workspace_version(context.workspace),
                    ) for c in context.request.test_commands)
                elif node.role == WorkerRole.REVIEWER:
                    decisions = (ReviewDecision(node.metadata["integration_task_id"], ReviewVerdict.ACCEPT,
                                                confidence=0.9),)
                patch = workspace_patch(context.workspace) if not role.read_only else ""
                if patch:
                    artifacts.append(context.artifacts.put_text("patch", patch, ".patch").id)
                return WorkerResult(
                    task_id=node.id, attempt=node.attempt, role=node.role, status=status,
                    summary=summary, artifact_ids=tuple(artifacts), changed_files=changed_paths(patch),
                    verifications=verifications, review_decisions=decisions,
                    metadata={**node.metadata, "workspace": str(context.workspace),
                              "workspace_version": workspace_version(context.workspace)},
                )

        return LocalWorker()


def _run(tmp_path, decisions, *, factory=None, issue="Return value 2", budget=None, **request_kwargs):
    root = _repo(tmp_path)
    manager = ScriptedManager(decisions)
    factory = factory or LocalFactory()
    runner = CodeSwarmOrchestrator(factory, keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = manager
    request = TaskRequest(issue, str(root), test_commands=(COMMAND,),
                          budget=budget or Budget(max_workers=40, max_retries_per_task=0), **request_kwargs)
    outcome = asyncio.run(runner.run(request))
    return outcome, factory, manager, runner


def _flow(first=None):
    return [first or _action("execute", acceptance_plan=_plan(), tasks=[_task()]),
            _action("integrate"), _action("verify"), _action("complete")]


def test_manager_v2_is_the_only_supported_architecture():
    assert CodeSwarmConfig().architecture_mode == "manager_v2"
    assert CodeSwarmOrchestrator(LocalFactory()).backend.architecture_mode == "manager_v2"
    with pytest.raises(ValueError, match="requires the manager_v2 architecture"):
        CodeSwarmOrchestrator(LocalFactory(), architecture_mode="retired")


def test_acceptance_revisions_preserve_user_conditions_and_tests(tmp_path):
    request = TaskRequest("Return value 2", str(tmp_path), acceptance_criteria=("Return value 2",),
                          test_commands=("pytest -q",))
    original = acceptance_plan(_plan(), request)
    revised = _plan(commands=[COMMAND, "pytest -q tests/test_api.py"])
    revised["revision_reason"] = "cover API regression"
    new = acceptance_plan(revised, request, original)
    assert new["version"] == 2
    assert required_commands(new, request) == ("pytest -q", COMMAND, "pytest -q tests/test_api.py")
    for field, value in [("required", False), ("behavior", "ignore failing behavior"),
                         ("commands", []), ("verification", "manual")]:
        bad = _plan()
        bad["criteria"][0][field] = value
        bad["revision_reason"] = "weaken it"
        with pytest.raises(ManagerError):
            acceptance_plan(bad, request, original)


def test_manager_executes_and_finishes_without_fixed_model_roles(tmp_path):
    outcome, factory, manager, _ = _run(tmp_path, _flow())
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.created == [WorkerRole.MANAGER, WorkerRole.INTEGRATOR, WorkerRole.TESTER]
    assert manager.states[-1]["manager"]["version"] != manager.states[0]["manager"]["version"]
    assert len(manager.states[-1]["manager"]["history"]) == 3
    assert (tmp_path / "repo" / "core.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("failure", ["stale", "outside", "manual", "empty"])
def test_completion_rejects_stale_scope_manual_and_empty_evidence(tmp_path, failure):
    plan = _plan(verification="manual" if failure == "manual" else "test",
                 commands=[] if failure == "empty" else None)
    flow = _flow(_action("execute", acceptance_plan=plan, tasks=[_task()]))
    flow.append(_action("blocked", reason="required evidence is unresolved"))
    outcome, _, manager, _ = _run(tmp_path, flow, factory=LocalFactory(
        stale=failure == "stale", outside=failure == "outside"))
    assert outcome.status != RunStatus.SUCCEEDED
    assert manager.states[-1]["manager"]["feedback"]


def test_reads_parallel_existing_writes_serial_and_full_patch_handoff(tmp_path):
    reads = [_task(mode="read", goal="inspect core.py"), _task(mode="read", goal="inspect api.py")]
    writes = [_task(), _task("api.py")]
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=reads),
            _action("delegate", tasks=writes), _action("integrate"),
            _action("verify"), _action("complete")]
    outcome, factory, _, _ = _run(tmp_path, flow, factory=LocalFactory(read_parallelism=2))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.max_active == {"read": 2, "write": 1}
    integrated = next((Path(outcome.run_directory) / "worktrees").glob("*-integrate"))
    assert (integrated / "core.py").read_text() == (integrated / "api.py").read_text() == "VALUE = 2\n"
    assert WorkerRole.AUDITOR not in factory.created and WorkerRole.REVIEWER not in factory.created


def test_failed_verification_drives_bounded_rework_preserving_state(tmp_path):
    flow = _flow()[:3] + [_action("execute", tasks=[_task(goal="repair the observed failing value")]),
                         _action("integrate"), _action("verify"), _action("complete")]
    outcome, factory, manager, _ = _run(tmp_path, flow, factory=LocalFactory(fail_once=True))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.writes == 2
    assert manager.states[3]["manager"]["feedback"] == ["fresh integrated verification failed"]
    assert manager.states[-1]["manager"]["reworks"] == 1


def test_risk_requires_independent_review_before_tests_and_after_verification(tmp_path):
    flow = _flow()[:3] + [_action("review"), _action("verify"), _action("complete"),
                         _action("review"), _action("complete")]
    outcome, factory, manager, _ = _run(tmp_path, flow, issue="Fix authentication: Return value 2")
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.created.count(WorkerRole.REVIEWER) == 2
    assert "before executing" in manager.states[3]["manager"]["feedback"][0]
    assert "independent review" in manager.states[6]["manager"]["feedback"][0]


def test_repeated_invalid_completion_stops_without_more_model_calls(tmp_path):
    flow = [_action("complete", acceptance_plan=_plan()), _action("complete"), _action("complete")]
    outcome, factory, manager, _ = _run(tmp_path, flow)
    assert outcome.status == RunStatus.FAILED
    assert not factory.created
    assert len(manager.states) == 3


def test_budget_reserves_closing_work_before_dispatch(tmp_path):
    flow = [_action("execute", acceptance_plan=_plan(), tasks=[_task()]),
            _action("blocked", reason="budget cannot support validated delivery")]
    outcome, factory, _, _ = _run(tmp_path, flow, budget=Budget(max_workers=5))
    assert outcome.status == RunStatus.FAILED
    assert not factory.created


def test_explicit_multi_request_cannot_be_labelled_with_one_worker(tmp_path):
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=[_task()]),
            _action("blocked", reason="provide actual bounded decomposition")]
    outcome, factory, manager, _ = _run(tmp_path, flow, force_multi_agent=True)
    assert outcome.status == RunStatus.FAILED
    assert not factory.created
    assert "two real bounded tasks" in manager.states[-1]["manager"]["feedback"][0]


def test_real_deterministic_services_do_not_call_model(tmp_path, monkeypatch):
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    async def no_model(*args, **kwargs):
        pytest.fail("deterministic integration and testing must not request a model")

    monkeypatch.setattr(LLMClient, "achat", no_model)
    outcome, _, _, _ = _run(tmp_path, _flow(), factory=LocalFactory(real_services=True))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason


def test_parallel_creation_uses_disjoint_new_directories_and_preserves_modules(tmp_path):
    tasks = [{**_task(path, mode="create"), "interface": "Export VALUE = 2 from module.py"}
             for path in ("feature_a/**", "feature_b/**")]
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=tasks), _action("integrate"),
            _action("execute", tasks=[_task()]), _action("integrate"),
            _action("verify"), _action("complete")]
    outcome, factory, _, _ = _run(tmp_path, flow, factory=LocalFactory(create_parallelism=2))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.max_active["write"] == 2
    integrated = Path(outcome.run_directory) / "worktrees" / outcome.selected_integrated_task_id
    assert (integrated / "feature_a/module.py").read_text() == "VALUE = 2\n"
    assert (integrated / "feature_b/module.py").read_text() == "VALUE = 2\n"


def test_checkpoint_replay_reuses_committed_action_after_crash(tmp_path, monkeypatch):
    import codeswarm_agent.manager_graph as module

    original = module._patch_preflight
    crash = [True]

    async def crash_once(*args):
        if crash.pop() if crash else False:
            raise RuntimeError("simulated crash after durable WorkerResult")
        return await original(*args)

    monkeypatch.setattr(module, "_patch_preflight", crash_once)
    root = _repo(tmp_path)
    factory = LocalFactory()
    manager = ScriptedManager(_flow())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = manager
    request = TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                          budget=Budget(max_workers=40))
    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(runner.run(request))
    directory = next((tmp_path / "runs").iterdir())
    outcome = asyncio.run(runner.resume(directory))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert factory.writes == 1
    assert len(manager.states) == 4


def test_unknown_manager_outcome_is_not_automatically_resent(tmp_path):
    root = _repo(tmp_path)
    runner = CodeSwarmOrchestrator(LocalFactory(), architecture_mode="manager_v2",
                                   run_root=tmp_path / "runs")
    calls = []

    def unavailable(prompt):
        calls.append(prompt)
        raise ConnectionError("response lost")

    runner.backend.route_model = unavailable
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root))))
    assert outcome.status == RunStatus.FAILED
    assert "outcome unknown" in outcome.failure_reason
    assert len(calls) == 1


def test_real_manager_execution_uses_same_tools_and_persistent_plan(tmp_path, monkeypatch):
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    observed = []

    async def scripted_edit(self, messages, *args, **kwargs):
        observed.append(messages)
        return LLMResponse(content="Implement current Manager action", model="test",
                           usage=LLMUsage(10, 10), tool_calls=[ToolCall(
                               name="str_replace_based_edit_tool", call_id="manager-edit",
                               arguments={"command": "str_replace", "path": "core.py",
                                          "old_str": "VALUE = 1", "new_str": "VALUE = 2"})])

    monkeypatch.setattr(LLMClient, "achat", scripted_edit)
    root = _repo(tmp_path)
    config = LocalFactory().config
    factory = TraeWorkerFactory(config, execution_backend=WindowsGuardedBackend())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager(_flow())
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(observed) == 1  # Integration and tests made no model requests.
    assert "Persistent Manager task state" in observed[0][0].content
    assert "Return value 2" in observed[0][0].content


def test_simple_rule_skips_initial_assignment_model_but_runs_all_gates(tmp_path):
    outcome, factory, manager, _ = _run(tmp_path, [_action("integrate"), _action("verify"),
                                                 _action("complete")], issue="Fix core.py to return value 2")
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(manager.states) == 3
    assert manager.states[0]["manager"]["initial_route"] == "rule-single"
    assert factory.created == [WorkerRole.MANAGER, WorkerRole.INTEGRATOR, WorkerRole.TESTER]


@pytest.mark.parametrize("change", ["cross_file", "import", "no_tests", "forced_multi", "risk"])
def test_initial_rule_uncertainty_never_forces_multiple_workers(tmp_path, change):
    from dataclasses import replace

    from codeswarm_agent.routing import single_agent_fast_path

    root = _repo(tmp_path)
    request = TaskRequest("Fix core.py to return value 2", str(root), test_commands=(COMMAND,))
    if change == "cross_file":
        request = replace(request, issue="Fix core.py and api.py together")
    elif change == "import":
        (root / "core.py").write_text("import api\nVALUE = 1\n", encoding="utf-8")
    elif change == "no_tests":
        request = replace(request, test_commands=())
    elif change == "forced_multi":
        request = replace(request, force_multi_agent=True)
    else:
        request = replace(request, issue="Fix authentication in core.py")
    assert single_agent_fast_path(request) is None


def test_probe_is_bounded_and_ignores_hidden_and_dependency_trees(tmp_path):
    from codeswarm_agent.routing import initial_probe

    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret/key.py").write_text("DO NOT READ", encoding="utf-8")
    for i in range(250):
        (tmp_path / f"module_{i:03}.py").write_text("x = 1\n" * 1000, encoding="utf-8")
    probe = initial_probe(str(tmp_path), "module_002.py", seconds=2)
    assert probe["inspected"] <= 200 and len(probe["files"]) <= 80
    assert len(probe["evidence"]) <= 12_000
    assert "DO NOT READ" not in probe["evidence"]
    assert "file:module_002.py" in probe["evidence"]


def test_same_gap_deduplicates_by_input_version_across_worker_identities():
    from codeswarm_agent.routing import RoutingControl

    control = RoutingControl()
    assert control.request("worker-a", "v1", "new_dependency", "shared API discovered", ["file:api.py"])
    duplicate = RoutingControl(seen=control.seen.copy())
    assert not duplicate.request("worker-b", "v1", "new_dependency", "shared API discovered", ["file:api.py"])
    assert not duplicate.pause_requested
    assert duplicate.request("worker-b", "v2", "new_dependency", "shared API discovered", ["file:api.py"])


def test_context_pressure_cannot_bypass_summary_prerequisite():
    from codeswarm_agent.routing import RoutingControl, RoutingRequestTool

    control = RoutingControl()
    result = asyncio.run(RoutingRequestTool(control, "worker", "v1").execute({
        "kind": "context_isolation", "reason": "context pressure", "evidence": ["context:75%"],
    }))
    assert result.error and "summary first" in result.error
    assert not control.pause_requested


def test_equivalent_assignment_does_not_restart_same_investigation(tmp_path):
    reads = [_task(mode="read", goal="inspect the core value")]
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=reads),
            _action("delegate", tasks=reads), _action("blocked", reason="duplicate work suppressed")]
    outcome, factory, manager, _ = _run(tmp_path, flow)
    assert outcome.status == RunStatus.FAILED
    assert factory.created.count(WorkerRole.EXPLORER) == 1
    assert "equivalent assignment" in manager.states[-1]["manager"]["feedback"][0]


def test_real_tool_boundary_handoff_preserves_patch_and_never_starts_queued_writer(tmp_path, monkeypatch):
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.tools.edit_tool import TextEditorTool
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    edits = []
    model_calls = []
    original = TextEditorTool.execute

    async def slow_edit(self, arguments):
        edits.append(("started", Path(arguments["path"]).name))
        await asyncio.sleep(0.02)
        result = await original(self, arguments)
        edits.append(("finished", Path(arguments["path"]).name))
        return result

    async def scripted_model(self, messages, *args, **kwargs):
        model_calls.append(messages)
        first = len(model_calls) == 1
        path = "core.py" if first else "api.py"
        calls = [ToolCall("str_replace_based_edit_tool", f"edit-{path}",
                          {"command": "str_replace", "path": path,
                           "old_str": "VALUE = 1", "new_str": "VALUE = 2"})]
        if first:
            calls.append(ToolCall("routing_request", "handoff", {
                "kind": "new_dependency", "reason": "core patch is ready, coordinate the remaining API task",
                "evidence": ["file:api.py"],
            }))
        return LLMResponse(content=f"Observed {path} change", model="test", usage=LLMUsage(20, 10),
                           tool_calls=calls)

    monkeypatch.setattr(TextEditorTool, "execute", slow_edit)
    monkeypatch.setattr(LLMClient, "achat", scripted_model)
    root = _repo(tmp_path)
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=[_task(), _task("api.py")]),
            _action("integrate"), _action("execute", tasks=[_task("api.py")],
                                         handoff_reason="reuse the completed core patch and finish API myself"),
            _action("integrate"), _action("verify"), _action("complete")]
    manager = ScriptedManager(flow)
    factory = TraeWorkerFactory(LocalFactory().config, execution_backend=WindowsGuardedBackend())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = manager
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40, max_retries_per_task=1))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(model_calls) == 2
    assert outcome.usage.model_calls == 8  # Six Manager decisions plus two actual tool-loop calls.
    assert edits == [("started", "core.py"), ("finished", "core.py"),
                     ("started", "api.py"), ("finished", "api.py")]
    handoff = manager.states[1]["manager"]
    assert len(handoff["remaining_tasks"]) == 2
    assert sorted(w["status"] for w in handoff["workers"].values()) == ["not_started", "yielded"]
    assert handoff["pending"] and handoff["reworks"] == 0
    assert manager.states[2]["manager"]["delivery"]
    assert outcome.mode == "single-agent"
    assert not (Path(outcome.run_directory) / "cancel.json").exists()
    raw = list(_saved_results(outcome.run_directory).values())
    assert not (Path(outcome.run_directory) / "blackboard.jsonl").exists()
    assert len([r for r in raw if r["role"] == "worker"]) == 1
    assert next(r for r in raw if r["role"] == "worker")["status"] == "blocked"


def test_parallel_readers_return_findings_and_queued_read_is_cancelled(tmp_path, monkeypatch):
    from codeswarm_agent.repository_tool import RepositoryReadTool
    from codeswarm_agent.routing import RoutingRequestTool
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    both_reading, release_peer = asyncio.Event(), asyncio.Event()
    started, finished, model_calls = [], [], []
    original_read, original_request = RepositoryReadTool.execute, RoutingRequestTool.execute

    async def wait_for_peer(self, arguments):
        path = arguments["path"]
        started.append(path)
        if len(started) == 2:
            both_reading.set()
        await asyncio.wait_for(both_reading.wait(), 5)
        if path == "api.py":
            await asyncio.wait_for(release_peer.wait(), 5)
        result = await original_read(self, arguments)
        finished.append(path)
        return result

    async def release_after_request(self, arguments):
        result = await original_request(self, arguments)
        release_peer.set()
        return result

    async def model(self, messages, *args, **kwargs):
        model_calls.append(messages)
        manager = "You are the Manager executing your own current action." in messages[0].content
        if manager:
            assert sorted(finished) == ["api.py", "core.py"]
            calls = [ToolCall("str_replace_based_edit_tool", "edit", {
                "command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})]
        else:
            path = "core.py" if "inspect core.py" in messages[0].content else "api.py"
            calls = [ToolCall("repository_read", f"read-{path}", {"operation": "read", "path": path})]
            if path == "core.py":
                calls.append(ToolCall("routing_request", "request", {
                    "kind": "remaining_shrank", "reason": "one cohesive Manager patch is sufficient",
                    "evidence": ["file:core.py"],
                }))
        return LLMResponse(content="partial investigation", model="test", usage=LLMUsage(20, 10), tool_calls=calls)

    monkeypatch.setattr(RepositoryReadTool, "execute", wait_for_peer)
    monkeypatch.setattr(RoutingRequestTool, "execute", release_after_request)
    monkeypatch.setattr(LLMClient, "achat", model)
    root = _repo(tmp_path)
    reads = [_task(mode="read", goal=f"inspect {path}") for path in ("core.py", "api.py", "test_core.py")]
    flow = [_action("delegate", acceptance_plan=_plan(), tasks=reads),
            _action("consolidate", reason="remaining investigations are obsolete; keep the two findings"),
            _action("execute", tasks=[_task()]), _action("integrate"), _action("verify"), _action("complete")]
    manager = ScriptedManager(flow)
    factory = TraeWorkerFactory(LocalFactory().config, execution_backend=WindowsGuardedBackend())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = manager
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40, max_parallel=2))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(model_calls) == 3
    assert sorted(started) == ["api.py", "core.py"]
    handoff = manager.states[1]["manager"]
    assert sorted(w["status"] for w in handoff["workers"].values()) == ["not_started", "yielded", "yielded"]
    assert len(handoff["remaining_tasks"]) == 3
    assert not manager.states[2]["manager"]["remaining_tasks"]
    assert manager.states[2]["manager"]["no_progress"] == 0


def test_failed_attempt_history_is_in_state_and_counted_once(tmp_path):
    from dataclasses import replace

    class RetryFactory(LocalFactory):
        def create(self, node, role):
            worker = super().create(node, role)
            run = worker.run

            async def attempt(context):
                result = await run(context)
                if node.role == WorkerRole.MANAGER:
                    return replace(result, status=WorkerStatus.FAILED if node.attempt == 0 else WorkerStatus.SUCCEEDED,
                                   usage=Usage(input_tokens=7, output_tokens=3, model_calls=1))
                return result

            worker.run = attempt
            return worker

    outcome, factory, _, _ = _run(tmp_path, _flow(), factory=RetryFactory(),
                                  budget=Budget(max_workers=40, max_retries_per_task=1))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    records = [r for r in _saved_results(outcome.run_directory).values() if r["role"] == "manager"]
    assert sorted((r["attempt"], r["status"]) for r in records) == [(0, "failed"), (1, "succeeded")]
    assert outcome.usage.input_tokens == 14 and outcome.usage.output_tokens == 6
    assert factory.writes == 2


def test_unconfirmed_task_reexecutes_in_clean_worktree(tmp_path, monkeypatch):
    from codeswarm_agent.langgraph_backend import WorkerNodeAdapter

    before = []

    class ObservedFactory(LocalFactory):
        def create(self, node, role):
            worker = super().create(node, role)
            run = worker.run

            async def observe(context):
                if node.role == WorkerRole.MANAGER:
                    before.append((context.workspace / "core.py").read_text())
                return await run(context)

            worker.run = observe
            return worker

    original = WorkerNodeAdapter._durable_attempt
    fail = [True]

    async def before_task_return(self, node):
        raw = await original(self, node)
        if fail.pop() if fail else False:
            raise RuntimeError("task output not yet committed")
        return raw

    monkeypatch.setattr(WorkerNodeAdapter, "_durable_attempt", before_task_return)
    root = _repo(tmp_path)
    factory = ObservedFactory()
    manager = ScriptedManager(_flow())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = manager
    with pytest.raises(RuntimeError, match="not yet committed"):
        asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                           budget=Budget(max_workers=40))))
    directory = next((tmp_path / "runs").iterdir())
    assert not _saved_results(directory)
    outcome = asyncio.run(runner.resume(directory))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert before == ["VALUE = 1\n", "VALUE = 1\n"]
    assert factory.writes == 2


@pytest.mark.parametrize("mode", ["normal", "incomplete", "replay"])
def test_actual_messages_consume_recall_once_and_restore_without_model_reissue(tmp_path, monkeypatch, mode):
    from codeswarm_agent.langgraph_backend import WorkerNodeAdapter
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    root = _repo(tmp_path)
    (root / "core.py").write_text("VALUE = 1\n# CORE_RECALL_MARKER\n", encoding="utf-8")
    (root / "api.py").write_text("VALUE = 1\n# API_RECALL_MARKER\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "recall fixtures")
    seen = []

    async def model(self, messages, *args, **kwargs):
        seen.append(messages)
        number = len(seen)
        if number == 2:
            body = "\n".join(m.tool_result.result or "" for m in messages if m.tool_result)
            assert "CORE_RECALL_MARKER" in body
        if number == 3:
            body = "\n".join(m.tool_result.result or "" for m in messages if m.tool_result)
            assert ("CORE_RECALL_MARKER" in body) == (mode == "incomplete")
            assert "API_RECALL_MARKER" in body
        if number <= 2:
            path = "core.py" if number == 1 else "api.py"
            calls = [ToolCall("repository_read", f"read-{number}", {"operation": "read", "path": path})]
        else:
            calls = [ToolCall("str_replace_based_edit_tool", "edit", {
                "command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})]
        return LLMResponse(content="Continue from current findings", model="test", usage=LLMUsage(10, 5),
                           finish_reason="length" if mode == "incomplete" and number == 2 else "stop", tool_calls=calls)

    monkeypatch.setattr(LLMClient, "achat", model)
    original = WorkerNodeAdapter._durable_attempt
    fail = [mode == "replay"]

    async def after_turns(self, node):
        raw = await original(self, node)
        if fail.pop() if fail else False:
            raise RuntimeError("crash after durable model turns")
        return raw

    monkeypatch.setattr(WorkerNodeAdapter, "_durable_attempt", after_turns)
    runner = CodeSwarmOrchestrator(
        TraeWorkerFactory(LocalFactory().config, execution_backend=WindowsGuardedBackend()),
        architecture_mode="manager_v2", keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager(_flow())
    request = TaskRequest("Return value 2", str(root), test_commands=(COMMAND,), budget=Budget(max_workers=40))
    if mode == "replay":
        with pytest.raises(RuntimeError, match="durable model turns"):
            asyncio.run(runner.run(request))
        outcome = asyncio.run(runner.resume(next((tmp_path / "runs").iterdir())))
    else:
        outcome = asyncio.run(runner.run(request))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(seen) == 3 and outcome.usage.model_calls == 7
    contexts = _saved_results(outcome.run_directory, "contexts")
    worker = contexts["manager-1-task-0@0"]
    assert set(worker["consumed_calls"]) == {"read-1", "read-2"}
    assert len(worker["interaction_refs"]) == 3
    assert "manager" in contexts
    assert all("_context" not in record["metadata"] for record in _saved_results(outcome.run_directory).values())


def test_real_query_tool_can_recall_consumed_source_again_without_automatic_search(tmp_path, monkeypatch):
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    root = _repo(tmp_path)
    (root / "core.py").write_text("VALUE = 1\n# QUERY_RECALL_MARKER\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "query fixture")
    seen = []

    async def model(self, messages, *args, **kwargs):
        seen.append(messages)
        number = len(seen)
        results = {m.tool_result.call_id: m.tool_result.result for m in messages if m.tool_result}
        if number == 2:
            assert "QUERY_RECALL_MARKER" in results["query-1"]
        if number == 3:
            assert "QUERY_RECALL_MARKER" not in results["query-1"]
            assert "QUERY_RECALL_MARKER" in results["query-2"]
        calls = ([ToolCall("repository_read", f"query-{number}", {"operation": "query", "pattern": "core.py"})]
                 if number < 3 else [ToolCall("str_replace_based_edit_tool", "edit", {
                     "command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})])
        return LLMResponse("Continue from current evidence", model="test", usage=LLMUsage(10, 5),
                           finish_reason="stop", tool_calls=calls)

    monkeypatch.setattr(LLMClient, "achat", model)
    runner = CodeSwarmOrchestrator(
        TraeWorkerFactory(LocalFactory().config, execution_backend=WindowsGuardedBackend()),
        architecture_mode="manager_v2", keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager(_flow())
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(seen) == 3 and outcome.usage.model_calls == 7
    logs = list(Path(outcome.run_directory).rglob("retrieval_queries.jsonl"))
    queries = [json.loads(line) for path in logs for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["trigger"] for item in queries].count("model-query") == 2
    assert len(queries) == 3


@pytest.mark.parametrize("authorization", ["allowed", "global-off", "role-off"])
def test_real_mcp_read_service_obeys_authorization_and_consumes_bodies(tmp_path, monkeypatch, authorization):
    import re

    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.config import MCPServerConfig
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient
    from trae_agent.utils.mcp_client import MCPClient

    root = _repo(tmp_path)
    (root / "core.py").write_text("VALUE = 1\n# MCP_RECALL_MARKER\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "MCP fixture")
    config = LocalFactory().config
    config.trae_agent.mcp_servers_config = {"local": MCPServerConfig(
        command="${CODESWARM_PYTHON}", args=["-m", "codeswarm_agent.mcp_server"],
        cwd="${CODESWARM_PACKAGE_ROOT}", timeout=20)}
    config.trae_agent.allow_mcp_servers = [] if authorization == "global-off" else ["local"]
    config.codeswarm.mcp_role_allowlist = {} if authorization == "role-off" else {"manager": ["local"]}
    config.codeswarm.mcp_tool_effects = {"local.repository_read": "read", "local.artifact_read": "read"}
    clients, seen = [], []
    original = MCPClient.connect_and_discover

    async def discover(self, *args, **kwargs):
        clients.append(self)
        return await original(self, *args, **kwargs)

    async def model(self, messages, model_config, tools, **kwargs):
        seen.append(messages)
        available = {tool.name for tool in tools}
        assert ("mcp__local__repository_read" in available) == (authorization == "allowed")
        number = len(seen)
        results = {message.tool_result.call_id: message.tool_result.result for message in messages if message.tool_result}
        if number == 1:
            calls = [ToolCall("mcp__local__repository_read" if authorization == "allowed" else "repository_read",
                              "read-source", {"operation": "read", "path": "core.py"})]
        elif number == 2 and authorization == "allowed":
            assert "MCP_RECALL_MARKER" in results["read-source"]
            reference = re.search(r"artifact=([^ ]+)", results["read-source"]).group(1)
            calls = [ToolCall("mcp__local__artifact_read", "read-artifact", {
                "artifact_id": reference, "start_line": 1, "end_line": 10})]
        else:
            if authorization == "allowed":
                assert "MCP_RECALL_MARKER" not in results["read-source"]
                assert "MCP_RECALL_MARKER" in results["read-artifact"]
            calls = [ToolCall("str_replace_based_edit_tool", "edit", {
                "command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})]
        return LLMResponse("Continue from evidence", model="test", usage=LLMUsage(10, 5), finish_reason="stop", tool_calls=calls)

    monkeypatch.setattr(MCPClient, "connect_and_discover", discover)
    monkeypatch.setattr(LLMClient, "achat", model)
    runner = CodeSwarmOrchestrator(TraeWorkerFactory(config, execution_backend=WindowsGuardedBackend()),
                                   architecture_mode="manager_v2", keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager(_flow())
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,), budget=Budget(max_workers=40))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(clients) == (1 if authorization == "allowed" else 0)
    assert all(not client.sessions and not client._owners for client in clients)
    if authorization == "allowed":
        worker = _saved_results(outcome.run_directory, "contexts")["manager-1-task-0@0"]
        assert set(worker["consumed_calls"]) == {"read-source", "read-artifact"}


@pytest.mark.parametrize("replay", [False, True])
def test_real_sdk_spans_cover_mcp_tools_and_cached_replay(tmp_path, monkeypatch, replay):
    import httpx
    import openai
    import requests
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    from codeswarm_agent.langgraph_backend import WorkerNodeAdapter
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.utils.config import MCPServerConfig

    root = _repo(tmp_path)
    calls = []
    exports = []

    def receive(transport, *, method, url, data, **kwargs):
        assert method == "POST"
        assert url == "https://api.smith.langchain.com/otel/v1/traces"
        exports.append(ExportTraceServiceRequest.FromString(data))
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        return response

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.setattr(requests.Session, "request", receive)

    async def http(request):
        calls.append(json.loads(request.content))
        number = len(calls)
        assert number <= 2, "cached model responses must not be sent again"
        name = "mcp__local__repository_read" if number == 1 else "str_replace_based_edit_tool"
        arguments = ({"operation": "read", "path": "core.py"} if number == 1 else
                     {"command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})
        return httpx.Response(200, json={"id": f"response-{number}", "object": "response", "created_at": 1,
            "model": "test", "status": "completed", "output": [{"type": "function_call", "id": f"function-{number}",
                "call_id": f"actual-tool-{number}", "name": name, "arguments": json.dumps(arguments), "status": "completed"}],
            "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})

    original_async = openai.AsyncOpenAI
    def sdk(**kwargs):
        return original_async(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(http)))
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk)
    original = WorkerNodeAdapter._durable_attempt
    crash = [replay]
    async def interrupt(self, node):
        raw = await original(self, node)
        if crash[0]:
            crash[0] = False
            raise RuntimeError("interrupt after real SDK responses")
        return raw
    monkeypatch.setattr(WorkerNodeAdapter, "_durable_attempt", interrupt)
    config = LocalFactory().config
    config.trae_agent.allow_mcp_servers = ["local"]
    config.trae_agent.mcp_servers_config = {"local": MCPServerConfig(command="${CODESWARM_PYTHON}",
        args=["-m", "codeswarm_agent.mcp_server"], cwd="${CODESWARM_PACKAGE_ROOT}", timeout=20)}
    config.codeswarm.mcp_role_allowlist = {"manager": ["local"]}
    config.codeswarm.mcp_tool_effects = {"local.repository_read": "read", "local.artifact_read": "read"}
    runner = CodeSwarmOrchestrator(TraeWorkerFactory(config, execution_backend=WindowsGuardedBackend()),
        architecture_mode="manager_v2", keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager(_flow())
    request = TaskRequest("Return value 2", str(root), test_commands=(COMMAND,), budget=Budget(max_workers=40))
    if replay:
        with pytest.raises(RuntimeError, match="real SDK"):
            asyncio.run(runner.run(request))
        outcome = asyncio.run(runner.resume(next((tmp_path / "runs").iterdir())))
    else:
        outcome = asyncio.run(runner.run(request))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    records = [json.loads(line) for file in (Path(outcome.run_directory) / "observability").glob("spans-*.jsonl")
               for line in file.read_text(encoding="utf-8").splitlines()]
    llm = [item for item in records if item["attributes"].get("openinference.span.kind") == "LLM"]
    assert len(llm) == len(calls) == 2
    assert len({item["context"]["trace_id"] for item in records}) == (2 if replay else 1)
    assert any(item["attributes"].get("tool.name") == "mcp__local__repository_read" for item in records)
    assert any(item["attributes"].get("codeswarm.origin_model_span_id") for item in records)
    assert any(item["attributes"].get("openinference.span.kind") == "RETRIEVER" for item in records)
    exported_spans = [item for batch in exports for resource in batch.resource_spans
                      for scope in resource.scope_spans for item in scope.spans]
    assert {item.span_id.hex() for item in exported_spans} == {
        item["context"]["span_id"].removeprefix("0x") for item in records}


@pytest.mark.parametrize("fail_first_summary", [False, True])
def test_manager_history_summary_is_budgeted_and_persisted(tmp_path, fail_first_summary):
    from codeswarm_agent.context import ContextSettings

    root = _repo(tmp_path)
    normal, summaries = [], []
    decisions = [_action("execute", acceptance_plan=_plan(), tasks=[_task(mode="read", goal="inspect part 0")],
                         reason="Investigate independent question 0 " + "x" * 1500)]
    decisions += [_action("execute", tasks=[_task(mode="read", goal=f"inspect part {i}")],
                          reason=f"Investigate independent question {i} " + "x" * 1500) for i in range(1, 4)]
    decisions += [_action("execute", tasks=[_task()]), _action("integrate"), _action("verify"), _action("complete")]
    queue = iter(decisions)

    def model(prompt):
        if prompt.startswith("[Progress summary]"):
            summaries.append(prompt)
            if fail_first_summary and len(summaries) == 1:
                raise ValueError("injected summary provider failure")
            return json.dumps({"problem": "Return value 2", "findings": ["Prior read actions completed"],
                               "decisions": ["Use one Manager writer"], "excluded": [], "changes": [],
                               "unfinished": ["Implement and verify"], "assumptions": [],
                               "next_steps": ["Inspect latest state"], "evidence_refs": []}), Usage(input_tokens=3, output_tokens=2, model_calls=1)
        state = json.loads(prompt.split("\nTask state:\n", 1)[1])
        normal.append(state)
        return json.dumps(next(queue)), Usage(input_tokens=7, output_tokens=3, model_calls=1)

    factory = LocalFactory()
    factory.context_settings = ContextSettings(context_window_tokens=16000, output_reserve_tokens=1024,
                                              safety_margin_tokens=1024, fixed_overhead_tokens=128, summary_max_tokens=500)
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    runner.backend.route_model = model
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=60))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert summaries and len(normal) == 8
    context = _saved_results(outcome.run_directory, "contexts")["manager"]
    assert context["summary_version"] >= 1 and context["covered_through"] > 0
    assert outcome.usage.model_calls == len(normal) + len(summaries)
    assert any("progress_summary" in state for state in normal)
    if fail_first_summary:
        assert summaries[0] == summaries[1]


@pytest.mark.parametrize("summary_mode", ["good", "invalid"])
def test_worker_summary_counts_usage_and_enables_context_handoff_only_after_summary(tmp_path, monkeypatch, summary_mode):
    from codeswarm_agent.context import ContextSettings
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    root = _repo(tmp_path)
    for name in ("extra0.py", "extra1.py"):
        (root / name).write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "summary input fixtures")
    normal, summaries = [], []

    async def model(self, messages, model_config, tools, **kwargs):
        if messages[0].content.startswith("Return a concise structured progress summary"):
            summaries.append(messages)
            content = json.dumps({
                "problem": "Return value 2", "findings": ["Observed source files; no current tests yet"],
                "decisions": ["Fix core"], "excluded": [], "changes": [], "unfinished": ["Implement and verify"],
                "assumptions": ["Unverified API hypothesis"], "next_steps": ["Edit core"], "evidence_refs": ["file:core.py"],
            }) if summary_mode == "good" else "{}"
            return LLMResponse(content=content, model="test", usage=LLMUsage(20, 10), finish_reason="stop")
        normal.append(messages)
        number = len(normal)
        if number <= 4:
            path = ("core.py", "api.py", "extra0.py", "extra1.py")[number - 1]
            calls = [ToolCall("repository_read", f"read-{number}", {"operation": "read", "path": path})]
            content = f"Unverified progress hypothesis {number}: " + "x" * 3200
        else:
            assert summaries
            routing = next(tool for tool in tools if tool.name == "routing_request")
            assert routing.summary_completed == (summary_mode == "good")
            calls = [ToolCall("str_replace_based_edit_tool", "edit", {
                "command": "str_replace", "path": "core.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"})]
            if summary_mode == "good":
                calls.append(ToolCall("routing_request", "context-handoff", {
                    "kind": "context_isolation", "reason": "After summary, evaluate whether API context still needs isolation",
                    "evidence": ["summary:1", "file:api.py"],
                }))
            content = "Core change is ready; evaluate remaining context need"
        return LLMResponse(content=content, model="test", usage=LLMUsage(10, 5), finish_reason="stop", tool_calls=calls)

    monkeypatch.setattr(LLMClient, "achat", model)
    factory = TraeWorkerFactory(LocalFactory().config, context_settings=ContextSettings(summary_max_tokens=700),
                                execution_backend=WindowsGuardedBackend())
    runner = CodeSwarmOrchestrator(factory, architecture_mode="manager_v2", keep_workspaces=True,
                                   run_root=tmp_path / "runs")
    flow = [_flow()[0], _action("integrate")]
    if summary_mode == "good":
        flow.append(_action("consolidate", reason="Core behavior is implemented; independent API work is unnecessary"))
    flow += [_action("verify"), _action("complete")]
    runner.backend.route_model = ScriptedManager(flow)
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40))))
    assert outcome.status == RunStatus.SUCCEEDED, outcome.failure_reason
    assert len(normal) == 5 and summaries
    assert outcome.usage.model_calls == len(flow) + 5 + len(summaries)
    assert outcome.usage.input_tokens == 50 + 20 * len(summaries)
    context = _saved_results(outcome.run_directory, "contexts")["manager-1-task-0@0"]
    if summary_mode == "good":
        assert context["summary_version"] >= 1 and context["covered_through"] > 0
        assert context["summary"]["assumptions"] == ["Unverified API hypothesis"]
        assert context["working_state"]["changes"]
    else:
        assert len(summaries) >= 2 and len(summaries) % 2 == 0
        assert all(summaries[i][1].content == summaries[i + 1][1].content for i in range(0, len(summaries), 2))
        assert context["summary"] is None and context["covered_through"] == 0


def test_failed_normal_request_leaves_tool_body_unconsumed(tmp_path, monkeypatch):
    from codeswarm_agent.safety.execution import WindowsGuardedBackend
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    calls = []

    async def model(self, messages, *args, **kwargs):
        calls.append(messages)
        if len(calls) == 2:
            assert any(m.tool_result and "VALUE = 1" in (m.tool_result.result or "") for m in messages)
            raise ValueError("provider rejected request")
        return LLMResponse(content="Inspect source", model="test", usage=LLMUsage(10, 5), finish_reason="stop",
                           tool_calls=[ToolCall("repository_read", "pending-read", {"operation": "read", "path": "core.py"})])

    monkeypatch.setattr(LLMClient, "achat", model)
    root = _repo(tmp_path)
    runner = CodeSwarmOrchestrator(TraeWorkerFactory(LocalFactory().config, execution_backend=WindowsGuardedBackend()),
                                   architecture_mode="manager_v2", keep_workspaces=True, run_root=tmp_path / "runs")
    runner.backend.route_model = ScriptedManager([_flow()[0], _action("blocked", reason="provider unavailable")])
    outcome = asyncio.run(runner.run(TaskRequest("Return value 2", str(root), test_commands=(COMMAND,),
                                                 budget=Budget(max_workers=40, max_retries_per_task=0))))
    assert outcome.status != RunStatus.SUCCEEDED and len(calls) == 2
    context = _saved_results(outcome.run_directory, "contexts")["manager-1-task-0@0"]
    assert "pending-read" in context["read_calls"] and not context["consumed_calls"]
