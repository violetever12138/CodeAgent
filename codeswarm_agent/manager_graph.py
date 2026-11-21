"""Manager-led, checkpointed action loop; deterministic services retain authority."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from langgraph.graph import END, START, StateGraph

from codeswarm_agent.context import ContextSettings
from codeswarm_agent.context.manager_context import prepare_manager_context
from codeswarm_agent.langgraph_backend import CodeSwarmState, RunContext, WorkerNodeAdapter
from codeswarm_agent.manager import (
    MANAGER_INSTRUCTIONS,
    ManagerDecision,
    ManagerError,
    parse_decision,
    required_commands,
    validate_tasks,
)
from codeswarm_agent.protocol import (
    ReviewVerdict,
    RunStatus,
    TaskNode,
    Usage,
    WorkerRole,
    WorkerStatus,
    to_jsonable,
)
from codeswarm_agent.result_state import StateResults
from codeswarm_agent.review import FinalResultGate
from codeswarm_agent.routing import (
    RoutingControl,
    assignment_signature,
    initial_probe,
    sensitive_issue,
    single_agent_fast_path,
)
from codeswarm_agent.runtime.completion import changed_paths, workspace_version
from codeswarm_agent.runtime.state import RuntimeBudgetExceeded
from codeswarm_agent.runtime.tool_policy import within_write_scope


class ManagerCallUnknown(ManagerError):
    """A request may have reached the provider; do not automatically resend it."""


async def _model_text(runtime: RunContext, prompt: str, route_model=None, *,
                      system_prompt="You are a precise repository task Manager. Output JSON only.",
                      with_completion=False, max_output_tokens=None):
    if route_model is not None:
        value = route_model(prompt)
        if asyncio.iscoroutine(value):
            value = await value
        if isinstance(value, tuple):
            result = (str(value[0]), value[1])
        else:
            result = (str(value), Usage(model_calls=1))
        return (*result, bool(result[0])) if with_completion else result
    from trae_agent.utils.llm_clients.llm_basics import LLMMessage
    from trae_agent.utils.llm_clients.llm_client import LLMClient

    config = getattr(runtime.worker_factory, "config", None)
    model = getattr(getattr(config, "trae_agent", None), "model", None)
    if model is None:
        raise ManagerError("Manager model is unavailable")
    if max_output_tokens is not None:
        model = copy.deepcopy(model)
        model.max_tokens = min(max_output_tokens, model.get_max_tokens_param())
        if model.max_completion_tokens is not None:
            model.max_completion_tokens = model.max_tokens
    client = LLMClient(model)
    started = time.monotonic()
    try:
        remaining = await runtime.budget.remaining_wall_time()
        response = await asyncio.wait_for(
            runtime.cancellation.run(client.achat(
                [LLMMessage(role="system", content=system_prompt),
                 LLMMessage(role="user", content=prompt)],
                model, tools=[], reuse_history=False,
            )),
            timeout=min(30.0, max(0.1, remaining * 0.1)),
        )
        usage = response.usage
        result = (response.content, Usage(
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            model_calls=1,
            wall_time_seconds=time.monotonic() - started,
        ))
        if with_completion:
            from codeswarm_agent.context.lifecycle import complete_response
            return (*result, complete_response(response))
        return result
    finally:
        await client.aclose()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _outstanding(m: dict, request) -> list[str]:
    """A compact progress view; the completion gate still reads actual evidence."""
    if not m["plan"]:
        return ["acceptance plan"]
    items = []
    if m.get("remaining_tasks"):
        items.append("unfinished assignments require disposition")
    if m["pending"]:
        items.append("integrate pending patches")
    if not m["delivery"]:
        items.append("integrated delivery")
    verified = set(m.get("verified_commands", []))
    for criterion in m["plan"]["criteria"]:
        if criterion["required"] and (criterion["verification"] == "manual"
                                      or not criterion["commands"]
                                      or not set(criterion["commands"]).issubset(verified)):
            items.append(criterion["id"])
    if not set(request.test_commands).issubset(verified):
        items.append("user test commands")
    if (m["risk"] or m.get("review_requested")) and m["review"] != [
        m["version"], m["plan"]["version"], m["verification"]
    ]:
        items.append("independent review")
    return items


async def _decision_call(runtime: RunContext, step: int, prompt: str, model) -> str:
    if not isinstance(runtime.results, StateResults):
        return await _cached_decision_call(runtime, step, prompt, model)
    from types import SimpleNamespace

    from langgraph.func import task

    async def turn():
        settings = getattr(runtime.worker_factory, "context_settings", None) or ContextSettings()
        ordinal = 0
        summary_usages = {}

        async def summarize(text, attempt):
            nonlocal ordinal
            number = ordinal
            ordinal += 1
            operation = f"{runtime.run_id}:manager-summary:{step}:{number}"

            async def summary_response():
                snapshot = await runtime.budget.snapshot()
                if int(snapshot["workers_started"]) + 2 > runtime.request.budget.max_workers:
                    raise ValueError("summary skipped to reserve the normal Manager request")
                from codeswarm_agent.context.packer import TokenCounter
                estimate = (TokenCounter().count(text) + settings.summary_max_tokens + settings.output_reserve_tokens
                            + int((settings.context_window_tokens - settings.output_reserve_tokens - settings.safety_margin_tokens)
                                  * settings.summary_trigger_ratio))
                limit = runtime.request.budget.max_total_tokens
                if limit is not None and runtime.budget.usage.total_tokens + estimate > limit:
                    raise ValueError("summary skipped to preserve the normal Manager token budget")
                if not await runtime.budget.claim_worker(operation):
                    raise ValueError("insufficient budget for Manager summary")
                try:
                    response, usage, complete = await _model_text(
                        runtime, text, model, system_prompt="Return a structured progress summary as JSON.",
                        max_output_tokens=settings.summary_max_tokens, with_completion=True,
                    )
                except Exception as exc:
                    from codeswarm_agent.cancellation import RunCancelled
                    if isinstance(exc, (RunCancelled, RuntimeBudgetExceeded)):
                        raise
                    # Save a failed read-only request too, so replay does not
                    # resend it or lose the invocation charge. Tokens are unknown.
                    response, usage, complete = str(exc), Usage(model_calls=1), False
                record = runtime.artifacts.put_text("manager-summary-response", response, ".json")
                return {"response": record.id, "usage": to_jsonable(usage), "complete": complete}

            saved = await task(name="codeswarm_manager_summary")(summary_response)()
            await runtime.budget.reconcile_committed(operation, Usage(**saved["usage"]))
            summary_usages[operation] = saved["usage"]
            if not saved["complete"]:
                raise ValueError("Manager summary response was incomplete")
            return runtime.artifacts.read_text(saved["response"])

        async def checkpoint(snapshot):
            async def save(value):
                return value
            return await task(name="codeswarm_manager_summary_state")(save)(snapshot)

        actual, lifecycle = await prepare_manager_context(runtime, prompt, summarize, checkpoint)
        lifecycle.state["revision"] += 1
        runtime.results.stage_context("manager", lifecycle.state.copy())
        if runtime.request.budget.max_total_tokens is not None and (
            runtime.budget.usage.total_tokens + lifecycle.state["last_input_tokens"] + settings.output_reserve_tokens
            > runtime.request.budget.max_total_tokens
        ):
            raise RuntimeBudgetExceeded("remaining run tokens cannot fit the Manager request and output reserve")
        response = await _cached_decision_call(runtime, step, actual, model)
        path = runtime.run_directory / f"manager-{step}.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        complete = cache.get("complete", True)
        snapshot = lifecycle.prospective(SimpleNamespace(content=response, tool_calls=None,
                                                        finish_reason="stop" if complete else "length"), {}, [])
        snapshot["probe_consumed"] = bool(snapshot.get("probe_consumed") or complete)
        return {"cache": path.name, "context": snapshot, "usage": cache["usage"],
                "summary_usages": summary_usages}

    saved = await task(name="codeswarm_manager_turn")(turn)()
    runtime.results.stage_context("manager", saved["context"])
    await runtime.budget.reconcile_committed(f"{runtime.run_id}:manager:{step}", Usage(**saved["usage"]))
    for operation, usage in saved["summary_usages"].items():
        await runtime.budget.reconcile_committed(operation, Usage(**usage))
    return json.loads((runtime.run_directory / saved["cache"]).read_text(encoding="utf-8"))["response"]


async def _cached_decision_call(runtime: RunContext, step: int, prompt: str, model) -> str:
    """Reuse a committed response across checkpoint replay; unknown calls stop safely."""
    path = runtime.run_directory / f"manager-{step}.json"
    pending = path.with_suffix(".pending")
    operation = f"{runtime.run_id}:manager:{step}"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        await runtime.budget.reconcile_committed(operation, Usage(**saved["usage"]))
        return saved["response"]
    if pending.exists():
        raise ManagerCallUnknown("previous Manager request outcome is unknown; inspect before resuming")
    if not await runtime.budget.claim_worker(operation):
        raise ManagerError("insufficient budget for Manager decision")
    pending.write_text(json.dumps({"operation": operation}), encoding="utf-8")
    try:
        response, usage, complete = await _model_text(
            runtime, prompt, model,
            system_prompt="You are the task's persistent Manager. Output JSON only.",
            with_completion=True,
        )
    except Exception as exc:
        from codeswarm_agent.cancellation import RunCancelled
        if isinstance(exc, RunCancelled):
            raise
        raise ManagerCallUnknown(f"Manager request outcome unknown: {exc}") from exc
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"response": response, "usage": to_jsonable(usage), "complete": complete},
                                    ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    pending.unlink(missing_ok=True)
    await runtime.budget.record(usage, operation)
    return response


def _disjoint_creation(tasks: list[dict]) -> bool:
    roots = [scope[:-3].rstrip("/") for task in tasks for scope in task["write_scope"]]
    return len(roots) == len(set(roots)) and not any(
        a.startswith(b + "/") or b.startswith(a + "/")
        for i, a in enumerate(roots) for b in roots[i + 1:]
    )


async def _patch_preflight(runtime: RunContext, node: TaskNode, result) -> tuple[list[str], bool]:
    issues = []
    patch_id = FinalResultGate._patch_artifact(result, runtime.artifacts)
    if not patch_id:
        return [f"{node.id}: missing actual patch artifact"], False
    patch = runtime.artifacts.read_text(patch_id)
    paths = changed_paths(patch)
    risk = bool(re.search(
        r"\b(?:eval\s*\(|exec\s*\(|os\.system\s*\(|subprocess\.|pickle\.loads\s*\(|chmod\s*\(|socket\.)",
        patch,
    ))
    if not paths or set(paths) != set(result.changed_files):
        issues.append(f"{node.id}: patch paths do not match result metadata")
    source = await runtime.results.result(str(node.metadata.get("workspace_from", "")))
    for path in paths:
        normalized = path.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if (PurePosixPath(normalized).is_absolute() or ".." in parts or ":" in normalized
                or ".git" in parts or any(p.startswith(".env") for p in parts)):
            issues.append(f"{node.id}: protected or unsafe patch path: {path}")
            continue
        risk |= sensitive_issue(path)
        if not within_write_scope(path, node.write_scope):
            # Rework exports the full patch against HEAD. Existing inherited files
            # may occur in it, but their bytes must remain unchanged by this action.
            before = Path(str(source.metadata.get("workspace", ""))) / path if source else None
            after = Path(str(result.metadata.get("workspace", ""))) / path
            before_bytes = before.read_bytes() if before and before.is_file() else None
            after_bytes = after.read_bytes() if after.is_file() else None
            if source is None or before_bytes != after_bytes:
                issues.append(f"{node.id}: changed path outside assigned scope: {path}")
        if node.metadata.get("task_mode") == "create" and (Path(runtime.request.repository) / path).exists():
            issues.append(f"{node.id}: creation modified an existing file: {path}")
    return issues, risk


def build_manager_graph(runtime: RunContext, saver, *, manager_model=None):
    adapter = WorkerNodeAdapter(runtime)

    async def initialize(state: CodeSwarmState) -> dict:
        started = not await runtime.budget.exhausted()
        fast = await asyncio.to_thread(single_agent_fast_path, runtime.request) if started else None
        probe = (await asyncio.to_thread(initial_probe, runtime.request.repository, runtime.request.issue,
                                        seconds=min(0.25, await runtime.budget.remaining_wall_time() * 0.05))
                 if started and not fast else {})
        if probe.get("evidence"):
            record = runtime.artifacts.put_text("manager-initial-probe", probe.pop("evidence"), ".txt")
            probe["evidence_artifact"] = record.id
        if started:
            await runtime.budget.record(Usage(tool_calls=1, wall_time_seconds=probe.get("elapsed_seconds", 0)),
                                        f"{runtime.run_id}:initial-route-probe")
        m = {"step": 0, "plan": None, "history": [], "pending": [],
             "delivery": None, "version": state["base_commit"],
             "verification": None, "review": None, "pre_review": None,
             "risk": sensitive_issue(runtime.request.issue), "feedback": [],
             "outstanding": ["acceptance plan", "integrated delivery"], "verified_commands": [],
             "no_progress": 0, "reworks": 0, "last_write_failed": False,
             "routing_seen": [], "routing_events": [], "workers": {}, "remaining_tasks": [],
             "initial_probe": probe, "initial_route": "rule-single" if fast else "manager-decides"}
        if fast:
            decision = parse_decision(json.dumps(fast), runtime.request, None)
            m.update(plan=decision.plan, action=fast, step=1,
                     history=[{"step": 0, "action": "execute", "reason": decision.reason,
                               "version": state["base_commit"], "source": "single-agent-rule"}])
        runtime.tracer.emit_once(f"{runtime.run_id}:started", "run_started",
                                 mode="single-agent", architecture_mode="manager_v2")
        runtime.tracer.emit_once(f"{runtime.run_id}:initial-route", "route_decided",
                                 mode="single-agent" if fast else "undecided", evidence=probe.get("files", []),
                                 reason=fast["reason"] if fast else "uncertain tasks go to Manager judgment")
        return {"mode": "single-agent", "phase": "manager",
                "manager": m, "budget_snapshot": await runtime.budget.snapshot(),
                **runtime.event_state()}

    async def decide(state: CodeSwarmState) -> dict:
        m = dict(state["manager"])
        if m["no_progress"] >= 3 or await runtime.budget.exhausted():
            return {"terminal_status": RunStatus.BUDGET_EXCEEDED.value if await runtime.budget.exhausted()
                    else RunStatus.FAILED.value,
                    "phase": "finished", "budget_snapshot": await runtime.budget.snapshot(),
                    "failure_reason": "; ".join(m["feedback"]) or "Manager stopped after repeated no progress"}
        results = await runtime.results.all_results()
        context = runtime.results.current.get().contexts.get("manager", {}) if isinstance(runtime.results, StateResults) else {}
        probe = m.get("initial_probe", {})
        evidence = ""
        if not context.get("probe_consumed") and (m["step"] == 0 or isinstance(runtime.results, StateResults)):
            evidence = runtime.artifacts.read_text(probe["evidence_artifact"]) if probe.get("evidence_artifact") else probe.get("evidence", "")
        prompt = MANAGER_INSTRUCTIONS + "\nTask state:\n" + json.dumps({
            "goal": runtime.request.issue,
            "user_acceptance": runtime.request.acceptance_criteria,
            "user_test_commands": runtime.request.test_commands,
            "force_multi_agent": runtime.request.force_multi_agent,
            "manager": m, "budget": await runtime.budget.snapshot(),
            "results": [{"task_id": r.task_id, "attempt": r.attempt, "role": r.role.value,
                         "status": r.status.value, "summary": r.summary[:2000],
                         "artifacts": r.artifact_ids, "unresolved": r.unresolved,
                         "verifications": to_jsonable(r.verifications),
                         "reviews": to_jsonable(r.review_decisions)} for r in results],
            "repository_evidence": evidence,
        }, ensure_ascii=False)
        try:
            response = await _decision_call(runtime, m["step"], prompt, manager_model)
            decision = parse_decision(response, runtime.request, m["plan"])
            if m["plan"] != decision.plan:
                m["review"] = None
            m["plan"] = decision.plan
            m["outstanding"] = _outstanding(m, runtime.request)
            m["action"] = decision.payload
            m["history"] = [*m["history"], {"step": m["step"], "action": decision.action,
                                            "reason": decision.reason, "version": m["version"]}]
            m["step"] += 1
            runtime.tracer.emit_once(f"{runtime.run_id}:manager-decision:{m['step']}",
                                     "manager_decided", action=decision.action,
                                     reason=decision.reason, version=m["version"],
                                     plan_version=m["plan"]["version"])
            return {"manager": m, "budget_snapshot": await runtime.budget.snapshot(), **runtime.event_state()}
        except ManagerCallUnknown as exc:
            return {"manager": m, "terminal_status": RunStatus.FAILED.value,
                    "phase": "finished", "failure_reason": str(exc),
                    "budget_snapshot": await runtime.budget.snapshot(),
                    **runtime.event_state()}
        except RuntimeBudgetExceeded as exc:
            return {"manager": m, "terminal_status": RunStatus.BUDGET_EXCEEDED.value,
                    "phase": "finished", "failure_reason": str(exc),
                    "budget_snapshot": await runtime.budget.snapshot(), **runtime.event_state()}
        except (ManagerError, ValueError) as exc:
            m["step"] += 1
            m["feedback"] = [str(exc)]
            m["no_progress"] += 1
            m["action"] = None
            return {"manager": m, "budget_snapshot": await runtime.budget.snapshot(), **runtime.event_state()}

    async def execute(state: CodeSwarmState) -> dict:
        m = dict(state["manager"])
        action = m["action"]
        if not action:
            return {}
        specs = dict(state.get("task_specs", {}))
        refs = {}
        mode = state.get("mode", "single-agent")
        before = _digest([m["version"], m["pending"], m["verification"], m["review"], m["pre_review"],
                          m.get("routing_seen", []), m.get("remaining_tasks", [])])
        feedback = []
        try:
            kind = action["action"]
            if m["pending"] and kind not in {"integrate", "blocked"}:
                raise ManagerError("integrate pending patches before taking another action")
            if kind in {"execute", "delegate"}:
                decision = ManagerDecision(kind, action["reason"], action, m["plan"])
                current = await runtime.results.result(m["delivery"]) if m["delivery"] else None
                tasks = validate_tasks(decision, str(current.metadata["workspace"])
                                       if current else runtime.request.repository)
                basis = _digest([m["version"], m.get("routing_seen", []),
                                 m.get("verified_commands", []),
                                 m["last_write_failed"]])
                signature = assignment_signature(basis, tasks)
                if signature in m.get("assignment_signatures", []):
                    raise ManagerError("equivalent assignment already considered at this delivery version; use new evidence")
                if len({assignment_signature(m["version"], [task]) for task in tasks}) != len(tasks):
                    raise ManagerError("duplicate Worker investigation or implementation assignment")
                if m.get("remaining_tasks") and not str(action.get("handoff_reason", "")).strip():
                    raise ManagerError("replacing unfinished assignments requires an explicit handoff_reason")
                if len(tasks) > (runtime.request.max_work_units or runtime.request.budget.max_work_units):
                    raise ManagerError("too many tasks for the work-unit budget")
                if runtime.request.force_multi_agent is False and kind == "delegate":
                    raise ManagerError("user requested single-agent execution")
                if (runtime.request.force_multi_agent is True and not m.get("delegated")
                        and (kind != "delegate" or len(tasks) < 2)):
                    raise ManagerError("explicit multi-agent request needs at least two real bounded tasks")
                writes = [t for t in tasks if t["mode"] != "read"]
                rework = bool(writes and (m["verification"] or m["last_write_failed"]))
                if rework and m["reworks"] >= runtime.request.budget.max_replans:
                    raise ManagerError("directed rework budget exhausted")
                snapshot = await runtime.budget.snapshot()
                reserve = 5 + 2 * int(bool(m["risk"]))
                if int(snapshot["workers_started"]) + len(tasks) + reserve > runtime.request.budget.max_workers:
                    raise ManagerError("insufficient budget after reserving integration, verification and completion")
                parallel_create = (len(writes) > 1 and len(writes) == len(tasks)
                                   and all(t["mode"] == "create" for t in writes)
                                   and m["delivery"] is None and _disjoint_creation(writes))
                if len(writes) > 1 and all(t["mode"] == "create" for t in writes) and not _disjoint_creation(writes):
                    raise ManagerError("new module write scopes overlap")
                if rework:
                    m["reworks"] += 1
                m["assignment_signatures"] = [*m.get("assignment_signatures", []), signature]
                for item in m.get("remaining_tasks", []):
                    m["workers"][item["task_id"]]["resolution"] = {
                        "kind": "superseded", "reason": action["handoff_reason"]}
                m["remaining_tasks"] = []
                control = RoutingControl(seen=set(m.get("routing_seen", [])))
                source = m["delivery"]
                m["last_execution_mode"] = "multi-agent" if kind == "delegate" else "single-agent"
                runtime.tracer.emit("route_decided", logical_operation=f"{runtime.run_id}:route-dispatch:{m['step']}",
                                         mode=m["last_execution_mode"],
                                         reason=action["reason"], input_version=m["version"], tasks=tasks,
                                         trigger_events=m.get("routing_events", [])[-8:],
                                         budget=snapshot)

                async def run_task(index: int, task: dict, workspace_from: str | None):
                    role = (WorkerRole.MANAGER if kind == "execute" else
                            WorkerRole.EXPLORER if task["mode"] == "read" else WorkerRole.WORKER)
                    # Only the Manager's execution identity changes. The same policy,
                    # budget, workspace and completion machinery handles every writer.
                    inherited = await runtime.results.result(workspace_from) if workspace_from else None
                    inherited_patch = FinalResultGate._patch_artifact(inherited, runtime.artifacts)
                    node = TaskNode(
                        id=f"manager-{m['step']}-task-{index}", role=role, goal=task["goal"],
                        depends_on=(workspace_from,) if workspace_from else (),
                        write_scope=tuple(task["write_scope"]),
                        max_attempts=runtime.request.budget.max_retries_per_task + 1,
                        metadata={"manager_action": True, "single_agent": kind == "execute",
                                  "task_mode": task["mode"], "read_scope": task["read_scope"],
                                  "criteria_ids": task["criteria_ids"], "interface": task["interface"],
                                  "deliverable": task["deliverable"],
                                  "input_version": runtime.artifacts.get(inherited_patch).sha256
                                  if inherited_patch else m["version"], "workspace_from": workspace_from,
                                  "inherited_paths": list(inherited.changed_files) if inherited else [],
                                  "allow_test_only": bool(task["write_scope"]) and all(
                                      p.startswith("tests/") or Path(p).name.startswith("test_")
                                      for p in task["write_scope"]),
                                  "manager_state": {"plan": m["plan"], "history": m["history"],
                                                    "feedback": m["feedback"]}},
                    )
                    specs[node.id] = to_jsonable(node)
                    result_refs = await adapter.execute(to_jsonable(node), routing_control=control)
                    result = await runtime.results.result(node.id)
                    return node, result, result_refs

                outputs = []
                preflight = {}
                if not writes or parallel_create:
                    # Waves respect the shared run concurrency budget, even when
                    # isolated worktrees would allow more simultaneous writers.
                    for start in range(0, len(tasks), runtime.request.budget.max_parallel):
                        if control.pause_requested:
                            break
                        outputs.extend(await asyncio.gather(*(
                            run_task(i, tasks[i], source) for i in range(start, min(
                                len(tasks), start + runtime.request.budget.max_parallel)))))
                else:
                    # Business-level serial writing: isolation never waives this.
                    for i, task in enumerate(tasks):
                        if control.pause_requested:
                            break
                        output = await run_task(i, task, source)
                        outputs.append(output)
                        node, result, _ = output
                        if result is None or (result.status != WorkerStatus.SUCCEEDED
                                              and not result.metadata.get("handoff_ready")):
                            break
                        if task["mode"] != "read" and (result.changed_files or not result.metadata.get("handoff_ready")):
                            issues, risk = await _patch_preflight(runtime, node, result)
                            preflight[node.id] = (issues, risk)
                            if issues:
                                feedback.extend(issues)
                                break
                            m["risk"] |= risk
                            source = node.id
                good_writes = []
                launched = {node.id for node, _, _ in outputs}
                for index, task in enumerate(tasks):
                    task_id = f"manager-{m['step']}-task-{index}"
                    if task_id not in launched:
                        m.setdefault("workers", {})[task_id] = {"status": "not_started", "task": task,
                                                               "input_version": m["version"]}
                        m["remaining_tasks"].append({"task_id": task_id, **task})
                for node, result, result_refs in outputs:
                    refs.update(result_refs)
                    yielded = bool(result and result.metadata.get("handoff_ready"))
                    task = tasks[int(node.id.rsplit("-", 1)[1])]
                    m.setdefault("workers", {})[node.id] = {
                        "status": "yielded" if yielded else result.status.value if result else "missing",
                        "task": task, "input_version": node.metadata["input_version"],
                        "artifacts": list(result.artifact_ids) if result else []}
                    if yielded:
                        m["remaining_tasks"].append({"task_id": node.id, **task})
                    if result is None or (result.status != WorkerStatus.SUCCEEDED and not yielded):
                        feedback.append(f"{node.id}: {result.summary if result else 'missing result'}")
                    elif task["mode"] != "read" and (result.changed_files or not yielded):
                        if node.id not in preflight:
                            preflight[node.id] = await _patch_preflight(runtime, node, result)
                        issues, risk = preflight[node.id]
                        feedback.extend(issues)
                        m["risk"] |= risk
                        if not issues:
                            good_writes.append(node.id)
                m["pending"] = good_writes if parallel_create else good_writes[-1:]
                if writes:
                    m["last_write_failed"] = bool(feedback)
                    m["verification"] = m["review"] = m["pre_review"] = None
                    m["verified_commands"] = []
                if kind == "delegate":
                    m["delegated"] = True
                    # The complete set of live Workers has now returned; Manager
                    # resumes naturally without a separate downgrade model call.
                    mode = "single-agent"
                elif kind == "execute":
                    mode = "single-agent"
                if not writes and not feedback:
                    # New read reports count as progress only once by their content.
                    signatures = set(m.get("read_reports", []))
                    observed = {_digest([r.summary, r.artifact_ids]) for _, r, _ in outputs if r}
                    if observed.issubset(signatures):
                        feedback.append("read actions repeated the same evidence without progress")
                    else:
                        m["read_reports"] = sorted(signatures | observed)
                        before = "new read evidence"
                m["routing_seen"] = sorted(control.seen)
                m["routing_events"] = [*m.get("routing_events", []), *control.events]
                if control.events:
                    feedback.extend(f"routing event {event['kind']}: {event['reason']}" for event in control.events)
                runtime.tracer.emit("routing_boundary", logical_operation=f"{runtime.run_id}:route-boundary:{m['step']}",
                                         from_mode="multi-agent" if kind == "delegate" else "single-agent",
                                         to_mode="single-agent", reason=action["reason"],
                                         events=control.events, remaining=m["remaining_tasks"],
                                         budget=await runtime.budget.snapshot())
            elif kind == "consolidate":
                if not m.get("remaining_tasks"):
                    raise ManagerError("no unfinished assignments to consolidate")
                for task in m["remaining_tasks"]:
                    worker = m["workers"][task["task_id"]]
                    worker["resolution"] = {"kind": "cancelled" if worker["status"] == "not_started"
                                            else "stopped_with_findings", "reason": action["reason"]}
                m["remaining_tasks"] = []
            elif kind == "integrate":
                if not m["pending"]:
                    raise ManagerError("no pending patches to integrate")
                node = TaskNode(id=f"manager-{m['step']}-integrate", role=WorkerRole.INTEGRATOR,
                                goal="Apply the current authorized delivery patches deterministically.",
                                depends_on=tuple(m["pending"]), write_scope=("**",),
                                metadata={"component_task_ids": m["pending"]})
                specs[node.id] = to_jsonable(node)
                refs.update(await adapter.execute(to_jsonable(node)))
                result = await runtime.results.result(node.id)
                patch = FinalResultGate._patch_artifact(result, runtime.artifacts)
                if not result or result.status != WorkerStatus.SUCCEEDED or not patch:
                    raise ManagerError("deterministic integration failed")
                m["delivery"] = node.id
                m["version"] = runtime.artifacts.get(patch).sha256
                m["pending"] = []
            elif kind in {"verify", "review"}:
                if kind == "review":
                    m["review_requested"] = True
                if not m["delivery"]:
                    raise ManagerError("an integrated delivery is required")
                commands = required_commands(m["plan"], runtime.request)
                if kind == "verify" and not commands:
                    raise ManagerError("no verification command; required behaviors remain unverified")
                if kind == "verify" and m["risk"] and m["pre_review"] != m["version"]:
                    raise ManagerError("risk rule requires independent review before executing changed code")
                pre = kind == "review" and not m["verification"]
                node = TaskNode(
                    id=f"manager-{m['step']}-{kind}",
                    role=WorkerRole.TESTER if kind == "verify" else WorkerRole.REVIEWER,
                    goal=("Run all acceptance commands against the current integrated delivery." if kind == "verify"
                          else "Independently inspect current patch, risk and acceptance coverage. "
                          + ("Security preflight only; tests have not run yet." if pre else "Judge fresh integrated test evidence.")),
                    depends_on=tuple(x for x in (m["delivery"], m["verification"]) if x),
                    metadata={"phase": "verify" if kind == "verify" else "preaudit" if pre else "review",
                              "manager_action": True,
                              "workspace_from": m["delivery"], "test_commands": list(commands),
                              "integrated_task_id": m["delivery"], "integration_task_id": m["delivery"],
                              "review_target_task_ids": [m["delivery"]],
                              "acceptance_plan": m["plan"], "input_version": m["version"]},
                )
                specs[node.id] = to_jsonable(node)
                refs.update(await adapter.execute(to_jsonable(node)))
                result = await runtime.results.result(node.id)
                if kind == "verify":
                    m["verification"] = node.id
                    passed = bool(result and FinalResultGate._tests_pass(
                        replace(runtime.request, test_commands=commands), result, m["version"]
                    ))
                    m["verified_commands"] = list(commands) if passed else []
                    if not passed:
                        feedback.append("fresh integrated verification failed")
                else:
                    decisions = result.review_decisions if result else ()
                    if (not result or result.status != WorkerStatus.SUCCEEDED or len(decisions) != 1
                            or decisions[0].target_task_id != m["delivery"]
                            or decisions[0].verdict != ReviewVerdict.ACCEPT):
                        feedback.extend(["independent review rejected or missing", *(
                            reason for d in decisions for reason in (*d.reasons, *d.missing_tests))])
                    elif pre:
                        m["pre_review"] = m["version"]
                    else:
                        m["review"] = [m["version"], m["plan"]["version"], m["verification"]]
                    m["review_decisions"] = to_jsonable(decisions)
            elif kind == "complete":
                if m.get("remaining_tasks"):
                    feedback.append("unfinished assignments require Manager disposition before completion")
                if m["last_write_failed"]:
                    feedback.append("the last modification failed and remains unresolved")
                commands = required_commands(m["plan"], runtime.request)
                integrated = await runtime.results.result(m["delivery"]) if m["delivery"] else None
                patch = FinalResultGate._patch_artifact(integrated, runtime.artifacts)
                if not integrated or not patch or integrated.status != WorkerStatus.SUCCEEDED:
                    feedback.append("no successful integrated artifact")
                elif runtime.artifacts.get(patch).sha256 != m["version"]:
                    feedback.append("current delivery version does not match the artifact")
                elif workspace_version(Path(integrated.metadata["workspace"])) != integrated.metadata.get("workspace_version"):
                    feedback.append("integrated workspace changed after artifact capture")
                tester = await runtime.results.result(m["verification"]) if m["verification"] else None
                if not commands or tester is None or not FinalResultGate._tests_pass(
                    replace(runtime.request, test_commands=commands), tester, m["version"]
                ):
                    feedback.append("required fresh verification is missing or failed")
                for criterion in m["plan"]["criteria"]:
                    if criterion["required"] and (criterion["verification"] == "manual" or not criterion["commands"]):
                        feedback.append(f"unconfirmed required criterion: {criterion['id']}")
                review_key = [m["version"], m["plan"]["version"], m["verification"]]
                if (m["risk"] or m.get("review_requested")) and m["review"] != review_key:
                    feedback.append("independent review of current delivery and verification is required")
                if runtime.request.force_multi_agent is True and not m.get("delegated"):
                    feedback.append("explicit multi-agent request has no actual delegation")
                if not feedback:
                    return {"manager": m, "terminal_status": RunStatus.SUCCEEDED.value,
                            "phase": "finished", "hard_gates_passed": True,
                            "selected_integrated_task_id": m["delivery"],
                            "selected_patch_artifact_id": patch,
                            "review_decisions": m.get("review_decisions", []),
                            "failure_reason": None, "budget_snapshot": await runtime.budget.snapshot(),
                            **runtime.event_state()}
            elif kind == "blocked":
                return {"manager": m, "terminal_status": RunStatus.FAILED.value,
                        "failure_reason": action["reason"], "phase": "finished"}
        except (ManagerError, ValueError, OSError) as exc:
            feedback.append(str(exc))
        m["feedback"] = feedback
        m["outstanding"] = _outstanding(m, runtime.request)
        after = _digest([m["version"], m["pending"], m["verification"], m["review"], m["pre_review"],
                         m.get("routing_seen", []), m.get("remaining_tasks", [])])
        # Compare semantic results, not newly allocated task IDs. Re-running a
        # failing test or producing the same patch must not reset stagnation.
        pending_versions = []
        for task_id in m["pending"]:
            result = await runtime.results.result(task_id)
            patch = FinalResultGate._patch_artifact(result, runtime.artifacts)
            pending_versions.append(runtime.artifacts.get(patch).sha256 if patch else None)
        verified = await runtime.results.result(m["verification"]) if m["verification"] else None
        signature = _digest([action["action"], m["version"], pending_versions,
                             [(v.command, v.passed, v.patch_sha256) for v in verified.verifications]
                             if verified else None,
                             bool(m["review"]), m["pre_review"], m.get("read_reports", []), feedback])
        seen = set(m.get("progress_signatures", []))
        stagnant = before == after or signature in seen
        m["no_progress"] = m["no_progress"] + 1 if stagnant else 0
        m["progress_signatures"] = sorted(seen | {signature})
        m["history"][-1] = {**m["history"][-1], "feedback": feedback,
                             "output_version": m["version"], "artifacts": list(refs)}
        runtime.tracer.emit_once(f"{runtime.run_id}:manager-action:{m['step']}",
                                 "manager_action_finished", action=action["action"],
                                 feedback=feedback, version=m["version"])
        return {"manager": m, "result_refs": refs, "task_specs": specs, "mode": mode,
                "replan_round": m["reworks"], "failure_reason": "; ".join(feedback) or None,
                "budget_snapshot": await runtime.budget.snapshot(), **runtime.event_state()}

    def after_decision(state):
        return END if state.get("terminal_status") else "execute"

    def after_action(state):
        return END if state.get("terminal_status") else "manager"

    builder = StateGraph(CodeSwarmState)
    runtime.add_node(builder, "initialize", initialize)
    runtime.add_node(builder, "manager", decide)
    runtime.add_node(builder, "execute", execute)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", lambda state: "execute" if state["manager"].get("action") else "manager")
    builder.add_conditional_edges("manager", after_decision)
    builder.add_conditional_edges("execute", after_action)
    return builder.compile(checkpointer=saver)
