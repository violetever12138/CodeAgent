"""LangGraph orchestration backend around the existing CodeSwarm workers."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, TypedDict

# The checkpoint database is local evidence. Restrict deserialization before importing
# the LangGraph serializer so a modified database cannot request arbitrary Python types.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.func import task as durable_task
from langgraph.types import Command
from langsmith import tracing_context

from codeagent.artifacts import ArtifactStore
from codeagent.budget import BudgetTracker
from codeagent.cancellation import CancellationToken, RunCancelled
from codeagent.context.lifecycle import merge_contexts
from codeagent.evidence.checkpoint import Checkpoint, CheckpointStore
from codeagent.protocol import (
    ApprovalAction,
    ApprovalRequest,
    RunStatus,
    TaskNode,
    TaskOutcome,
    TaskRequest,
    TaskStatus,
    Usage,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
    review_decision_from_json,
    task_node_from_json,
    task_request_from_json,
    to_jsonable,
    worker_result_from_json,
)
from codeagent.result_state import (
    RESULT_SCHEMA_VERSION,
    RESULT_STORAGE,
    ResultView,
    StateResults,
    compact_result,
    merge_results,
)
from codeagent.roles import get_role_spec
from codeagent.telemetry import TraceSettings, current_operation, execution, span
from codeagent.telemetry import operation as observe_operation
from codeagent.trace import EventTracer
from codeagent.worker import WorkerContext, WorkerFactory
from codeagent.workspace import GitWorkspaceManager, WorkspaceManager

LANGGRAPH_BACKEND_VERSION = "langgraph-cooperative-v1"


def _merge_result_refs(
    left: dict[str, dict[str, Any]] | None,
    right: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(f"conflicting LangGraph result update: {key}")
        merged[key] = value
    return merged


class CodeSwarmState(TypedDict, total=False):
    contexts: Annotated[dict[str, dict[str, Any]], merge_contexts]
    results: Annotated[dict[str, dict[str, Any]], merge_results]
    manager: dict[str, Any]
    schema_version: int
    run_id: str
    request: dict[str, Any]
    base_commit: str
    mode: str
    phase: str
    replan_round: int
    task_specs: dict[str, dict[str, Any]]
    developer_tasks: list[dict[str, Any]]
    closing_tasks: dict[str, dict[str, Any]]
    result_refs: Annotated[dict[str, dict[str, Any]], _merge_result_refs]
    active_task: dict[str, Any]
    ready_task_ids: list[str]
    developer_phase: str
    review_decisions: list[dict[str, Any]]
    finalization_reasons: list[str]
    requires_replan: bool
    hard_gates_passed: bool
    selected_integrated_task_id: str | None
    selected_patch_artifact_id: str | None
    pending_approval: dict[str, Any] | None
    approval_decision: dict[str, Any] | None
    budget_snapshot: dict[str, Any]
    event_sequence: int
    event_checksum: str | None
    terminal_status: str
    failure_reason: str | None
    planning_signals: dict[str, Any] | None
    planning_decision: dict[str, Any] | None
    work_unit_signatures: list[str]
    replan_reason: str | None
    replan_workspace_from: str | None
    routing_error: str | None
    preflight_passed: bool
    preflight_reasons: list[str]
    audit_required: bool
    preaudit_required: bool
    preaudit_passed: bool
    verification_commands: list[str]
    verification_missing: bool
    replan_failed: bool
    budget_admission_failed: bool


@dataclass
class RunContext:
    request: TaskRequest
    run_id: str
    run_directory: Path
    worker_factory: WorkerFactory
    artifacts: ArtifactStore
    results: StateResults
    tracer: EventTracer
    budget: BudgetTracker
    workspace_manager: WorkspaceManager
    cancellation: CancellationToken

    def event_state(self) -> dict[str, Any]:
        return {
            "event_sequence": self.tracer.store.sequence,
            "event_checksum": self.tracer.store.last_checksum,
        }

    def add_node(self, builder, name, handler):
        handler = self.results.wrap(handler, self)
        if inspect.isfunction(handler):
            handler = observe_operation(f"graph.{name}", "CHAIN")(handler)
        builder.add_node(name, handler)


def _result_key(task_id: str, attempt: int) -> str:
    return f"{task_id}@{attempt}"


def _result_ref(result: WorkerResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "attempt": result.attempt,
        "role": result.role.value,
        "status": result.status.value,
        "summary": result.summary[:1000],
        "artifact_ids": list(result.artifact_ids),
        "evidence_ids": [item.id for item in result.evidence],
        "changed_files": list(result.changed_files),
    }


def _latest_ref(state: CodeSwarmState, task_id: str) -> dict[str, Any] | None:
    matches = [
        value
        for value in (state.get("results") or state.get("result_refs", {})).values()
        if value.get("task_id") == task_id
    ]
    return max(matches, key=lambda item: int(item.get("attempt", 0)), default=None)


class WorkerNodeAdapter:
    """Idempotent node boundary around the unchanged Worker implementation."""

    def __init__(self, runtime: RunContext) -> None:
        self.runtime = runtime
        self._routing: ContextVar = ContextVar("worker_routing_control", default=None)
        self._attempt_task = durable_task(name="codeswarm_worker_attempt")(self._durable_attempt)

    async def execute(self, raw_node: dict[str, Any], *, routing_control=None) -> dict[str, dict[str, Any]]:
        updates = {}
        token = self._routing.set(routing_control)
        try:
            for attempt in range(raw_node.get("attempt", 0), raw_node.get("max_attempts", 1)):
                existing = await self.runtime.results.result_attempt(raw_node["id"], attempt)
                if existing is None:
                    attempt_node = {**raw_node, "attempt": attempt, "max_attempts": attempt + 1}
                    raw = (await self._attempt_task(attempt_node) if self.runtime.results.current.get().durable_attempts
                           else await self._durable_attempt(attempt_node))
                    existing = worker_result_from_json(raw)
                    await self.runtime.results.commit(existing)
                    existing = await self.runtime.results.result_attempt(raw_node["id"], attempt)
                operation_id = f"{self.runtime.run_id}:{existing.task_id}:{attempt}"
                await self.runtime.budget.reconcile_committed(operation_id, existing.usage)
                updates[_result_key(existing.task_id, attempt)] = to_jsonable(existing)
                if routing_control:
                    routing_control.restore(existing.metadata.get("routing_events", []))
                if existing.status == WorkerStatus.SUCCEEDED or existing.metadata.get("handoff_ready"):
                    break
            return updates
        finally:
            self._routing.reset(token)

    async def _durable_attempt(self, raw_node: dict) -> dict:
        # Keep unconfirmed changes out of the parent node's view. Only the task's
        # returned, framework-managed output may populate that view on replay.
        results = self.runtime.results
        view = ResultView(results.current.get().values, contexts=results.current.get().contexts)
        token = results.current.set(view)
        try:
            await self._execute_attempt(raw_node, routing_control=self._routing.get())
            result = await results.result_attempt(raw_node["id"], raw_node["attempt"])
            raw = to_jsonable(compact_result(result, self.runtime.artifacts))
            key = _result_key(raw_node["id"], raw_node["attempt"])
            if key in view.contexts:
                raw["metadata"]["_context"] = view.contexts[key]
            return raw
        finally:
            results.current.reset(token)

    async def _execute_attempt(self, raw_node: dict[str, Any], *, routing_control=None) -> dict[str, dict[str, Any]]:
        template = task_node_from_json(raw_node)
        updates: dict[str, dict[str, Any]] = {}
        for attempt in range(template.attempt, template.max_attempts):
            node = task_node_from_json({**raw_node, "attempt": attempt})
            key = _result_key(node.id, attempt)
            operation_id = f"{self.runtime.run_id}:{node.id}:{attempt}"
            existing = await self.runtime.results.result_attempt(node.id, attempt)
            if existing is not None:
                await self.runtime.budget.reconcile_committed(operation_id, existing.usage)
                updates[key] = _result_ref(existing)
                if routing_control:
                    routing_control.restore(existing.metadata.get("routing_events", []))
                if existing.status == WorkerStatus.SUCCEEDED or existing.metadata.get("handoff_ready"):
                    break
                continue

            if not await self.runtime.budget.claim_worker(operation_id):
                self.runtime.cancellation.cancel("run budget exhausted")
                raise RunCancelled("run budget exhausted")

            reset = getattr(self.runtime.workspace_manager, "reset", None)
            if reset is not None:
                await reset(node.id)
            workspace = await self.runtime.workspace_manager.acquire(node)
            if node.role in {WorkerRole.REVIEWER, WorkerRole.AUDITOR}:
                integration_id = str(node.metadata.get("integration_task_id", ""))
                integrated = await self.runtime.results.result(integration_id)
                related_ids = {
                    integration_id, *node.depends_on,
                    *node.metadata.get("component_task_ids", ()),
                }
                if integrated:
                    related_ids.add(str(integrated.metadata.get("security_task_id", "")))
                dependency_results = tuple(
                    item for item in await self.runtime.results.all_results()
                    if item.task_id in related_ids
                )
            else:
                dependency_results = await self.runtime.results.results_for(node.depends_on)
            evidence = await self.runtime.results.evidence()
            if node.role in {WorkerRole.REVIEWER, WorkerRole.AUDITOR}:
                evidence = tuple(item for item in evidence if item.task_id in related_ids)
            role = get_role_spec(node.role)
            if node.role == WorkerRole.MANAGER and node.metadata.get("task_mode") == "read":
                role = replace(role, read_only=True, allowed_tools=get_role_spec(WorkerRole.EXPLORER).allowed_tools,
                               system_prompt=role.system_prompt + "\nThis Manager action is read-only. Report findings; do not edit code.\n")
            worker = self.runtime.worker_factory.create(node, role)
            self._emit_execution(
                f"{operation_id}:started",
                "worker_started",
                task_id=node.id,
                worker_id=worker.worker_id,
                parent_task_ids=node.depends_on,
                role=node.role.value,
                workspace=str(workspace),
                attempt=attempt,
                orchestration_backend="langgraph",
            )
            context = WorkerContext(
                run_id=self.runtime.run_id,
                request=(
                    replace(self.runtime.request, test_commands=tuple(node.metadata["test_commands"]))
                    if "test_commands" in node.metadata else self.runtime.request
                ),
                node=node,
                role=role,
                workspace=workspace,
                dependency_results=dependency_results,
                shared_evidence=evidence,
                artifacts=self.runtime.artifacts,
                run_directory=self.runtime.run_directory,
                deadline_at=time.monotonic()
                + await self.runtime.budget.remaining_wall_time(),
                cancellation=self.runtime.cancellation,
                routing_control=routing_control,
                durable_context=True,
                context_state=self.runtime.results.current.get().contexts.get(key),
                remaining_tokens=(max(0, self.runtime.request.budget.max_total_tokens
                                      - self.runtime.budget.usage.total_tokens)
                                  if self.runtime.request.budget.max_total_tokens is not None else None),
            )
            try:
                with span(f"worker.{node.role.value}", "AGENT", **{
                    "codeswarm.task_id": node.id, "codeswarm.attempt": attempt,
                    "codeswarm.worker_id": worker.worker_id, "codeswarm.operation_id": operation_id}) as observed:
                    result = await worker.run(context)
                    observed.attrs(**{"codeswarm.status": result.status.value})
                    if result.status == WorkerStatus.FAILED:
                        observed.error(RuntimeError("; ".join(result.unresolved) or "worker failed"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = self._failure_result(node, str(exc))
            await self._commit(result, operation_id)
            updates[key] = _result_ref(result)
            if result.status == WorkerStatus.SUCCEEDED or result.metadata.get("handoff_ready"):
                break
        return updates

    async def _commit(self, result: WorkerResult, operation_id: str) -> None:
        await self.runtime.results.commit(result)
        await self.runtime.budget.record(result.usage, operation_id)
        worker_id = str(result.metadata.get("worker_id", ""))
        self._emit_execution(
            f"{operation_id}:finished",
            "task_completed" if result.status == WorkerStatus.SUCCEEDED else "task_failed",
            task_id=result.task_id,
            worker_id=worker_id or None,
            status=result.status.value,
            attempt=result.attempt,
            usage=result.usage,
            orchestration_backend="langgraph",
        )

    def _emit_execution(self, operation: str, event: str, **data) -> None:
        self.runtime.tracer.emit(event, logical_operation=operation, **data)

    @staticmethod
    def _failure_result(
        node: TaskNode,
        reason: str,
        status: WorkerStatus = WorkerStatus.FAILED,
    ) -> WorkerResult:
        return WorkerResult(
            task_id=node.id,
            attempt=node.attempt,
            role=node.role,
            status=status,
            summary=reason,
            unresolved=(reason,),
            usage=Usage(),
            metadata={**node.metadata, "worker_id": f"failed-{uuid.uuid4().hex[:8]}"},
        )


class LangGraphOrchestrationBackend:
    """Persistent cooperative orchestration with existing CodeSwarm Workers as nodes."""

    def __init__(
        self,
        worker_factory: WorkerFactory,
        *,
        run_root: str | Path | None = None,
        keep_workspaces: bool = False,
        workspace_manager_factory=None,
        route_model=None,
    ) -> None:
        self.worker_factory = worker_factory
        self.run_root = Path(run_root).resolve() if run_root else None
        self.keep_workspaces = keep_workspaces
        self.workspace_manager_factory = workspace_manager_factory
        self.architecture_mode = "manager_v2"
        self.route_model = route_model

    async def run(self, request: TaskRequest) -> TaskOutcome:
        run_id = uuid.uuid4().hex
        repository = Path(request.repository).resolve()
        root = self.run_root or repository.parent / ".codeswarm-runs" / repository.name
        run_directory = root / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        base_commit = self._repository_base(request)
        (run_directory / "request.json").write_text(
            json.dumps(to_jsonable(request), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "base_commit": base_commit,
                    "orchestration_backend": LANGGRAPH_BACKEND_VERSION,
                    "state_schema_version": RESULT_SCHEMA_VERSION,
                    "result_storage": RESULT_STORAGE,
                    "architecture_mode": self.architecture_mode,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        runtime = self._runtime(request, run_id, run_directory)
        initial: CodeSwarmState = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "request": to_jsonable(request),
            "base_commit": base_commit,
            "phase": "created",
            "replan_round": 0,
            "task_specs": {},
            "developer_tasks": [],
            "closing_tasks": {},
            "results": {},
            "review_decisions": [],
            "finalization_reasons": [],
            "requires_replan": False,
            "hard_gates_passed": False,
            "pending_approval": None,
            "budget_snapshot": await runtime.budget.snapshot(),
            **runtime.event_state(),
        }
        try:
            return await self._drive(runtime, initial)
        except Exception:
            runtime.artifacts.close()
            raise

    async def resume(
        self,
        run_directory: str | Path,
        *,
        approval_decision: str | None = None,
        approval_comment: str | None = None,
    ) -> TaskOutcome:
        directory = Path(run_directory).resolve()
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("orchestration_backend") != LANGGRAPH_BACKEND_VERSION:
            raise ValueError("run was not created by the LangGraph backend")
        request = task_request_from_json(
            json.loads((directory / "request.json").read_text(encoding="utf-8"))
        )
        if (metadata.get("architecture_mode") != self.architecture_mode
                or metadata.get("result_storage") != RESULT_STORAGE):
            raise ValueError("this release resumes Manager V2 runs with LangGraph result storage")
        observed_base = self._repository_base(request)
        if observed_base != metadata["base_commit"]:
            raise ValueError("repository base commit does not match the persisted run")
        runtime = self._runtime(request, str(metadata["run_id"]), directory)
        try:
            return await self._drive(
                runtime,
                None,
                approval_decision=approval_decision,
                approval_comment=approval_comment,
            )
        except Exception:
            runtime.artifacts.close()
            raise

    async def _drive(
        self, runtime, graph_input, *, approval_decision=None, approval_comment=None,
    ):
        config = getattr(getattr(runtime.worker_factory, "config", None), "codeswarm", None)
        settings = TraceSettings(**getattr(config, "tracing", {}))
        # CodeSwarm exports its own redacted spans. Suppress LangGraph's automatic
        # LangSmith runs so graph state and model inputs are not sent a second time.
        with tracing_context(enabled=False), execution(
            runtime.run_id, runtime.run_directory, runtime.artifacts, settings,
            **{"codeswarm.resumed": graph_input is None,
               "codeswarm.checkpoint": str(runtime.run_directory / "checkpoints.sqlite")},
        ) as observed:
            outcome = await self._drive_impl(runtime, graph_input, approval_decision=approval_decision,
                                             approval_comment=approval_comment)
            observed.attrs(**{"codeswarm.status": outcome.status.value})
            if outcome.status in {RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BUDGET_EXCEEDED}:
                observed.error(RuntimeError(outcome.failure_reason or outcome.status.value))
            return outcome

    async def _drive_impl(
        self,
        runtime: RunContext,
        graph_input: CodeSwarmState | None,
        *,
        approval_decision: str | None = None,
        approval_comment: str | None = None,
    ) -> TaskOutcome:
        database = runtime.run_directory / "checkpoints.sqlite"
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.conn.execute("PRAGMA journal_mode=WAL")
            await saver.conn.execute("PRAGMA busy_timeout=30000")
            await saver.setup()
            from codeagent.manager_graph import build_manager_graph

            graph = build_manager_graph(runtime, saver, manager_model=self.route_model)
            config = {
                "configurable": {"thread_id": runtime.run_id},
                # Framework slots include waiting parent/response/summary tasks.
                # Actual Worker admission remains bounded by dispatch and budget.
                "max_concurrency": max(8, runtime.request.budget.max_parallel * 4),
                "recursion_limit": 200,
            }
            if graph_input is None:
                snapshot = await graph.aget_state(config)
                current_operation().attrs(**{"codeswarm.resume_checkpoint_id":
                    (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")})
                if not snapshot.values:
                    raise ValueError("no LangGraph checkpoint exists for this run")
                persisted = dict(snapshot.values)
                runtime.tracer.store.validate_cursor(
                    int(persisted.get("event_sequence", 0)),
                    persisted.get("event_checksum"),
                )
                runtime.budget = BudgetTracker.restore(
                    runtime.request.budget, persisted.get("budget_snapshot")
                )
                pending_interrupts = [
                    item
                    for task in snapshot.tasks
                    for item in getattr(task, "interrupts", ())
                ]
                if pending_interrupts:
                    if approval_decision is None:
                        return self._paused_outcome(runtime, persisted, pending_interrupts[0].value)
                    interrupt_payload = pending_interrupts[0].value
                    allowed_actions = {
                        str(item)
                        for item in (
                            interrupt_payload.get("allowed_actions", ())
                            if isinstance(interrupt_payload, dict)
                            else ()
                        )
                    }
                    if approval_decision not in allowed_actions:
                        # Validate before invoking the graph.  A rejected resume command
                        # must not be persisted as a failed task input and poison the next
                        # legitimate resume attempt for this thread.
                        raise ValueError(
                            f"approval action is not allowed: {approval_decision}"
                        )
                    graph_input = Command(
                        resume={
                            "action": approval_decision,
                            "comment": approval_comment or "",
                        }
                    )
                else:
                    if approval_decision is not None:
                        raise ValueError("run is not waiting for human approval")
                    graph_input = None

            try:
                remaining = await runtime.budget.remaining_wall_time()
                result = await runtime.cancellation.run(
                    asyncio.wait_for(graph.ainvoke(graph_input, config=config), remaining)
                )
            except asyncio.TimeoutError:
                runtime.cancellation.cancel("run wall-clock budget exhausted")
                exc = RunCancelled("run wall-clock budget exhausted")
                runtime.tracer.emit_once(
                    f"{runtime.run_id}:cancelled", "run_cancelled", reason=exc.reason
                )
                outcome = TaskOutcome(
                    run_id=runtime.run_id,
                    status=RunStatus.CANCELLED,
                    mode="cancelled",
                    run_directory=str(runtime.run_directory),
                    usage=runtime.budget.usage,
                    failure_reason=exc.reason,
                    checkpoint_thread_id=runtime.run_id,
                )
                await runtime.workspace_manager.cleanup()
                runtime.artifacts.close()
                return outcome
            except RunCancelled as exc:
                runtime.tracer.emit_once(
                    f"{runtime.run_id}:cancelled", "run_cancelled", reason=exc.reason
                )
                outcome = TaskOutcome(
                    run_id=runtime.run_id,
                    status=RunStatus.CANCELLED,
                    mode="cancelled",
                    run_directory=str(runtime.run_directory),
                    usage=runtime.budget.usage,
                    failure_reason=exc.reason,
                    checkpoint_thread_id=runtime.run_id,
                )
                await runtime.workspace_manager.cleanup()
                runtime.artifacts.close()
                return outcome
            interrupts = result.get("__interrupt__", ())
            if interrupts:
                outcome = self._paused_outcome(runtime, result, interrupts[0].value)
                runtime.tracer.emit_once(
                    f"{runtime.run_id}:paused:{result.get('replan_round', 0)}",
                    "run_paused",
                    reason=outcome.pending_approval.reason if outcome.pending_approval else "",
                )
                self._write_compatibility_state(runtime, result, outcome)
                runtime.artifacts.close()
                return outcome

            outcome = self._outcome_from_state(runtime, result)
            runtime.tracer.emit_once(
                f"{runtime.run_id}:finished",
                "run_finished",
                status=outcome.status.value,
                integrated_task=outcome.selected_integrated_task_id,
                orchestration_backend="langgraph",
            )
            self._write_compatibility_state(runtime, result, outcome)
            await runtime.workspace_manager.cleanup()
            runtime.artifacts.close()
            return outcome

    def _runtime(
        self, request: TaskRequest, run_id: str, run_directory: Path,
    ) -> RunContext:
        artifacts = ArtifactStore(run_directory / "artifacts")
        results = StateResults()
        tracer = EventTracer(run_id, run_directory / "trace.jsonl")
        budget = BudgetTracker(request.budget)
        if self.workspace_manager_factory:
            workspace = self.workspace_manager_factory(request, run_directory)
        else:
            workspace = GitWorkspaceManager(
                request.repository,
                run_directory / "worktrees",
                request.base_commit,
                keep=self.keep_workspaces,
            )
        return RunContext(
            request=request,
            run_id=run_id,
            run_directory=run_directory,
            worker_factory=self.worker_factory,
            artifacts=artifacts,
            results=results,
            tracer=tracer,
            budget=budget,
            workspace_manager=workspace,
            cancellation=CancellationToken(run_directory / "cancel.json"),
        )

    @staticmethod
    def _repository_base(request: TaskRequest) -> str:
        result = subprocess.run(
            ["git", "rev-parse", request.base_commit or "HEAD"],
            cwd=request.repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else request.base_commit or "unknown"

    def _paused_outcome(
        self, runtime: RunContext, state: dict[str, Any], payload: dict[str, Any]
    ) -> TaskOutcome:
        request = ApprovalRequest(
            reason=str(payload["reason"]),
            allowed_actions=tuple(
                ApprovalAction(item) for item in payload.get("allowed_actions", ())
            ),
            hard_gates_passed=bool(payload.get("hard_gates_passed", False)),
            replan_round=int(payload.get("replan_round", 0)),
        )
        return TaskOutcome(
            run_id=runtime.run_id,
            status=RunStatus.AWAITING_APPROVAL,
            mode=str(state.get("mode", "multi-agent")),
            run_directory=str(runtime.run_directory),
            selected_integrated_task_id=state.get("selected_integrated_task_id"),
            selected_patch_artifact_id=state.get("selected_patch_artifact_id"),
            usage=runtime.budget.usage,
            failure_reason=None,
            pending_approval=request,
            checkpoint_thread_id=runtime.run_id,
        )

    def _outcome_from_state(
        self, runtime: RunContext, state: CodeSwarmState
    ) -> TaskOutcome:
        decisions = tuple(
            review_decision_from_json(item)
            for item in state.get("review_decisions", ())
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for ref in (state.get("results") or state.get("result_refs", {})).values()
                for evidence_id in (ref.get("evidence_ids") or [e["id"] for e in ref.get("evidence", [])])
            )
        )
        return TaskOutcome(
            run_id=runtime.run_id,
            status=RunStatus(state.get("terminal_status", RunStatus.FAILED.value)),
            mode=state.get("mode", "multi-agent"),
            run_directory=str(runtime.run_directory),
            selected_integrated_task_id=state.get("selected_integrated_task_id"),
            selected_patch_artifact_id=state.get("selected_patch_artifact_id"),
            evidence_ids=evidence_ids,
            review_decisions=decisions,
            usage=runtime.budget.usage,
            failure_reason=state.get("failure_reason"),
            checkpoint_thread_id=runtime.run_id,
        )

    def _write_compatibility_state(
        self,
        runtime: RunContext,
        state: dict[str, Any],
        outcome: TaskOutcome,
    ) -> None:
        specs: dict[str, dict[str, Any]] = {}
        specs.update(state.get("task_specs", {}))
        specs.update({item["id"]: item for item in state.get("developer_tasks", [])})
        specs.update(state.get("closing_tasks", {}))
        nodes = []
        for task_id, raw in specs.items():
            latest = _latest_ref(state, task_id)
            status = (
                TaskStatus.SUCCEEDED.value
                if latest and latest["status"] == WorkerStatus.SUCCEEDED.value
                else TaskStatus.FAILED.value
                if latest
                else TaskStatus.PENDING.value
            )
            nodes.append({**raw, "status": status, "attempt": int(latest["attempt"]) if latest else 0})
        graph_snapshot = {
            "version": len(state.get("results") or state.get("result_refs", {})),
            "orchestration_backend": LANGGRAPH_BACKEND_VERSION,
            "nodes": nodes,
        }
        (runtime.run_directory / "task_graph.json").write_text(
            json.dumps(graph_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if outcome.status != RunStatus.AWAITING_APPROVAL:
            (runtime.run_directory / "final_outcome.json").write_text(
                json.dumps(to_jsonable(outcome), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        checkpoint = Checkpoint(
            schema_version=2,
            run_id=runtime.run_id,
            event_sequence=runtime.tracer.store.sequence,
            event_checksum=runtime.tracer.store.last_checksum,
            repository_base=str(state.get("base_commit", "unknown")),
            graph=graph_snapshot,
            budget=state.get("budget_snapshot", {}),
            artifact_ids=tuple(record.id for record in runtime.artifacts.records()),
        )
        CheckpointStore(runtime.run_directory / "checkpoint.json").save(checkpoint)
