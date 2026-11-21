"""Worker contracts and an adapter that gives every role a private Trae Agent runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from codeagent.artifacts import ArtifactStore
from codeagent.cancellation import CancellationToken, RunCancelled
from codeagent.context import ContextEngine, ContextSettings
from codeagent.context.dense import EmbeddingProvider
from codeagent.context.lifecycle import (
    complete_response,
    message_json,
    response_from_json,
    response_json,
)
from codeagent.integration_tool import PatchApplyTool
from codeagent.mcp_config import bind_mcp_configs
from codeagent.protocol import (
    Evidence,
    ReviewDecision,
    TaskNode,
    TaskRequest,
    Usage,
    VerificationRecord,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
    to_jsonable,
)
from codeagent.roles import RoleSpec
from codeagent.routing import RoutingControl, RoutingRequestTool
from codeagent.runtime.completion import (
    CompletionGate,
    changed_paths,
    is_test_command,
    parse_review_decisions,
    workspace_patch,
    workspace_version,
)
from codeagent.runtime.recovery import ModelRecovery, tool_feedback
from codeagent.runtime.state import (
    Phase,
    RuntimeBudgetExceeded,
    RuntimeController,
    RuntimeDeadlineExceeded,
    RuntimeLimits,
    StopReason,
)
from codeagent.runtime.tool_policy import ToolPolicy, read_tool_name
from codeagent.safety.execution import DockerBackend, GuardedBashTool, WindowsGuardedBackend
from codeagent.safety.sandbox_profile import SandboxProfile
from codeagent.telemetry import current_operation, span

ROLE_STEP_LIMITS = {
    WorkerRole.MANAGER: 16,
    WorkerRole.WORKER: 16,
    WorkerRole.AUDITOR: 2,
    WorkerRole.EXPLORER: 3,
    WorkerRole.TESTER: 3,
    WorkerRole.REVIEWER: 2,
    WorkerRole.SECURITY: 2,
    WorkerRole.INTEGRATOR: 10,
    WorkerRole.DEVELOPER: 16,
}


class DeveloperReadGuard:
    """Bound repeated repository exploration between successful code edits."""

    def __init__(self, limit: int = 8) -> None:
        self.limit = limit
        self.seen: set[str] = set()
        self.queries = 0

    def claim(self, arguments: dict) -> str | None:
        key = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        query = arguments.get("operation") == "query"
        if key in self.seen and not query:
            return "This repository_read was already observed. Use the earlier result and edit the assigned files."
        if len(self.seen) + self.queries >= self.limit:
            return (
                f"Repository exploration limit ({self.limit}) reached before an edit. "
                "Use the supplied code and observations to implement the assigned change."
            )
        if query:
            self.queries += 1  # Explicit recall may be needed again after consumption.
        else:
            self.seen.add(key)
        return None

    def after_tool(self, name: str, success: bool, arguments: dict | None = None) -> None:
        normalized = name.lower().replace("_", "")
        if not success and normalized == "repositoryread" and arguments is not None:
            self.seen.discard(json.dumps(
                arguments, sort_keys=True, ensure_ascii=False, default=str
            ))
        if success and normalized in {
            "strreplacebasededittool", "jsonedittool",
        }:
            self.seen.clear()
            self.queries = 0


def security_rejection_reasons(
    decisions: tuple[ReviewDecision, ...],
) -> tuple[str, ...]:
    """Return fail-closed reasons for any component Security did not accept."""
    return tuple(
        reason
        for item in decisions
        if item.verdict.value != "accept"
        for reason in (item.reasons or (f"security verdict: {item.verdict.value}",))
    )


@dataclass(frozen=True)
class WorkerContext:
    run_id: str
    request: TaskRequest
    node: TaskNode
    role: RoleSpec
    workspace: Path
    dependency_results: tuple[WorkerResult, ...]
    shared_evidence: tuple[Evidence, ...]
    artifacts: ArtifactStore
    run_directory: Path
    deadline_at: float | None = None
    cancellation: CancellationToken | None = None
    routing_control: RoutingControl | None = None
    durable_context: bool = False
    context_state: dict | None = None
    remaining_tokens: int | None = None


class Worker(Protocol):
    worker_id: str

    async def run(self, context: WorkerContext) -> WorkerResult: ...


class WorkerFactory(Protocol):
    def create(self, node: TaskNode, role: RoleSpec) -> Worker: ...


class RoleToolExecutor:
    """Validate and schedule tool calls before any delegate side effect."""

    def __init__(
        self, delegate, role: RoleSpec, workspace: Path, observer=None,
        controller: RuntimeController | None = None, on_result=None,
        write_scope: tuple[str, ...] = (), additional_allowed: tuple[str, ...] = (),
        cancellation: CancellationToken | None = None,
        read_guard: DeveloperReadGuard | None = None,
    ) -> None:
        self.delegate = delegate
        self.role = role
        self.workspace = workspace.resolve()
        self.observer = observer
        self.controller = controller
        self.on_result = on_result
        self.cancellation = cancellation
        self.read_guard = read_guard
        available = list(delegate.tools.values()) if hasattr(delegate, "tools") else []
        self.policy = ToolPolicy(
            role, self.workspace, available, write_scope,
            additional_allowed=additional_allowed,
        )

    async def close_tools(self):
        return await self.delegate.close_tools()

    async def execute_tool_call(self, tool_call):
        with span(f"tool.{tool_call.name}", "TOOL", **{"tool.name": tool_call.name,
                  "codeswarm.tool_call_id": tool_call.call_id}) as observed:
            origin = None
            if observed.session:
                origin = observed.session.origins.get(tool_call.call_id)
                if origin and origin.active:
                    observed.attrs(**{"codeswarm.origin_model_span_id": f"{origin.active.get_span_context().span_id:016x}",
                                      "codeswarm.origin_response_ref": origin.output_ref})
            observed.attrs(**{"codeswarm.arguments_ref": origin.output_ref if origin else observed.body("tool-arguments", tool_call.arguments)})
            result = await self._execute_tool_call(tool_call)
            success = getattr(result, "success", None)
            observed.attrs(**{"codeswarm.success": success})
            if success is False:
                observed.error(RuntimeError(getattr(result, "error", None) or "tool failed"))
            return result

    async def _execute_tool_call(self, tool_call):
        try:
            if self.cancellation:
                self.cancellation.raise_if_cancelled()
            timeout = self.controller.before_tool(tool_call.name, tool_call.call_id) \
                if self.controller else None
        except (RuntimeDeadlineExceeded, RuntimeBudgetExceeded) as exc:
            if self.cancellation:
                self.cancellation.cancel(str(exc))
            raise RunCancelled(str(exc)) from exc
        decision = self.policy.validate(tool_call)
        read_denial = None
        if decision.allowed and self.read_guard and read_tool_name(tool_call.name) == "repositoryread":
            read_denial = self.read_guard.claim(tool_call.arguments)
        current_operation().attrs(**{"codeswarm.executed": bool(decision.allowed and not read_denial)})
        if not decision.allowed or read_denial:
            from trae_agent.tools.base import ToolResult

            result = ToolResult(
                call_id=tool_call.call_id, name=tool_call.name, success=False,
                error=(f"READ_GUARD: {read_denial}" if read_denial else
                       f"POLICY/{decision.code}: {decision.reason}"), id=tool_call.id,
            )
            if self.controller:
                self.controller.emit("tool_policy_denied", tool=tool_call.name,
                                     call_id=tool_call.call_id,
                                     code="REPEATED_READ" if read_denial else decision.code,
                                     rule_id=decision.rule_id, risk=decision.risk)
        else:
            try:
                normalized_name = tool_call.name.lower().replace("_", "")
                raw_path = tool_call.arguments.get("path")
                if (
                    normalized_name in {"strreplacebasededittool", "jsonedittool"}
                    and isinstance(raw_path, str)
                    and not Path(raw_path).is_absolute()
                ):
                    tool_call.arguments["path"] = str((self.workspace / raw_path).resolve())
                operation = self.delegate.execute_tool_call(tool_call)
                if self.cancellation:
                    operation = self.cancellation.run(operation)
                result = await asyncio.wait_for(operation, timeout) if timeout else await operation
                if self.read_guard:
                    self.read_guard.after_tool(
                        read_tool_name(tool_call.name), result.success, tool_call.arguments
                    )
            except asyncio.TimeoutError as exc:
                reason = "tool request timed out"
                if self.cancellation:
                    self.cancellation.cancel(reason)
                raise RunCancelled(reason) from exc
        if self.observer:
            result = self.observer(tool_call, result)
            engine = getattr(self.observer, "__self__", None)
            raw = getattr(getattr(engine, "reducer", None), "by_call_id", {}).get(tool_call.call_id)
            current_operation().output_ref = raw.artifact_id if raw else None
            current_operation().event("tool_observation", raw_ref=raw.artifact_id if raw else None,
                                      preview=(result.result or result.error or "")[:600])
        if self.on_result:
            self.on_result(tool_call, result)
        if self.controller:
            self.controller.after_tool(tool_call.name, tool_call.call_id,
                                       result.success, result.error)
        return result

    async def parallel_tool_call(self, tool_calls):
        results = []
        for batch in self.policy.partition(tool_calls):
            if len(batch) == 1:
                results.append(await self.execute_tool_call(batch[0]))
            else:
                results.extend(await asyncio.gather(
                    *(self.execute_tool_call(call) for call in batch)
                ))
        return results

    async def sequential_tool_call(self, tool_calls):
        return [await self.execute_tool_call(call) for call in tool_calls]


class TraeWorkerFactory:
    def __init__(
        self,
        config,
        context_settings: ContextSettings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        sandbox_image: str | None = None,
        sandbox_profile: SandboxProfile | None = None,
        execution_backend: WindowsGuardedBackend | DockerBackend | None = None,
    ) -> None:
        self.config = config
        self.context_settings = context_settings
        self.embedding_provider = embedding_provider
        self.sandbox_image = sandbox_image
        self.sandbox_profile = sandbox_profile
        self.execution_backend = execution_backend

    def create(self, node: TaskNode, role: RoleSpec) -> Worker:
        return TraeAgentWorker(
            copy.deepcopy(self.config), role, self.context_settings, self.embedding_provider,
            self.sandbox_image, self.sandbox_profile,
            self.execution_backend,
        )


class TraeAgentWorker:
    def __init__(
        self,
        config,
        role: RoleSpec,
        context_settings: ContextSettings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        sandbox_image: str | None = None,
        sandbox_profile: SandboxProfile | None = None,
        execution_backend: WindowsGuardedBackend | DockerBackend | None = None,
    ) -> None:
        self.config = config
        self.role = role
        self.context_settings = context_settings
        self.embedding_provider = embedding_provider
        self.sandbox_image = sandbox_image
        self.sandbox_profile = sandbox_profile
        self.execution_backend = execution_backend
        self.worker_id = f"{role.role.value}-{uuid.uuid4().hex[:10]}"

    async def run(self, context: WorkerContext) -> WorkerResult:
        from codeagent.context.artifact_tool import ArtifactReadTool
        from codeagent.repository_tool import RepositoryReadTool
        from codeagent.trajectory import ReferenceTrajectoryRecorder
        from trae_agent.agent.trae_agent import TraeAgent
        from trae_agent.tools import tools_registry
        from trae_agent.tools.base import ToolCall, ToolExecutor
        from trae_agent.tools.docker_tool_executor import DockerToolExecutor

        agent_config = copy.deepcopy(self.config.trae_agent)
        if agent_config is None:
            raise ValueError("Trae agent configuration is required")
        role_step_limit = ROLE_STEP_LIMITS[context.role.role]
        if (
            context.role.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER}
            and not context.node.metadata.get("single_agent")
        ):
            role_step_limit = min(role_step_limit, 10)
        agent_config.max_steps = min(agent_config.max_steps, role_step_limit)
        provider_retries = agent_config.model.max_retries
        agent_config.model.max_retries = 0  # CodeSwarm owns classified retries.
        controller = RuntimeController(
            run_id=context.run_id, task_id=context.node.id, worker_id=self.worker_id,
            trace_path=context.run_directory / "workers" / self.worker_id / "runtime.jsonl",
            limits=RuntimeLimits(
                deadline_at=context.deadline_at or (
                    time.monotonic() + context.request.budget.max_wall_time_seconds
                ),
                max_tool_calls=context.request.budget.max_tool_calls,
                max_tokens=context.remaining_tokens if context.remaining_tokens is not None else context.request.budget.max_total_tokens,
            ),
        )
        # Keep model-call recovery inside the Worker.  LangGraph deliberately does
        # not retry Worker nodes because that would duplicate tools, artifacts, and
        # accounting.  Even when the provider SDK is configured for zero retries,
        # transient transport failures receive bounded CodeSwarm-owned recovery.
        recovery = ModelRecovery(max_attempts=min(4, max(3, provider_retries + 1)))
        gate = CompletionGate(
            role=context.role.role, workspace=context.workspace,
            metadata={**context.node.metadata, "write_scope": context.node.write_scope},
            test_commands=context.request.test_commands,
        )
        context_engine = await asyncio.to_thread(
            ContextEngine,
            workspace=context.workspace,
            trace_root=context.run_directory / "context" / self.worker_id,
            artifacts=context.artifacts,
            role=context.role.role.value,
            issue=context.request.issue,
            goal=context.node.goal,
            acceptance=context.request.acceptance_criteria,
            test_commands=context.request.test_commands,
            dependency_results=context.dependency_results,
            shared_evidence=context.shared_evidence,
            model=agent_config.model.model,
            settings=replace(
                self.context_settings or ContextSettings(),
                output_reserve_tokens=max(
                    (self.context_settings or ContextSettings()).output_reserve_tokens,
                    agent_config.model.get_max_tokens_param(),
                ),
            ),
            embedding_provider=self.embedding_provider,
        )
        if context.context_state:
            context_engine.restore_state(context.context_state)
        routing_tool = None
        if context.durable_context:
            async def checkpoint_summary(snapshot):
                from langgraph.func import task
                async def save(value):
                    return value
                return await task(name="codeswarm_summary_state")(save)(snapshot)
            context_engine.summary_checkpoint = checkpoint_summary

        def counts():
            return {key: getattr(controller, key) for key in (
                "model_attempts", "model_calls", "input_tokens", "output_tokens")}

        def restore_counts(before, delta):
            for key, value in delta.items():
                setattr(controller, key, max(getattr(controller, key), before[key] + value))

        async def summary_request(prompt, attempt):
            from trae_agent.utils.llm_clients.llm_basics import LLMMessage
            model = copy.deepcopy(agent_config.model)
            model.max_tokens = min(context_engine.settings.summary_max_tokens, model.get_max_tokens_param())
            if model.max_completion_tokens is not None:
                model.max_completion_tokens = model.max_tokens
            estimate = (context_engine.counter.count(prompt) + model.max_tokens
                        + int(context_engine.lifecycle.available * context_engine.settings.summary_trigger_ratio)
                        + agent_config.model.get_max_tokens_param())
            if controller.limits.max_tokens is not None and (
                controller.input_tokens + controller.output_tokens + estimate > controller.limits.max_tokens
            ):
                raise ValueError("summary skipped to reserve remaining run tokens for the normal request")
            before = counts()

            async def send():
                response = await ModelRecovery(max_attempts=1).call(lambda: agent.llm_client.achat(
                    [LLMMessage(role="system", content="Return a concise structured progress summary as JSON."),
                     LLMMessage(role="user", content=prompt)], model, tools=[], reuse_history=False,
                ), controller)
                record = context.artifacts.put_text("summary-response", json.dumps(response_json(response), ensure_ascii=False), ".json")
                return {"response": record.id, "delta": {k: counts()[k] - before[k] for k in before}}

            prior_failure = controller.failure_reason
            try:
                if context.durable_context:
                    from langgraph.func import task
                    saved = await task(name="codeswarm_summary_response")(send)()
                else:
                    saved = await send()
            except asyncio.TimeoutError:
                controller.remaining_seconds()  # The run deadline is still a hard stop.
                controller.failure_reason = prior_failure
                raise
            restore_counts(before, saved["delta"])
            response = response_from_json(json.loads(context.artifacts.read_text(saved["response"])))
            if not complete_response(response):
                raise ValueError("summary response was incomplete")
            controller.emit("summary_response", response_artifact=saved["response"], attempt=attempt)
            return response.content

        class RoleTraeAgent(TraeAgent):
            def get_system_prompt(inner_self) -> str:
                prompt = context.role.system_prompt
                if context.node.metadata.get("manager_action"):
                    prompt += "\n[Current action contract]\n" + json.dumps({
                        "task_id": context.node.id, "goal": context.node.goal,
                        "write_scope": context.node.write_scope,
                        **{key: context.node.metadata.get(key) for key in (
                            "input_version", "read_scope", "criteria_ids", "interface",
                            "review_target_task_ids", "phase",
                        )},
                    }, ensure_ascii=False)
                if context.node.metadata.get("manager_state"):
                    prompt += "\n[Persistent Manager task state]\n" + json.dumps(
                        context.node.metadata["manager_state"], ensure_ascii=False
                    )
                if context.node.metadata.get("acceptance_plan"):
                    prompt += "\n[Acceptance plan]\n" + json.dumps(
                        context.node.metadata["acceptance_plan"], ensure_ascii=False
                    )
                if context.node.metadata.get("phase") == "preaudit":
                    prompt += "\nSecurity preflight only: assess whether the patch is safe to test; final tests have not run yet.\n"
                if context.routing_control:
                    prompt += ("\nUse routing_request only for a concrete new dependency, independent task, "
                               "need for help or changed work boundary. Cite actual evidence. It requests "
                               "Manager judgment and preserves partial work; it does not finish the task. "
                               "Context pressure alone is not a reason to delegate: summarize/clean first.\n")
                return prompt

            def prepare_model_messages(inner_self, messages, execution):
                inner_self._context_execution = execution
                return list(messages)

            def reuse_model_history(inner_self) -> bool:
                return False

            def llm_indicates_task_completed(inner_self, llm_response):
                if super().llm_indicates_task_completed(llm_response):
                    return True
                implicit = (
                    not llm_response.tool_calls
                    and gate.accepts_evidence_backed_completion(llm_response.content)
                )
                if implicit:
                    inner_self._codeswarm_implicit_completion = True
                    controller.emit(
                        "implicit_completion_detected",
                        role=context.role.role.value,
                        reason="role deliverable satisfies deterministic evidence gate",
                    )
                return implicit

            async def model_chat(inner_self, messages, model_config, tools, reuse_history):
                before = counts()
                controller.transition(Phase.THINKING)

                async def normal_turn():
                    actual = await context_engine.prepare_messages(
                        inner_self.get_system_prompt(), inner_self._context_execution, messages, tools,
                        summary_request if context.durable_context else None,
                    )
                    if routing_tool:
                        routing_tool.summary_completed = context_engine.lifecycle.state["summary_version"] > 0
                    tools_tokens = sum(context_engine.counter.count(json.dumps(tool.json_definition(), ensure_ascii=False)) for tool in tools)
                    estimate = context_engine.lifecycle.cost(actual, tools_tokens) + model_config.get_max_tokens_param()
                    if controller.limits.max_tokens is not None and (
                        controller.input_tokens + controller.output_tokens + estimate > controller.limits.max_tokens
                    ):
                        raise RuntimeBudgetExceeded("remaining run tokens cannot fit the normal request and output reserve")
                    input_record = context.artifacts.put_text("model-input", json.dumps({
                        "messages": [message_json(message) for message in actual],
                        "tools": [tool.json_definition() for tool in tools], "model": model_config.model,
                    }, ensure_ascii=False), ".json")
                    response = await recovery.call(lambda: inner_self.llm_client.achat(
                        actual, model_config, tools, reuse_history=False), controller)
                    record = context.artifacts.put_text("model-response", json.dumps(response_json(response), ensure_ascii=False), ".json")
                    snapshot = context_engine.prospective_state(response)
                    snapshot["interaction_refs"] = [*snapshot.get("interaction_refs", []),
                                                     {"input": input_record.id, "response": record.id}]
                    return {"response": record.id, "context": snapshot,
                            "delta": {k: counts()[k] - before[k] for k in before}}

                try:
                    if context.durable_context:
                        from langgraph.func import task
                        operation = task(name="codeswarm_model_turn")(normal_turn)()
                    else:
                        operation = normal_turn()
                    saved = await context.cancellation.run(operation) if context.cancellation else await operation
                    restore_counts(before, saved["delta"])
                    if context.durable_context:
                        # Only the combined, framework-managed output can advance consumption.
                        context_engine.restore_state(saved["context"])
                    if routing_tool:
                        routing_tool.summary_completed = context_engine.lifecycle.state["summary_version"] > 0
                    return response_from_json(json.loads(context.artifacts.read_text(saved["response"])))
                except (RuntimeDeadlineExceeded, RuntimeBudgetExceeded, asyncio.TimeoutError) as exc:
                    if context.cancellation:
                        context.cancellation.cancel(str(exc))
                    raise RunCancelled(str(exc)) from exc

            async def _tool_call_handler(inner_self, tool_calls, step):
                if tool_calls and context.routing_control and context.routing_control.pause_requested:
                    from trae_agent.tools.base import ToolResult
                    from trae_agent.utils.llm_clients.llm_basics import LLMMessage
                    step.tool_calls = tool_calls
                    step.tool_results = [ToolResult(
                        call_id=call.call_id, name=call.name, id=call.id, success=False,
                        error="HANDOFF_NOT_STARTED: no new tool batch after reassignment request",
                    ) for call in tool_calls]
                    return [LLMMessage(role="user", tool_result=result) for result in step.tool_results]
                if not tool_calls:
                    from trae_agent.utils.llm_clients.llm_basics import LLMMessage

                    decision = gate.evaluate(
                        step.llm_response.content if step.llm_response else ""
                    )
                    return [LLMMessage(
                        role="user",
                        content=(
                            "The role deliverable was not accepted: "
                            + "; ".join(decision.reasons)
                            + ". Correct these specific issues before finishing. "
                            "For work-unit IDs, use only lowercase letters, digits, "
                            "hyphens or underscores."
                        ),
                    )]
                messages = await super()._tool_call_handler(tool_calls, step)
                if tool_calls:
                    controller.after_tool_batch(
                        any(not result.success for result in step.tool_results or [])
                    )
                return messages

            async def _run_llm_step(inner_self, step, messages, execution):
                def handoff():
                    from trae_agent.agent.agent_basics import AgentState
                    # COMPLETED ends the local loop; success stays false because
                    # yielding is not evidence that the assigned goal was solved.
                    execution.success = False
                    execution.agent_state = AgentState.COMPLETED
                    execution.stop_reason = StopReason.HANDOFF.value
                    execution.final_result = (
                        step.llm_response.content if step.llm_response else None
                    ) or self._last_model_content(execution) or "Partial observations returned for Manager handoff."
                    if step.tool_results:
                        execution.final_result += "\nObserved tool results:\n" + "\n".join(
                            f"{result.name}: success={result.success}; {(result.result or result.error or '')[:250]}"
                            for result in step.tool_results[-4:]
                        )

                if context.routing_control and context.routing_control.pause_requested:
                    handoff()
                    return messages
                messages = await super()._run_llm_step(step, messages, execution)
                if context.routing_control:
                    for result in step.tool_results or []:
                        if not result.success and not (result.error or "").startswith("HANDOFF_NOT_STARTED"):
                            context.routing_control.request(
                                context.node.id, str(context.node.metadata.get("input_version", "")),
                                "scope_violation" if "POLICY/PERMISSION_DENIED" in (result.error or "")
                                else "tool_failure", (result.error or "tool failed")[:1000],
                                [f"tool:{result.name}:{(result.error or '')[:300]}"],
                            )
                    if (context.routing_control.pause_requested and not execution.success
                            and not execution.cleanup_errors):
                        handoff()
                        return messages
                if (
                    not execution.success
                    and gate.accepts_post_tool_completion()
                ):
                    from trae_agent.agent.agent_basics import AgentState

                    execution.success = True
                    execution.agent_state = AgentState.COMPLETED
                    execution.stop_reason = StopReason.COMPLETED.value
                    execution.final_result = (
                        step.llm_response.content
                        if step.llm_response and step.llm_response.content
                        else "Developer patch accepted from fresh required-test evidence."
                    )
                    inner_self._codeswarm_implicit_completion = True
                    controller.emit(
                        "post_tool_evidence_completion",
                        role=context.role.role.value,
                        reason="business patch and fresh required tests satisfy the gate",
                    )
                return messages

            def reflect_on_result(inner_self, tool_results):
                return tool_feedback(tool_results)

            def _is_task_completed(inner_self, llm_response):
                controller.before_completion()
                decision = gate.evaluate(llm_response.content)
                if not decision.accepted:
                    controller.completion_rejected(decision.reasons)
                return decision.accepted

            def task_incomplete_message(inner_self):
                return (
                    "Completion rejected by the deterministic gate: "
                    + "; ".join(gate.last_decision.reasons)
                    + ". Address the missing evidence before calling task_done again."
                )

            async def execute_task(inner_self):
                try:
                    execution = await super().execute_task()
                except asyncio.CancelledError:
                    controller.finish(StopReason.CANCELLED)
                    raise
                except Exception:
                    controller.finish(controller.failure_reason or StopReason.ERROR)
                    raise
                if (
                    not execution.success
                    and execution.stop_reason == StopReason.MAX_STEPS.value
                    and gate.accepts_terminal_evidence_completion(
                        self._last_model_content(execution)
                    )
                ):
                    from trae_agent.agent.agent_basics import AgentState

                    execution.success = True
                    execution.agent_state = AgentState.COMPLETED
                    execution.stop_reason = StopReason.COMPLETED.value
                    execution.final_result = (
                        self._last_model_content(execution)
                        or "Role deliverable accepted from deterministic tool evidence."
                    )
                    inner_self._codeswarm_implicit_completion = True
                    controller.emit(
                        "terminal_evidence_completion",
                        role=context.role.role.value,
                        reason="step limit reached after evidence gate was satisfied",
                    )
                reason = controller.failure_reason or StopReason(
                    execution.stop_reason or "error"
                )
                controller.finish(reason)
                execution.stop_reason = reason.value
                return execution

        started = time.monotonic()
        agent_config.tools = []
        codeswarm_config = getattr(self.config, "codeswarm", None)
        role_allowlist = getattr(codeswarm_config, "mcp_role_allowlist", {}) or {}
        configured_for_role = set(role_allowlist.get(context.role.role.value, ()))
        globally_allowed = set(agent_config.allow_mcp_servers or ())
        agent_config.allow_mcp_servers = sorted(configured_for_role & globally_allowed)
        agent_config.mcp_servers_config = bind_mcp_configs(
            agent_config.mcp_servers_config, context.workspace, context.artifacts.root)
        for name in agent_config.allow_mcp_servers:
            if name in agent_config.mcp_servers_config:
                service = agent_config.mcp_servers_config[name]
                service.timeout = min(service.timeout or 60, controller.remaining_seconds())
        docker_config = None
        if isinstance(self.execution_backend, DockerBackend):
            docker_config = self.execution_backend.agent_config(context.workspace)
        elif self.sandbox_image:
            profile = self.sandbox_profile or SandboxProfile()
            profile.validate_image(self.sandbox_image)
            docker_config = {
                "image": self.sandbox_image,
                "workspace_dir": str(context.workspace),
                "security_options": profile.docker_options(),
            }
        agent = RoleTraeAgent(agent_config, docker_config=docker_config, docker_keep=False)
        agent._codeswarm_cancellation = context.cancellation
        agent._codeswarm_mcp_effects = dict(
            getattr(codeswarm_config, "mcp_tool_effects", {}) or {}
        )
        if agent_config.allow_mcp_servers:
            await agent.initialise_mcp()
            for server, reason in agent.mcp_errors.items():
                controller.emit("mcp_connection_failed", server=server, reason=reason)
            undeclared_effects = [
                f"{tool.server_name}.{tool.tool.name}"
                for tool in agent.mcp_tools
                if f"{tool.server_name}.{tool.tool.name}"
                not in agent._codeswarm_mcp_effects
            ]
            if undeclared_effects:
                await agent.cleanup_mcp_clients()
                raise ValueError(
                    "MCP tool effect is not declared: " + ", ".join(undeclared_effects)
                )
        provider = agent_config.model.model_provider.provider
        tools = []
        for tool_name in context.role.allowed_tools:
            if tool_name == "repository_read":
                tools.append(RepositoryReadTool(context.workspace, provider, query=context_engine.query_repository))
            elif tool_name == "artifact_read":
                tools.append(ArtifactReadTool(context.artifacts, provider))
            elif tool_name == "bash":
                worker_temp = (
                    Path(tempfile.gettempdir())
                    / "codeswarm-agent"
                    / context.run_id[:8]
                    / self.worker_id
                )
                if isinstance(self.execution_backend, WindowsGuardedBackend):
                    tools.append(GuardedBashTool(
                        self.execution_backend,
                        model_provider=provider,
                        cwd=context.workspace,
                        temp_dir=worker_temp,
                    ))
                else:
                    tools.append(tools_registry[tool_name](
                        model_provider=provider,
                        cwd=str(context.workspace),
                        # Windows has a legacy path-length limit in parts of pytest/tmp_path.
                        # Keep the randomized worker temp root short and outside nested worktrees.
                        temp_dir=str(worker_temp),
                    ))
            elif tool_name == "patch_apply":
                tools.append(PatchApplyTool(
                    context.workspace,
                    context.artifacts,
                    self._component_patch_artifacts(context),
                    provider,
                ))
            else:
                tools.append(tools_registry[tool_name](model_provider=provider))
        for mcp_tool in agent.mcp_tools:
            tools.append(mcp_tool)
        if context.routing_control:
            routing_tool = RoutingRequestTool(
                context.routing_control, context.node.id,
                str(context.node.metadata.get("input_version", "")), provider,
            )
            tools.append(routing_tool)
        def record_tool(call, result):
            observation = context_engine.reducer.by_call_id.get(call.call_id)
            gate.record(call, result, observation.artifact_id if observation else None)

        base_executor = ToolExecutor(tools)
        if agent.docker_manager:
            base_executor = DockerToolExecutor(
                original_executor=base_executor,
                docker_manager=agent.docker_manager,
                docker_tools=["bash", "str_replace_based_edit_tool", "json_edit_tool"],
                host_workspace_dir=str(context.workspace),
                container_workspace_dir=agent.docker_manager.container_workspace,
            )
        role_executor = RoleToolExecutor(
            base_executor, context.role, context.workspace,
            context_engine.observe, controller, record_tool, context.node.write_scope,
            (*tuple(tool.name for tool in agent.mcp_tools),
             *(("routing_request",) if context.routing_control else ())),
            cancellation=context.cancellation,
            read_guard=(DeveloperReadGuard() if context.role.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER}
                        else None),
        )
        agent.replace_tools(tools, cast(ToolExecutor, role_executor))
        trajectory_path = (
            context.run_directory
            / "workers"
            / self.worker_id
            / f"attempt-{context.node.attempt}.json"
        )
        from trae_agent.utils.request_observer import observer as request_observer
        from trae_agent.utils.trajectory_recorder import TrajectoryRecorder
        recorder = ReferenceTrajectoryRecorder if context.durable_context or request_observer.get() else TrajectoryRecorder
        agent.set_trajectory_recorder(recorder(str(trajectory_path)))
        agent.new_task(
            context.node.goal,
            {
                "project_path": str(context.workspace),
                "issue": self._build_worker_prompt(context),
            },
            # The worker already installed the exact role tool set. Passing an explicit
            # empty list prevents Trae from restoring its broad defaults for tool-free
            # roles such as Reviewer.
            tool_names=[],
        )
        verify_integrity = (
            context.role.role == WorkerRole.TESTER
            and context.node.metadata.get("phase") == "verify"
        )
        fingerprint_before = (
            self._tracked_workspace_fingerprint(context.workspace) if verify_integrity else None
        )
        deterministic_tester = (
            context.role.role == WorkerRole.TESTER
            and bool(context.request.test_commands)
        )
        deterministic_integrator = context.role.role == WorkerRole.INTEGRATOR
        if deterministic_integrator:
            from trae_agent.agent.agent_basics import (
                AgentExecution,
                AgentState,
                AgentStep,
                AgentStepState,
            )

            controller.transition(Phase.THINKING)
            component_ids = tuple(context.node.metadata.get("component_task_ids", ()))
            controller.emit(
                "deterministic_integrator_started",
                component_task_ids=component_ids,
            )
            calls = [
                ToolCall(
                    name="patch_apply",
                    call_id=f"integrate-{uuid.uuid4().hex}",
                    arguments={"task_id": task_id},
                )
                for task_id in component_ids
            ]
            integration_results = await role_executor.sequential_tool_call(calls)
            controller.after_tool_batch(any(not item.success for item in integration_results))
            decision = gate.evaluate("")
            summary = "; ".join(
                f"{task_id}: {'applied' if result.success else 'failed'}"
                for task_id, result in zip(component_ids, integration_results, strict=True)
            )
            if decision.reasons:
                summary += "; gate: " + "; ".join(decision.reasons)
            execution = AgentExecution(
                task=context.node.goal,
                steps=[AgentStep(
                    step_number=1,
                    state=(
                        AgentStepState.COMPLETED if decision.accepted else AgentStepState.ERROR
                    ),
                    tool_calls=calls,
                    tool_results=integration_results,
                )],
                final_result=summary,
                success=decision.accepted,
                agent_state=(AgentState.COMPLETED if decision.accepted else AgentState.ERROR),
                stop_reason=(
                    StopReason.COMPLETED.value if decision.accepted else StopReason.ERROR.value
                ),
            )
            controller.emit(
                "deterministic_integrator_finished",
                accepted=decision.accepted,
                reasons=decision.reasons,
            )
            controller.finish(StopReason.COMPLETED if decision.accepted else StopReason.ERROR)
            cleanup_errors = []
            try:
                await role_executor.close_tools()
            except Exception as exc:
                cleanup_errors.append(f"tools: {exc}")
            try:
                await agent.cleanup_mcp_clients()
            except Exception as exc:
                cleanup_errors.append(f"mcp: {exc}")
            execution.cleanup_errors = tuple(cleanup_errors)
            if cleanup_errors and execution.success:
                execution.success = False
                execution.agent_state = AgentState.ERROR
                execution.stop_reason = StopReason.RESOURCE_CLEANUP_ERROR.value
        elif deterministic_tester:
            from trae_agent.agent.agent_basics import (
                AgentExecution,
                AgentState,
                AgentStep,
                AgentStepState,
            )

            controller.transition(Phase.THINKING)
            controller.emit(
                "deterministic_tester_started",
                commands=context.request.test_commands,
                phase=context.node.metadata.get("phase"),
            )
            calls = [
                ToolCall(
                    name="bash",
                    call_id=f"test-{uuid.uuid4().hex}",
                    arguments={"command": command, "restart": False},
                )
                for command in context.request.test_commands
            ]
            test_results = await role_executor.sequential_tool_call(calls)
            controller.after_tool_batch(any(not item.success for item in test_results))
            decision = gate.evaluate("")
            test_summary = "; ".join(
                f"{command}: {'passed' if result.success else 'failed'}"
                for command, result in zip(
                    context.request.test_commands, test_results, strict=True
                )
            )
            if decision.reasons:
                test_summary += "; gate: " + "; ".join(decision.reasons)
            execution = AgentExecution(
                task=context.node.goal,
                steps=[AgentStep(
                    step_number=1,
                    state=(
                        AgentStepState.COMPLETED
                        if decision.accepted else AgentStepState.ERROR
                    ),
                    tool_calls=calls,
                    tool_results=test_results,
                )],
                final_result=test_summary,
                success=decision.accepted,
                agent_state=(
                    AgentState.COMPLETED if decision.accepted else AgentState.ERROR
                ),
                stop_reason=(
                    StopReason.COMPLETED.value if decision.accepted else StopReason.ERROR.value
                ),
            )
            controller.emit(
                "deterministic_tester_finished",
                accepted=decision.accepted,
                reasons=decision.reasons,
            )
            controller.finish(
                StopReason.COMPLETED if decision.accepted else StopReason.ERROR
            )
            cleanup_errors = []
            try:
                await role_executor.close_tools()
            except Exception as exc:
                cleanup_errors.append(f"tools: {exc}")
            try:
                await agent.cleanup_mcp_clients()
            except Exception as exc:
                cleanup_errors.append(f"mcp: {exc}")
            execution.cleanup_errors = tuple(cleanup_errors)
            if cleanup_errors and execution.success:
                execution.success = False
                execution.agent_state = AgentState.ERROR
                execution.stop_reason = StopReason.RESOURCE_CLEANUP_ERROR.value
        else:
            # Only the synchronous provider request runs in a thread. The async Agent loop and
            # tool lifecycle remain cancellable in the scheduler event loop.
            execution = await agent.execute_task()
        summary = execution.final_result or self._last_model_content(execution) or ""
        if execution.cleanup_errors:
            summary = (
                summary + "\nResource cleanup failed: "
                + "; ".join(execution.cleanup_errors)
            ).strip()
        evidence, artifact_ids, verifications = self._collect_tool_evidence(
            context, execution, context_engine, gate
        )
        changed_files: tuple[str, ...] = ()
        if (context.role.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER, WorkerRole.INTEGRATOR}
                and not context.role.read_only):
            patch = self._get_workspace_patch(context.workspace)
            if patch.strip():
                record = context.artifacts.put_text("patch", patch, ".patch")
                artifact_ids.append(record.id)
                changed_files = self._changed_files(patch)
                evidence.append(
                    Evidence(
                        id=f"evidence-{uuid.uuid4().hex}",
                        producer=self.worker_id,
                        task_id=context.node.id,
                        kind="patch",
                        claim=f"Component modifies {len(changed_files)} file(s).",
                        artifact_id=record.id,
                        base_commit=context.request.base_commit,
                    )
                )
        decisions = (
            self._parse_review_decisions(summary)
            if self.role.role in {WorkerRole.SECURITY, WorkerRole.REVIEWER, WorkerRole.AUDITOR}
            else ()
        )
        worker_usage = Usage(
            input_tokens=controller.input_tokens,
            output_tokens=controller.output_tokens,
            model_calls=controller.model_attempts,
            tool_calls=controller.tool_calls,
            wall_time_seconds=time.monotonic() - started,
        )
        status = WorkerStatus.SUCCEEDED if execution.success else WorkerStatus.FAILED
        handoff_ready = bool(execution.stop_reason == StopReason.HANDOFF.value and not execution.cleanup_errors)
        if handoff_ready:
            status = WorkerStatus.BLOCKED
        unresolved: tuple[str, ...] = () if execution.success else (
            summary or execution.stop_reason or "worker failed",
        )
        security_reasons = security_rejection_reasons(decisions)
        if context.role.role == WorkerRole.SECURITY and security_reasons:
            status = WorkerStatus.FAILED
            unresolved = security_reasons
        if verify_integrity:
            fingerprint_after = self._tracked_workspace_fingerprint(context.workspace)
            if fingerprint_before != fingerprint_after:
                status = WorkerStatus.FAILED
                unresolved = ("tester modified tracked integrated files",)
        if context.durable_context and controller.model_attempts:
            context_engine.refresh_workspace()
        return WorkerResult(
            task_id=context.node.id,
            attempt=context.node.attempt,
            role=context.role.role,
            status=status,
            summary=summary,
            evidence=tuple(evidence),
            artifact_ids=tuple(dict.fromkeys(artifact_ids)),
            changed_files=changed_files,
            unresolved=unresolved,
            usage=worker_usage,
            verifications=tuple(verifications),
            review_decisions=decisions,
            metadata={
                **context.node.metadata,
                "worker_id": self.worker_id,
                "workspace": str(context.workspace),
                "workspace_version": workspace_version(context.workspace),
                "stop_reason": execution.stop_reason,
                "cleanup_errors": execution.cleanup_errors,
                "completion_gate": gate.last_decision.reasons,
                "implicit_completion": bool(
                    getattr(agent, "_codeswarm_implicit_completion", False)
                ),
                "runtime_trace": str(controller.trace_path),
                **({"_context": {**context_engine.lifecycle.state,
                                 "revision": context_engine.lifecycle.state["revision"] + 1,
                                 "read_calls": context_engine.read_calls,
                                 "working_state": asdict(context_engine.state)}}
                   if context.durable_context and controller.model_attempts else {}),
                **({"handoff_ready": handoff_ready,
                    "routing_events": [event for event in context.routing_control.events
                                       if event["task_id"] == context.node.id]}
                   if context.routing_control else {}),
            },
        )

    def _build_worker_prompt(self, context: WorkerContext) -> str:
        sections = [
            f"[Original issue]\n{context.request.issue}",
            f"[Assigned role]\n{context.role.role.value}",
            f"[Assigned goal]\n{context.node.goal}",
            f"[Workspace]\n{context.workspace}",
        ]
        if context.request.acceptance_criteria:
            sections.append(
                "[Acceptance criteria]\n- " + "\n- ".join(context.request.acceptance_criteria)
            )
        if context.request.test_commands:
            sections.append(
                "[Required test commands]\n- "
                + "\n- ".join(context.request.test_commands)
            )
        if context.role.role == WorkerRole.EXPLORER:
            sections.append(
                "[Efficiency contract]\nThe supplied code chunks are your primary evidence. "
                "Make one targeted repository_read to anchor the report; do not list the "
                "repository or reread a path/range already present in context. Then return one "
                "concise final report with file and line references and call task_done in that "
                "same response. Plain status updates and repeated verification are forbidden."
            )
        elif context.role.role == WorkerRole.TESTER:
            phase = context.node.metadata.get("phase")
            instruction = (
                "Run each required test command once. A failing command is successful "
                "reproduction evidence; summarize it and call task_done immediately."
                if phase == "reproduce"
                else "Run each required test command once, report the exact result, and call "
                "task_done immediately without additional exploration."
            )
            sections.append(f"[Efficiency contract]\n{instruction}")
        elif context.role.role in {WorkerRole.REVIEWER, WorkerRole.AUDITOR}:
            review_targets = tuple(context.node.metadata.get("review_target_task_ids", ()))
            component_targets = tuple(context.node.metadata.get("component_task_ids", ()))
            if context.role.role == WorkerRole.AUDITOR:
                if context.node.metadata.get("phase") == "preaudit":
                    sections.append(
                        "[Security preaudit]\nOnly assess whether the integrated patch is safe "
                        "to execute in the test sandbox. No test result exists yet; do not "
                        "judge final functional correctness. Return a decision for the "
                        "integration target."
                    )
                integrated = next((item for item in context.dependency_results
                                   if item.task_id == context.node.metadata.get("integration_task_id")), None)
                if integrated:
                    patch_id = next((artifact_id for artifact_id in integrated.artifact_ids
                                     if context.artifacts.get(artifact_id).kind == "patch"), None)
                    if patch_id:
                        sections.append("[Integrated patch]\n" +
                                        context.artifacts.read_text(patch_id)[:30000])
                verification = next((item for item in context.dependency_results
                                     if item.role == WorkerRole.TESTER and
                                     item.metadata.get("phase") == "verify"), None)
                if verification:
                    sections.append("[Fresh verification]\n" + json.dumps(
                        [to_jsonable(item) for item in verification.verifications],
                        ensure_ascii=False))
            sections.append(
                "[Efficiency contract]\nJudge the supplied patch and verification evidence. "
                "You have no tools because all admissible evidence is already in context. Do not "
                "announce an investigation or request more evidence. Return only the required "
                "decisions JSON; the deterministic gate will end the task."
            )
            sections.append(
                "[Decision identity contract]\nSet `target_task_id` to the integrated result "
                f"task ID exactly: {list(review_targets)}. Do not use a developer task ID there. "
                "Only a `replan` decision may name developer tasks in `rework_task_ids`, chosen "
                f"from: {list(component_targets)}."
            )
        elif context.role.role == WorkerRole.SECURITY:
            patch_sections: list[str] = []
            for task_id, artifact_id in self._component_patch_artifacts(context).items():
                component = next(
                    (item for item in context.dependency_results if item.task_id == task_id),
                    None,
                )
                scope = component.metadata.get("write_scope", ()) if component else ()
                changed = component.changed_files if component else ()
                patch_sections.append(
                    f"## {task_id}\nDeclared scope: {scope}\nChanged files: {changed}\n"
                    f"Patch artifact: {artifact_id}\n{context.artifacts.read_text(artifact_id)}"
                )
            if patch_sections:
                sections.append("[Component patches]\n" + "\n\n".join(patch_sections))
            sections.append(
                "[Efficiency contract]\nThe exact component patches and scopes are supplied "
                "above. You have no tools. Return the complete structured security decisions in "
                "your first response; do not repeat functional review or propose an alternative "
                "implementation."
            )
        elif context.role.role == WorkerRole.INTEGRATOR:
            ordered = self._component_patch_artifacts(context)
            sections.append(
                "[Integration order]\n- " + "\n- ".join(
                    f"{task_id}: {artifact_id}" for task_id, artifact_id in ordered.items()
                )
            )
            sections.append(
                "[Efficiency contract]\nCall patch_apply exactly once for each listed task in "
                "order. Resolve only concrete merge seams, then call task_done. Final testing and "
                "approval belong to later roles."
            )
        elif context.role.role in {WorkerRole.MANAGER, WorkerRole.DEVELOPER, WorkerRole.WORKER}:
            required_paths = context.node.metadata.get("required_paths", ())
            if required_paths and not context.node.metadata.get("sole_work_unit"):
                sections.append(
                    "[Assigned deliverable files]\n" + ", ".join(required_paths)
                    + "\nEach named file must be addressed before this component is complete."
                )
            sections.append(
                "[Efficiency contract]\nPrefer supplied evidence over repeated reads. Apply the "
                "smallest patch using an absolute workspace path, run only the exact required "
                "tests, then call task_done immediately. Do not create ad-hoc verification "
                "scripts or perform extra checks after the required tests pass. Repeated "
                "repository reads are blocked; after eight distinct reads, edit the code "
                "before exploring further. Use "
                "repository_read rather than shell listing commands to inspect files."
            )
        return "\n\n".join(sections)

    def _collect_tool_evidence(
        self, context: WorkerContext, execution, context_engine=None,
        gate: CompletionGate | None = None,
    ):
        evidence: list[Evidence] = []
        artifact_ids: list[str] = []
        verifications: list[VerificationRecord] = []
        integrated_patch_sha = self._verified_patch_sha(context)
        for step in execution.steps:
            calls = step.tool_calls or []
            results = step.tool_results or []
            for call, result in zip(calls, results, strict=False):
                output = result.result or result.error or ""
                observation = (
                    context_engine.reducer.by_call_id.get(call.call_id)
                    if context_engine else None
                )
                artifact_id = observation.artifact_id if observation else context.artifacts.put_text(
                    f"{context.node.id}-{call.name}", output or "<empty>", ".log"
                ).id
                artifact_ids.append(artifact_id)
                evidence_kind, claim, source_path = self._evidence_description(call, result)
                evidence.append(
                    Evidence(
                        id=f"evidence-{uuid.uuid4().hex}",
                        producer=self.worker_id,
                        task_id=context.node.id,
                        kind=evidence_kind,
                        claim=claim,
                        artifact_id=artifact_id,
                        source_path=source_path,
                        base_commit=context.request.base_commit,
                        confidence=1.0,
                    )
                )
                if call.name.lower().replace("_", "") == "bash":
                    command = str(call.arguments.get("command", ""))
                    if self._looks_like_test(command):
                        recorded = next(
                            (item for item in gate.evidence if item.call_id == call.call_id),
                            None,
                        ) if gate else None
                        verifications.append(
                            VerificationRecord(
                                command=command,
                                passed=result.success,
                                artifact_id=artifact_id,
                                exit_code=0 if result.success else 1,
                                workspace_version=recorded.workspace_version if recorded else None,
                                patch_sha256=integrated_patch_sha,
                            )
                        )
        return evidence, artifact_ids, verifications

    @staticmethod
    def _verified_patch_sha(context: WorkerContext) -> str | None:
        target_id = context.node.metadata.get("integrated_task_id") or context.node.metadata.get(
            "candidate_task_id"
        )
        if not isinstance(target_id, str):
            return None
        integrated = next(
            (result for result in context.dependency_results if result.task_id == target_id),
            None,
        )
        if integrated is None:
            return None
        for artifact_id in integrated.artifact_ids:
            try:
                record = context.artifacts.get(artifact_id)
            except (KeyError, ValueError):
                continue
            if record.kind == "patch":
                return record.sha256
        return None

    @staticmethod
    def _component_patch_artifacts(context: WorkerContext) -> dict[str, str]:
        ordered = context.node.metadata.get("component_task_ids", [])
        if not isinstance(ordered, list):
            return {}
        by_task = {result.task_id: result for result in context.dependency_results}
        patches: dict[str, str] = {}
        for task_id in ordered:
            result = by_task.get(str(task_id))
            if result is None:
                continue
            for artifact_id in result.artifact_ids:
                try:
                    if context.artifacts.get(artifact_id).kind == "patch":
                        patches[str(task_id)] = artifact_id
                        break
                except (KeyError, ValueError):
                    continue
        return patches

    @staticmethod
    def _last_model_content(execution) -> str:
        for step in reversed(execution.steps):
            if step.llm_response and step.llm_response.content:
                return step.llm_response.content
        return ""

    @staticmethod
    def _looks_like_test(command: str) -> bool:
        return is_test_command(command)

    @staticmethod
    def _changed_files(patch: str) -> tuple[str, ...]:
        return changed_paths(patch)

    @staticmethod
    def _evidence_description(call, result) -> tuple[str, str, str | None]:
        normalized = call.name.lower().replace("_", "")
        if normalized == "bash":
            command = str(call.arguments.get("command", ""))
            kind = "test_result" if TraeAgentWorker._looks_like_test(command) else "command_result"
            return kind, f"command `{command}` returned success={result.success}", None
        if normalized == "repositoryread":
            operation = str(call.arguments.get("operation", "read"))
            path = call.arguments.get("path")
            claim = f"{operation} evidence from {path or 'repository root'}"
            return "repository_evidence", claim, str(path) if path else None
        if normalized == "ckg":
            identifier = str(call.arguments.get("identifier", ""))
            return "symbol_evidence", f"code graph lookup for `{identifier}`", None
        return "tool_result", f"{call.name} returned success={result.success}", None

    @staticmethod
    def _get_workspace_patch(workspace: Path) -> str:
        return workspace_patch(workspace)

    @staticmethod
    def _tracked_workspace_fingerprint(workspace: Path) -> str:
        return workspace_version(workspace)

    @staticmethod
    def _parse_review_decisions(content: str) -> tuple[ReviewDecision, ...]:
        return parse_review_decisions(content)
