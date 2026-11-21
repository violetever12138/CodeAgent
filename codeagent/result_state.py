"""Worker results live in LangGraph state; this module provides node-local views."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps

from langgraph.types import Command

from codeagent.context.lifecycle import merge_contexts
from codeagent.protocol import WorkerResult, to_jsonable, worker_result_from_json

RESULT_STORAGE = "langgraph_state_v1"
RESULT_SCHEMA_VERSION = 4


def result_key(task_id: str, attempt: int) -> str:
    return f"{task_id}@{attempt}"


def merge_results(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key != result_key(value["task_id"], value["attempt"]):
            raise ValueError(f"result identity does not match key: {key}")
        if key in merged and merged[key] != value:
            raise ValueError(f"conflicting LangGraph result update: {key}")
        merged[key] = value
    evidence = {}
    for value in merged.values():
        for item in value.get("evidence", []):
            if item["id"] in evidence and evidence[item["id"]] != item:
                raise ValueError(f"conflicting result evidence: {item['id']}")
            evidence[item["id"]] = item
    return merged


def compact_result(result: WorkerResult, artifacts) -> WorkerResult:
    """Externalize large reports; never copy private Manager history into results."""
    metadata = dict(result.metadata)
    metadata.pop("manager_state", None)
    plan = metadata.pop("acceptance_plan", None)
    if plan:
        metadata["acceptance_plan_version"] = plan.get("version")
    ids = list(result.artifact_ids)
    summary = result.summary
    if len(summary) > 4000:
        record = artifacts.put_text("worker-report", summary, ".md")
        ids.append(record.id)
        metadata["summary_artifact_id"] = record.id
        summary = summary[:4000] + f"\n[Full report: {record.id}]"
    unresolved = result.unresolved
    if any(len(item) > 1000 for item in unresolved):
        record = artifacts.put_text("worker-unresolved", "\n\n".join(unresolved), ".md")
        ids.append(record.id)
        metadata["unresolved_artifact_id"] = record.id
        unresolved = tuple(item[:1000] for item in unresolved)
    return replace(result, summary=summary, unresolved=unresolved, metadata=metadata,
                   artifact_ids=tuple(dict.fromkeys(ids)))


class ResultView:
    def __init__(self, values: dict, *, durable_attempts: bool = True, contexts=None):
        self.values = dict(values)
        self.updates = {}
        self.durable_attempts = durable_attempts
        self.contexts = dict(contexts or {})
        self.context_updates = {}


class StateResults:
    """A scoped read view and node delta, not a second persistent result store."""

    def __init__(self):
        self.current: ContextVar[ResultView] = ContextVar("codeswarm_result_view")

    async def commit(self, result: WorkerResult) -> bool:
        """Stage a node-local delta; only the framework checkpointer persists it."""
        view = self.current.get()
        key = result_key(result.task_id, result.attempt)
        raw = to_jsonable(result)
        snapshot = raw["metadata"].pop("_context", None)
        if snapshot:
            self.stage_context(key, snapshot)
        updated = merge_results(view.values, {key: raw})
        fresh = key not in view.values
        view.values = updated
        if fresh:
            view.updates[key] = raw
        return fresh

    def stage_context(self, key, snapshot):
        view = self.current.get()
        view.contexts = merge_contexts(view.contexts, {key: snapshot})
        view.context_updates[key] = snapshot

    async def result_attempt(self, task_id: str, attempt: int) -> WorkerResult | None:
        raw = self.current.get().values.get(result_key(task_id, attempt))
        return worker_result_from_json(raw) if raw else None

    async def result(self, task_id: str) -> WorkerResult | None:
        matches = [r for r in self.current.get().values.values() if r["task_id"] == task_id]
        raw = max(matches, key=lambda r: r["attempt"], default=None)
        return worker_result_from_json(raw) if raw else None

    async def results_for(self, task_ids) -> tuple[WorkerResult, ...]:
        return tuple([result for task_id in task_ids if (result := await self.result(task_id))])

    async def all_results(self) -> tuple[WorkerResult, ...]:
        ids = dict.fromkeys(r["task_id"] for r in self.current.get().values.values())
        return await self.results_for(ids)

    async def evidence(self, evidence_ids=()) -> tuple:
        all_evidence = {item.id: item for raw in self.current.get().values.values()
                        for item in worker_result_from_json(raw).evidence}
        return tuple(all_evidence[key] for key in evidence_ids if key in all_evidence) if evidence_ids else tuple(all_evidence.values())

    def wrap(self, handler, runtime):
        if not inspect.isfunction(handler):
            return handler  # Compiled subgraphs scope their own nodes.

        @wraps(handler)
        async def node(state):
            # Send branches already are independently checkpointed Worker nodes.
            # Nested tasks are needed for multi-action nodes such as Manager.
            view = ResultView(state.get("results", {}), durable_attempts=not bool(state.get("active_task")),
                              contexts=state.get("contexts", {}))
            token = self.current.set(view)
            try:
                for raw in view.values.values():
                    result = worker_result_from_json(raw)
                    await runtime.budget.reconcile_committed(
                        f"{runtime.run_id}:{result.task_id}:{result.attempt}", result.usage)
                output = handler(state)
                if inspect.isawaitable(output):
                    output = await output
                if isinstance(output, Command):
                    update = dict(output.update or {})
                else:
                    update = dict(output or {})
                # Existing graph handlers use this name for their returned deltas.
                # New runs persist the full records only in the results channel.
                supplied = update.pop("result_refs", {})
                update["results"] = merge_results(view.updates, supplied)
                update["contexts"] = merge_contexts(view.context_updates, update.get("contexts", {}))
                if "budget_snapshot" in update:
                    update["budget_snapshot"] = await runtime.budget.snapshot()
                if isinstance(output, Command):
                    return replace(output, update=update)
                return update
            finally:
                self.current.reset(token)

        return node
