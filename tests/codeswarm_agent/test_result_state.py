"""Single-source results, durable attempt recovery, and workspace side effects."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from codeswarm_agent.artifacts import ArtifactStore
from codeswarm_agent.budget import BudgetTracker
from codeswarm_agent.cancellation import CancellationToken
from codeswarm_agent.langgraph_backend import CodeSwarmState, WorkerNodeAdapter
from codeswarm_agent.protocol import (
    Budget,
    Evidence,
    TaskNode,
    TaskRequest,
    Usage,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
    to_jsonable,
)
from codeswarm_agent.result_state import ResultView, StateResults, compact_result, merge_results
from codeswarm_agent.trace import EventTracer


def _result(task="work", attempt=0, **kwargs):
    return WorkerResult(task, attempt, WorkerRole.WORKER, WorkerStatus.SUCCEEDED, "completed", **kwargs)


def test_reducer_keeps_attempts_and_latest_is_numeric_not_arrival():
    result_store = StateResults()

    async def scenario():
        token = result_store.current.set(ResultView({}))
        try:
            for attempt in (2, 0, 1):
                await result_store.commit(_result(attempt=attempt))
            assert (await result_store.result("work")).attempt == 2
            assert (await result_store.result_attempt("work", 0)).attempt == 0
            assert len(result_store.current.get().values) == 3
            assert not await result_store.commit(_result(attempt=2))
        finally:
            result_store.current.reset(token)

    asyncio.run(scenario())


def test_reducer_rejects_result_and_evidence_conflicts():
    result = _result(evidence=(Evidence("e", "worker", "work", "observation", "fact"),))
    original = {"work@0": to_jsonable(result)}
    assert merge_results(original, original) == original
    with pytest.raises(ValueError, match="conflicting LangGraph result"):
        merge_results(original, {"work@0": to_jsonable(replace(result, summary="changed"))})
    other = _result("other", evidence=(Evidence("e", "worker", "other", "observation", "different"),))
    with pytest.raises(ValueError, match="conflicting result evidence"):
        merge_results(original, {"other@0": to_jsonable(other)})
    with pytest.raises(ValueError, match="identity"):
        merge_results({}, {"wrong@0": to_jsonable(result)})


def test_large_worker_report_is_externalized_without_private_history(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    try:
        result = replace(_result(), summary="large report " * 2000,
                         metadata={"manager_state": {"history": "private"},
                                   "acceptance_plan": {"version": 2}, "workspace": "workspace"})
        compact = compact_result(result, artifacts)
        assert len(compact.summary) < 4200
        assert artifacts.read_text(compact.metadata["summary_artifact_id"]) == result.summary
        assert "manager_state" not in compact.metadata and "acceptance_plan" not in compact.metadata
        assert compact.metadata["acceptance_plan_version"] == 2
    finally:
        artifacts.close()


class TestWorkspace:
    async def reset(self, task_id):
        pass

    async def acquire(self, node):
        return Path(".").resolve()


def test_framework_pending_task_is_reused_after_parent_crash_with_fresh_runtime(tmp_path):
    counts = {}

    class Factory:
        def create(self, node, role):
            class Worker:
                worker_id = f"worker-{node.id}"

                async def run(self, context):
                    counts[node.id] = counts.get(node.id, 0) + 1
                    return _result(node.id, node.attempt, usage=Usage(input_tokens=7, output_tokens=3,
                                                                    model_calls=1))

            return Worker()

    def runtime():
        return SimpleNamespace(
            results=StateResults(), budget=BudgetTracker(Budget(max_workers=10)),
            run_id="run", run_directory=tmp_path,
            request=TaskRequest("inspect", str(tmp_path)), worker_factory=Factory(),
            workspace_manager=TestWorkspace(), cancellation=CancellationToken(),
            artifacts=ArtifactStore(tmp_path / "artifacts"), tracer=EventTracer("run", tmp_path / "trace.jsonl"),
        )

    async def drive(resume):
        run = runtime()
        adapter = WorkerNodeAdapter(run)

        async def execute(state):
            await adapter.execute(to_jsonable(TaskNode("a", WorkerRole.WORKER, "first")))
            if not resume:
                raise RuntimeError("after task saved, before parent returned")
            await adapter.execute(to_jsonable(TaskNode("b", WorkerRole.WORKER, "second")))
            return {"budget_snapshot": await run.budget.snapshot()}

        try:
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
                builder = StateGraph(CodeSwarmState)
                builder.add_node("execute", run.results.wrap(execute, run))
                builder.add_edge(START, "execute")
                builder.add_edge("execute", END)
                graph = builder.compile(checkpointer=saver)
                return await graph.ainvoke(None if resume else {"results": {}},
                                           {"configurable": {"thread_id": "run"}})
        finally:
            run.artifacts.close()

    with pytest.raises(RuntimeError, match="parent returned"):
        asyncio.run(drive(False))
    state = asyncio.run(drive(True))
    assert counts == {"a": 1, "b": 1}
    assert set(state["results"]) == {"a@0", "b@0"}
    assert not state.get("result_refs")
    assert state["budget_snapshot"]["workers_started"] == 2
    assert state["budget_snapshot"]["usage"]["input_tokens"] == 14
    assert state["budget_snapshot"]["usage"]["output_tokens"] == 6
    assert not (tmp_path / "blackboard.jsonl").exists()
