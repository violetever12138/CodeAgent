import asyncio
import copy
import json
from dataclasses import replace

import pytest

from codeagent.artifacts import ArtifactStore
from codeagent.context.engine import ContextSettings
from codeagent.context.lifecycle import ContextLifecycle, merge_contexts
from codeagent.context.packer import ContextBudgetError, TokenCounter
from trae_agent.tools.base import ToolCall, ToolResult
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


def _report(**changes):
    raw = {"problem": "Fix value", "findings": ["Historical evidence: test failed on v1"],
           "decisions": ["Use the observed source; retain required behavior"], "excluded": ["Unrelated migration"],
           "changes": [], "unfinished": ["Verify current code"], "assumptions": ["Unverified: same API"],
           "next_steps": ["Run required tests on current version"], "evidence_refs": ["artifact:test-v1"]}
    return json.dumps({**raw, **changes})


@pytest.fixture
def lifecycle(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    settings = ContextSettings(context_window_tokens=8000, output_reserve_tokens=1000,
                               safety_margin_tokens=1000, fixed_overhead_tokens=64, summary_max_tokens=800)
    value = ContextLifecycle(TokenCounter(), settings, artifacts)
    yield value
    artifacts.close()


def test_below_threshold_keeps_more_than_fixed_history_limit_without_summary(lifecycle):
    calls = []

    async def summary(prompt, attempt):
        calls.append(prompt)
        return _report()

    groups = [(i, [LLMMessage("assistant", content=f"finding {i}")]) for i in range(1, 9)]
    messages = asyncio.run(lifecycle.prepare([LLMMessage("system", "hard user constraints")], groups, [], 0, summary))
    assert len(messages) == 9 and not calls
    assert lifecycle.state["covered_through"] == 0


def test_threshold_generates_structured_summary_and_preserves_recent_tool_pair(lifecycle):
    calls = []

    async def summary(prompt, attempt):
        calls.append(prompt)
        return _report()

    pair = [LLMMessage("assistant", tool_call=ToolCall("repository_read", "read", {"path": "core.py"})),
            LLMMessage("user", tool_result=ToolResult("read", "repository_read", True, "current code"))]
    groups = [(i, [LLMMessage("assistant", "old finding " + "x" * 1200)]) for i in range(1, 5)] + [(5, pair)]
    messages = asyncio.run(lifecycle.prepare([LLMMessage("system", "MANDATORY_CONSTRAINT")], groups, [], 0, summary,
                                           workspace_version="v2"))
    assert len(calls) == 1
    assert lifecycle.state["covered_through"] > 0 and lifecycle.state["summary_version"] == 1
    assert lifecycle.state["summary_workspace_version"] == "v2"
    assert messages[0].content == "MANDATORY_CONSTRAINT"
    assert [m.tool_call.call_id for m in messages if m.tool_call] == ["read"]
    assert [m.tool_result.call_id for m in messages if m.tool_result] == ["read"]
    assert "Historical evidence" in messages[1].content and "Unverified" in messages[1].content
    assert lifecycle.cost(messages) < lifecycle.available * .75


def test_safe_above_target_does_not_trigger_repeated_summary(lifecycle):
    lifecycle.settings = replace(lifecycle.settings, summary_max_tokens=1800)
    calls = []

    async def summary(prompt, attempt):
        calls.append(attempt)
        return _report(problem="summary " + "s" * 900)

    groups = [(i, [LLMMessage("assistant", "x" * 1000)]) for i in range(1, 4)]
    groups.append((4, [LLMMessage("assistant", "recent " + "r" * 2200)]))
    messages = asyncio.run(lifecycle.prepare([LLMMessage("system", "constraints " * 15)], groups, [], 0, summary))
    assert len(calls) == 1
    assert lifecycle.available * .5 < lifecycle.cost(messages) < lifecycle.available * .75


@pytest.mark.parametrize("failure", ["exception", "invalid_json", "wrong_fields"])
def test_summary_failure_retries_once_and_keeps_all_uncovered_history(lifecycle, failure):
    previous = copy.deepcopy(lifecycle.state)
    calls = []

    async def summary(prompt, attempt):
        calls.append(attempt)
        if failure == "exception":
            raise RuntimeError("summary provider unavailable")
        return "{" if failure == "invalid_json" else '{"problem":"drop everything"}'

    groups = [(i, [LLMMessage("assistant", f"finding {i}: " + "x" * 1000)]) for i in range(1, 6)]
    messages = asyncio.run(lifecycle.prepare([LLMMessage("system", "keep user goal")], groups, [], 0, summary))
    assert calls == [0, 1] and lifecycle.state == previous
    assert len(messages) == 6
    assert all(any(f"finding {i}:" in (m.content or "") for m in messages) for i in range(1, 6))


def test_failed_summary_cannot_silently_drop_history_to_fit_hard_budget(lifecycle):
    async def failure(prompt, attempt):
        raise ValueError("summary failed")

    groups = [(i, [LLMMessage("assistant", "x" * 1000)]) for i in range(1, 9)]
    with pytest.raises(ContextBudgetError, match="history retained"):
        asyncio.run(lifecycle.prepare([LLMMessage("system", "keep user goal")], groups, [], 0, failure))
    assert lifecycle.state["summary"] is None and lifecycle.state["covered_through"] == 0


def test_consumption_advances_only_after_combined_response_snapshot_is_restored(lifecycle):
    response = LLMResponse("complete response", finish_reason="stop")
    saved = lifecycle.prospective(response, {"r1": "artifact-1"}, ["chunk-1"])
    assert not lifecycle.state["consumed_calls"]
    lifecycle.restore(saved)
    raw = ToolResult("r1", "repository_read", False, "RAW_RECALL_BODY", "RAW_ERROR_BODY", "provider-id")
    clean = lifecycle.sanitize(LLMMessage("user", tool_result=raw)).tool_result
    assert "RAW_RECALL_BODY" not in clean.result and "RAW_ERROR_BODY" not in clean.error
    assert clean.call_id == raw.call_id and clean.id == raw.id and not clean.success
    assert raw.result == "RAW_RECALL_BODY" and raw.error == "RAW_ERROR_BODY"
    assert lifecycle.state["consumed_chunks"] == ["chunk-1"]


def test_incomplete_and_cancelled_requests_do_not_consume(lifecycle):
    saved = lifecycle.prospective(LLMResponse("partial", finish_reason="length"), {"r": "a"}, ["c"])
    assert not saved["consumed_calls"] and not saved["consumed_chunks"]

    async def cancelled(prompt, attempt):
        raise asyncio.CancelledError()

    groups = [(i, [LLMMessage("assistant", "x" * 1100)]) for i in range(1, 6)]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(lifecycle.prepare([LLMMessage("system", "rules")], groups, [], 0, cancelled))
    assert lifecycle.state["covered_through"] == 0 and not lifecycle.state["consumed_calls"]


def test_context_reducer_keeps_workers_separate_and_rejects_same_revision_conflict():
    original = {"a@0": {"revision": 2, "summary": "A"}, "b@0": {"revision": 1, "summary": "B"}}
    assert merge_contexts(original, {"a@0": {"revision": 1, "summary": "old"}}) == original
    assert merge_contexts(original, {"a@0": {"revision": 3, "summary": "new"}})["b@0"] == original["b@0"]
    with pytest.raises(ValueError, match="conflicting context"):
        merge_contexts(original, {"a@0": {"revision": 2, "summary": "different"}})
