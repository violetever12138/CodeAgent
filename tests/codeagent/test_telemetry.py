"""No paid APIs: real SDK/OTel, local transport and bounded exporter failures."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time

import httpx
import openai
import pytest

from codeagent.artifacts import ArtifactStore
from codeagent.runtime.recovery import ModelRecovery
from codeagent.runtime.state import RuntimeController, RuntimeLimits
from codeagent.telemetry import TraceSettings, execution, span
from codeagent.trajectory import ReferenceTrajectoryRecorder
from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.openai_client import OpenAIClient
from trae_agent.utils.request_observer import observer


def rows(root):
    return [json.loads(line) for file in (root / "observability").glob("spans-*.jsonl")
            for line in file.read_text(encoding="utf-8").splitlines()]


def config():
    return ModelConfig(model="test", model_provider=ModelProvider(provider="openai", api_key="unused-test-key"),
                       temperature=0, top_p=1, top_k=0, parallel_tool_calls=False, max_retries=0)


def response():
    return {"id": "resp-test", "object": "response", "created_at": 1, "model": "test", "status": "completed",
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "actual SDK answer", "annotations": []}]}],
            "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def test_real_sdk_retries_are_distinct_and_actual_payload_is_redacted(tmp_path):
    calls = []
    async def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "retry", "type": "rate_limit_error"}}) if len(calls) == 1 else httpx.Response(200, json=response())

    async def run():
        artifacts = ArtifactStore(tmp_path / "artifacts")
        controller = RuntimeController(run_id="run", task_id="task", worker_id="worker", trace_path=tmp_path / "runtime.jsonl",
                                       limits=RuntimeLimits(deadline_at=time.monotonic() + 30))
        client = OpenAIClient(config())
        client.async_client = openai.AsyncOpenAI(api_key="sk-not-recorded-secret", base_url="https://sdk.invalid/v1",
            max_retries=2, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            with execution("run", tmp_path, artifacts), span("worker", "AGENT"):
                raw = await ModelRecovery(max_attempts=2, base_delay=.001).call(
                    lambda: client._create_openai_response_async([{"role": "user", "content": "api_key=secret-value"}], config(), None), controller)
                assert raw.status == "completed"
        finally:
            await client.aclose()
        records = rows(tmp_path)
        llm = [item for item in records if item["attributes"].get("openinference.span.kind") == "LLM"]
        assert len(calls) == len(llm) == 2
        assert [item["attributes"]["codeswarm.request_attempt"] for item in llm] == [1, 2]
        assert len({item["attributes"]["codeswarm.logical_call_id"] for item in llm}) == 1
        assert llm[0]["status"]["status_code"] == "ERROR"
        request = json.loads(artifacts.read_text(llm[-1]["attributes"]["codeswarm.input_ref"]))
        assert request["model"] == calls[-1]["model"]
        assert "secret-value" not in json.dumps(request)
        body = json.loads(artifacts.read_text(llm[-1]["attributes"]["codeswarm.response_ref"]))
        assert body == raw.model_dump(mode="json", by_alias=True)
        assert not any("sk-not-recorded-secret" in file.read_text(encoding="utf-8") for file in artifacts.root.glob("*.json"))
        artifacts.close()

    asyncio.run(run())


def test_parallel_parentage_and_resume_create_new_trace(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    async def run(resumed=False):
        with execution("same-run", tmp_path, artifacts, **{"codeswarm.resumed": resumed}):
            async def work(number):
                with span(f"graph.worker-{number}"), span(f"worker-{number}", "AGENT"):
                    await asyncio.sleep(.01)
                    with span(f"tool-{number}", "TOOL"):
                        pass
            await asyncio.gather(work(1), work(2))
    asyncio.run(run())
    asyncio.run(run(True))
    records = rows(tmp_path)
    roots = [item for item in records if item["name"] == "run"]
    assert len(roots) == 2 and roots[0]["context"]["trace_id"] != roots[1]["context"]["trace_id"]
    assert all(item["attributes"]["session.id"] == "same-run" for item in records)
    for root in roots:
        segment = [item for item in records if item["context"]["trace_id"] == root["context"]["trace_id"]]
        indexed = {item["context"]["span_id"]: item for item in segment}
        for item in segment:
            if item["name"].startswith("tool-"):
                assert indexed[item["parent_id"]]["name"] == item["name"].replace("tool", "worker")
    artifacts.close()


def test_artifact_failure_does_not_fail_business_work(tmp_path, monkeypatch):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    def fail(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(artifacts, "put_text", fail)
    with execution("artifact-failure", tmp_path, artifacts), observer.get().request("test", {"model": "test"}) as request:
        request.response({"result": "business succeeded"})
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert not status["complete"] and status["missing"]["artifact"] == 2
    artifacts.close()


def test_queue_full_and_bounded_shutdown_are_visible(tmp_path):
    from opentelemetry.sdk.trace.export import SpanExportResult
    started, release = threading.Event(), threading.Event()
    class Blocking:
        def export(self, spans):
            started.set()
            release.wait(3)
            return SpanExportResult.SUCCESS
        def shutdown(self):
            pass
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with execution("queue", tmp_path, artifacts, TraceSettings(queue_size=1, batch_size=1, flush_timeout=.05)):
        observer.get().sink.exporter = Blocking()
        with span("first"):
            pass
        assert started.wait(1)
        for number in range(5):
            with span(f"queued-{number}"):
                pass
    release.set()
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert not status["complete"] and status["missing"]["queue_full"] >= 1
    assert status["missing"]["flush_timeout"] == 1
    artifacts.close()


def test_unavailable_langsmith_keeps_local_spans(tmp_path, monkeypatch):
    import requests

    def unavailable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("test transport is unavailable")

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.setattr(requests.Session, "request", unavailable)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with execution("offline", tmp_path, artifacts, TraceSettings(export_timeout=.1, flush_timeout=.5)), span("business"):
        pass
    assert rows(tmp_path)
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert not status["complete"] and status["missing"]
    artifacts.close()


def test_reference_trajectory_does_not_repeat_messages_or_write_each_step(tmp_path):
    from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
    recorder = ReferenceTrajectoryRecorder(str(tmp_path / "trajectory.json"))
    recorder.start_recording("task", "test", "test", 4)
    recorder.record_llm_interaction([LLMMessage("user", "LARGE_BODY")], LLMResponse("LARGE_RESPONSE"), "test", "test")
    recorder.record_agent_step(1, "completed", llm_messages=[LLMMessage("user", "LARGE_BODY")], llm_response=LLMResponse("LARGE_RESPONSE"))
    assert not (tmp_path / "trajectory.json").exists()
    assert "LARGE_BODY" not in (tmp_path / "trajectory.jsonl").read_text()
    recorder.finalize_recording(True, "summary")
    assert (tmp_path / "trajectory.json").exists()
    assert "LARGE_RESPONSE" not in (tmp_path / "trajectory.json").read_text()


@pytest.mark.parametrize("endpoint", ["https://external.example", "http://127.0.0.1:6006", None])
def test_langsmith_bad_configuration_keeps_local_spans(tmp_path, monkeypatch, endpoint):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    if endpoint:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", endpoint)
    else:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with contextlib.closing(ArtifactStore(tmp_path / "artifacts")) as artifacts:
        with execution("bad-config", tmp_path, artifacts), span("business"):
            pass
    assert rows(tmp_path)
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert status["missing"] == {"langsmith_configuration": 1}


@pytest.mark.parametrize("agent_id", [None, "codeswarm-test"])
def test_langsmith_exports_spans_with_destination_headers(tmp_path, monkeypatch, agent_id):
    import requests
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    calls = []

    def receive(transport, *, method, url, headers, data, **kwargs):
        assert method == "POST"
        calls.append((url, requests.structures.CaseInsensitiveDict(headers), ExportTraceServiceRequest.FromString(data)))
        response = requests.Response()
        response.status_code = 200
        response._content = b""
        return response

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com/")
    monkeypatch.setenv("LANGSMITH_PROJECT", "codeswarm-otlp-test")
    monkeypatch.delenv("LANGSMITH_AGENT_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_AGENT_ENVIRONMENT", raising=False)
    if agent_id:
        monkeypatch.setenv("LANGSMITH_AGENT_ID", agent_id)
    monkeypatch.setattr(requests.Session, "request", receive)
    with contextlib.closing(ArtifactStore(tmp_path / "artifacts")) as artifacts:
        with execution("langsmith", tmp_path, artifacts), span("business"):
            pass

    assert calls
    spans = []
    for url, headers, batch in calls:
        assert url == "https://eu.api.smith.langchain.com/otel/v1/traces"
        assert headers["x-api-key"] == "test-langsmith-key"
        if agent_id:
            assert headers["Langsmith-Agent-Id"] == agent_id
            assert headers["Langsmith-Agent-Environment"] == "local"
            assert "Langsmith-Project" not in headers
        else:
            assert headers["Langsmith-Project"] == "codeswarm-otlp-test"
        spans.extend(item for resource in batch.resource_spans
                     for scope in resource.scope_spans for item in scope.spans)
    by_name = {item.name: item for item in spans}
    assert by_name["business"].parent_span_id == by_name["run"].span_id
    assert "test-langsmith-key" not in json.dumps(rows(tmp_path))


def test_summary_span_records_threshold_coverage_and_accepted_content(tmp_path):
    from codeagent.context.engine import ContextSettings
    from codeagent.context.lifecycle import SUMMARY_FIELDS, ContextLifecycle
    from codeagent.context.packer import TokenCounter
    from trae_agent.utils.llm_clients.llm_basics import LLMMessage
    artifacts = ArtifactStore(tmp_path / "artifacts")
    settings = ContextSettings(context_window_tokens=8000, output_reserve_tokens=1000,
        safety_margin_tokens=1000, fixed_overhead_tokens=64, summary_max_tokens=800)
    lifecycle = ContextLifecycle(TokenCounter(), settings, artifacts)
    async def summarize(prompt, attempt):
        return json.dumps({name: "summary" if name == "problem" else [] for name in SUMMARY_FIELDS})
    groups = [(i, [LLMMessage("assistant", "x" * 1200)]) for i in range(1, 5)]
    with execution("summary", tmp_path, artifacts):
        asyncio.run(lifecycle.prepare([LLMMessage("system", "constraint")], groups, [], 0, summarize, workspace_version="v2"))
    summary = next(item for item in rows(tmp_path) if item["name"] == "summary")
    assert summary["attributes"]["summary.before_tokens"] >= summary["attributes"]["summary.trigger_tokens"]
    assert summary["attributes"]["summary.after_tokens"] < summary["attributes"]["summary.before_tokens"]
    assert summary["attributes"]["summary.version"] == 1
    assert artifacts.read_text(summary["attributes"]["summary.content_ref"])
    artifacts.close()


def test_missing_otel_dependency_is_visible_and_does_not_fail_execution(tmp_path, monkeypatch):
    import sys
    artifacts = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    completed = []
    with execution("missing", tmp_path, artifacts):
        completed.append(True)
    assert completed == [True]
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert not status["complete"] and status["missing"]["initialization"] == 1
    artifacts.close()


def test_prefixed_and_quoted_credentials_are_removed_from_trace_text():
    from trae_agent.utils.redaction import redact_text
    text = 'DEEPSEEK_API_KEY="opaque secret with spaces"; AZURE_ACCESS_TOKEN=opaque-token; normal=value'
    cleaned = redact_text(text)
    assert "opaque" not in cleaned and "with spaces" not in cleaned
    assert "normal=value" in cleaned
    assert redact_text(cleaned) == cleaned
    arguments = json.loads(redact_text('{"api_key":"opaque value", "path":"src/app.py"}'))
    assert arguments == {"api_key": "[REDACTED]", "path": "src/app.py"}


def test_large_event_fields_are_externalized_and_previewed(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with execution("large", tmp_path, artifacts), span("tool", "TOOL") as observed:
        observed.event("large_failure", error="LARGE_ERROR" * 2000)
    record = next(item for item in rows(tmp_path) if item["name"] == "tool")
    event = record["events"][0]["attributes"]
    assert len(event["error"]) < 1100
    assert "LARGE_ERROR" * 2000 in artifacts.read_text(event["codeswarm.details_ref"])
    artifacts.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("completion_limit", [False, True])
def test_compatible_sdk_payload_keeps_flat_token_parameters(tmp_path, asynchronous, completion_limit):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from trae_agent.utils.llm_clients.openai_compatible_base import OpenAICompatibleClient
    model = config()
    model.max_tokens = 123
    if completion_limit:
        model.model_provider.provider = "azure"
        model.model = "gpt-5"
        model.max_completion_tokens = 333
    raw = {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    create = AsyncMock(return_value=raw) if asynchronous else Mock(return_value=raw)
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.provider_config = SimpleNamespace(get_provider_name=lambda: "fixture")
    client.message_history = [{"role": "user", "content": "question"}]
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.client = client.async_client = sdk
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with execution("compatible", tmp_path, artifacts):
        if asynchronous:
            asyncio.run(client._create_response_async(model, None))
        else:
            client._create_response(model, None)
    sent = create.call_args.kwargs
    key = "max_completion_tokens" if completion_limit else "max_tokens"
    assert sent[key] == (333 if completion_limit else 123)
    assert all(isinstance(key, str) for key in sent)
    wire = next(item for item in rows(tmp_path) if item["attributes"].get("openinference.span.kind") == "LLM")
    assert json.loads(artifacts.read_text(wire["attributes"]["codeswarm.input_ref"]))[key] == sent[key]
    artifacts.close()


def test_google_retry_control_preserves_existing_request_options(tmp_path):
    from google.genai import types

    from trae_agent.utils.request_observer import google_config
    original = types.GenerateContentConfig(http_options=types.HttpOptions(timeout=9876, headers={"X-Test": "yes"}))
    artifacts = ArtifactStore(tmp_path / "artifacts")
    with execution("google", tmp_path, artifacts):
        changed = google_config(original)
    assert changed.http_options.timeout == 9876 and changed.http_options.headers == {"X-Test": "yes"}
    assert changed.http_options.retry_options.attempts == 1 and original.http_options.retry_options is None
    artifacts.close()


def test_provider_response_with_union_schema_keeps_tool_origin_and_complete_status(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    raw = {"tools": [{"type": "function", "parameters": {"type": "object", "properties": {
        "path": {"type": ["string", "null"]}}}}],
        "choices": [{"message": {"tool_calls": [{"id": "actual-call", "type": "function",
            "function": {"name": "repository_read", "arguments": "{}"}}]}}]}
    with execution("union-schema", tmp_path, artifacts) as root:
        session = root.session
        with session.request("openai", {"model": "test"}) as observed:
            observed.response(raw)
        assert session.origins["actual-call"] is observed
        assert json.loads(artifacts.read_text(observed.output_ref)) == raw
    status = json.loads(next((tmp_path / "observability").glob("status-*.json")).read_text())
    assert status["complete"] and status["missing"] == {}
    artifacts.close()
