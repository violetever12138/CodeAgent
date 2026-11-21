"""Manual OpenTelemetry spans with local capture and optional LangSmith export."""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import queue
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from urllib.parse import urlparse

from trae_agent.utils.redaction import redact_value
from trae_agent.utils.request_observer import last_request, logical_call, observer


LANGSMITH_CLOUD_HOSTS = {
    "api.smith.langchain.com",
    "eu.api.smith.langchain.com",
    "apac.api.smith.langchain.com",
    "aws.api.smith.langchain.com",
}


@dataclass(frozen=True)
class TraceSettings:
    enabled: bool = True
    project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT") or "codeswarm-agent")
    queue_size: int = 1024
    batch_size: int = 64
    export_timeout: float = 3.0
    flush_timeout: float = 5.0

    def __post_init__(self):
        if min(self.queue_size, self.batch_size, self.export_timeout, self.flush_timeout) <= 0:
            raise ValueError("trace limits must be positive")


def _langsmith_exporter(settings: TraceSettings):
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY is required when LangSmith tracing is enabled")
    base_url = os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    url = urlparse(base_url)
    if (url.scheme != "https" or url.netloc not in LANGSMITH_CLOUD_HOSTS
            or url.path not in {"", "/"} or url.params or url.query or url.fragment):
        raise ValueError("LANGSMITH_ENDPOINT must be a LangSmith Cloud HTTPS API base URL")

    headers = {"x-api-key": api_key}
    agent_id = os.getenv("LANGSMITH_AGENT_ID")
    if agent_id:
        headers["Langsmith-Agent-Id"] = agent_id
        headers["Langsmith-Agent-Environment"] = os.getenv("LANGSMITH_AGENT_ENVIRONMENT") or "local"
    else:
        headers["Langsmith-Project"] = settings.project

    import requests
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    transport = requests.Session()
    transport.trust_env = False
    transport.max_redirects = 0
    return OTLPSpanExporter(
        endpoint=f"{base_url.rstrip('/')}/otel/v1/traces",
        headers=headers,
        timeout=settings.export_timeout,
        session=transport,
    )


def plain(value):
    if hasattr(value, "model_dump"):
        return plain(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items() if type(v).__name__ not in {"NotGiven", "Omit"}}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Operation:
    def __init__(self, session=None, active=None, name="", kind=""):
        self.session, self.active, self.name, self.kind = session, active, name, kind
        self.input_ref = self.output_ref = None
        self.parent = None
        self.requests = 0
        self.responses = []

    def attrs(self, **values):
        if not self.active:
            return
        try:
            for key, value in self._bounded_values(values, 4096).items():
                self.active.set_attribute(key, value)
        except Exception as exc:
            self.session.missing("attributes", exc)

    def event(self, name, **values):
        if not self.active:
            return
        try:
            self.active.add_event(name, self._bounded_values(values, 1000))
        except Exception as exc:
            self.session.missing("event", exc)

    def _bounded_values(self, values, limit):
        clean = redact_value(plain(values))
        encoded = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                   for key, value in clean.items() if value is not None}
        oversized = {key: clean[key] for key, value in encoded.items() if isinstance(value, str) and len(value) > limit}
        if oversized:
            reference = self.output_ref or self.body("span-details", oversized)
            if reference:
                encoded["codeswarm.details_ref"] = reference
            for key in oversized:
                encoded[key] = encoded[key][:limit] + " [see artifact]"
        return encoded

    def body(self, kind, value):
        if not self.session:
            return None
        try:
            record = self.session.artifacts.put_text(kind, json.dumps(redact_value(plain(value)), ensure_ascii=False), ".json")
            return record.id
        except Exception as exc:
            self.session.missing("artifact", exc)
            return None

    def error(self, exc):
        self.attrs(**{"codeswarm.error.type": type(exc).__name__, "codeswarm.error.message": str(exc)})
        if self.active:
            try:
                from opentelemetry.trace import Status, StatusCode
                self.active.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            except Exception:
                pass

    def response(self, value):
        try:
            self._response(value)
        except Exception as exc:
            if self.session:
                self.session.missing("response", exc)

    def _response(self, value):
        data = plain(value)
        self.output_ref = self.body("llm-wire-response", data)
        if self.session:
            with self.session._lock:
                ancestor = self.parent
                while ancestor:
                    ancestor.responses.append(self.output_ref)
                    ancestor = ancestor.parent
        usage = (data.get("usage") or data.get("usage_metadata") or data.get("usageMetadata") or {}) if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        self.attrs(**{"codeswarm.response_ref": self.output_ref, "output.value": json.dumps({"artifact_id": self.output_ref,
                      "preview": json.dumps(redact_value(data), ensure_ascii=False)[:600]}, ensure_ascii=False),
                      "output.mime_type": "application/json", "llm.token_count.prompt": usage.get("input_tokens", usage.get("prompt_tokens",
                          usage.get("prompt_token_count", usage.get("promptTokenCount", data.get("prompt_eval_count"))))),
                      "llm.token_count.completion": usage.get("output_tokens", usage.get("completion_tokens",
                          usage.get("candidates_token_count", usage.get("candidatesTokenCount", data.get("eval_count")))))})
        if isinstance(data, dict):
            choices = data.get("choices") or [{}]
            self.attrs(**{"codeswarm.finish_reason": data.get("status") or data.get("stop_reason") or choices[0].get("finish_reason")})
        if self.session:
            # Provider-native IDs are supplemented with normalized IDs by the compact recorder.
            self.session.register_response(data, self)


class _Sink:
    def __init__(self, session):
        self.session = session
        self.pending = queue.Queue(maxsize=session.settings.queue_size)
        self.stopping = threading.Event()
        self.exporter = None
        if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
            try:
                self.exporter = _langsmith_exporter(session.settings)
            except Exception as exc:
                session.missing("langsmith_configuration", exc)
        self.worker = threading.Thread(target=self._work, name="codeswarm-trace-export", daemon=True)
        self.worker.start()

    def on_start(self, span, parent_context=None):
        pass

    def _on_ending(self, span):
        pass

    def on_end(self, span):
        if self.stopping.is_set():
            self.session.missing("late_span")
            return
        try:
            self.pending.put_nowait(span)
        except queue.Full:
            self.session.missing("queue_full")

    def _work(self):
        while not self.stopping.is_set() or not self.pending.empty():
            try:
                first = self.pending.get(timeout=.05)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < self.session.settings.batch_size:
                try:
                    batch.append(self.pending.get_nowait())
                except queue.Empty:
                    break
            try:
                with self.session.span_path.open("a", encoding="utf-8") as stream:
                    for item in batch:
                        stream.write(json.dumps(json.loads(item.to_json()), ensure_ascii=False) + "\n")
                if self.exporter:
                    from opentelemetry.sdk.trace.export import SpanExportResult
                    if self.exporter.export(batch) != SpanExportResult.SUCCESS:
                        self.session.missing("export_failed")
            except Exception as exc:
                self.session.missing("export_or_local_write", exc)
            finally:
                for _ in batch:
                    self.pending.task_done()

    def force_flush(self, timeout_millis=2000):
        deadline = time.monotonic() + timeout_millis / 1000
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.01)
        return not self.pending.unfinished_tasks

    def shutdown(self):
        self.stopping.set()
        self.worker.join(self.session.settings.flush_timeout)
        if self.worker.is_alive():
            self.session.missing("flush_timeout")
        elif self.exporter:
            self.exporter.shutdown()


class TraceSession:
    def __init__(self, run_id, directory, artifacts, settings=None):
        self.run_id, self.directory, self.artifacts = run_id, Path(directory), artifacts
        self.settings = settings or TraceSettings()
        self.segment_id = uuid.uuid4().hex
        self.root = self.directory / "observability"
        self.span_path = self.root / f"spans-{self.segment_id}.jsonl"
        self.status_path = self.root / f"status-{self.segment_id}.json"
        self.failures, self.origins = {}, {}
        self.request_attempts = {}
        self._lock = threading.RLock()
        self.provider = self.tracer = self.sink = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps({"run_id": run_id, "segment_id": self.segment_id,
                "complete": False, "active": True}), encoding="utf-8")
        except Exception as exc:
            self.missing("directory", exc)
            return
        if self.settings.enabled:
            try:
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.sampling import ALWAYS_ON
                self.provider = TracerProvider(resource=Resource.create({"service.name": "codeswarm-agent",
                    "openinference.project.name": self.settings.project}), sampler=ALWAYS_ON, shutdown_on_exit=False)
                self.sink = _Sink(self)
                self.provider.add_span_processor(self.sink)
                self.tracer = self.provider.get_tracer("codeswarm.manual", "1")
            except Exception as exc:
                self.missing("initialization", exc)

    def missing(self, reason, exc=None):
        try:
            with self._lock:
                self.failures[reason] = self.failures.get(reason, 0) + 1
                self.status_path.write_text(json.dumps({"run_id": self.run_id, "segment_id": self.segment_id,
                    "complete": False, "missing": self.failures,
                    "last_error": redact_value(str(exc)) if exc else reason}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    @contextlib.contextmanager
    def span(self, name, kind="CHAIN", **attrs):
        operation, context = Operation(), None
        if self.tracer:
            try:
                from opentelemetry.context import Context
                context = self.tracer.start_as_current_span(name, context=Context() if name == "run" else None,
                    record_exception=False, set_status_on_exception=False)
                active = context.__enter__()
                operation = Operation(self, active, name, kind)
                operation.parent = None if name == "run" else _current_operation.get()
                operation.attrs(**{"openinference.span.kind": kind, "session.id": self.run_id,
                    "codeswarm.run_id": self.run_id, "codeswarm.segment_id": self.segment_id,
                    "codeswarm.run_directory": str(self.directory),
                    "metadata": {"run_id": self.run_id, "segment_id": self.segment_id}, **attrs})
            except Exception as exc:
                self.missing("span_start", exc)
        token = _current_operation.set(operation)
        try:
            yield operation
        except BaseException as exc:
            operation.error(exc)
            raise
        finally:
            operation.attrs(**{"codeswarm.actual_sdk_requests": operation.requests})
            _current_operation.reset(token)
            if context:
                try:
                    context.__exit__(None, None, None)
                except Exception as exc:
                    self.missing("span_end", exc)

    @contextlib.contextmanager
    def request(self, provider, payload):
        parent = current_operation()
        logical = logical_call.get() or (f"{parent.active.get_span_context().span_id:016x}" if parent.active else uuid.uuid4().hex)
        with self._lock:
            attempt = self.request_attempts.get(logical, 0) + 1
            self.request_attempts[logical] = attempt
        with self.span(f"llm.{provider}", "LLM", **{"llm.provider": provider}) as operation:
            with self._lock:
                ancestor = operation
                while ancestor:
                    ancestor.requests += 1
                    ancestor = ancestor.parent
            operation.input_ref = operation.body("llm-wire-input", payload)
            operation.attrs(**{"codeswarm.input_ref": operation.input_ref,
                "input.value": json.dumps({"artifact_id": operation.input_ref,
                    "preview": json.dumps(redact_value(plain(payload)), ensure_ascii=False)[:600]}, ensure_ascii=False), "input.mime_type": "application/json",
                "llm.model_name": payload.get("model") if isinstance(payload, dict) else None})
            operation.attrs(**{"codeswarm.logical_call_id": logical, "codeswarm.request_attempt": attempt})
            last_request.set(operation)
            yield operation

    def register_response(self, data, operation):
        if not operation.active:
            return
        def visit(value):
            if isinstance(value, dict):
                item_type = value.get("type")
                if isinstance(item_type, str) and item_type in {"function", "function_call", "tool_use"}:
                    key = value.get("call_id") or value.get("id")
                    if key:
                        self.origins[key] = operation
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        with self._lock:
            visit(data)

    def close(self):
        if self.sink:
            self.sink.shutdown()
        if not self.failures:
            with contextlib.suppress(Exception):
                self.status_path.write_text(json.dumps({"run_id": self.run_id, "segment_id": self.segment_id,
                    "complete": bool(self.tracer), "missing": {}}, ensure_ascii=False), encoding="utf-8")


_current_operation = ContextVar("codeswarm_current_operation", default=None)


def current_operation():
    return _current_operation.get() or Operation()


def span(name, kind="CHAIN", **attrs):
    session = observer.get()
    return session.span(name, kind, **attrs) if session else contextlib.nullcontext(Operation())


def event(name, **attrs):
    current_operation().event(name, **attrs)


@contextlib.contextmanager
def execution(run_id, directory, artifacts, settings=None, **attributes):
    settings = settings or TraceSettings()
    if not settings.enabled:
        yield Operation()
        return
    try:
        session = TraceSession(run_id, directory, artifacts, settings)
    except Exception:
        yield Operation()
        return
    token = observer.set(session)
    try:
        with session.span("run", "AGENT", **attributes) as active:
            yield active
    finally:
        observer.reset(token)
        try:
            session.close()
        except Exception as exc:
            session.missing("shutdown", exc)


def operation(name, kind="CHAIN", attributes=None):
    def decorate(function):
        @functools.wraps(function)
        async def asynchronous(*args, **kwargs):
            values = attributes(*args, **kwargs) if attributes else {}
            with span(name, kind, **values):
                return await function(*args, **kwargs)
        @functools.wraps(function)
        def synchronous(*args, **kwargs):
            values = attributes(*args, **kwargs) if attributes else {}
            with span(name, kind, **values):
                return function(*args, **kwargs)
        return asynchronous if inspect.iscoroutinefunction(function) else synchronous
    return decorate
