"""Per-worker context lifecycle around every model and tool call."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from codeswarm_agent.artifacts import ArtifactStore
from codeswarm_agent.context.dense import (
    BruteForceDenseIndex,
    EmbeddingProvider,
    SQLiteEmbeddingCache,
    USearchDenseIndex,
    encoding_identity,
)
from codeswarm_agent.context.fusion import fuse
from codeswarm_agent.context.indexer import RepositoryIndex
from codeswarm_agent.context.lifecycle import ContextLifecycle
from codeswarm_agent.context.observation import ObservationReducer
from codeswarm_agent.context.packer import ContextBudgetError, TokenCounter, pack
from codeswarm_agent.context.protocol import ContextItem, ContextPackage, RetrievalHit
from codeswarm_agent.context.reranker import HybridReranker
from codeswarm_agent.context.working_state import WorkingState
from codeswarm_agent.runtime.tool_policy import read_tool_name
from codeswarm_agent.telemetry import current_operation, event, span

if TYPE_CHECKING:
    from trae_agent.tools.base import ToolCall, ToolResult
    from trae_agent.utils.llm_clients.llm_basics import LLMMessage


@dataclass(frozen=True)
class ContextSettings:
    context_window_tokens: int = 32_768
    output_reserve_tokens: int = 4_096
    fixed_overhead_tokens: int = 2_048
    safety_margin_tokens: int = 2_048
    max_history_steps: int = 3
    max_observation_chars: int = 2_800
    top_chunks: int = 12
    reranker_mode: str = "feature"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    reranker_top_n: int = 16
    reranker_timeout: float = 4.0
    dense_backend: str = "brute_force"
    summary_trigger_ratio: float = 0.75
    summary_target_ratio: float = 0.50
    summary_max_tokens: int = 2048

    def __post_init__(self) -> None:
        if min(
            self.context_window_tokens, self.output_reserve_tokens,
            self.fixed_overhead_tokens, self.safety_margin_tokens,
            self.max_history_steps, self.max_observation_chars, self.top_chunks,
            self.reranker_top_n, self.reranker_timeout,
        ) <= 0:
            raise ValueError("context settings must be positive")
        if self.reranker_mode not in {"none", "feature", "hybrid"}:
            raise ValueError("reranker mode must be none, feature or hybrid")
        if self.dense_backend not in {"brute_force", "usearch"}:
            raise ValueError("dense backend must be brute_force or usearch")
        if not 0 < self.summary_target_ratio < self.summary_trigger_ratio < 1:
            raise ValueError("summary ratios must satisfy 0 < target < trigger < 1")
        if self.summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens must be positive")


class ContextEngine:
    def __init__(
        self,
        *,
        workspace: Path,
        trace_root: Path,
        artifacts: ArtifactStore,
        role: str,
        issue: str,
        goal: str,
        acceptance: tuple[str, ...],
        test_commands: tuple[str, ...],
        dependency_results: tuple,
        shared_evidence: tuple,
        model: str,
        settings: ContextSettings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.trace_root = trace_root.resolve()
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifacts
        self.role = role
        self.issue = issue
        self.goal = goal
        self.acceptance = acceptance
        self.test_commands = test_commands
        self.dependency_results = dependency_results
        self.shared_evidence = shared_evidence
        self.settings = settings or ContextSettings()
        self.counter = TokenCounter(model)
        self.index = RepositoryIndex(self.workspace, self.trace_root)
        self.index.refresh()
        self.embedding_provider = embedding_provider
        self.reranker = HybridReranker(
            mode=self.settings.reranker_mode,
            model=self.settings.reranker_model,
            revision=self.settings.reranker_revision,
            candidate_count=self.settings.reranker_top_n,
            top_n=self.settings.top_chunks,
            timeout_seconds=self.settings.reranker_timeout,
            cache_path=self.trace_root.parent / "reranker-cache.sqlite3",
        )
        self._dense: BruteForceDenseIndex | USearchDenseIndex | None = None
        self._dense_version: str | None = None
        self._dense_identity: str | None = None
        self._dense_effective_backend = "not-requested" if embedding_provider is not None else "disabled"
        self._dense_fallback_reason: str | None = None
        self._embedding_cache = SQLiteEmbeddingCache(
            self.trace_root.parent / "embedding-cache.sqlite3"
        )
        self.reducer = ObservationReducer(artifacts, self.settings.max_observation_chars)
        self.state = WorkingState(current_plan=goal)
        self._dirty_paths: set[str] = set()
        self._unknown_changes = False
        self._call_number = 0
        self._total_package_tokens = 0
        self._total_exclusions = 0
        self._reranker_calls = 0
        self._reranker_fallbacks = 0
        self.last_package: ContextPackage | None = None
        self.lifecycle = ContextLifecycle(self.counter, self.settings, self.artifacts)
        self.read_calls: dict[str, dict] = {}
        self._package_only = False
        self._query_lock = RLock()

    def observe(self, call: ToolCall, result: ToolResult) -> ToolResult:
        reduced = self.reducer.reduce(call, result)
        observation = self.reducer.by_call_id[call.call_id]
        normalized = read_tool_name(call.name)
        command = str(call.arguments.get("command", ""))
        is_test = normalized == "bash" and bool(re.search(
            r"\b(pytest|unittest|tox|jest|vitest|cargo test|go test|npm test)\b",
            command, re.I,
        ))
        source_path = None
        file_hash = None
        if normalized == "repositoryread" and call.arguments.get("operation") == "read":
            raw_path = call.arguments.get("path")
            if isinstance(raw_path, str):
                candidate = (self.workspace / raw_path).resolve()
                if self.workspace in candidate.parents:
                    source_path = candidate.relative_to(self.workspace).as_posix()
                    file_hash = self.index.file_hash(source_path)
        if normalized in {"repositoryread", "artifactread", "ckg"}:
            self.read_calls[call.call_id] = {"artifact_id": observation.artifact_id,
                                            "source_path": source_path, "file_hash": file_hash,
                                            "repository_version": self.index.repository_version,
                                            "version_bound": normalized in {"repositoryread", "ckg"} and source_path is None}
        claim = (
            f"{call.name} `{command[:180]}` success={result.success} "
            f"version={self.index.repository_version} artifact={observation.artifact_id}"
        )
        if not result.success and normalized not in {"repositoryread", "artifactread", "ckg"}:
            claim += f" excerpt={observation.preview[-700:]}"
        if is_test:
            claim += f" test_status={'passed' if result.success else 'failed'}"
        if not result.success and normalized not in {"repositoryread", "artifactread", "ckg"}:
            self.lifecycle.state["retrieval_failure"] = f"{call.name} {command[:300]}\n{(result.error or result.result or '')[-1200:]}"
        self.state.add_observation(
            claim=claim, artifact_id=observation.artifact_id,
            repository_version=self.index.repository_version, source_path=source_path,
            file_hash=file_hash, is_failure=not result.success, is_test=is_test,
        )
        if normalized in {"strreplacebasededittool", "jsonedittool"}:
            raw_path = call.arguments.get("path")
            if isinstance(raw_path, str) and raw_path:
                candidate = Path(raw_path)
                if candidate.is_absolute():
                    try:
                        raw_path = candidate.resolve().relative_to(self.workspace).as_posix()
                    except ValueError:
                        raw_path = ""
                if raw_path:
                    self._dirty_paths.add(str(raw_path).replace("\\", "/"))
            else:
                self._unknown_changes = True
        elif normalized in {"bash", "patchapply"}:
            self._unknown_changes = True
        return reduced

    def history_groups(self, execution):
        from trae_agent.utils.llm_clients.llm_basics import LLMMessage
        stale = {call_id for call_id, info in self.read_calls.items()
                 if (info["source_path"] and info["file_hash"] != self.index.file_hash(info["source_path"]))
                 or (info.get("version_bound") and info.get("repository_version") != self.index.repository_version)}
        groups = []
        for index, step in enumerate(execution.steps, 1):
            group = []
            if step.llm_response and step.llm_response.content:
                group.append(LLMMessage(role="assistant", content=step.llm_response.content))
            group.extend(LLMMessage(role="assistant", tool_call=call) for call in step.tool_calls or [])
            group.extend(self.lifecycle.sanitize(LLMMessage(role="user", tool_result=result), stale)
                         for result in step.tool_results or [])
            if getattr(step, "reflection", None):
                group.append(LLMMessage(role="assistant", content=step.reflection))
            groups.append((index, group))
        return groups

    async def prepare_messages(self, system_prompt, execution, pending, tools, summarize):
        self._package_only = True
        recent = self.history_groups(execution)[-1:]
        self._recent_reserve = self.counter.messages([
            *self.lifecycle.summary_message(), *(m for _, group in recent for m in group),
            *self._pending_tail(execution, pending),
        ])
        try:
            prefix = await asyncio.to_thread(self.model_messages, system_prompt, execution, pending, tools)
        finally:
            self._package_only = False
        groups = self.history_groups(execution)
        tail = self._pending_tail(execution, pending)
        tools_tokens = sum(self.counter.count(json.dumps(tool.json_definition(), ensure_ascii=False)) for tool in tools)
        messages = await self.lifecycle.prepare(prefix, groups, tail, tools_tokens, summarize,
                                                workspace_version=self.index.repository_version,
                                                checkpoint=getattr(self, "summary_checkpoint", None))
        estimated = self.lifecycle.cost(messages, tools_tokens)
        event("context_prepared", input_tokens=estimated, covered_through=self.lifecycle.state["covered_through"],
              summary_version=self.lifecycle.state["summary_version"])
        self.last_package = replace(self.last_package, compacted_steps=self.lifecycle.state["covered_through"],
                                    estimated_request_tokens=estimated)
        self._write_json(self.trace_root / "context_packages" / f"step-{self._call_number}.json", asdict(self.last_package))
        metrics = json.loads((self.trace_root / "metrics.json").read_text(encoding="utf-8"))
        metrics.update(latest_estimated_request_tokens=estimated,
                       summary_version=self.lifecycle.state["summary_version"],
                       covered_through=self.lifecycle.state["covered_through"],
                       summary_error=self.lifecycle.last_summary_error)
        self._write_json(self.trace_root / "metrics.json", metrics)
        self._last_messages = messages
        return messages

    @staticmethod
    def _pending_tail(execution, pending):
        # Initial task constraints come from the pinned package. Only genuine
        # new feedback is appended, never an entire regenerated raw history.
        return list(pending[-1:]) if (execution.steps and not execution.steps[-1].tool_calls
                                      and pending and pending[-1].role == "user"
                                      and pending[-1].content) else []

    def prospective_state(self, response):
        visible = {m.tool_result.call_id for m in self._last_messages if m.tool_result}
        calls = {key: info["artifact_id"] for key, info in self.read_calls.items() if key in visible}
        chunks = [item.id for item in self.last_package.items if item.kind == "code"]
        snapshot = self.lifecycle.prospective(response, calls, chunks)
        from codeswarm_agent.context.lifecycle import complete_response
        if complete_response(response) and snapshot.get("retrieval"):
            snapshot["retrieval"]["pending"] = {}
        snapshot.update(read_calls=self.read_calls, working_state=asdict(self.state))
        return snapshot

    def restore_state(self, snapshot):
        from codeswarm_agent.context.working_state import Fact
        self.refresh_workspace()
        self.lifecycle.restore(snapshot)
        self.read_calls = dict(snapshot.get("read_calls", {}))
        if snapshot.get("working_state"):
            raw = dict(snapshot["working_state"])
            raw["accepted_facts"] = [Fact(**item) for item in raw["accepted_facts"]]
            self.state = WorkingState(**raw)
            self.state.invalidate({fact.source_path: self.index.file_hash(fact.source_path)
                                   for fact in self.state.accepted_facts if fact.source_path},
                                  self.index.repository_version)

    def refresh_workspace(self):
        if self._dirty_paths or self._unknown_changes:
            dirty = set(self._dirty_paths)
            if self._unknown_changes:
                dirty.update(self.index.discover_changed_paths())
            changed = self.index.refresh(dirty)
            if changed:
                self.state.invalidate(
                    {path: self.index.file_hash(path) for path in changed},
                    self.index.repository_version,
                )
                self.state.changes.extend(changed)
                # A test run before this diff is historical evidence, not current verification.
                self.state.verification = [f"STALE after code change: {item}"
                                           for item in self.state.verification[-8:]]
            self._dirty_paths.clear()
            self._unknown_changes = False

    def model_messages(
        self,
        system_prompt: str,
        execution,
        pending: list[LLMMessage],
        tools: list,
    ) -> list[LLMMessage]:
        from trae_agent.utils.llm_clients.llm_basics import LLMMessage
        self._call_number += 1
        self.refresh_workspace()
        history, compacted = self._bounded_history(execution, pending)
        tool_schema_tokens = sum(
            self.counter.count(json.dumps(tool.json_definition(), ensure_ascii=False))
            for tool in tools
        )
        available = (
            self.settings.context_window_tokens
            - self.settings.output_reserve_tokens
            - self.settings.fixed_overhead_tokens
            - self.settings.safety_margin_tokens
            - self.counter.count(system_prompt)
            - tool_schema_tokens
            - self.counter.messages(history)
            - (getattr(self, "_recent_reserve", 0) if self._package_only else 0)
        )
        package = self._build_package(available)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=package.render()),
            *history,
        ]
        estimated_request = (
            self.counter.messages(messages)
            + tool_schema_tokens
            + self.settings.fixed_overhead_tokens
        )
        if (
            estimated_request + self.settings.output_reserve_tokens
            > self.settings.context_window_tokens
        ):
            raise ContextBudgetError(
                f"estimated request {estimated_request} plus output reserve "
                f"{self.settings.output_reserve_tokens} exceeds configured window "
                f"{self.settings.context_window_tokens}"
            )
        package = replace(
            package, compacted_steps=compacted,
            estimated_request_tokens=estimated_request,
        )
        self.last_package = package
        self._total_package_tokens += package.input_tokens
        self._total_exclusions += len(package.excluded)
        self._write_json(
            self.trace_root / "context_packages" / f"step-{self._call_number}.json",
            asdict(package),
        )
        if compacted:
            old_steps = execution.steps[:min(compacted, len(execution.steps))]
            history_record = self.artifacts.put_text(
                "compacted-history",
                json.dumps([
                    {
                        "step": step.step_number if hasattr(step, "step_number") else index + 1,
                        "response": step.llm_response.content if step.llm_response else None,
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments}
                            for call in step.tool_calls or []
                        ],
                        "tool_results": [
                            {"name": result.name, "success": result.success,
                             "result": result.result, "error": result.error}
                            for result in step.tool_results or []
                        ],
                    }
                    for index, step in enumerate(old_steps)
                ], ensure_ascii=False),
                ".json",
            )
            self.state.artifact_refs.append(history_record.id)
            self._append("compaction_records.jsonl", {
                "step": self._call_number, "compacted_steps": compacted,
                "working_state": self.state.compact(),
                "history_artifact_id": history_record.id,
                "artifact_refs": self.state.artifact_refs[-30:],
            })
        self._write_json(self.trace_root / "metrics.json", {
            "model_calls": self._call_number,
            "latest_estimated_request_tokens": estimated_request,
            "total_package_tokens": self._total_package_tokens,
            "excluded_items": self._total_exclusions,
            "tokenizer": self.counter.name,
            "repository_version": self.index.repository_version,
            "embedding_enabled": self.embedding_provider is not None,
            "dense": {
                "requested_backend": self.settings.dense_backend,
                "effective_backend": self._dense_effective_backend,
                "fallback": self._dense_fallback_reason is not None,
                "fallback_reason": self._dense_fallback_reason,
            },
            "index": self.index.metrics,
            "reranker": {
                "requested_mode": self.settings.reranker_mode,
                "model": self.settings.reranker_model,
                "revision": self.settings.reranker_revision,
                "calls": self._reranker_calls,
                "fallbacks": self._reranker_fallbacks,
                "fallback_rate": (
                    self._reranker_fallbacks / self._reranker_calls
                    if self._reranker_calls else 0.0
                ),
            },
        })
        return messages

    def _dense_search(self, question):
        version = self.index.repository_version
        identity = encoding_identity(self.embedding_provider, "document")
        if self._dense_version != version or self._dense_identity != identity:
            self._dense_fallback_reason = None
            if self.settings.dense_backend == "usearch":
                try:
                    self._dense = USearchDenseIndex(
                        self.index.chunks, self.embedding_provider, self._embedding_cache,
                        index_path=self.trace_root.parent / "dense-index.usearch")
                except Exception as exc:
                    self._dense_fallback_reason = f"ANN fallback: {type(exc).__name__}: {exc}"
                    self._dense = BruteForceDenseIndex(self.index.chunks, self.embedding_provider, self._embedding_cache)
            else:
                self._dense = BruteForceDenseIndex(self.index.chunks, self.embedding_provider, self._embedding_cache)
            self._dense_version = version
            self._dense_identity = identity
        hits = self._dense.search(question, top_n=30)
        self._dense_effective_backend = "usearch" if isinstance(self._dense, USearchDenseIndex) else "brute_force"
        return hits

    def _retrieve(self, question, trigger, scope=""):
        from codeswarm_agent.context.retrieval import retrieve
        version = self.index.repository_version
        question = question.replace("\\", "/").replace(self.workspace.as_posix() + "/", "")
        current = tuple(c for c in self.index.chunks if c.repository_version == version
                        and self.index.file_hash(c.path) == c.file_hash)
        recall = retrieve(current, question, scope=scope,
                          dense_search=self._dense_search if self.embedding_provider is not None else None)
        self._dense_fallback_reason = recall.dense_fallback or self._dense_fallback_reason
        if recall.dense_fallback:
            self._dense_effective_backend = "unavailable" if self.embedding_provider is not None else "disabled"
        recalled = fuse(recall.rankings, recall.query.lexical, top_n=40)
        outcome = self.reranker.rerank(recalled, recall.query.semantic, role=self.role,
                                      failure=self.state.latest_failure, changed_paths=tuple(self.state.changes[-30:]))
        self._reranker_calls += int(bool(recalled))
        self._reranker_fallbacks += int(outcome.fallback)
        status = {"status": recall.status, "reason": recall.reason, "expanded": int(recall.expanded),
                  "dense_fallback": recall.dense_fallback, "source_version": version}
        self._append("retrieval_queries.jsonl", {"step": self._call_number, "version": version,
                                                "trigger": trigger, "scope": scope, **asdict(recall.query), **status})
        self._append("retrieval_results.jsonl", {"step": self._call_number, "version": version, **status,
            "channels": {name: [{"chunk_id": c.id, "path": c.path, "score": score} for c, score in values]
                         for name, values in recall.rankings.items()}})
        self._append("fusion_trace.jsonl", {"step": self._call_number, "version": version,
            "recall_hits": [{"chunk_id": h.chunk.id, "score": h.score, "channels": h.channels, "ranks": h.ranks} for h in recalled],
            "reranker": {key: getattr(outcome, key) for key in ("requested_mode", "effective_mode", "model", "revision", "fallback", "fallback_reason", "latency_ms")},
            "hits": [{"chunk_id": h.chunk.id, "path": h.chunk.path, "symbol": h.chunk.symbol, "score": h.score,
                      "channels": h.channels, "ranks": h.ranks} for h in outcome.hits]})
        observed = current_operation()
        report = {"query": question, "scope": scope, "trigger": trigger, "version": version, **status,
                  "channels": {key: [{"id": c.id, "score": score} for c, score in values] for key, values in recall.rankings.items()},
                  "ranked": [{"id": h.chunk.id, "ranks": h.ranks, "score": h.score} for h in outcome.hits]}
        observed.attrs(**{"codeswarm.retrieval_ref": observed.body("retrieval-report", report),
                          "retrieval.status": recall.status, "retrieval.trigger": trigger,
                          "input.value": question[:600], "input.mime_type": "text/plain"})
        return list(outcome.hits), status

    def query_repository(self, question, scope=""):
        with self._query_lock, span("retrieval.query", "RETRIEVER") as observed:
            self.refresh_workspace()
            hits, status = self._retrieve(question[:3000], "model-query", scope)
            version = self.index.repository_version
            observed.attrs(**{"retrieval.selected_ids": [hit.chunk.id for hit in hits[:6]], "codeswarm.workspace_version": version})
        return json.dumps({**status, "workspace_version": version,
            "candidates": [{"path": hit.chunk.path, "symbol": hit.chunk.symbol,
                            "start_line": hit.chunk.start_line, "end_line": hit.chunk.end_line,
                            "text": hit.chunk.text[:2800]} for hit in hits[:6]]}, ensure_ascii=False)

    def _recall_signal(self):
        failure = self.lifecycle.state.get("retrieval_failure", "")
        signal = ("error:" + failure) if failure else "initial:" + self.goal
        return hashlib.sha256((self.goal + "\n" + signal).encode()).hexdigest()

    def _build_package(self, budget_tokens: int) -> ContextPackage:
        if self._recall_signal() in self.lifecycle.state.get("retrieval", {}).get("seen", []):
            return self._build_package_impl(budget_tokens)
        with span("retrieval.automatic", "RETRIEVER", **{"codeswarm.workspace_version": self.index.repository_version}) as observed:
            package = self._build_package_impl(budget_tokens)
            observed.attrs(**{"retrieval.selected_ids": [item.id for item in package.items if item.kind == "code"],
                              "retrieval.excluded": [asdict(item) for item in package.excluded]})
            return package

    def _build_package_impl(self, budget_tokens: int) -> ContextPackage:
        version = self.index.repository_version
        retrieval = self.lifecycle.state.setdefault("retrieval", {"seen": [], "pending": {}, "status": None})
        failure = self.lifecycle.state.get("retrieval_failure", "")
        signal = self._recall_signal()
        if signal not in retrieval["seen"]:
            question = (failure + "\nCurrent goal: " + self.goal[:800]) if failure else self.goal
            if self.issue != self.goal and not failure:
                question += "\n" + self.issue[:1000]
            hits, status = self._retrieve(question, "new-error" if failure else "initial-or-goal-change")
            retrieval["seen"].append(signal)
            retrieval["pending"].update({hit.chunk.id: f"chunk-{hit.chunk.id}@{signal[:16]}"
                                         for hit in hits[:self.settings.top_chunks]})
            retrieval["status"] = status
        current = {chunk.id: chunk for chunk in self.index.chunks
                   if chunk.repository_version == version and self.index.file_hash(chunk.path) == chunk.file_hash}
        retrieval["pending"] = {key: item_id for key, item_id in retrieval["pending"].items() if key in current}
        hits = [RetrievalHit(current[key], .01, (), {}) for key in retrieval["pending"]
                if retrieval["pending"][key] not in self.lifecycle.state["consumed_chunks"]]
        required = [
            ContextItem("issue", "task", self.issue, "user issue", "stable", 100, True),
            ContextItem("goal", "goal", self.goal, "assigned task", "stable", 100, True),
            ContextItem("workspace", "workspace", str(self.workspace),
                        "runtime", "stable", 100, True),
            ContextItem("workspace-version", "workspace_version", version,
                        "runtime", "stable", 100, True),
        ]
        if retrieval.get("status"):
            status = dict(retrieval["status"])
            if status.get("source_version") != version:
                status.update(status="stale", reason="prior recall belongs to an older version; query current code if needed")
            required.append(ContextItem("retrieval-status", "retrieval_status", json.dumps(status, ensure_ascii=False), "retrieval rules", version, 100, True))
        if self.acceptance:
            required.append(ContextItem(
                "acceptance", "acceptance", "\n".join(self.acceptance),
                "user acceptance", "stable", 100, True,
            ))
        if self.test_commands:
            required.append(ContextItem(
                "tests", "required_tests", "\n".join(self.test_commands),
                "user test commands", "stable", 100, True,
            ))
        if self.state.latest_failure:
            required.append(ContextItem(
                "latest-failure", "failure", self.state.latest_failure,
                "observed tool result", version, 90, True,
            ))
        if self.role in {"reviewer", "tester"} and self.dependency_results:
            dependencies = [
                f"{result.task_id}: {result.status.value}; "
                f"artifacts={','.join(result.artifact_ids[:8])}"
                for result in self.dependency_results[-25:]
            ]
            required.append(ContextItem(
                "dependency-index", "dependency_status", "\n".join(dependencies),
                "blackboard references", "stable", 90, True,
            ))
        if self.role == "reviewer":
            for result in self.dependency_results:
                for verification in result.verifications:
                    try:
                        output = self.artifacts.read_text(verification.artifact_id)[:2000]
                    except (KeyError, ValueError, OSError):
                        output = "<test output unavailable>"
                    required.append(ContextItem(
                        f"verification-{verification.artifact_id}", "verification",
                        f"command={verification.command} passed={verification.passed} "
                        f"workspace_version={verification.workspace_version} "
                        f"patch_sha256={verification.patch_sha256}\n"
                        f"Untrusted test output excerpt:\n{output}",
                        f"blackboard:{result.task_id}", "stable", 95, True,
                        artifact_id=verification.artifact_id,
                    ))
        if self.role in {"developer", "tester", "reviewer"}:
            diff = self._current_diff()
            if diff:
                record = self.artifacts.put_text("working-diff", diff, ".patch")
                required.append(ContextItem(
                    "current-diff", "current_diff",
                    f"artifact={record.id}\n{diff[:3500]}" +
                    ("\n[diff continues in artifact]" if len(diff) > 3500 else ""),
                    "live workspace diff", version, 90, True, artifact_id=record.id,
                ))
        candidates: list[ContextItem] = []
        if self.state.artifact_refs:
            required.append(ContextItem(
                "working-state", "working_state", self.state.compact(),
                "deterministic reducer", version, 100, True,
            ))
        for result in self.dependency_results:
            refs = ", ".join(result.artifact_ids)
            candidates.append(ContextItem(
                f"dependency-{result.task_id}", "dependency",
                f"role={result.role.value} status={result.status.value}\n"
                f"summary={result.summary[:3000]}\nartifacts={refs}",
                f"blackboard:{result.task_id}", "stable", 85 if self.role == "reviewer" else 55,
            ))
            for artifact_id in result.artifact_ids:
                try:
                    record = self.artifacts.get(artifact_id)
                except KeyError:
                    continue
                if record.kind != "patch":
                    continue
                patch = self.artifacts.read_text(artifact_id)
                if len(patch) > 24_000:
                    patch = (
                        patch[:12_000]
                        + "\n[...patch omitted; read artifact...]\n"
                        + patch[-4_000:]
                    )
                candidates.append(ContextItem(
                    f"patch-{artifact_id}", "patch", patch,
                    f"artifact:{artifact_id}", "stable",
                    95 if self.role in {"reviewer", "tester"} else 60,
                    artifact_id=artifact_id,
                ))
        for evidence in self.shared_evidence[-30:]:
            if self.role == "reviewer" and evidence.task_id == "reproduce":
                # Base-revision reproduction is historical and can contradict a
                # fresh integrated test. Keep it out of the final verdict context.
                continue
            candidates.append(ContextItem(
                f"evidence-{evidence.id}", "evidence",
                f"CROSS-WORKER CLAIM (verify against current workspace): "
                f"{evidence.kind}: {evidence.claim} [artifact={evidence.artifact_id}]",
                f"blackboard:{evidence.task_id}", "stable", 45,
                artifact_id=evidence.artifact_id,
            ))
        for hit in hits[: self.settings.top_chunks]:
            chunk = hit.chunk
            item_id = retrieval["pending"][chunk.id]
            if item_id in self.lifecycle.state["consumed_chunks"]:
                continue
            candidates.append(ContextItem(
                id=item_id, kind="code",
                text=f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                     f"symbol={chunk.symbol or '-'}\n{chunk.text}",
                source=f"repository:{chunk.path}", repository_version=version,
                priority=40 + hit.score * 1000, file_hash=chunk.file_hash,
            ))
        arguments = {"required": required, "candidates": candidates,
                     "repository_version": version, "role": self.role, "counter": self.counter}
        target = min(budget_tokens, int(self.lifecycle.available * self.settings.summary_target_ratio)) if self._package_only else budget_tokens
        try:
            return pack(**arguments, budget_tokens=target)
        except ContextBudgetError:
            if target == budget_tokens:
                raise
            return pack(**arguments, budget_tokens=budget_tokens)

    def _bounded_history(
        self, execution, pending: list[LLMMessage]
    ) -> tuple[list[LLMMessage], int]:
        if self._package_only:
            return [], 0
        groups = self.history_groups(execution)
        return [m for i, group in groups if i > self.lifecycle.state["covered_through"] for m in group] + self._pending_tail(execution, pending), self.lifecycle.state["covered_through"]

    def _current_diff(self) -> str:
        result = subprocess.run(
            ["git", "--no-pager", "diff", "--binary"], cwd=self.workspace,
            capture_output=True, text=True, timeout=20, check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    def _append(self, name: str, payload: dict) -> None:
        with (self.trace_root / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
