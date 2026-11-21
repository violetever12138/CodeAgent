"""Offline context-engine checks; no model API or vector service is required."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from codeagent.artifacts import ArtifactStore
from codeagent.context.dense import DenseIndex
from codeagent.context.engine import ContextEngine, ContextSettings
from codeagent.context.fusion import fuse, symbol_search
from codeagent.context.indexer import RepositoryIndex
from codeagent.context.lexical import BM25Index, code_terms
from codeagent.context.observation import ObservationReducer
from codeagent.context.packer import ContextBudgetError, TokenCounter, pack
from codeagent.context.protocol import CodeChunk, ContextItem, RetrievalHit
from codeagent.context.reranker import HybridReranker
from codeagent.context.working_state import WorkingState
from codeagent.evaluation.context_metrics import RelevanceLabel, retrieval_metrics
from codeagent.protocol import (
    Evidence,
    VerificationRecord,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "auth.py").write_text(
        "import time\n\ndef authenticate_user(token):\n"
        "    if token == 'expired':\n        raise ValueError('expired token')\n"
        "    return True\n",
        encoding="utf-8",
    )
    (root / "tests.md").write_text("acceptance login expiry\n" * 90, encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    return root


def test_content_hash_invalidates_same_dirty_status(tmp_path):
    root = _repository(tmp_path)
    index = RepositoryIndex(root, tmp_path / "trace")
    index.refresh()
    assert "authenticate_user" in {chunk.symbol for chunk in index.chunks}
    first = index.repository_version
    source = root / "auth.py"
    source.write_text(source.read_text(encoding="utf-8").replace(
        "authenticate_user", "authenticate_login"
    ), encoding="utf-8")
    assert index.refresh() == ("auth.py",)
    second = index.repository_version
    source.write_text(source.read_text(encoding="utf-8").replace(
        "authenticate_login", "authenticate_session"
    ), encoding="utf-8")
    assert index.refresh() == ("auth.py",)
    assert len({first, second, index.repository_version}) == 3
    assert "authenticate_session" in {chunk.symbol for chunk in index.chunks}
    assert "authenticate_login" not in {chunk.symbol for chunk in index.chunks}
    manifest = json.loads((tmp_path / "trace" / "index_manifest.json").read_text())
    assert manifest["repository_version"] == index.repository_version


def test_incremental_refresh_only_rechunks_dirty_file_and_tombstones_delete(tmp_path):
    root = _repository(tmp_path)
    index = RepositoryIndex(root, tmp_path / "trace")
    index.refresh()
    scanned = index.metrics["scanned_files"]
    rechunked = index.metrics["rechunked_files"]
    source = root / "auth.py"
    source.write_text(source.read_text().replace("return True", "return token != ''"))

    assert index.refresh(("auth.py",)) == ("auth.py",)
    assert index.metrics["scanned_files"] - scanned == 1
    assert index.metrics["rechunked_files"] - rechunked == 1
    assert not (tmp_path / "trace" / "chunks.jsonl").exists()

    source.unlink()
    assert index.refresh(("auth.py",)) == ("auth.py",)
    assert index.file_hash("auth.py") is None
    reopened = RepositoryIndex(root, tmp_path / "trace-reopened",
                               store_path=index.store.path)
    assert reopened.file_hash("auth.py") is None


def test_base_snapshot_is_built_once_and_incremental_matches_clean_rebuild(tmp_path):
    root = _repository(tmp_path)
    shared_store = tmp_path / "run" / "repository-index.sqlite3"
    first = RepositoryIndex(root, tmp_path / "run" / "context" / "worker-a",
                            store_path=shared_store)
    first.refresh()
    worktree = tmp_path / "worker-b"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")

    overlay = RepositoryIndex(worktree, tmp_path / "run" / "context" / "worker-b",
                              store_path=shared_store)
    overlay.refresh()
    assert overlay.metrics["full_rebuilds"] == 0
    assert overlay.metrics["scanned_files"] == 0

    source = worktree / "auth.py"
    source.write_text(source.read_text().replace("return True", "return bool(token)"))
    overlay.refresh(("auth.py",))
    clean = RepositoryIndex(worktree, tmp_path / "clean-trace",
                            store_path=tmp_path / "clean.sqlite3")
    clean.refresh()
    assert [asdict(chunk) for chunk in overlay.chunks] == [
        asdict(chunk) for chunk in clean.chunks
    ]


def test_bm25_symbol_and_rrf_channels(tmp_path):
    index = RepositoryIndex(_repository(tmp_path), tmp_path / "trace")
    index.refresh()
    assert "authenticate" in code_terms("authenticateUser_token")
    bm25 = BM25Index(index.chunks).search("authenticate_user expired token")
    symbols = symbol_search(index.chunks, "authenticate_user")
    hits = fuse({"bm25": bm25, "symbol": symbols}, "authenticate_user")
    assert hits and hits[0].chunk.path == "auth.py"
    assert "bm25" in hits[0].channels
    assert any("symbol" in hit.channels for hit in hits)


class _FakeEmbeddings:
    model_id = "fake-v1"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "auth" in text else [0.0, 1.0] for text in texts]


def test_dense_is_explicit_optional_channel(tmp_path):
    index = RepositoryIndex(_repository(tmp_path), tmp_path / "trace")
    index.refresh()
    provider = _FakeEmbeddings()
    cache = {}
    dense = DenseIndex(index.chunks, provider, cache)
    assert dense.search("auth login")[0][0].path == "auth.py"
    first_calls = provider.calls
    DenseIndex(index.chunks, provider, cache)
    assert provider.calls == first_calls


def test_packer_preserves_mandatory_and_reports_exclusions():
    counter = TokenCounter()
    mandatory = ContextItem("issue", "task", "must retain", "user", "stable", 100, True)
    stale = ContextItem("old", "code", "old implementation", "repo", "old", 50)
    huge = ContextItem("huge", "code", "x" * 4000, "repo", "new", 20)
    package = pack(
        required=[mandatory], candidates=[stale, huge], budget_tokens=300,
        repository_version="new", role="developer", counter=counter,
    )
    assert [item.id for item in package.items] == ["issue"]
    assert {item.id: item.reason for item in package.excluded} == {
        "old": "stale_version", "huge": "token_budget",
    }
    with pytest.raises(ContextBudgetError):
        pack(required=[mandatory], candidates=[], budget_tokens=5,
             repository_version="new", role="developer", counter=counter)


def test_packer_counts_rendered_json_escaping_within_budget():
    counter = TokenCounter()
    required = ContextItem(
        "workspace", "workspace", r"D:\repo\nested\workspace",
        r"runtime\source", "stable", 100, True,
    )
    candidate = ContextItem(
        "trace", "evidence", r"path=D:\repo\tests\test_app.py" * 20,
        r"blackboard\worker", "stable", 50,
    )
    package = pack(
        required=[required], candidates=[candidate], budget_tokens=500,
        repository_version="stable", role="reviewer", counter=counter,
    )

    assert counter.count(package.render()) <= package.input_tokens <= 500


@dataclass
class _Call:
    name: str
    call_id: str
    arguments: dict


@dataclass
class _Result:
    call_id: str
    name: str
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None


def test_long_observation_keeps_tail_failure_and_full_artifact(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    reducer = ObservationReducer(artifacts, max_preview_chars=900)
    raw = "warning\n" * 2000 + "\nAssertionError: tail failure\n"
    result = reducer.reduce(_Call("bash", "call-1", {}), _Result("call-1", "bash", False,
                                                                raw, "exit 1"))
    artifact_id = reducer.by_call_id["call-1"].artifact_id
    assert "AssertionError: tail failure" in result.result
    assert len(artifacts.read_text(artifact_id)) > len(result.result)
    assert "AssertionError: tail failure" in artifacts.read_lines(artifact_id, 2000, 2003)


def test_artifact_detects_tampering_and_refuses_outside_path(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    record = artifacts.put_text("log", "trusted evidence")
    Path(record.path).write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.read_text(record.id)
    with sqlite3.connect(artifacts.catalog_path) as connection:
        connection.execute(
            "UPDATE artifacts SET filename = ? WHERE id = ?", ("../outside.txt", record.id)
        )
    reopened = ArtifactStore(artifacts.root)
    with pytest.raises(ValueError, match="escapes store"):
        reopened.get(record.id)

    legacy_root = tmp_path / "legacy-artifacts"
    legacy_root.mkdir()
    (legacy_root / "index.json").write_text(json.dumps([
        {**record.__dict__, "path": str(tmp_path / "outside.txt")}
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes store"):
        ArtifactStore(legacy_root)


def test_artifact_preserves_crlf_bytes_used_by_content_hash(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    content = "command output\r\nsecond line\r\n"

    record = artifacts.put_text("log", content, ".log")

    assert Path(record.path).read_bytes() == content.encode("utf-8")
    assert artifacts.read_text(record.id) == content


def test_artifact_logical_key_tracks_latest_without_rewriting_json(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    first = artifacts.put_text("report", "v1", logical_key="final-report")
    second = artifacts.put_text("report", "v2", logical_key="final-report")

    latest = artifacts.latest("final-report")
    assert latest.id == second.id
    assert latest.supersedes_id == first.id
    assert artifacts.metrics()["full_catalog_rewrites"] == 0
    assert not (artifacts.root / "index.json").exists()


def test_artifact_catalog_migrates_legacy_and_supports_concurrent_workers(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    payload = b"legacy evidence"
    digest = hashlib.sha256(payload).hexdigest()
    artifact_id = f"log-{digest[:16]}"
    artifact_path = root / f"{artifact_id}.txt"
    artifact_path.write_bytes(payload)
    (root / "index.json").write_text(json.dumps([{
        "id": artifact_id, "kind": "log", "path": str(artifact_path),
        "size": len(payload), "sha256": digest,
    }]), encoding="utf-8")
    migrated = ArtifactStore(root)
    assert migrated.read_text(artifact_id) == "legacy evidence"
    assert (root / "index.legacy.json").exists()

    workers = [ArtifactStore(root) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(
            lambda item: workers[item % 4].put_text("worker-log", f"value-{item}"),
            range(40),
        ))
    assert len({record.id for record in records}) == 40
    assert ArtifactStore(root).metrics()["artifact_count"] == 41


def test_feature_reranker_diversifies_adjacent_chunks_and_offline_metrics(tmp_path):
    def chunk(identifier, path, start, text, symbol=None):
        return CodeChunk(
            identifier, path, "python", symbol, start, start + 20, text,
            identifier, "version", "symbol",
        )

    chunks = [
        chunk("a", "auth.py", 1, "authenticate expired token", "authenticate"),
        chunk("b", "auth.py", 10, "authenticate expired token branch", "authenticate"),
        chunk("c", "tests/test_auth.py", 1, "test authenticate expired token", "test_expired"),
    ]
    hits = [RetrievalHit(item, 1 / rank, ("bm25",), {"bm25": rank})
            for rank, item in enumerate(chunks, 1)]
    reranker = HybridReranker(mode="feature", top_n=2,
                              cache_path=tmp_path / "rerank.sqlite3")
    outcome = reranker.rerank(hits, "authenticate expired token", role="tester")

    assert {hit.chunk.path for hit in outcome.hits} == {"auth.py", "tests/test_auth.py"}
    metrics = retrieval_metrics(
        outcome.hits, RelevanceLabel(("auth.py", "tests/test_auth.py"), ()),
        top_n=2, current_repository_version="version",
    )
    assert metrics == {
        "recall_at_2": 1.0, "mrr_at_2": 1.0, "ndcg_at_2": 1.0,
        "stale_hit_rate": 0.0, "file_diversity": 1.0,
    }


def test_hybrid_reranker_records_explicit_fallback(tmp_path, monkeypatch):
    chunk = CodeChunk("a", "auth.py", "python", "authenticate", 1, 2,
                      "def authenticate(): pass", "hash", "version")
    hit = RetrievalHit(chunk, 1.0, ("bm25",), {"bm25": 1})
    reranker = HybridReranker(mode="hybrid", top_n=1,
                              cache_path=tmp_path / "rerank.sqlite3")
    monkeypatch.setattr(reranker, "_cross_scores",
                        lambda *_: (_ for _ in ()).throw(ImportError("model unavailable")))

    outcome = reranker.rerank([hit], "authenticate", role="developer")

    assert outcome.effective_mode == "feature"
    assert outcome.fallback is True
    assert "model unavailable" in outcome.fallback_reason


def test_working_state_never_promotes_hypothesis_and_invalidates_file():
    state = WorkingState(hypotheses=["race condition"])
    state.add_observation(claim="auth.py has authenticate_user", artifact_id="a1",
                          repository_version="v1", source_path="auth.py", file_hash="old")
    assert "HYPOTHESIS (unverified): race condition" in state.compact()
    state.invalidate({"auth.py": "new"}, "v2")
    assert state.accepted_facts[0].status == "stale"
    assert "auth.py has authenticate_user" not in state.compact()


def test_reviewer_receives_live_diff_and_fresh_verification_output(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    (root / "auth.py").write_text("def authenticate_user(token):\n    return False\n", encoding="utf-8")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    log = artifacts.put_text("test-output", "1 passed in 0.1s", ".log")
    tester = WorkerResult(
        task_id="test-integrated-0", attempt=0, role=WorkerRole.TESTER,
        status=WorkerStatus.SUCCEEDED, summary="pytest -q: passed",
        verifications=(VerificationRecord(
            command="pytest -q", passed=True, artifact_id=log.id,
            workspace_version="patched-v1", patch_sha256="a" * 64,
        ),),
    )
    module = types.ModuleType("trae_agent.utils.llm_clients.llm_basics")

    @dataclass
    class Message:
        role: str
        content: str | None = None
        tool_call: object | None = None
        tool_result: object | None = None

    module.LLMMessage = Message
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = ContextEngine(
        workspace=root, trace_root=tmp_path / "context", artifacts=artifacts,
        role="reviewer", issue="Fix auth", goal="Review integrated patch",
        acceptance=("auth fixed",), test_commands=("pytest -q",),
        dependency_results=(tester,), shared_evidence=(Evidence(
            id="base-failure", producer="tester", task_id="reproduce",
            kind="test_result", claim="BASE TEST FAILED: current patch still broken",
        ),), model="unknown",
    )
    messages = engine.model_messages("System", SimpleNamespace(steps=[]), [], [])
    context = messages[1].content
    assert "return False" in context
    assert "1 passed in 0.1s" in context
    assert "patch_sha256=" in context
    assert "BASE TEST FAILED" not in context


def test_context_package_keeps_uncovered_history_and_constraints_below_threshold(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    engine = ContextEngine(
        workspace=root, trace_root=tmp_path / "context", artifacts=artifacts,
        role="developer", issue="Fix expired token in authenticate_user",
        goal="Change auth.py", acceptance=("expired token must be rejected",),
        test_commands=("pytest tests/test_auth.py",), dependency_results=(),
        shared_evidence=(), model="unknown",
        settings=ContextSettings(max_history_steps=1),
    )
    fake_module = types.ModuleType("trae_agent.utils.llm_clients.llm_basics")

    @dataclass
    class Message:
        role: str
        content: str | None = None
        tool_call: object | None = None
        tool_result: object | None = None

    fake_module.LLMMessage = Message
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    old_steps = [SimpleNamespace(
        llm_response=SimpleNamespace(content="searched"), tool_calls=[], tool_results=[]
    ) for _ in range(4)]
    old_steps[-1].reflection = "Tool failure observations; correct the arguments"
    execution = SimpleNamespace(steps=old_steps)
    messages = engine.model_messages("System rules", execution, [], [])
    assert messages[0].role == "system"
    assert "expired token must be rejected" in messages[1].content
    assert "authenticate_user" in messages[1].content
    assert "pytest tests/test_auth.py" in messages[1].content
    assert any(message.content == old_steps[-1].reflection for message in messages)
    assert engine.last_package.compacted_steps == 0
    assert sum(message.content == "searched" for message in messages) == 4
    assert engine.last_package.estimated_request_tokens < 32768 - 4096
    assert (tmp_path / "context" / "context_packages" / "step-1.json").exists()
    assert not (tmp_path / "context" / "compaction_records.jsonl").exists()
    assert (tmp_path / "context" / "retrieval_results.jsonl").exists()
    assert (tmp_path / "context" / "metrics.json").exists()


def test_worker_context_refreshes_after_edit_and_stales_old_evidence(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    module = types.ModuleType("trae_agent.utils.llm_clients.llm_basics")

    @dataclass
    class Message:
        role: str
        content: str | None = None
        tool_call: object | None = None
        tool_result: object | None = None

    module.LLMMessage = Message
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engine = ContextEngine(
        workspace=root, trace_root=tmp_path / "context",
        artifacts=ArtifactStore(tmp_path / "artifacts"), role="developer",
        issue="Fix authenticate_user", goal="Fix auth.py", acceptance=(),
        test_commands=(), dependency_results=(), shared_evidence=(), model="unknown",
    )
    engine.observe(
        _Call("repository_read", "read-1", {"operation": "read", "path": "auth.py"}),
        _Result("read-1", "repository_read", True, "authenticate_user exists"),
    )
    initial = engine.index.repository_version
    engine.state.verification.append("pytest passed on original version")
    source = root / "auth.py"
    source.write_text(source.read_text().replace("authenticate_user", "authenticate_session"))
    engine.observe(_Call("str_replace_based_edit_tool", "edit-1", {}),
                   _Result("edit-1", "str_replace_based_edit_tool", True, "edited"))
    engine.model_messages("System", SimpleNamespace(steps=[]), [], [])
    assert engine.index.repository_version != initial
    assert "authenticate_session" in {chunk.symbol for chunk in engine.index.chunks}
    assert engine.state.accepted_facts[0].status == "stale"
    engine.lifecycle.state.update(summary={"problem": "Historical model claim", "findings": ["all tests passed"]},
                                  summary_version=1, summary_workspace_version=initial)
    actual = asyncio.run(engine.prepare_messages("System", SimpleNamespace(steps=[]), [], [], None))
    text = "\n".join(message.content or "" for message in actual)
    assert "Model-authored progress; verify against current program facts" in text
    assert "STALE after code change: pytest passed on original version" in text
    assert engine.index.repository_version in text and initial in text
