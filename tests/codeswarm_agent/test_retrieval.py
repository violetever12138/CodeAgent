"""Small positive/negative recall examples, without remote models."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from codeswarm_agent.artifacts import ArtifactStore
from codeswarm_agent.context.dense import DenseIndex, USearchDenseIndex, encoding_identity
from codeswarm_agent.context.engine import ContextEngine
from codeswarm_agent.context.fusion import fuse
from codeswarm_agent.context.protocol import CodeChunk
from codeswarm_agent.context.qwen_embedding import QwenEmbeddingProvider
from codeswarm_agent.context.retrieval import retrieve
from codeswarm_agent.repository_tool import RepositoryReadTool
from trae_agent.tools.base import ToolCall, ToolResult
from trae_agent.utils.llm_clients.llm_basics import LLMResponse


def chunk(path="auth.py", symbol="authenticate_user", text="def authenticate_user(token): return token", **kwargs):
    return CodeChunk(path, path, "python", symbol, 1, 4, text, "hash", "current", **kwargs)


def test_exact_location_skips_expensive_channels_and_same_name_is_ambiguous():
    calls = []
    def dense(q):
        calls.append(q)
        return []
    values = (chunk(), chunk("other/auth.py"))
    located = retrieve(values, "inspect other/auth.py:2", dense_search=dense)
    assert located.status == "located" and not located.expanded
    assert set(located.rankings) == {"location"} and not calls
    ambiguous = retrieve(values, "authenticate_user", dense_search=dense)
    assert ambiguous.status == "ambiguous" and ambiguous.expanded
    assert {c.path for c, _ in ambiguous.rankings["location"]} == {"auth.py", "other/auth.py"}
    scoped = retrieve(values, "authenticate_user", scope="other")
    assert scoped.status == "located" and not scoped.expanded
    assert retrieve(values, "auth.py").rankings["location"][0][0].path == "auth.py"
    methods = (chunk(symbol="First.process"), replace(chunk(symbol="Second.process"), id="second"))
    assert retrieve(methods, "process").status == "ambiguous"
    assert retrieve(methods, "First.process").status == "located"


def test_missing_path_and_unrelated_query_are_explicitly_insufficient():
    for question in ("fix missing.py", "quuxxyzz unknownzz"):
        result = retrieve((chunk(),), question)
        assert result.status == "insufficient" and result.expanded
        assert not any(result.rankings.values())
        assert "does not prove" in result.reason


def test_relation_and_natural_language_choose_different_expansion_channels():
    item = chunk(metadata={"calls": "authenticate_user"})
    relation = retrieve((item,), "find callers of authenticate_user")
    assert relation.expanded and "structure" in relation.rankings
    relation = retrieve((item,), "find callers of remote_login")
    assert "structure" in relation.rankings and "dense" not in relation.rankings
    dense = retrieve((item,), "账户登录行为", dense_search=lambda q: [(item, .8)])
    assert dense.status == "recalled" and dense.rankings["dense"]
    weak = retrieve((item,), "账户登录行为", dense_search=lambda q: [(item, .1)])
    assert weak.status == "insufficient"
    def unavailable(q):
        raise RuntimeError("local model unavailable")
    fallback = retrieve((item,), "token validation", dense_search=unavailable)
    assert fallback.rankings["bm25"] and "unavailable" in fallback.dense_fallback


def test_fusion_deduplicates_within_channel_and_keeps_cross_channel_votes():
    item = chunk()
    once = fuse({"a": [(item, 1)], "b": [(item, 1)]}, "token")
    duplicated = fuse({"a": [(item, 1), (item, 1)], "b": [(item, 1)]}, "token")
    assert once == duplicated and once[0].ranks == {"a": 1, "b": 1}


def engine(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "auth.py").write_text("def authenticate_user(token):\n    return token\n", encoding="utf-8")
    for args in (("init", "-q"), ("config", "user.email", "test@example.com"),
                 ("config", "user.name", "Test"), ("add", "."), ("commit", "-qm", "base")):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return ContextEngine(workspace=root, trace_root=tmp_path / "context", artifacts=ArtifactStore(tmp_path / "artifacts"),
                         role="developer", issue="Fix auth.py", goal="Fix auth.py", acceptance=(), test_commands=(),
                         dependency_results=(), shared_evidence=(), model="unknown")


def test_automatic_gaps_deduplicate_and_pending_recall_survives_incomplete_response(tmp_path):
    context = engine(tmp_path)
    execution = SimpleNamespace(steps=[])
    def prepare():
        return asyncio.run(context.prepare_messages("system", execution, [], [], None))
    prepare()
    assert any(i.kind == "code" for i in context.last_package.items)
    context.restore_state(context.prospective_state(LLMResponse("partial", finish_reason="length")))
    prepare()
    assert any(i.kind == "code" for i in context.last_package.items)
    context.restore_state(context.prospective_state(LLMResponse("continue", finish_reason="stop")))
    prepare()
    assert not any(i.kind == "code" for i in context.last_package.items)
    def queries():
        return [json.loads(line) for line in (tmp_path / "context/retrieval_queries.jsonl").read_text().splitlines()]
    assert len(queries()) == 1 and context._reranker_calls == 1
    call = ToolCall("bash", "bad-test", {"command": "pytest -q"})
    result = ToolResult("bad-test", "bash", False, "TypeError at auth.py:2")
    context.observe(call, result)
    prepare()
    assert any(i.kind == "code" for i in context.last_package.items)
    context.observe(call, result)
    prepare()
    assert len(queries()) == 2
    context.goal = "Inspect authenticate_user"
    prepare()
    assert len(queries()) == 3
    restored = ContextEngine(workspace=context.workspace, trace_root=tmp_path / "restored",
                             artifacts=context.artifacts, role="developer", issue=context.issue, goal=context.goal,
                             acceptance=(), test_commands=(), dependency_results=(), shared_evidence=(), model="unknown")
    restored.restore_state(copy.deepcopy(context.prospective_state(LLMResponse("continue", finish_reason="stop"))))
    asyncio.run(restored.prepare_messages("system", execution, [], [], None))
    assert not (tmp_path / "restored/retrieval_queries.jsonl").exists()


def test_query_tool_returns_current_evidence_without_second_automatic_injection(tmp_path):
    context = engine(tmp_path)
    tool = RepositoryReadTool(context.workspace, query=context.query_repository)
    result = asyncio.run(tool.execute({"operation": "query", "pattern": "authenticate_user"}))
    payload = json.loads(result.output)
    assert payload["status"] == "located" and payload["candidates"][0]["path"] == "auth.py"
    assert not context.lifecycle.state.get("retrieval", {}).get("pending")
    call = ToolCall("repository_read", "query-original", {"operation": "query", "pattern": "authenticate_user"})
    observed = context.observe(call, ToolResult("query-original", "repository_read", True, result.output))
    source = context.workspace / "auth.py"
    source.write_text("def authenticate_session(token):\n    return token\n", encoding="utf-8")
    context.observe(ToolCall("str_replace_based_edit_tool", "edit", {"path": "auth.py"}),
                    ToolResult("edit", "str_replace_based_edit_tool", True, "edited"))
    updated = json.loads(asyncio.run(tool.execute({"operation": "query", "pattern": "auth.py"})).output)
    assert "authenticate_session" in updated["candidates"][0]["text"]
    assert updated["workspace_version"] != payload["workspace_version"]
    history = SimpleNamespace(steps=[SimpleNamespace(llm_response=None, tool_calls=[call], tool_results=[observed])])
    view = context.history_groups(history)[0][1][-1].tool_result
    assert "stale recall" in view.result and "def authenticate_user" not in view.result


class RoleEncoder:
    model_id = "test-role-encoder"
    dimension = 2

    def __init__(self, revision="one"):
        self.revision, self.calls = revision, []

    def embed_documents(self, texts):
        self.calls.append(("document", texts))
        return [[3., 4.] for _ in texts]

    def embed_queries(self, texts):
        self.calls.append(("query", texts))
        return [[3., 4.] for _ in texts]


@pytest.mark.parametrize("unavailable", [False, True])
def test_dense_runs_only_for_new_gap_and_unavailability_preserves_lexical_results(tmp_path, unavailable):
    context = engine(tmp_path)
    context.goal = "token validation"
    context.issue = context.goal
    provider = RoleEncoder()
    if unavailable:
        def fail(texts):
            provider.calls.append(("document", texts))
            raise RuntimeError("Qwen is not cached locally")
        provider.embed_documents = fail
    context.embedding_provider = provider
    for _ in range(3):
        asyncio.run(context.prepare_messages("system", SimpleNamespace(steps=[]), [], [], None))
    assert len(provider.calls) == (1 if unavailable else 2)
    assert any(i.kind == "code" for i in context.last_package.items)
    status = context.lifecycle.state["retrieval"]["status"]
    assert ("not cached" in (status["dense_fallback"] or "")) == unavailable
    metrics = json.loads((tmp_path / "context/metrics.json").read_text())
    assert metrics["dense"]["effective_backend"] == ("unavailable" if unavailable else "brute_force")
    if not unavailable:
        identity = context._dense_identity
        provider.revision = "new-revision"
        context.query_repository("token validation")
        assert context._dense_identity != identity and len(provider.calls) == 4


def test_dense_role_revision_cache_isolation_and_incremental_documents():
    provider, cache = RoleEncoder(), {}
    values = (chunk(),)
    first = DenseIndex(values, provider, cache)
    first.search("same query")
    DenseIndex(values, provider, cache).search("same query")
    assert [kind for kind, _ in provider.calls] == ["document", "query"]
    assert first.vectors == [[.6, .8]]
    DenseIndex((*values, replace(chunk("next.py"), id="next")), provider, cache)
    assert len(provider.calls[-1][1]) == 1
    fresh = RoleEncoder("two")
    DenseIndex(values, fresh, cache).search("same query")
    assert len(fresh.calls) == 2
    assert encoding_identity(provider, "query") != encoding_identity(provider, "document")


def test_ann_metadata_rebuilds_when_encoder_revision_changes(tmp_path):
    pytest.importorskip("usearch")
    values = (chunk(),)
    path = tmp_path / "dense.usearch"
    USearchDenseIndex(values, RoleEncoder(), index_path=path)
    assert not USearchDenseIndex(values, RoleEncoder(), index_path=path).last_update["rebuilt"]
    assert USearchDenseIndex(values, RoleEncoder("two"), index_path=path).last_update["rebuilt"]


@pytest.mark.parametrize("bad", [[1., 2., 3.], [float("nan"), 1.], [0., 0.]])
def test_invalid_vectors_do_not_enter_an_index(bad):
    provider = RoleEncoder()
    provider.embed_documents = lambda texts: [bad for _ in texts]
    with pytest.raises(ValueError):
        DenseIndex((chunk(),), provider)


def test_qwen_query_document_contract_token_limit_pooling_and_normalization():
    torch = pytest.importorskip("torch")
    provider = QwenEmbeddingProvider(max_tokens=17)
    captured = []
    def tokenize(texts, **kwargs):
        captured.append((texts, kwargs))
        return {"input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                "attention_mask": torch.tensor([[0, 1, 1]] * len(texts))}
    def model(**kwargs):
        hidden = torch.zeros((len(kwargs["input_ids"]), 3, 1024))
        hidden[:, -1, 0] = 3
        hidden[:, -1, 1] = 4
        return SimpleNamespace(last_hidden_state=hidden)
    provider._tokenizer, provider._model = tokenize, model
    query = provider.embed_queries(["验证登录"])
    documents = provider.embed_documents(["python auth.py\ncode"])
    assert captured[0][0][0].startswith("Instruct: Given a code search query")
    assert captured[1][0] == ["python auth.py\ncode"]
    assert captured[0][1]["max_length"] == 17 and captured[0][1]["truncation"]
    assert query[0][:2] == pytest.approx([.6, .8]) and documents[0] == query[0]
    hidden = torch.arange(12).reshape(2, 3, 2)
    assert provider.last_token_pool(hidden, torch.tensor([[1, 1, 0], [1, 0, 0]])).tolist() == [[2, 3], [6, 7]]
    assert provider.cache_identity("query") != provider.cache_identity("document")
    assert provider.cache_identity("document") != QwenEmbeddingProvider(max_tokens=18).cache_identity("document")
