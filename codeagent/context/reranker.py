"""Explainable feature reranking with an optional local cross-encoder stage."""

from __future__ import annotations

import concurrent.futures
import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import ClassVar

from codeagent.context.lexical import code_terms
from codeagent.context.protocol import CodeChunk, RetrievalHit


@dataclass(frozen=True)
class RerankOutcome:
    hits: tuple[RetrievalHit, ...]
    requested_mode: str
    effective_mode: str
    model: str | None
    revision: str | None
    fallback: bool
    fallback_reason: str | None
    latency_ms: float
    feature_scores: dict[str, float]


class RerankCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS rerank_cache (
                       model TEXT NOT NULL,
                       query_hash TEXT NOT NULL,
                       content_hash TEXT NOT NULL,
                       score REAL NOT NULL,
                       PRIMARY KEY(model, query_hash, content_hash)
                   )"""
            )

    def scores(
        self, model: str, query: str, chunks: list[CodeChunk], scorer,
    ) -> list[float]:
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        content_hashes = [hashlib.sha256(chunk.text.encode()).hexdigest() for chunk in chunks]
        result: list[float | None] = [None] * len(chunks)
        missing: list[int] = []
        with self._lock:
            for index, content_hash in enumerate(content_hashes):
                row = self._connection.execute(
                    "SELECT score FROM rerank_cache WHERE model=? AND query_hash=? AND content_hash=?",
                    (model, query_hash, content_hash),
                ).fetchone()
                if row is None:
                    missing.append(index)
                else:
                    result[index] = float(row[0])
        if missing:
            fresh = scorer([chunks[index] for index in missing])
            if len(fresh) != len(missing):
                raise ValueError("cross-encoder returned an unexpected score count")
            with self._lock, self._connection:
                for index, score in zip(missing, fresh, strict=True):
                    numeric = float(score)
                    result[index] = numeric
                    self._connection.execute(
                        "INSERT OR REPLACE INTO rerank_cache VALUES (?, ?, ?, ?)",
                        (model, query_hash, content_hashes[index], numeric),
                    )
        return [float(score) for score in result]


class FeatureReranker:
    def score(
        self,
        hits: list[RetrievalHit],
        query: str,
        *,
        role: str,
        failure: str | None = None,
        changed_paths: tuple[str, ...] = (),
        dependency_paths: tuple[str, ...] = (),
    ) -> dict[str, float]:
        query_terms = set(code_terms(query))
        failure_text = failure or ""
        failure_paths = set(re.findall(r"(?:[A-Za-z]:)?[^\s:\"']+\.(?:py|js|ts|go|rs)", failure_text))
        related = {path.replace("\\", "/") for path in (*changed_paths, *dependency_paths)}
        scores: dict[str, float] = {}
        for hit in hits:
            chunk = hit.chunk
            path_terms = set(code_terms(chunk.path))
            symbol_terms = set(code_terms(chunk.symbol or ""))
            structure_terms = set(code_terms(" ".join(chunk.metadata.values())))
            exact_symbol = 1.0 if chunk.symbol and chunk.symbol.lower() in query.lower() else 0.0
            symbol_overlap = len(query_terms & symbol_terms) / max(1, len(symbol_terms))
            path_overlap = len(query_terms & path_terms) / max(1, len(path_terms))
            structural_overlap = len(query_terms & structure_terms) / max(1, len(query_terms))
            role_match = 0.0
            is_test = "test" in chunk.path.lower() or chunk.metadata.get("test_relation") == "test"
            if role == "tester" and is_test:
                role_match = 0.8
            elif role in {"developer", "explorer", "integrator"} and not is_test:
                role_match = 0.25
            failure_match = 1.0 if any(path.replace("\\", "/") in chunk.path for path in failure_paths) else 0.0
            proximity = 1.0 if chunk.path in related else 0.0
            freshness = 0.2 if chunk.repository_version else 0.0
            scores[chunk.id] = (
                3.0 * exact_symbol + 1.7 * symbol_overlap + 1.1 * path_overlap
                + 1.4 * structural_overlap + role_match + 2.0 * failure_match
                + 1.5 * proximity + freshness
            )
        return scores

    @staticmethod
    def order(hits: list[RetrievalHit], scores: dict[str, float]) -> list[RetrievalHit]:
        return sorted(
            hits,
            key=lambda hit: (-scores.get(hit.chunk.id, 0.0), -hit.score,
                             hit.chunk.path, hit.chunk.start_line),
        )


class HybridReranker:
    _models: ClassVar[dict[str, object]] = {}
    _models_lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        *,
        mode: str = "feature",
        model: str = "BAAI/bge-reranker-base",
        revision: str | None = None,
        candidate_count: int = 16,
        top_n: int = 12,
        timeout_seconds: float = 4.0,
        cache_path: str | Path | None = None,
    ) -> None:
        if mode not in {"none", "feature", "hybrid"}:
            raise ValueError("reranker mode must be none, feature or hybrid")
        self.mode = mode
        self.model = model
        self.revision = revision
        self.candidate_count = candidate_count
        self.top_n = top_n
        self.timeout_seconds = timeout_seconds
        self.features = FeatureReranker()
        self.cache = RerankCache(cache_path) if cache_path else None
        self._cross_encoder = None

    def preload(self) -> None:
        """Load the pinned local model before latency-budgeted request processing."""
        if self.mode == "hybrid":
            self._cross_encoder = self._load_cross_encoder()

    def rerank(
        self,
        hits: list[RetrievalHit],
        query: str,
        *,
        role: str,
        failure: str | None = None,
        changed_paths: tuple[str, ...] = (),
        dependency_paths: tuple[str, ...] = (),
    ) -> RerankOutcome:
        started = time.perf_counter()
        if self.mode == "none":
            ranked = tuple(
                RetrievalHit(
                    chunk=hit.chunk, score=1.0 / rank,
                    channels=hit.channels, ranks=hit.ranks,
                )
                for rank, hit in enumerate(hits[:self.top_n], 1)
            )
            return RerankOutcome(
                hits=ranked, requested_mode="none", effective_mode="none",
                model=None, revision=None, fallback=False, fallback_reason=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                feature_scores={},
            )
        feature_scores = self.features.score(
            hits, query, role=role, failure=failure, changed_paths=changed_paths,
            dependency_paths=dependency_paths,
        )
        feature_order = self.features.order(hits, feature_scores)
        fallback_reason = None
        cross_order: list[RetrievalHit] = []
        effective_mode = "feature"
        if self.mode == "hybrid" and feature_order:
            candidates = feature_order[:self.candidate_count]
            try:
                scores = self._cross_scores(query, [hit.chunk for hit in candidates])
                cross_order = [
                    hit for _, hit in sorted(
                        zip(scores, candidates, strict=True),
                        key=lambda pair: (-pair[0], pair[1].chunk.path,
                                          pair[1].chunk.start_line),
                    )
                ]
                effective_mode = "hybrid"
            except (ImportError, OSError, RuntimeError, ValueError, TimeoutError) as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"[:500]
        fused = self._rank_fusion(hits, feature_order, cross_order)
        diversified = self._mmr(fused, self.top_n)
        ranked = tuple(
            RetrievalHit(
                chunk=hit.chunk,
                score=1.0 / rank,
                channels=hit.channels,
                ranks=hit.ranks,
            )
            for rank, hit in enumerate(diversified, 1)
        )
        return RerankOutcome(
            hits=ranked, requested_mode=self.mode,
            effective_mode=effective_mode, model=self.model if self.mode == "hybrid" else None,
            revision=self.revision if self.mode == "hybrid" else None,
            fallback=self.mode == "hybrid" and effective_mode != "hybrid",
            fallback_reason=fallback_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            feature_scores=feature_scores,
        )

    def _cross_scores(self, query: str, chunks: list[CodeChunk]) -> list[float]:
        def calculate(selected: list[CodeChunk]) -> list[float]:
            if self._cross_encoder is None:
                self._cross_encoder = self._load_cross_encoder()
            values = self._cross_encoder.predict(
                [(query, chunk.text) for chunk in selected], batch_size=16,
                show_progress_bar=False,
            )
            return [float(value) for value in values]

        def scorer(selected: list[CodeChunk]) -> list[float]:
            return self._within_timeout(lambda: calculate(selected))
        if self.cache:
            model_key = f"{self.model}@{self.revision or 'default'}"
            return self.cache.scores(model_key, query, chunks, scorer)
        return scorer(chunks)

    def _load_cross_encoder(self):
        key = f"{self.model}@{self.revision or 'default'}"
        with self._models_lock:
            if key in self._models:
                return self._models[key]
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("install the context-ml optional dependency") from exc
            model = CrossEncoder(
                self.model, revision=self.revision, max_length=512,
                local_files_only=True,
            )
            self._models[key] = model
            return model

    def _within_timeout(self, callback):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(callback)
        try:
            return future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"reranker exceeded {self.timeout_seconds:.1f}s budget") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _rank_fusion(
        original: list[RetrievalHit], feature: list[RetrievalHit], cross: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        by_id = {hit.chunk.id: hit for hit in original}
        scores: dict[str, float] = dict.fromkeys(by_id, 0.0)
        for weight, ranking in ((1.0, original), (1.5, feature), (2.0, cross)):
            for rank, hit in enumerate(ranking, 1):
                scores[hit.chunk.id] += weight / (40 + rank)
        return sorted(
            by_id.values(),
            key=lambda hit: (-scores[hit.chunk.id], hit.chunk.path, hit.chunk.start_line),
        )

    @staticmethod
    def _mmr(hits: list[RetrievalHit], top_n: int) -> list[RetrievalHit]:
        selected: list[RetrievalHit] = []
        remaining = list(hits)
        base_rank = {hit.chunk.id: 1 / (index + 1) for index, hit in enumerate(hits)}
        while remaining and len(selected) < top_n:
            def value(hit: RetrievalHit) -> float:
                similarity = max(
                    (HybridReranker._similarity(hit.chunk, item.chunk) for item in selected),
                    default=0.0,
                )
                return 0.60 * base_rank[hit.chunk.id] - 0.40 * similarity

            chosen = max(remaining, key=lambda hit: (value(hit), hit.chunk.id))
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    @staticmethod
    def _similarity(left: CodeChunk, right: CodeChunk) -> float:
        left_terms = set(code_terms(left.text))
        right_terms = set(code_terms(right.text))
        lexical = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
        if left.path != right.path:
            return lexical * 0.5
        overlap = max(0, min(left.end_line, right.end_line) - max(left.start_line, right.start_line) + 1)
        span = max(left.end_line, right.end_line) - min(left.start_line, right.start_line) + 1
        return max(lexical, overlap / max(1, span))
