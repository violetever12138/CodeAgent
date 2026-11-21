"""Optional embedding channel; never substitutes a lexical score for semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Protocol

from codeagent.context.protocol import CodeChunk


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def encoding_identity(provider, role):
    identity = getattr(provider, "cache_identity", None)
    return identity(role) if identity else json.dumps({
        "schema": "code-embedding-v2", "model": provider.model_id, "role": role,
        "revision": getattr(provider, "revision", "unspecified"),
        "dimension": getattr(provider, "dimension", None),
        "endpoint": getattr(provider, "base_url", None), "document_format": "language-path-symbol-body-v2",
    }, sort_keys=True)


def encode(provider, texts, role):
    method = getattr(provider, "embed_queries" if role == "query" else "embed_documents", None)
    return (method or provider.embed)(texts)


def normalized(vector, dimension=None):
    values = [float(value) for value in vector]
    if not values or (dimension is not None and len(values) != dimension) or not all(map(math.isfinite, values)):
        raise ValueError("invalid embedding dimension or non-finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        raise ValueError("zero embedding vector")
    return [value / norm for value in values]


class OpenAIEmbeddingProvider:
    """Uses the already-declared OpenAI dependency when explicitly configured."""

    def __init__(self, model_id: str, api_key: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model_id, input=texts)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class SQLiteEmbeddingCache:
    """Persistent model/text-hash embedding cache shared by workers in a run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        with self._connection:
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                       model TEXT NOT NULL, content_hash TEXT NOT NULL,
                       vector_json TEXT NOT NULL,
                       PRIMARY KEY(model, content_hash)
                   )"""
            )

    def get(self, key: tuple[str, str]) -> list[float] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT vector_json FROM embeddings WHERE model=? AND content_hash=?", key,
            ).fetchone()
        return [float(value) for value in json.loads(row[0])] if row else None

    def put(self, key: tuple[str, str], vector: list[float]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                (*key, json.dumps(vector, separators=(",", ":"))),
            )


class BruteForceDenseIndex:
    def __init__(
        self,
        chunks: tuple[CodeChunk, ...],
        provider: EmbeddingProvider,
        cache: dict[tuple[str, str], list[float]] | SQLiteEmbeddingCache | None = None,
        batch_size: int = 32,
    ) -> None:
        self.chunks = chunks
        self.provider = provider
        self.cache = cache if cache is not None else {}
        texts = {
            chunk.id: f"{chunk.language} {chunk.path} {chunk.symbol or ''}\n{chunk.text}"
            for chunk in chunks
        }
        keys = {
            chunk.id: (encoding_identity(provider, "document"), hashlib.sha256(texts[chunk.id].encode()).hexdigest())
            for chunk in chunks
        }
        missing = [chunk for chunk in chunks if self._get(keys[chunk.id]) is None]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            vectors = encode(provider, [texts[chunk.id] for chunk in batch], "document")
            if len(vectors) != len(batch):
                raise ValueError("embedding provider returned the wrong vector count")
            for chunk, vector in zip(batch, vectors, strict=True):
                self._put(keys[chunk.id], normalized(vector, getattr(provider, "dimension", None)))
        self.vectors = [self._get(keys[chunk.id]) or [] for chunk in chunks]
        dimension = getattr(provider, "dimension", None) or (len(self.vectors[0]) if self.vectors else None)
        self.vectors = [normalized(vector, dimension) for vector in self.vectors]

    def _get(self, key: tuple[str, str]) -> list[float] | None:
        if isinstance(self.cache, SQLiteEmbeddingCache):
            return self.cache.get(key)
        return self.cache.get(key)

    def _put(self, key: tuple[str, str], vector: list[float]) -> None:
        if isinstance(self.cache, SQLiteEmbeddingCache):
            self.cache.put(key, vector)
        else:
            self.cache[key] = vector

    def search(self, query: str, top_n: int = 30) -> list[tuple[CodeChunk, float]]:
        if not self.chunks:
            return []
        vector = self.query_vector(query)
        scored = [(chunk, self._cosine(vector, candidate))
                  for chunk, candidate in zip(self.chunks, self.vectors, strict=True)]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line))[
            :top_n
        ]

    def query_vector(self, query):
        query_key = (encoding_identity(self.provider, "query"), hashlib.sha256(query.encode()).hexdigest())
        vector = self._get(query_key)
        if vector is None:
            query_vectors = encode(self.provider, [query], "query")
            if len(query_vectors) != 1:
                raise ValueError("embedding provider returned the wrong query vector count")
            vector = normalized(query_vectors[0], len(self.vectors[0]))
            self._put(query_key, vector)
        return normalized(vector, len(self.vectors[0]))

    @staticmethod
    def _cosine(first: list[float], second: list[float]) -> float:
        if len(first) != len(second):
            raise ValueError("embedding dimensions differ")
        denominator = math.sqrt(sum(x * x for x in first)) * math.sqrt(
            sum(x * x for x in second)
        )
        return sum(x * y for x, y in zip(first, second, strict=True)) / denominator \
            if denominator else 0.0


_ANN_LOCKS: dict[str, RLock] = {}
_ANN_LOCKS_GUARD = RLock()
_ANN_REBUILDS: set[str] = set()
_ANN_REBUILD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codeswarm-ann")


def _ann_lock(path: Path) -> RLock:
    with _ANN_LOCKS_GUARD:
        return _ANN_LOCKS.setdefault(str(path.resolve()), RLock())


class USearchDenseIndex(BruteForceDenseIndex):
    """Persistent HNSW index with exact cosine reranking of ANN candidates."""

    SCHEMA_VERSION = 2
    TOMBSTONE_REBUILD_RATIO = 0.20

    def __init__(
        self,
        chunks: tuple[CodeChunk, ...],
        provider: EmbeddingProvider,
        cache: dict[tuple[str, str], list[float]] | SQLiteEmbeddingCache | None = None,
        *,
        index_path: str | Path,
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ) -> None:
        super().__init__(chunks, provider, cache)
        self.index_path = Path(index_path).resolve()
        self.metadata_path = self.index_path.with_suffix(".json")
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self._index = None
        self._label_to_position: dict[int, int] = {}
        self.last_update: dict[str, int | bool] = {
            "added": 0, "removed": 0, "tombstones": 0, "rebuilt": False,
            "rebuild_scheduled": False,
        }
        if chunks:
            self._load_or_build()

    def _base_metadata(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_id": self.provider.model_id,
            "encoding_identity": encoding_identity(self.provider, "document"),
            "dimension": len(self.vectors[0]) if self.vectors else 0,
            "metric": "cos",
            "dtype": "f32",
            "connectivity": self.connectivity,
            "expansion_add": self.expansion_add,
            "expansion_search": self.expansion_search,
        }

    def _entry(self, position: int, label: int) -> dict[str, str | int]:
        chunk = self.chunks[position]
        return {
            "label": label,
            "path": chunk.path,
            "file_hash": chunk.file_hash,
            "content_hash": hashlib.sha256(chunk.text.encode()).hexdigest(),
        }

    def _new_index(self, Index, dimension: int):
        return Index(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
        )

    def _persist(self, index, metadata: dict) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        nonce = f"{os.getpid()}.{id(self)}"
        temporary = self.index_path.with_suffix(self.index_path.suffix + f".{nonce}.tmp")
        metadata_tmp = self.metadata_path.with_suffix(f".json.{nonce}.tmp")
        try:
            payload = index.save()
            if not isinstance(payload, (bytes, bytearray)):
                raise RuntimeError("USearch did not return an in-memory index payload")
            temporary.write_bytes(payload)
            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.index_path)
            metadata_tmp.replace(self.metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)

    def _rebuild(self, Index, base: dict, *, tombstones: int = 0):
        import numpy as np

        index = self._new_index(Index, base["dimension"])
        labels = np.arange(len(self.vectors), dtype=np.uint64)
        if self.vectors:
            index.add(labels, np.asarray(self.vectors, dtype=np.float32), threads=1)
        entries = {
            chunk.id: self._entry(position, int(labels[position]))
            for position, chunk in enumerate(self.chunks)
        }
        metadata = {
            **base,
            "repository_version": self.chunks[0].repository_version if self.chunks else "",
            "entries": entries,
            "next_label": len(entries),
            "tombstones": 0,
        }
        self._persist(index, metadata)
        self._label_to_position = {
            int(entry["label"]): position
            for position, chunk in enumerate(self.chunks)
            if (entry := entries.get(chunk.id)) is not None
        }
        self.last_update = {
            "added": len(entries),
            "removed": tombstones,
            "tombstones": 0,
            "rebuilt": True,
            "rebuild_scheduled": False,
        }
        return index

    def _schedule_rebuild(self, Index, base: dict) -> bool:
        key = str(self.index_path)
        with _ANN_LOCKS_GUARD:
            if key in _ANN_REBUILDS:
                return False
            _ANN_REBUILDS.add(key)
        vectors = tuple(tuple(vector) for vector in self.vectors)
        chunks = self.chunks

        def rebuild() -> None:
            try:
                import numpy as np

                with _ann_lock(self.index_path):
                    index = self._new_index(Index, base["dimension"])
                    labels = np.arange(len(vectors), dtype=np.uint64)
                    if vectors:
                        index.add(labels, np.asarray(vectors, dtype=np.float32), threads=1)
                    entries = {
                        chunk.id: {
                            "label": int(labels[position]),
                            "path": chunk.path,
                            "file_hash": chunk.file_hash,
                            "content_hash": hashlib.sha256(chunk.text.encode()).hexdigest(),
                        }
                        for position, chunk in enumerate(chunks)
                    }
                    self._persist(index, {
                        **base,
                        "repository_version": chunks[0].repository_version if chunks else "",
                        "entries": entries,
                        "next_label": len(entries),
                        "tombstones": 0,
                    })
            finally:
                with _ANN_LOCKS_GUARD:
                    _ANN_REBUILDS.discard(key)

        _ANN_REBUILD_EXECUTOR.submit(rebuild)
        return True

    def _load_or_build(self) -> None:
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise RuntimeError("USearch is not installed; install the context-ann extra") from exc
        base = self._base_metadata()
        with _ann_lock(self.index_path):
            current = None
            if self.metadata_path.exists():
                try:
                    current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    current = None
            compatible = bool(
                current
                and all(current.get(key) == value for key, value in base.items())
                and isinstance(current.get("entries"), dict)
                and self.index_path.exists()
            )
            if not compatible:
                self._index = self._rebuild(Index, base)
                return
            index = self._new_index(Index, base["dimension"])
            try:
                index.load(self.index_path.read_bytes())
            except Exception:
                self._index = self._rebuild(Index, base)
                return

            import numpy as np

            entries: dict[str, dict] = dict(current["entries"])
            current_ids = {chunk.id for chunk in self.chunks}
            removed_ids = sorted(set(entries) - current_ids)
            removed = 0
            for chunk_id in removed_ids:
                label = int(entries.pop(chunk_id)["label"])
                index.remove(label)
                removed += 1

            next_label = int(current.get("next_label", 0))
            additions: list[tuple[int, int]] = []
            for position, chunk in enumerate(self.chunks):
                if chunk.id not in entries:
                    label = next_label
                    next_label += 1
                    entries[chunk.id] = self._entry(position, label)
                    additions.append((position, label))
            if additions:
                index.add(
                    np.asarray([label for _, label in additions], dtype=np.uint64),
                    np.asarray([self.vectors[position] for position, _ in additions], dtype=np.float32),
                    threads=1,
                )
            tombstones = int(current.get("tombstones", 0)) + removed
            total_slots = len(entries) + tombstones
            if total_slots and tombstones / total_slots > self.TOMBSTONE_REBUILD_RATIO:
                metadata = {
                    **base,
                    "repository_version": self.chunks[0].repository_version,
                    "entries": entries,
                    "next_label": next_label,
                    "tombstones": tombstones,
                }
                self._persist(index, metadata)
                self._label_to_position = {
                    int(entries[chunk.id]["label"]): position
                    for position, chunk in enumerate(self.chunks)
                }
                scheduled = self._schedule_rebuild(Index, base)
                self.last_update = {
                    "added": len(additions),
                    "removed": removed,
                    "tombstones": tombstones,
                    "rebuilt": False,
                    "rebuild_scheduled": scheduled,
                }
                self._index = index
                return
            metadata = {
                **base,
                "repository_version": self.chunks[0].repository_version,
                "entries": entries,
                "next_label": next_label,
                "tombstones": tombstones,
            }
            if removed or additions or metadata != current:
                self._persist(index, metadata)
            self._label_to_position = {
                int(entries[chunk.id]["label"]): position
                for position, chunk in enumerate(self.chunks)
            }
            self.last_update = {
                "added": len(additions),
                "removed": removed,
                "tombstones": tombstones,
                "rebuilt": False,
                "rebuild_scheduled": False,
            }
            self._index = index

    def search(self, query: str, top_n: int = 30) -> list[tuple[CodeChunk, float]]:
        if not self.chunks or self._index is None:
            return []
        vector = self.query_vector(query)
        candidate_count = min(len(self.chunks), max(top_n * 4, 64))
        import numpy as np

        matches = self._index.search(np.asarray(vector, dtype=np.float32), candidate_count)
        labels = getattr(matches, "keys", matches)
        candidates = []
        for raw_label in labels:
            label = int(raw_label)
            position = self._label_to_position.get(label)
            if position is not None:
                candidates.append(
                    (self.chunks[position], self._cosine(vector, self.vectors[position]))
                )
        return sorted(
            candidates, key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line)
        )[:top_n]


# Backward-compatible import used by existing extensions and tests.
DenseIndex = BruteForceDenseIndex
