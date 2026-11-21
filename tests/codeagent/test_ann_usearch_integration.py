from __future__ import annotations

import hashlib
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from codeagent.context.dense import BruteForceDenseIndex, USearchDenseIndex
from codeagent.context.protocol import CodeChunk

pytest.importorskip("usearch")


class DeterministicProvider:
    model_id = "deterministic-16-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) - 127.5
            vector /= np.linalg.norm(vector)
            vectors.append(vector.tolist())
        return vectors


def make_chunk(number: int, version: str = "v1") -> CodeChunk:
    text = f"def symbol_{number}(): return '{version}'"
    return CodeChunk(
        id=f"chunk-{number}-{version}",
        path=f"module_{number}.py",
        language="python",
        symbol=f"symbol_{number}",
        start_line=1,
        end_line=1,
        text=text,
        file_hash=f"hash-{number}-{version}",
        repository_version=version,
    )


def test_real_usearch_persistence_incremental_update_and_corruption_rebuild(tmp_path: Path):
    provider = DeterministicProvider()
    cache: dict[tuple[str, str], list[float]] = {}
    index_path = tmp_path / "dense.usearch"
    chunks = tuple(make_chunk(index) for index in range(8))

    first = USearchDenseIndex(chunks, provider, cache, index_path=index_path)
    expected = [chunk.id for chunk, _ in first.search("symbol 3", top_n=4)]
    loaded = USearchDenseIndex(chunks, provider, cache, index_path=index_path)
    assert [chunk.id for chunk, _ in loaded.search("symbol 3", top_n=4)] == expected
    assert loaded.last_update["rebuilt"] is False

    changed_chunks = (*chunks[:-1], make_chunk(7, "v2"))
    changed = USearchDenseIndex(changed_chunks, provider, cache, index_path=index_path)
    assert changed.last_update["added"] == 1
    assert changed.last_update["removed"] == 1
    assert changed.search("symbol 7 v2", top_n=4)

    index_path.write_bytes(b"corrupt")
    repaired = USearchDenseIndex(changed_chunks, provider, cache, index_path=index_path)
    assert repaired.last_update["rebuilt"] is True
    assert repaired.search("symbol 1", top_n=4)


@pytest.mark.ann_benchmark
@pytest.mark.skipif(
    os.environ.get("CODESWARM_RUN_ANN_BENCHMARK") != "1",
    reason="set CODESWARM_RUN_ANN_BENCHMARK=1 for the 10k real-USearch benchmark",
)
def test_real_usearch_recall_and_10k_latency(tmp_path: Path):
    rng = np.random.default_rng(20260924)
    matrix = rng.normal(size=(10_000, 64)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    class MatrixProvider:
        model_id = "matrix-64-v1"

        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = []
            for text in texts:
                marker = text.rsplit("vector-", 1)[-1].split()[0]
                vectors.append(matrix[int(marker)].tolist())
            return vectors

    chunks = tuple(
        CodeChunk(
            id=f"chunk-{index}", path=f"pkg/m{index}.py", language="python",
            symbol=f"s{index}", start_line=1, end_line=1, text=f"vector-{index}",
            file_hash=f"hash-{index}", repository_version="benchmark-v1",
        )
        for index in range(len(matrix))
    )
    provider = MatrixProvider()
    cache: dict[tuple[str, str], list[float]] = {}
    brute = BruteForceDenseIndex(chunks, provider, cache)
    ann = USearchDenseIndex(chunks, provider, cache, index_path=tmp_path / "10k.usearch")
    query_ids = tuple(range(0, 10_000, 397))

    recalls = []
    brute_latencies = []
    ann_latencies = []
    for query_id in query_ids:
        query = f"vector-{query_id}"
        started = time.perf_counter()
        exact = brute.search(query, top_n=12)
        brute_latencies.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        approximate = ann.search(query, top_n=12)
        ann_latencies.append((time.perf_counter() - started) * 1000)
        exact_ids = {chunk.id for chunk, _ in exact}
        approximate_ids = {chunk.id for chunk, _ in approximate}
        recalls.append(len(exact_ids & approximate_ids) / 12)

    recall_at_12 = sum(recalls) / len(recalls)
    brute_p95 = statistics.quantiles(brute_latencies, n=20)[18]
    ann_p95 = statistics.quantiles(ann_latencies, n=20)[18]
    print(
        f"ANN_BENCHMARK recall_at_12={recall_at_12:.6f} "
        f"brute_p95_ms={brute_p95:.3f} ann_p95_ms={ann_p95:.3f}"
    )
    assert recall_at_12 >= 0.98
    assert ann_p95 < brute_p95
