"""Rank-based fusion and a transparent, light rule reranker."""

from __future__ import annotations

from collections import defaultdict

from codeagent.context.lexical import code_terms
from codeagent.context.protocol import CodeChunk, RetrievalHit


def symbol_search(chunks: tuple[CodeChunk, ...], query: str, top_n: int = 30):
    terms = set(code_terms(query))
    if not terms:
        return []
    matches: list[tuple[CodeChunk, float]] = []
    for chunk in chunks:
        if not chunk.symbol:
            continue
        symbol_terms = set(code_terms(chunk.symbol))
        overlap = terms & symbol_terms
        if overlap:
            exact = 3.0 if chunk.symbol.lower() in terms else 0.0
            matches.append((chunk, exact + len(overlap) / max(1, len(symbol_terms))))
    return sorted(matches, key=lambda pair: (-pair[1], pair[0].path))[:top_n]


def structural_search(chunks: tuple[CodeChunk, ...], query: str, top_n: int = 30):
    """Recall imports/calls/references without requiring a graph database."""
    terms = set(code_terms(query))
    if not terms:
        return []
    matches: list[tuple[CodeChunk, float]] = []
    for chunk in chunks:
        structural = set(code_terms(" ".join(chunk.metadata.values())))
        overlap = terms & structural
        if overlap:
            matches.append((chunk, len(overlap) / max(1, len(terms))))
    return sorted(matches, key=lambda pair: (-pair[1], pair[0].path,
                                               pair[0].start_line))[:top_n]


def fuse(
    rankings: dict[str, list[tuple[CodeChunk, float]]],
    query: str,
    *,
    top_n: int = 40,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    chunks: dict[str, CodeChunk] = {}
    query_terms = set(code_terms(query))
    for channel, ranking in rankings.items():
        seen = set()
        for chunk, _ in ranking:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            rank = len(seen)
            chunks[chunk.id] = chunk
            scores[chunk.id] += 1 / (rrf_k + rank)
            ranks[chunk.id][channel] = rank
    hits: list[RetrievalHit] = []
    for chunk_id, chunk in chunks.items():
        boost = 0.0
        if chunk.symbol and chunk.symbol.lower() in query_terms:
            boost += 0.01
        if any(term in code_terms(chunk.path) for term in query_terms):
            boost += 0.002
        if "test" in query_terms and "test" in chunk.path.lower():
            boost += 0.002
        hits.append(RetrievalHit(
            chunk=chunk, score=scores[chunk_id] + boost,
            channels=tuple(sorted(ranks[chunk_id])), ranks=dict(ranks[chunk_id]),
        ))
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.path,
                                          hit.chunk.start_line))[:top_n]
