"""Offline retrieval metrics computed from evaluator-only relevance labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from codeswarm_agent.context.protocol import RetrievalHit


@dataclass(frozen=True)
class RelevanceLabel:
    paths: tuple[str, ...]
    symbols: tuple[str, ...] = ()


def retrieval_metrics(
    hits: Iterable[RetrievalHit],
    relevant: RelevanceLabel,
    *,
    top_n: int = 12,
    current_repository_version: str | None = None,
) -> dict[str, float]:
    selected = tuple(hits)[:top_n]
    relevant_paths = {path.replace("\\", "/") for path in relevant.paths}
    relevant_symbols = {symbol.casefold() for symbol in relevant.symbols}

    relevant_count = max(1, len(relevant_paths) + len(relevant_symbols))
    found_labels: set[tuple[str, str]] = set()
    flags: list[bool] = []
    for hit in selected:
        path = hit.chunk.path.replace("\\", "/")
        labels: set[tuple[str, str]] = set()
        if path in relevant_paths:
            labels.add(("path", path))
        if hit.chunk.symbol and hit.chunk.symbol.casefold() in relevant_symbols:
            labels.add(("symbol", hit.chunk.symbol.casefold()))
        novel = labels - found_labels
        flags.append(bool(novel))
        found_labels.update(novel)
    reciprocal_rank = next((1 / rank for rank, matched in enumerate(flags, 1) if matched), 0.0)
    dcg = sum((1.0 if matched else 0.0) / math.log2(rank + 1)
              for rank, matched in enumerate(flags, 1))
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(relevant_count, top_n) + 1))
    stale = [
        hit for hit in selected
        if current_repository_version is not None
        and hit.chunk.repository_version != current_repository_version
    ]
    return {
        f"recall_at_{top_n}": min(1.0, len(found_labels) / relevant_count),
        f"mrr_at_{top_n}": reciprocal_rank,
        f"ndcg_at_{top_n}": dcg / ideal if ideal else 0.0,
        "stale_hit_rate": len(stale) / len(selected) if selected else 0.0,
        "file_diversity": (
            len({hit.chunk.path for hit in selected}) / len(selected) if selected else 0.0
        ),
    }
