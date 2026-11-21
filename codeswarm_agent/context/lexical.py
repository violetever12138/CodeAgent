"""Field-aware BM25 with code-identifier tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from codeswarm_agent.context.protocol import CodeChunk

_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[\u4e00-\u9fff]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def code_terms(value: str) -> list[str]:
    terms: list[str] = []
    for match in _WORD.finditer(value):
        raw = match.group()
        terms.append(raw.lower())
        for part in raw.split("_"):
            terms.extend(piece.lower() for piece in _CAMEL.split(part) if piece)
    return terms


class BM25Index:
    def __init__(self, chunks: tuple[CodeChunk, ...], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.fields: dict[str, list[Counter[str]]] = {}
        self.lengths: dict[str, list[int]] = {}
        self.document_frequency: dict[str, int] = defaultdict(int)
        for field in ("path", "symbol", "body"):
            values = [
                chunk.path if field == "path" else
                (chunk.symbol or "") if field == "symbol" else chunk.text
                for chunk in chunks
            ]
            counts = [Counter(code_terms(value)) for value in values]
            self.fields[field] = counts
            self.lengths[field] = [sum(count.values()) for count in counts]
        for index in range(len(chunks)):
            terms = set().union(*(set(self.fields[field][index]) for field in self.fields))
            for term in terms:
                self.document_frequency[term] += 1

    def search(self, query: str, top_n: int = 50) -> list[tuple[CodeChunk, float]]:
        terms = list(dict.fromkeys(code_terms(query)))
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        averages = {
            field: max(1.0, sum(lengths) / n)
            for field, lengths in self.lengths.items()
        }
        weights = {"path": 2.5, "symbol": 4.0, "body": 1.0}
        scored: list[tuple[CodeChunk, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = 0.0
            for term in terms:
                df = self.document_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log1p((n - df + 0.5) / (df + 0.5))
                for field, weight in weights.items():
                    tf = self.fields[field][index].get(term, 0)
                    if not tf:
                        continue
                    length = self.lengths[field][index]
                    norm = self.k1 * (1 - self.b + self.b * length / averages[field])
                    score += weight * idf * (tf * (self.k1 + 1)) / (tf + norm)
            if score:
                scored.append((chunk, score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0].path, pair[0].start_line))[
            :top_n
        ]
