"""Bounded, phase-sensitive queries from the current goal and failure state."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchQuery:
    lexical: str
    symbols: str
    semantic: str
    path_hints: tuple[str, ...]


_PATH = re.compile(r"(?<![\w./-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+")
_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]{2,}\b")


def plan_query(issue: str, goal: str, role: str, latest_failure: str = "") -> SearchQuery:
    parts = [goal, latest_failure[:1500], issue[:1500]]
    if role == "tester":
        parts = [latest_failure[:1500], goal, issue[:1000]]
    elif role == "reviewer":
        parts = [issue[:1500], goal, latest_failure[:1000]]
    source = "\n".join(part for part in parts if part)
    symbols = list(dict.fromkeys(_SYMBOL.findall(source)))[:24]
    paths = tuple(dict.fromkeys(_PATH.findall(source)))[:12]
    return SearchQuery(
        lexical=" ".join(symbols + list(paths)),
        symbols=" ".join(symbols),
        semantic=source[:3000],
        path_hints=paths,
    )
