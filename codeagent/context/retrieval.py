"""Rule-based localization and one bounded expansion; no model router."""

from __future__ import annotations

import re
from dataclasses import dataclass

from codeagent.context.fusion import structural_search, symbol_search
from codeagent.context.lexical import BM25Index, code_terms
from codeagent.context.query_planner import SearchQuery, plan_query

_PATH = re.compile(r"(?<![\w./-])([\w./-]+\.(?:py|jsx?|tsx?|go|rs|java|cpp|c|h|json|yaml|yml))(?!\w)(?:[:](\d+))?")
_STOP = {"a", "an", "the", "to", "of", "in", "on", "and", "or", "is", "are", "be", "for", "with", "this", "that", "fix", "bug", "issue", "change", "implement", "return", "function", "code", "file", "module", "please", "use", "add", "from", "at", "as", "it", "by"}


@dataclass
class Recall:
    query: SearchQuery
    rankings: dict
    expanded: bool
    status: str
    reason: str
    dense_fallback: str | None = None


def retrieve(chunks, question, *, scope="", dense_search=None):
    question = question[:3000]
    query = plan_query("", question, "worker")
    chunks = tuple(c for c in chunks if not scope or c.path == scope or c.path.startswith(scope.rstrip("/") + "/"))
    paths = list(_PATH.finditer(question.replace("\\", "/")))
    rankings = {}
    located = []
    relation = bool(re.search(r"\b(callers?|calls?|imports?|references?|dependenc\w*)\b|调用|引用|依赖|导入", question, re.I))
    if paths:
        for match in paths:
            path, line = match.group(1), match.group(2)
            candidates = [c for c in chunks if c.path == path]
            if not candidates:
                candidates = [c for c in chunks if c.path.endswith("/" + path)]
            if line:
                candidates = [c for c in candidates if c.start_line <= int(line) <= c.end_line]
            located.extend((c, 1.0) for c in candidates)
    else:
        words = set(re.findall(r"\b[A-Za-z_]\w*\b", question))
        qualified = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", question))
        exact = [(c, 1.0) for c in chunks if c.symbol in qualified]
        located = exact or [(c, 1.0) for c in chunks if c.symbol and c.symbol.split(".")[-1] in words]
    identities = {(c.path, None if paths else c.symbol) for c, _ in located}
    if located:
        rankings["location"] = located[:30]
        if len(identities) == 1 and not relation:
            return Recall(query, rankings, False, "located", "current workspace path/symbol match")
    # Ambiguous locations or missing evidence get exactly one broader pass.
    meaningful = set(code_terms(question)) - _STOP
    lexical = BM25Index(chunks).search(" ".join(sorted(meaningful)), top_n=40)
    rankings["bm25"] = [(c, score) for c, score in lexical
                        if meaningful & set(code_terms(c.text + " " + c.path + " " + (c.symbol or "")))]
    if relation:
        rankings["structure"] = structural_search(chunks, " ".join(sorted(meaningful)), top_n=30)
        rankings["symbol"] = symbol_search(chunks, query.symbols, top_n=30)
    fallback = None
    if not paths and not located and not relation:
        if dense_search:
            try:
                valid_ids = {c.id for c in chunks}
                rankings["dense"] = [(c, score) for c, score in dense_search(question)
                                     if c.id in valid_ids and score >= .45]
            except Exception as exc:
                fallback = f"{type(exc).__name__}: {exc}"
        else:
            fallback = "Dense is not configured"
    ambiguous = len(identities) > 1
    # A broad lexical hit cannot validate a missing explicit path.
    sufficient = any(rankings.values()) and (not paths or bool(located))
    status = "ambiguous" if ambiguous else "recalled" if sufficient else "insufficient"
    reason = ("multiple locations match; verify the intended module/symbol before editing" if ambiguous else
              "one bounded expansion" if sufficient else
              "not enough evidence found; this does not prove the implementation is absent")
    if not sufficient:
        rankings = {key: [] for key in rankings}
    return Recall(query, rankings, True, status, reason, fallback)
