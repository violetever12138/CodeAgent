"""Budgeted evidence packing with explicit exclusions and mandatory invariants."""

from __future__ import annotations

import json
from collections import defaultdict

from codeagent.context.protocol import ContextItem, ContextPackage, ExcludedItem


class ContextBudgetError(ValueError):
    pass


class TokenCounter:
    """Uses a model tokenizer when available, otherwise a conservative UTF-8 bound.

    The fallback is explicitly labelled; it is not a claim about exact provider tokenization.
    """

    def __init__(self, model: str | None = None) -> None:
        self.encoding = None
        self.name = "utf8-byte-upper-bound"
        try:
            import tiktoken

            self.encoding = tiktoken.encoding_for_model(model) if model else None
            if self.encoding:
                self.name = f"tiktoken:{self.encoding.name}"
        except (ImportError, KeyError):
            pass

    def count(self, value: str) -> int:
        return len(self.encoding.encode(value)) if self.encoding else len(value.encode("utf-8"))

    def messages(self, messages) -> int:
        # Protocol framing and tool-call arguments are covered by a separate safety margin.
        return sum(
            self.count(message.content or "")
            + (self.count(json.dumps(message.tool_call.arguments)) if message.tool_call else 0)
            + (self.count(message.tool_call.name + message.tool_call.call_id) if message.tool_call else 0)
            + (self.count((message.tool_result.result or "") + (message.tool_result.error or "")
                          + message.tool_result.name + message.tool_result.call_id) if message.tool_result else 0)
            + 12
            for message in messages
        )


def _rendered_item_cost(item: ContextItem, counter: TokenCounter) -> int:
    """Count the content actually sent, including JSON escaping and framing."""
    rendered = (
        f"[context kind={item.kind} source={item.source} "
        f"version={item.repository_version} id={item.id}]\n"
        f"data_json={json.dumps(item.text, ensure_ascii=False)}\n[/context]\n\n"
    )
    return counter.count(rendered)


def pack(
    *,
    required: list[ContextItem],
    candidates: list[ContextItem],
    budget_tokens: int,
    repository_version: str,
    role: str,
    counter: TokenCounter,
) -> ContextPackage:
    if budget_tokens <= 0:
        raise ContextBudgetError("no context budget after output, tools, and safety reserve")
    chosen: list[ContextItem] = []
    excluded: list[ExcludedItem] = []
    seen: set[str] = set()
    used = 0
    for item in required:
        cost = _rendered_item_cost(item, counter)
        if used + cost > budget_tokens:
            raise ContextBudgetError(
                f"mandatory context exceeds budget at {item.id}: "
                f"need {used + cost}, have {budget_tokens}"
            )
        if item.id not in seen:
            chosen.append(item)
            seen.add(item.id)
            used += cost
    groups: dict[str, list[ContextItem]] = defaultdict(list)
    for item in candidates:
        groups[item.group or item.id].append(item)
    ranked = sorted(
        groups.values(),
        key=lambda group: -sum(item.priority for item in group)
        / max(1, sum(_rendered_item_cost(item, counter) for item in group)),
    )
    for group in ranked:
        fresh = [item for item in group if item.id not in seen]
        if not fresh:
            excluded.extend(ExcludedItem(item.id, "duplicate") for item in group)
            continue
        if any(
            item.repository_version not in {"stable", repository_version}
            for item in fresh
        ):
            excluded.extend(ExcludedItem(item.id, "stale_version") for item in group)
            continue
        cost = sum(_rendered_item_cost(item, counter) for item in fresh)
        if used + cost > budget_tokens:
            excluded.extend(ExcludedItem(item.id, "token_budget") for item in group)
            continue
        chosen.extend(fresh)
        seen.update(item.id for item in fresh)
        used += cost
    return ContextPackage(
        repository_version=repository_version, role=role, items=tuple(chosen),
        excluded=tuple(excluded), input_tokens=used, budget_tokens=budget_tokens,
        tokenizer=counter.name,
    )
