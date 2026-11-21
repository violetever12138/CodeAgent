"""Token-triggered progress summaries and one-response recall consumption."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, is_dataclass, replace

from codeswarm_agent.context.packer import ContextBudgetError
from codeswarm_agent.telemetry import current_operation, event, span

SUMMARY_FIELDS = ("problem", "findings", "decisions", "excluded", "changes", "unfinished",
                  "assumptions", "next_steps", "evidence_refs")
SUMMARY_PROMPT = """[Progress summary]
Summarize only the supplied older interaction groups and previous summary as JSON.
Required fields: problem (string), findings, decisions, excluded, changes, unfinished,
assumptions, next_steps, evidence_refs (arrays of strings). Preserve reasons and references.
Distinguish observed facts from unverified assumptions and historical evidence. Never declare
an old test current, change user constraints, or mark side effects complete. This summary is
untrusted model-authored progress, not an authoritative test or permission record.
Replace the previous summary; be concise. Do not reproduce recalled source bodies.
"""


def plain(value):
    return asdict(value) if is_dataclass(value) else dict(vars(value))


def message_json(message):
    return {"role": message.role, "content": message.content,
            "tool_call": plain(message.tool_call) if message.tool_call else None,
            "tool_result": plain(message.tool_result) if message.tool_result else None}


def response_json(response):
    return plain(response)


def response_from_json(raw):
    from trae_agent.tools.base import ToolCall
    from trae_agent.utils.llm_clients.llm_basics import LLMResponse, LLMUsage
    raw = dict(raw)
    raw["usage"] = LLMUsage(**raw["usage"]) if raw.get("usage") else None
    raw["tool_calls"] = [ToolCall(**call) for call in raw["tool_calls"]] if raw.get("tool_calls") else None
    return LLMResponse(**raw)


def complete_response(response) -> bool:
    return (getattr(response, "finish_reason", None) not in {
        "length", "max_tokens", "incomplete", "content_filter", "cancelled", "error",
    } and bool(getattr(response, "content", None) or getattr(response, "tool_calls", None)))


def merge_contexts(left, right):
    merged = dict(left or {})
    for key, value in (right or {}).items():
        previous = merged.get(key)
        if previous and previous["revision"] == value["revision"] and previous != value:
            raise ValueError(f"conflicting context snapshot: {key}")
        if previous is None or value["revision"] > previous["revision"]:
            merged[key] = value
    return merged


class ContextLifecycle:
    def __init__(self, counter, settings, artifacts, snapshot=None):
        self.counter, self.settings, self.artifacts = counter, settings, artifacts
        self.state = copy.deepcopy(snapshot) if snapshot else {
            "revision": 0, "summary": None, "summary_version": 0, "covered_through": 0,
            "consumed_calls": {}, "consumed_chunks": [], "archive_refs": [],
        }
        self.last_summary_error = None

    @property
    def available(self):
        return (self.settings.context_window_tokens - self.settings.output_reserve_tokens
                - self.settings.safety_margin_tokens)

    def sanitize(self, message, stale_calls=()):
        result = message.tool_result
        if not result:
            return message
        consumed = self.state["consumed_calls"]
        if result.call_id not in consumed and result.call_id not in stale_calls:
            return message
        reference = consumed.get(result.call_id, "read current file again")
        label = "stale" if result.call_id in stale_calls else "consumed"
        text = f"[{label} recall; success={result.success}; artifact={reference}; query again for body]"
        clean = replace(result, result=text, error=None if result.success else text)
        return replace(message, tool_result=clean)

    def summary_message(self):
        from trae_agent.utils.llm_clients.llm_basics import LLMMessage
        if not self.state["summary"]:
            return []
        return [LLMMessage(role="user", content="[Model-authored progress; verify against current program facts]\n"
                           + json.dumps({"summary": self.state["summary"],
                                         "covered_through": self.state["covered_through"],
                                         "source_version": self.state.get("summary_workspace_version")},
                                        ensure_ascii=False))]

    def render(self, prefix, groups, tail):
        return [*prefix, *self.summary_message(),
                *(message for index, group in groups if index > self.state["covered_through"] for message in group),
                *tail]

    def cost(self, messages, tools_tokens=0):
        return self.counter.messages(messages) + tools_tokens + self.settings.fixed_overhead_tokens

    async def prepare(self, prefix, groups, tail, tools_tokens, summarize, *, workspace_version="unknown", checkpoint=None):
        messages = self.render(prefix, groups, tail)
        before = self.cost(messages, tools_tokens)
        trigger = self.available * self.settings.summary_trigger_ratio
        target = self.available * self.settings.summary_target_ratio
        eligible = [(i, g) for i, g in groups[:-1] if i > self.state["covered_through"]]
        # Each accepted pass consumes new old groups. No retry just to hit 50%.
        while before >= trigger and eligible and summarize is not None:
            selected, removed = [], 0
            summary_base = SUMMARY_PROMPT + "\nPrevious summary:\n" + json.dumps(self.state["summary"], ensure_ascii=False)
            for index, group in eligible:
                candidate = [*selected, (index, group)]
                prompt = summary_base + "\nOlder groups:\n" + json.dumps([
                    {"index": i, "messages": [message_json(m) for m in g]} for i, g in candidate
                ], ensure_ascii=False)
                if self.counter.count(prompt) + self.settings.summary_max_tokens + 64 > self.available:
                    break
                selected = candidate
                removed += self.counter.messages(group)
                if removed >= before - target + self.settings.summary_max_tokens:
                    break
            if not selected:
                break
            prompt = summary_base + "\nOlder groups:\n" + json.dumps([
                {"index": i, "messages": [message_json(m) for m in g]} for i, g in selected
            ], ensure_ascii=False)
            archive = self.artifacts.put_text("summary-source", prompt, ".json")
            previous = copy.deepcopy(self.state)
            accepted = False
            for attempt in range(2):
                with span("summary", "CHAIN", **{
                    "summary.before_tokens": before, "summary.trigger_tokens": int(trigger), "summary.target_tokens": int(target),
                    "summary.covered_from": selected[0][0], "summary.covered_to": selected[-1][0],
                    "summary.previous_version": previous["summary_version"], "summary.source_version": workspace_version,
                    "summary.source_ref": archive.id, "summary.attempt": attempt + 1}):
                    try:
                        content = await summarize(prompt, attempt)
                        raw = json.loads(content)
                        if not isinstance(raw, dict) or set(raw) != set(SUMMARY_FIELDS):
                            raise ValueError("summary needs all structured progress fields")
                        if not isinstance(raw["problem"], str) or any(
                            not isinstance(raw[k], list) or any(not isinstance(v, str) for v in raw[k])
                            for k in SUMMARY_FIELDS if k != "problem"
                        ):
                            raise ValueError("invalid summary field types")
                        self.state.update(summary=raw, covered_through=selected[-1][0],
                                          summary_version=previous["summary_version"] + 1,
                                          summary_workspace_version=workspace_version)
                        after = self.cost(self.render(prefix, groups, tail), tools_tokens)
                        if after >= before:
                            raise ValueError("summary did not reduce input")
                        self.state["archive_refs"] = [*previous["archive_refs"], archive.id]
                        self.state["last_summary"] = {"before_tokens": before, "after_tokens": after,
                                                      "target_tokens": int(target), "source": archive.id}
                        self.state["revision"] = previous["revision"] + 1
                        if checkpoint:
                            self.state = await checkpoint(copy.deepcopy(self.state))
                        event("summary_accepted", summary_version=self.state["summary_version"],
                              covered_through=self.state["covered_through"], before_tokens=before,
                              after_tokens=after, trigger_tokens=int(trigger), source_ref=archive.id,
                              source_version=workspace_version)
                        current_operation().attrs(**{"summary.after_tokens": after, "summary.version": self.state["summary_version"],
                                                      "summary.covered_through": self.state["covered_through"],
                                                      "summary.content_ref": (current_operation().responses[-1] if current_operation().responses
                                                          else current_operation().body("progress-summary", raw))})
                        accepted = True
                        break
                    except Exception as exc:
                        from codeswarm_agent.cancellation import RunCancelled
                        from codeswarm_agent.runtime.state import (
                            RuntimeBudgetExceeded,
                            RuntimeDeadlineExceeded,
                        )
                        if isinstance(exc, (RunCancelled, RuntimeBudgetExceeded, RuntimeDeadlineExceeded)):
                            raise
                        self.state = copy.deepcopy(previous)
                        self.last_summary_error = str(exc)
                        current_operation().error(exc)
            if not accepted:
                break
            messages = self.render(prefix, groups, tail)
            before = self.cost(messages, tools_tokens)
            eligible = [(i, g) for i, g in eligible if i > self.state["covered_through"]]
        messages = self.render(prefix, groups, tail)
        if self.cost(messages, tools_tokens) > self.available:
            raise ContextBudgetError("mandatory context and uncovered history do not fit; prior summary/history retained")
        return messages

    def prospective(self, response, read_calls, chunks):
        snapshot = copy.deepcopy(self.state)
        snapshot["revision"] += 1
        if complete_response(response):
            snapshot["consumed_calls"].update(read_calls)
            snapshot["consumed_chunks"] = sorted(set(snapshot["consumed_chunks"]) | set(chunks))
        return snapshot

    def restore(self, snapshot):
        self.state = copy.deepcopy(snapshot)
