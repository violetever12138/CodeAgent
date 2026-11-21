"""Apply the same progress-summary rules to Manager decision history."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

from codeswarm_agent.context.engine import ContextSettings
from codeswarm_agent.context.lifecycle import ContextLifecycle
from codeswarm_agent.context.packer import ContextBudgetError, TokenCounter


async def prepare_manager_context(runtime, prompt, summarize, checkpoint=None):
    from trae_agent.utils.llm_clients.llm_basics import LLMMessage

    instruction, raw = prompt.split("\nTask state:\n", 1)
    data = json.loads(raw)
    history = data["manager"].get("history", [])
    manager = copy.deepcopy(data["manager"])
    for key in ("history", "action", "initial_probe", "progress_signatures", "assignment_signatures"):
        manager.pop(key, None)
    data["manager"] = {**manager, "history": []}
    # Operation ledgers remain in trusted run state. Their full ID lists are
    # redundant model input; retain the actual limits, usage and remaining time.
    if "budget" in data:
        data["budget"] = {key: value for key, value in data["budget"].items()
                          if key not in {"claimed_operations", "recorded_operations"}}
    snapshot = runtime.results.current.get().contexts.get("manager")
    settings = getattr(runtime.worker_factory, "context_settings", None) or ContextSettings()
    model = getattr(getattr(getattr(runtime.worker_factory, "config", None), "trae_agent", None), "model", None)
    if model:
        settings = replace(settings, output_reserve_tokens=max(settings.output_reserve_tokens, model.get_max_tokens_param()))
    counter = TokenCounter(model.model if model else None)
    lifecycle = ContextLifecycle(counter, settings, runtime.artifacts, snapshot)
    if lifecycle.state.get("probe_consumed"):
        data["repository_evidence"] = ""
    fixed = instruction + "\nTask state:\n" + json.dumps(data, ensure_ascii=False)
    system = "You are the task's persistent Manager. Output JSON only."
    groups = [(index, [LLMMessage(role="assistant", content=json.dumps(item, ensure_ascii=False))])
              for index, item in enumerate(history, 1)]
    await lifecycle.prepare([LLMMessage(role="system", content=system), LLMMessage(role="user", content=fixed)],
                            groups, [], 0, summarize, workspace_version=manager["version"], checkpoint=checkpoint)
    data["manager"]["history"] = history[lifecycle.state["covered_through"]:]
    if lifecycle.state["summary"]:
        data["progress_summary"] = {"model_authored": True, "content": lifecycle.state["summary"],
                                    "covered_through": lifecycle.state["covered_through"],
                                    "source_version": lifecycle.state.get("summary_workspace_version")}
    actual = instruction + "\nTask state:\n" + json.dumps(data, ensure_ascii=False)
    estimated = lifecycle.cost([LLMMessage(role="system", content=system), LLMMessage(role="user", content=actual)])
    if estimated > lifecycle.available:
        raise ContextBudgetError("Manager current constraints and uncovered history exceed the input budget")
    lifecycle.state["last_input_tokens"] = estimated
    return actual, lifecycle
