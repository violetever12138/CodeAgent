"""Append references instead of rewriting complete conversations every step."""

import json
from datetime import datetime

from trae_agent.utils.redaction import redact_value
from trae_agent.utils.request_observer import last_request
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class ReferenceTrajectoryRecorder(TrajectoryRecorder):
    def __init__(self, path):
        super().__init__(path)
        self._finalized = False

    def _append(self, item):
        try:
            with self.trajectory_path.with_suffix(".jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact_value(item), ensure_ascii=False) + "\n")
        except Exception:
            from codeagent.telemetry import current_operation
            session = current_operation().session
            if session:
                session.missing("trajectory_append")

    def record_llm_interaction(self, messages, response, provider, model, tools=None):
        observed = last_request.get()
        item = {"type": "llm_reference", "at": datetime.now().isoformat(), "provider": provider, "model": model,
                "input_ref": observed.input_ref if observed else None, "response_ref": observed.output_ref if observed else None}
        if observed and observed.active:
            item["span_id"] = f"{observed.active.get_span_context().span_id:016x}"
            observed.session.register_response([{"type": "function", "call_id": call.call_id}
                                                for call in response.tool_calls or []], observed)
        self.trajectory_data["llm_interactions"].append(item)
        self._append(item)

    def record_agent_step(self, step_number, state, llm_messages=None, llm_response=None,
                          tool_calls=None, tool_results=None, reflection=None, error=None):
        item = {"type": "step", "step_number": step_number, "state": state,
                "tool_call_ids": [call.call_id for call in tool_calls or []],
                "tool_status": [{"call_id": result.call_id, "success": result.success} for result in tool_results or []],
                "error": error[:600] if error else None}
        self.trajectory_data["agent_steps"].append(item)
        self._append(item)

    def update_lakeview(self, step_number, lakeview_summary):
        self._append({"type": "lakeview", "step_number": step_number, "summary": lakeview_summary})

    def finalize_recording(self, success, final_result=None):
        self._finalized = True
        records = self.trajectory_data["llm_interactions"]
        self.trajectory_data["final_result_ref"] = records[-1].get("response_ref") if records else None
        super().finalize_recording(success, final_result[:600] if final_result else None)

    def save_trajectory(self):
        if getattr(self, "_finalized", False):
            super().save_trajectory()
