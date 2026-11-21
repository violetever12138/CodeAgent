"""Frozen configuration for reproducible, same-budget Agent comparisons."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationManifest:
    run_id: str
    git_commit: str
    agent_variant: str
    task_set_version: str
    model_provider: str
    model_name: str
    model_revision: str | None
    temperature: float
    seed: int | None
    budget: dict[str, int | float | None]
    sandbox_profile: str
    prompt_version: str
    policy_version: str
    harness_version: str
    schema_version: int = 2
    architecture: str = "langgraph-manager-v2"
    orchestration_backend: str = "langgraph"
    langgraph_version: str = "1.2.12"
    checkpoint_backend: str = "sqlite-3.1.1"
    orchestration_state_schema: int = 4
    recovery_count: int = 0
    context_profile: str = "incremental-hybrid-v2"
    reranker: dict[str, Any] = field(default_factory=dict)
    index_schema: int = 2
    artifact_schema: int = 2
    environment: dict[str, Any] = field(default_factory=dict)
    dependency_lock_hash: str = "unrecorded"
    task_set_checksum: str = "unrecorded"
    execution_backend: str = "unrecorded"
    execution_strength: str = "unrecorded"
    docker_inspect_verified: bool = False
    router_profile: str = "manager_v2"
    dense_backend: str = "brute_force"
    ann_schema: int = 0
    mcp_profile: str = "disabled"
    cancellation_protocol: str = "async-v1"
    docker_backend_version: str = "session-v1"
    experiment_factors: tuple[str, ...] = ()
    architecture_mode: str = "manager_v2"
    auditor_schema: int = 0
    planning_decision_schema: int = 1

    def __post_init__(self) -> None:
        required = (
            self.run_id, self.git_commit, self.agent_variant, self.task_set_version,
            self.model_provider, self.model_name, self.sandbox_profile,
            self.prompt_version, self.policy_version, self.harness_version,
        )
        if any(not value.strip() for value in required):
            raise ValueError("manifest identity fields cannot be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.schema_version not in {1, 2, 3, 4}:
            raise ValueError("unsupported evaluation manifest schema")
        object.__setattr__(self, "experiment_factors", tuple(self.experiment_factors))
        for name, value in self.budget.items():
            if value is not None and (value < 0 or (
                value == 0 and name not in {"max_replans", "max_retries_per_task"}
            )):
                raise ValueError(f"budget {name} must be positive")


class ManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def save(self, manifest: EvaluationManifest) -> str:
        body = asdict(manifest)
        digest = hashlib.sha256(self._canonical(body)).hexdigest()
        payload = {**body, "checksum": digest}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)
        return digest

    def load(self) -> EvaluationManifest:
        raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        observed = raw.pop("checksum", None)
        expected = hashlib.sha256(self._canonical(raw)).hexdigest()
        if observed != expected:
            raise ValueError("evaluation manifest checksum mismatch")
        raw.setdefault("schema_version", 1)
        return EvaluationManifest(**raw)

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
