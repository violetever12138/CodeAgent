"""Deterministic task memory; observed facts never arise from a free-text summary."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fact:
    claim: str
    artifact_id: str
    repository_version: str
    source_path: str | None = None
    file_hash: str | None = None
    status: str = "observed"


@dataclass
class WorkingState:
    accepted_facts: list[Fact] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    current_plan: str = ""
    changes: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    latest_failure: str = ""

    def add_observation(
        self,
        *,
        claim: str,
        artifact_id: str,
        repository_version: str,
        source_path: str | None = None,
        file_hash: str | None = None,
        is_failure: bool = False,
        is_test: bool = False,
        changed_files: tuple[str, ...] = (),
    ) -> None:
        self.accepted_facts.append(Fact(
            claim=claim, artifact_id=artifact_id, repository_version=repository_version,
            source_path=source_path, file_hash=file_hash,
        ))
        self.accepted_facts = self.accepted_facts[-40:]
        self.artifact_refs.append(artifact_id)
        self.artifact_refs = list(dict.fromkeys(self.artifact_refs[-80:]))
        if is_failure:
            self.latest_failure = claim
        if is_test:
            self.verification.append(claim)
            self.verification = self.verification[-20:]
        self.changes.extend(changed_files)
        self.changes = list(dict.fromkeys(self.changes[-40:]))

    def invalidate(self, file_hashes: dict[str, str | None], repository_version: str) -> None:
        self.accepted_facts = [
            Fact(**{**fact.__dict__, "status": "stale"})
            if (
                fact.source_path in file_hashes
                and fact.file_hash != file_hashes[fact.source_path]
            ) or (fact.source_path is None and fact.repository_version != repository_version)
            else fact
            for fact in self.accepted_facts
        ]

    def compact(self) -> str:
        facts = [fact for fact in self.accepted_facts if fact.status == "observed"][-12:]
        lines = ["[Deterministic working state; hypotheses are not facts]"]
        lines.extend(f"FACT: {fact.claim} [artifact={fact.artifact_id}]" for fact in facts)
        lines.extend(f"HYPOTHESIS (unverified): {item}" for item in self.hypotheses[-5:])
        lines.extend(f"REJECTED: {item}" for item in self.rejected_hypotheses[-5:])
        lines.extend(f"TEST: {item}" for item in self.verification[-8:])
        lines.extend(f"CHANGED: {item}" for item in self.changes[-12:])
        lines.extend(f"OPEN: {item}" for item in self.open_questions[-5:])
        lines.append(f"CURRENT PLAN: {self.current_plan or 'follow assigned goal'}")
        return "\n".join(lines)
