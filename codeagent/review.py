"""Hard final gate for the one integrated patch produced by cooperative workers."""

from __future__ import annotations

from dataclasses import dataclass

from codeagent.artifacts import ArtifactStore
from codeagent.protocol import (
    ReviewDecision,
    ReviewVerdict,
    TaskRequest,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)


@dataclass(frozen=True)
class FinalizationResult:
    integrated: WorkerResult | None
    patch_artifact_id: str | None
    decisions: tuple[ReviewDecision, ...]
    requires_replan: bool
    reasons: tuple[str, ...]


class FinalResultGate:
    """Require security, integrated-patch tests, and independent final review."""

    def finalize(
        self,
        request: TaskRequest,
        results: tuple[WorkerResult, ...],
        artifacts: ArtifactStore,
    ) -> FinalizationResult:
        reviewers = [
            result for result in results
            if result.role == WorkerRole.REVIEWER
            and result.status == WorkerStatus.SUCCEEDED
        ]
        reviewer = max(
            reviewers,
            key=lambda item: int(item.metadata.get("round", 0)),
            default=None,
        )
        decisions = reviewer.review_decisions if reviewer else ()
        integrations = [
            result for result in results
            if result.role == WorkerRole.INTEGRATOR
            and result.status == WorkerStatus.SUCCEEDED
        ]
        latest_integration = max(
            integrations,
            key=lambda item: int(item.metadata.get("round", 0)),
            default=None,
        )
        # A malformed or exhausted Reviewer must not erase independently durable
        # integration, security, and test evidence.  Recover the target from the
        # latest successful integration so the caller can route an otherwise clean
        # run to human approval instead of misreporting an Integrator failure.
        integration_id = (
            str(reviewer.metadata.get("integration_task_id"))
            if reviewer
            else (latest_integration.task_id if latest_integration else "")
        )
        integrated = next(
            (
                result for result in results
                if result.task_id == integration_id
                and result.role == WorkerRole.INTEGRATOR
                and result.status == WorkerStatus.SUCCEEDED
            ),
            None,
        )
        reasons: list[str] = []
        patch_id = self._patch_artifact(integrated, artifacts) if integrated else None
        if integrated is None:
            reasons.append("integrator did not produce a successful result")
        elif patch_id is None:
            reasons.append("integrated result has no patch artifact")

        tester = next(
            (
                result for result in results
                if result.role == WorkerRole.TESTER
                and result.metadata.get("phase") == "verify"
                and result.metadata.get("integrated_task_id") == integration_id
            ),
            None,
        )
        test_failed = False
        if patch_id is not None:
            patch_sha = artifacts.get(patch_id).sha256
            if tester is None or not self._tests_pass(request, tester, patch_sha):
                reasons.append("integrated result failed the deterministic test gate")
                test_failed = tester is not None

        if integrated is not None:
            security = next(
                (
                    result for result in results
                    if result.task_id == integrated.metadata.get("security_task_id")
                    and result.role == WorkerRole.SECURITY
                ),
                None,
            )
            expected = set(integrated.metadata.get("component_task_ids", []))
            security_by_target = {
                item.target_task_id: item for item in security.review_decisions
            } if security else {}
            rejected = [
                task_id for task_id in expected
                if security_by_target.get(task_id) is None
                or security_by_target[task_id].verdict != ReviewVerdict.ACCEPT
            ]
            if rejected:
                reasons.append(f"security gate rejected component tasks: {sorted(rejected)}")

        final_decision = next(
            (item for item in decisions if item.target_task_id == integration_id),
            None,
        )
        if final_decision is None:
            reasons.append("missing reviewer decision for integrated result")
        elif final_decision.verdict != ReviewVerdict.ACCEPT:
            reasons.append(f"reviewer verdict is {final_decision.verdict.value}")

        requires_replan = any(
            item.verdict == ReviewVerdict.REPLAN for item in decisions
        ) or bool(integrated and patch_id and test_failed)
        if not reasons and integrated and patch_id:
            return FinalizationResult(
                integrated=integrated,
                patch_artifact_id=patch_id,
                decisions=decisions,
                requires_replan=False,
                reasons=(),
            )
        return FinalizationResult(
            integrated=None,
            patch_artifact_id=None,
            decisions=decisions,
            requires_replan=requires_replan,
            reasons=tuple(reasons) or ("no valid integrated result",),
        )

    @staticmethod
    def _patch_artifact(
        result: WorkerResult | None, artifacts: ArtifactStore
    ) -> str | None:
        if result is None:
            return None
        for artifact_id in result.artifact_ids:
            try:
                if artifacts.get(artifact_id).kind == "patch":
                    return artifact_id
            except (KeyError, ValueError):
                continue
        return None

    @staticmethod
    def _tests_pass(
        request: TaskRequest, result: WorkerResult, patch_sha256: str
    ) -> bool:
        if result.status != WorkerStatus.SUCCEEDED or not result.verifications:
            return False
        latest: dict[str, bool] = {}
        current_version = result.metadata.get("workspace_version")
        for verification in result.verifications:
            latest[" ".join(verification.command.split())] = (
                verification.passed
                and bool(current_version)
                and verification.workspace_version == current_version
                and verification.patch_sha256 == patch_sha256
            )
        if request.test_commands:
            return all(
                latest.get(" ".join(command.split()), False)
                for command in request.test_commands
            )
        return bool(latest) and all(latest.values())


class PawFinalResultGate(FinalResultGate):
    """Fresh deterministic gates for both single- and multi-Worker runs."""

    def finalize_round(
        self,
        request: TaskRequest,
        results: tuple[WorkerResult, ...],
        artifacts: ArtifactStore,
        *,
        integration_id: str,
        verification_id: str,
        auditor_id: str | None,
        preflight_passed: bool,
    ) -> FinalizationResult:
        by_id = {item.task_id: item for item in results}
        integrated = by_id.get(integration_id)
        tester = by_id.get(verification_id)
        auditor = by_id.get(auditor_id) if auditor_id else None
        reasons: list[str] = []
        if not preflight_passed:
            reasons.append("deterministic patch preflight failed")
        if integrated is None or integrated.role != WorkerRole.INTEGRATOR or integrated.status != WorkerStatus.SUCCEEDED:
            reasons.append("integrated patch missing or failed")
            integrated = None
        patch_id = self._patch_artifact(integrated, artifacts)
        if patch_id is None:
            reasons.append("integrated patch artifact missing")
        elif tester is None or not self._tests_pass(request, tester, artifacts.get(patch_id).sha256):
            reasons.append("fresh integrated verification failed")
        decisions = (
            auditor.review_decisions
            if auditor and auditor.role == WorkerRole.AUDITOR
            and auditor.status == WorkerStatus.SUCCEEDED else ()
        )
        if auditor_id:
            decision = next((item for item in decisions if item.target_task_id == integration_id), None)
            if decision is None:
                reasons.append("auditor decision missing")
            elif decision.verdict != ReviewVerdict.ACCEPT:
                reasons.append(f"auditor verdict is {decision.verdict.value}")
        requires_replan = any(item.verdict == ReviewVerdict.REPLAN for item in decisions)
        requires_replan = requires_replan or bool(
            integrated and patch_id and "fresh integrated verification failed" in reasons
        )
        return FinalizationResult(
            integrated=integrated if not reasons else None,
            patch_artifact_id=patch_id if not reasons else None,
            decisions=decisions,
            requires_replan=requires_replan,
            reasons=tuple(reasons),
        )
