from codeagent.artifacts import ArtifactStore
from codeagent.protocol import (
    ReviewDecision,
    ReviewVerdict,
    TaskRequest,
    VerificationRecord,
    WorkerResult,
    WorkerRole,
    WorkerStatus,
)
from codeagent.review import FinalResultGate


def _results(artifacts, *, test_passed=True, patch_sha=None, security_accept=True):
    patch = artifacts.put_text("patch", "diff --git a/a.py b/a.py\n", ".patch")
    integrated = WorkerResult(
        task_id="integrate-0",
        attempt=0,
        role=WorkerRole.INTEGRATOR,
        status=WorkerStatus.SUCCEEDED,
        summary="integrated",
        artifact_ids=(patch.id,),
        changed_files=("a.py",),
        metadata={
            "component_task_ids": ["develop-core"],
            "security_task_id": "security-0",
        },
    )
    security = WorkerResult(
        task_id="security-0",
        attempt=0,
        role=WorkerRole.SECURITY,
        status=WorkerStatus.SUCCEEDED,
        summary="security review",
        review_decisions=(ReviewDecision(
            "develop-core",
            ReviewVerdict.ACCEPT if security_accept else ReviewVerdict.REPLAN,
        ),),
    )
    tester = WorkerResult(
        task_id="test-integrated-0",
        attempt=0,
        role=WorkerRole.TESTER,
        status=WorkerStatus.SUCCEEDED,
        summary="test result",
        verifications=(VerificationRecord(
            "pytest -q",
            passed=test_passed,
            workspace_version="v1",
            patch_sha256=patch_sha or patch.sha256,
        ),),
        metadata={
            "phase": "verify",
            "integrated_task_id": "integrate-0",
            "workspace_version": "v1",
        },
    )
    reviewer = WorkerResult(
        task_id="review-0",
        attempt=0,
        role=WorkerRole.REVIEWER,
        status=WorkerStatus.SUCCEEDED,
        summary="accept",
        review_decisions=(ReviewDecision(
            "integrate-0", ReviewVerdict.ACCEPT, confidence=0.99
        ),),
        metadata={"integration_task_id": "integrate-0", "round": 0},
    )
    return integrated, security, tester, reviewer


def _request(tmp_path):
    return TaskRequest(
        issue="fix",
        repository=str(tmp_path),
        test_commands=("pytest -q",),
    )


def test_reviewer_cannot_override_failed_integrated_test_gate(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    final = FinalResultGate().finalize(
        _request(tmp_path), _results(artifacts, test_passed=False), artifacts
    )

    assert final.integrated is None
    assert "test gate" in final.reasons[0]
    assert final.requires_replan


def test_passing_security_test_and_review_accept_the_integrated_patch(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    final = FinalResultGate().finalize(
        _request(tmp_path), _results(artifacts), artifacts
    )

    assert final.integrated is not None
    assert final.integrated.task_id == "integrate-0"
    assert final.patch_artifact_id


def test_test_evidence_for_another_patch_is_rejected(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    final = FinalResultGate().finalize(
        _request(tmp_path), _results(artifacts, patch_sha="0" * 64), artifacts
    )

    assert final.integrated is None
    assert "test gate" in final.reasons[0]


def test_security_rejection_cannot_be_overridden_by_final_review(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    final = FinalResultGate().finalize(
        _request(tmp_path), _results(artifacts, security_accept=False), artifacts
    )

    assert final.integrated is None
    assert "security gate rejected" in final.reasons[0]
    assert not final.requires_replan


def test_missing_reviewer_preserves_successful_hard_gate_evidence(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    integrated, security, tester, _reviewer = _results(artifacts)

    final = FinalResultGate().finalize(
        _request(tmp_path), (integrated, security, tester), artifacts
    )

    assert final.integrated is None
    assert final.patch_artifact_id is None
    assert final.reasons == ("missing reviewer decision for integrated result",)
