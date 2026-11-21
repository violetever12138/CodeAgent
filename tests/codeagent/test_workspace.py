import asyncio
import subprocess

from codeagent.protocol import TaskNode, WorkerRole
from codeagent.workspace import GitWorkspaceManager


def test_developers_and_verifier_get_isolated_clean_worktrees(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "codeswarm@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CodeSwarm Test"], cwd=repository, check=True
    )
    (repository / "module.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repository, check=True, capture_output=True)
    manager = GitWorkspaceManager(repository, tmp_path / "worktrees")
    developer_a = TaskNode("developer-a", WorkerRole.DEVELOPER, "core component")
    developer_b = TaskNode("developer-b", WorkerRole.DEVELOPER, "interface component")
    tester_a = TaskNode(
        "tester-a",
        WorkerRole.TESTER,
        "verify a",
        metadata={"phase": "verify", "integrated_task_id": "developer-a", "workspace_from": "developer-a"},
    )

    async def scenario():
        workspace_a, workspace_b = await asyncio.gather(
            manager.acquire(developer_a), manager.acquire(developer_b)
        )
        (workspace_a / "module.py").write_text("value = 2\n", encoding="utf-8")
        (workspace_a / "new_feature.py").write_text("enabled = True\n", encoding="utf-8")
        tester_workspace = await manager.acquire(tester_a)
        assert workspace_a != workspace_b
        assert tester_workspace != workspace_a
        assert (tester_workspace / "module.py").read_text(encoding="utf-8") == "value = 2\n"
        assert (tester_workspace / "new_feature.py").read_text(encoding="utf-8") == "enabled = True\n"
        assert (workspace_b / "module.py").read_text(encoding="utf-8") == "value = 1\n"
        reviewer = TaskNode("review-a", WorkerRole.REVIEWER, "review integrated result",
                            metadata={"workspace_from": "developer-a"})
        assert await manager.acquire(reviewer) == workspace_a
        await manager.reset(reviewer.id)
        assert workspace_a.exists()
        assert (workspace_a / "module.py").read_text(encoding="utf-8") == "value = 2\n"
        await manager.cleanup()
        assert not workspace_a.exists()
        assert not workspace_b.exists()
        assert not tester_workspace.exists()

    asyncio.run(scenario())
