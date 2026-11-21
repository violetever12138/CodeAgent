import asyncio
from types import SimpleNamespace

from codeagent.protocol import WorkerRole
from codeagent.roles import DEFAULT_ROLES
from codeagent.worker import RoleToolExecutor


class RecordingExecutor:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    async def execute_tool_call(self, call):
        self.calls.append(call)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return call.name

    async def close_tools(self):
        return None


def test_stateful_developer_tools_are_forced_to_run_sequentially(tmp_path):
    delegate = RecordingExecutor()
    executor = RoleToolExecutor(
        delegate,
        DEFAULT_ROLES[WorkerRole.DEVELOPER],
        tmp_path,
    )
    calls = [
        SimpleNamespace(
            name="str_replace_based_edit_tool",
            arguments={},
            call_id=str(index),
            id=None,
        )
        for index in range(2)
    ]

    results = asyncio.run(executor.parallel_tool_call(calls))

    assert results == ["str_replace_based_edit_tool", "str_replace_based_edit_tool"]
    assert delegate.max_active == 1


def test_edit_paths_are_normalized_after_workspace_policy_check(tmp_path):
    delegate = RecordingExecutor()
    executor = RoleToolExecutor(
        delegate,
        DEFAULT_ROLES[WorkerRole.DEVELOPER],
        tmp_path,
    )
    call = SimpleNamespace(
        name="str_replace_based_edit_tool",
        arguments={"path": "src/app.py"},
        call_id="edit-1",
        id=None,
    )

    asyncio.run(executor.execute_tool_call(call))

    assert delegate.calls[0].arguments["path"] == str((tmp_path / "src/app.py").resolve())
