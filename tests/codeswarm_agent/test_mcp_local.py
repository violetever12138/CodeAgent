"""Real stdio service and schema/result boundary checks, without model APIs."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from codeswarm_agent.artifacts import ArtifactStore
from codeswarm_agent.cancellation import CancellationToken, RunCancelled
from codeswarm_agent.mcp_config import bind_mcp_configs
from codeswarm_agent.protocol import WorkerRole
from codeswarm_agent.roles import DEFAULT_ROLES
from codeswarm_agent.runtime.tool_policy import ToolPolicy
from trae_agent.tools.base import ToolCall, ToolExecutor
from trae_agent.tools.mcp_tool import MCPTool
from trae_agent.utils.config import MCPServerConfig
from trae_agent.utils.mcp_client import MCPClient, MCPServerStatus

PACKAGE_ROOT = Path(__file__).parents[2]


def _config(workspace, artifacts):
    return bind_mcp_configs({"local": MCPServerConfig(
        command="${CODESWARM_PYTHON}", args=["-m", "codeswarm_agent.mcp_server"],
        cwd="${CODESWARM_PACKAGE_ROOT}", timeout=20,
        env={"PYTHONUTF8": "1"})}, workspace, artifacts)["local"]


def test_real_stdio_business_tools_discover_read_and_reject_path_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "code.py").write_text("first line\nsecond line\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    record = artifacts.put_text("test", "artifact one\nartifact two\n", ".txt")

    async def run():
        client, tools = MCPClient(), []
        try:
            await client.connect_and_discover("local", _config(root, artifacts.root), tools, "openai")
            assert {t.tool.name for t in tools} == {"repository_read", "artifact_read"}
            assert all(t.tool.annotations.readOnlyHint for t in tools)
            executor = ToolExecutor(tools)
            response = await executor.execute_tool_call(ToolCall("mcp__local__repository_read", "read", {
                "operation": "read", "path": "code.py"}))
            assert response.success and "second line" in response.result
            raw = json.loads(response.result)
            assert raw["content"][0]["text"] == raw["structuredContent"]["output"]
            artifact = await client.call_tool("artifact_read", {"artifact_id": record.id, "start_line": 1, "end_line": 2}, "local")
            assert "artifact two" in artifact.content[0].text
            for args in ({"operation": "read", "path": str(outside)}, {"operation": "write", "path": "code.py"}):
                rejected = await executor.execute_tool_call(ToolCall("mcp__local__repository_read", "bad", args))
                assert not rejected.success and rejected.error
            unknown = await client.call_tool("task_done", {}, "local")
            assert unknown.isError
            assert "query" not in next(t for t in tools if t.tool.name == "repository_read").get_input_schema()["properties"]["operation"]["enum"]
        finally:
            await client.cleanup_all()
        assert not client.sessions and not client.exit_stacks and not client._owners
        assert client.get_mcp_server_status("local") == MCPServerStatus.DISCONNECTED

    asyncio.run(run())
    artifacts.close()


def test_mcp_schema_preserves_nested_enum_union_and_host_validates_it(tmp_path):
    schema = {"type": "object", "$defs": {"item": {"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["fast", "safe"]}}, "required": ["mode"], "additionalProperties": False}},
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/item"}},
                       "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["items"], "additionalProperties": False}
    tool = MCPTool(SimpleNamespace(), types.Tool(name="nested", inputSchema=schema), "openai", server_name="local")
    assert tool.get_input_schema() == schema and not tool.strict_schema
    copy = tool.get_input_schema()
    copy["required"] = []
    assert tool.get_input_schema()["required"] == ["items"]
    policy = ToolPolicy(DEFAULT_ROLES[WorkerRole.EXPLORER], tmp_path, [tool], additional_allowed=(tool.name,))
    assert policy.validate(ToolCall(tool.name, "ok", {"items": [{"mode": "safe"}]})).allowed
    assert not policy.validate(ToolCall(tool.name, "bad", {"items": [{"mode": "other"}]})).allowed
    assert not policy.validate(ToolCall(tool.name, "bad", {"items": "wrong"})).allowed
    from trae_agent.utils.config import ModelConfig
    from trae_agent.utils.llm_clients.google_client import GoogleClient
    from trae_agent.utils.llm_clients.openai_client import OpenAIClient
    openai = OpenAIClient.__new__(OpenAIClient)
    _, declared = openai._prepare_request([], [tool], False)
    assert declared[0]["strict"] is False and declared[0]["parameters"] == schema
    google = GoogleClient.__new__(GoogleClient)
    google.system_instruction, google.message_history = None, []
    model = ModelConfig(model="fixture", model_provider="fixture", temperature=0, top_p=1,
                        top_k=0, parallel_tool_calls=False, max_retries=0)
    _, declared, _, _ = google._prepare_request([], model, [tool], False)
    assert declared.tools[0].function_declarations[0].parameters_json_schema == schema


@pytest.mark.parametrize("kind", ["texts", "structured", "image", "empty", "error"])
def test_complete_mcp_results_and_error_status(kind):
    content = [types.TextContent(type="text", text="one"), types.TextContent(type="text", text="two")]
    structured = None
    if kind == "structured":
        structured = {"details": [1, 2, 3]}
    if kind == "image":
        content.append(types.ImageContent(type="image", data=base64.b64encode(b"image bytes").decode(), mimeType="image/png"))
    if kind == "empty":
        content = []
    response = types.CallToolResult(content=content, structuredContent=structured, isError=kind == "error")
    client = SimpleNamespace(call_tool=AsyncMock(return_value=response))
    tool = MCPTool(client, types.Tool(name="result", inputSchema={"type": "object"}))
    result = asyncio.run(ToolExecutor([tool]).execute_tool_call(ToolCall(tool.name, "result", {})))
    assert result.success == (kind != "error")
    text = result.error if kind == "error" else result.result
    if kind == "texts" or kind == "error":
        assert text == "one\n\ntwo"
    elif kind == "empty":
        assert text == ""
    else:
        decoded = json.loads(text)
        assert len(decoded["content"]) == len(content)
        assert decoded["structuredContent"] == structured


def _alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            status = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("mode", ["normal", "timeout", "cancel", "parallel-timeout"])
def test_real_child_lifecycle_timeout_and_cancellation(tmp_path, mode):
    pid_path = tmp_path / "pid.txt"
    script = tmp_path / "fixture.py"
    script.write_text('''import asyncio, os, sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
Path(sys.argv[1]).write_text(str(os.getpid()))
server = FastMCP("lifecycle-test")
@server.tool()
async def slow() -> str:
    await asyncio.sleep(30)
    return "finished"
server.run(transport="stdio")
''', encoding="utf-8")

    async def run():
        cancellation = CancellationToken()
        client = MCPClient(cancellation)
        config = MCPServerConfig(command=sys.executable, args=[str(script), str(pid_path)], cwd=str(PACKAGE_ROOT), timeout=20)
        await client.connect_and_discover("fixture", config, [], None)
        pid = int(pid_path.read_text())
        assert _alive(pid)
        try:
            if mode == "timeout":
                client.server_timeouts["fixture"] = .1
                with pytest.raises(TimeoutError):
                    await client.call_tool("slow", {}, "fixture")
                assert not cancellation.cancelled
            elif mode == "parallel-timeout":
                client.server_timeouts["fixture"] = .1
                results = await asyncio.gather(*(client.call_tool("slow", {}, "fixture") for _ in range(2)), return_exceptions=True)
                assert all(isinstance(result, TimeoutError) for result in results)
            elif mode == "cancel":
                pending = asyncio.create_task(client.call_tool("slow", {}, "fixture"))
                await asyncio.sleep(.1)
                cancellation.cancel("test cancellation")
                with pytest.raises(RunCancelled):
                    await pending
        finally:
            await client.cleanup_all()
        assert not _alive(pid)
        assert not client.sessions and not client._owners

    asyncio.run(run())


def test_startup_timeout_closes_a_child_that_never_initializes(tmp_path):
    pid_path = tmp_path / "pid.txt"
    script = tmp_path / "startup.py"
    script.write_text("import os, sys, time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n")

    async def run():
        client = MCPClient()
        config = MCPServerConfig(command=sys.executable, args=[str(script), str(pid_path)], timeout=1.5)
        with pytest.raises(TimeoutError):
            await client.connect_and_discover("startup", config, [], None)
        assert not _alive(int(pid_path.read_text()))
        assert not client._owners and not client.sessions

    asyncio.run(run())


def test_configuration_binding_does_not_mutate_or_cross_worker_directories(tmp_path):
    config = {"local": MCPServerConfig(command="${CODESWARM_PYTHON}", cwd="${CODESWARM_PACKAGE_ROOT}",
                                      env={"CODESWARM_WORKSPACE": "wrong"})}
    first = bind_mcp_configs(config, tmp_path / "one", tmp_path / "artifacts")
    second = bind_mcp_configs(config, tmp_path / "two", tmp_path / "artifacts")
    assert first["local"].env["CODESWARM_WORKSPACE"] != second["local"].env["CODESWARM_WORKSPACE"]
    assert config["local"].env["CODESWARM_WORKSPACE"] == "wrong"


def test_mcp_read_alias_cannot_bypass_native_read_guard_by_case_or_prefix(tmp_path):
    from codeswarm_agent.repository_tool import RepositoryReadTool
    from codeswarm_agent.worker import DeveloperReadGuard, RoleToolExecutor
    client = SimpleNamespace(call_tool=AsyncMock(return_value=types.CallToolResult(
        content=[types.TextContent(type="text", text="body")], isError=False)))
    native = RepositoryReadTool(tmp_path)
    remote = MCPTool(client, types.Tool(name="repository_read", inputSchema=native.get_input_schema()), server_name="local")
    executor = RoleToolExecutor(ToolExecutor([native, remote]), DEFAULT_ROLES[WorkerRole.DEVELOPER], tmp_path,
                                additional_allowed=(remote.name,), read_guard=DeveloperReadGuard())

    async def run():
        args = {"operation": "read", "path": "code.py"}
        first = await executor.execute_tool_call(ToolCall("MCP__local__Repository_Read", "one", args))
        duplicate = await executor.execute_tool_call(ToolCall("repository_read", "two", args))
        assert first.success and not duplicate.success and "READ_GUARD" in duplicate.error
        assert client.call_tool.await_count == 1

    asyncio.run(run())
