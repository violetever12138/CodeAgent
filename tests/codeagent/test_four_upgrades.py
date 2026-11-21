from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from codeagent.cancellation import CancellationToken, RunCancelled
from codeagent.context import ContextSettings
from codeagent.context.dense import USearchDenseIndex
from codeagent.context.protocol import CodeChunk
from codeagent.protocol import WorkerRole
from codeagent.roles import get_role_spec
from codeagent.runtime.tool_policy import ToolPolicy
from codeagent.safety import DockerBackend, SandboxProfile
from trae_agent.utils.config import CodeSwarmConfig, ConfigError
from trae_agent.utils.llm_clients.retry_utils import async_retry_with
from trae_agent.utils.mcp_client import MCPClient, MCPServerStatus
from trae_agent.utils.redaction import REDACTED, redact_value


@pytest.mark.asyncio
async def test_cancellation_interrupts_inflight_remote_wait(tmp_path: Path):
    token = CancellationToken(tmp_path / "cancel.json")
    cleaned = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task = asyncio.create_task(token.run(blocked()))
    await asyncio.sleep(0)
    token.cancel("operator stop")
    with pytest.raises(RunCancelled, match="operator stop"):
        await asyncio.wait_for(task, 1)
    assert cleaned.is_set()
    assert (tmp_path / "cancel.json").exists()


def test_mcp_policy_requires_role_allowlist_and_denies_external_write(tmp_path: Path):
    role = get_role_spec(WorkerRole.EXPLORER)
    read_tool = SimpleNamespace(
        name="mcp__docs__search", policy_effect="read", parameters=[]
    )
    external_tool = SimpleNamespace(
        name="mcp__tracker__create", policy_effect="external_write", parameters=[]
    )
    policy = ToolPolicy(
        role, tmp_path, [read_tool, external_tool],
        additional_allowed=(read_tool.name, external_tool.name),
    )
    read = policy.validate(SimpleNamespace(name=read_tool.name, arguments={}))
    external = policy.validate(SimpleNamespace(name=external_tool.name, arguments={}))
    assert read.allowed
    assert not external.allowed
    assert external.rule_id == "MCP-EXTERNAL-WRITE-DENY"


def test_codeswarm_config_defaults_and_validates_effects():
    assert CodeSwarmConfig().architecture_mode == "manager_v2"
    assert CodeSwarmConfig().dense_backend == "brute_force"
    with pytest.raises(ConfigError, match="invalid MCP tool effects"):
        CodeSwarmConfig(mcp_tool_effects={"server.tool": "unknown"})
    assert ContextSettings(dense_backend="usearch").dense_backend == "usearch"


def test_docker_backend_emits_hardened_per_worker_configuration(tmp_path: Path):
    image = "example/agent@sha256:" + "a" * 64
    backend = DockerBackend(image, SandboxProfile())
    config = backend.agent_config(tmp_path)
    assert config["image"] == image
    assert config["workspace_dir"] == str(tmp_path.resolve())
    options = config["security_options"]
    assert options["network_disabled"] is True
    assert options["read_only"] is True
    assert options["cap_drop"] == ["ALL"]


@pytest.mark.asyncio
async def test_async_provider_retry_never_retries_cancelled_request():
    calls = 0

    async def blocked():
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    request = async_retry_with(blocked, max_retries=4)
    task = asyncio.create_task(request())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1


@pytest.mark.asyncio
async def test_mcp_cancellation_closes_only_target_server(tmp_path: Path):
    token = CancellationToken(tmp_path / "cancel.json")
    client = MCPClient(token)
    async def blocked_call(*_args):
        await asyncio.Event().wait()

    session = SimpleNamespace(call_tool=AsyncMock(side_effect=blocked_call))
    stack = SimpleNamespace(aclose=AsyncMock())
    client.sessions["docs"] = session
    client.session = session
    client.exit_stacks["docs"] = stack
    client.server_timeouts["docs"] = 30
    client.update_mcp_server_status("docs", MCPServerStatus.CONNECTED)

    task = asyncio.create_task(client.call_tool("search", {}, server_name="docs"))
    await asyncio.sleep(0)
    token.cancel("stop MCP")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    stack.aclose.assert_awaited_once()
    assert client.get_mcp_server_status("docs") == MCPServerStatus.DISCONNECTED


def test_trace_redaction_covers_headers_environment_and_nested_arguments():
    payload = redact_value({
        "headers": {"Authorization": "Bearer abc.def", "X-Api-Key": "top-secret"},
        "env": {"ACCESS_TOKEN": "token-value", "SAFE": "visible"},
        "text": "password=hunter2 sk-abcdefghijklmnop",
    })
    assert payload["headers"]["Authorization"] == REDACTED
    assert payload["headers"]["X-Api-Key"] == REDACTED
    assert payload["env"]["ACCESS_TOKEN"] == REDACTED
    assert payload["env"]["SAFE"] == "visible"
    assert "hunter2" not in payload["text"]
    assert "sk-" not in payload["text"]


def test_usearch_index_incremental_tombstones_and_compaction(tmp_path: Path, monkeypatch):
    import numpy as np

    storage: dict[str, dict[int, list[float]]] = {}

    class FakeMatches:
        def __init__(self, keys):
            self.keys = np.asarray(keys, dtype=np.uint64)

    class FakeIndex:
        def __init__(self, **_kwargs):
            self.values: dict[int, list[float]] = {}

        def add(self, keys, vectors, **_kwargs):
            for key, vector in zip(np.atleast_1d(keys), np.atleast_2d(vectors), strict=True):
                self.values[int(key)] = [float(value) for value in vector]

        def remove(self, key):
            self.values.pop(int(key), None)

        def save(self, path=None):
            if path is None:
                return json.dumps(self.values).encode()
            storage[str(path)] = dict(self.values)
            Path(path).write_text("fake", encoding="utf-8")

        def load(self, path):
            if isinstance(path, bytes):
                self.values = {
                    int(key): value for key, value in json.loads(path.decode()).items()
                }
            else:
                self.values = dict(storage[str(path)])

        def search(self, vector, count):
            ranked = sorted(
                self.values,
                key=lambda key: -sum(
                    left * right for left, right in zip(vector, self.values[key], strict=True)
                ),
            )
            return FakeMatches(ranked[:count])

    # Mirror atomic file replacement in the in-memory fake persistence layer.
    original_replace = Path.replace

    def replace(source, target):
        if str(source) in storage:
            storage[str(target)] = storage.pop(str(source))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    package = ModuleType("usearch")
    module = ModuleType("usearch.index")
    module.Index = FakeIndex
    package.index = module
    monkeypatch.setitem(sys.modules, "usearch", package)
    monkeypatch.setitem(sys.modules, "usearch.index", module)

    class Provider:
        model_id = "fixture-v1"

        def embed(self, texts):
            return [[float(len(text)), float(sum(map(ord, text)) % 97)] for text in texts]

    def chunk(number: int, version: str = "v1") -> CodeChunk:
        text = f"chunk {number} {version}"
        return CodeChunk(
            id=f"{number}-{version}", path=f"f{number}.py", language="python",
            symbol=None, start_line=1, end_line=1, text=text,
            file_hash=f"hash-{number}-{version}", repository_version=version,
        )

    index_path = tmp_path / "dense.usearch"
    initial = USearchDenseIndex(tuple(chunk(i) for i in range(4)), Provider(), index_path=index_path)
    assert initial.last_update["rebuilt"] is True

    changed = USearchDenseIndex(
        (chunk(0, "v2"), chunk(1), chunk(2), chunk(3)), Provider(), index_path=index_path
    )
    assert changed.last_update == {
        "added": 1, "removed": 1, "tombstones": 1, "rebuilt": False,
        "rebuild_scheduled": False,
    }
    metadata = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["tombstones"] == 1
    assert changed.search("chunk 0 v2", top_n=2)

    compacted = USearchDenseIndex(
        (chunk(0, "v2"), chunk(1), chunk(2)), Provider(), index_path=index_path
    )
    assert compacted.last_update["rebuild_scheduled"] is True
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        metadata = json.loads(index_path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["tombstones"] == 0:
            break
        time.sleep(0.01)
    assert metadata["tombstones"] == 0
