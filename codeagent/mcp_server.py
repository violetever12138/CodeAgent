"""Local stdio adapter for existing read-only business tools."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from codeagent.artifacts import ArtifactStore
from codeagent.context.artifact_tool import ArtifactReadTool
from codeagent.protocol import WorkerRole
from codeagent.repository_tool import RepositoryReadTool
from codeagent.roles import RoleSpec
from codeagent.runtime.tool_policy import ToolPolicy


def create_server(workspace: Path, artifacts: ArtifactStore | None = None):
    tools = [RepositoryReadTool(workspace)]
    if artifacts is not None:
        tools.append(ArtifactReadTool(artifacts))
    by_name = {tool.name: tool for tool in tools}
    role = RoleSpec(WorkerRole.EXPLORER, "Local MCP read service", "", tuple(by_name), True)
    policy = ToolPolicy(role, workspace, tools)
    server = Server("codeswarm-local-read", version="1")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=tool.name, description=tool.description, inputSchema=tool.get_input_schema(),
                           annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                                             idempotentHint=True, openWorldHint=False))
                for tool in tools]

    @server.call_tool()
    async def call_tool(name, arguments):
        from trae_agent.tools.base import ToolCall
        decision = policy.validate(ToolCall(name, "mcp-read", arguments))
        if not decision.allowed:
            raise ValueError(f"{decision.code}: {decision.reason}")
        result = await by_name[name].execute(arguments)
        if result.error_code or result.error:
            raise ValueError(result.error or "business tool failed")
        # Keep both the readable result and machine-readable envelope.
        return ([types.TextContent(type="text", text=result.output or "")],
                {"output": result.output, "error": result.error, "error_code": result.error_code})

    return server


def _bound_path(value, environment_name, *, required=False):
    bound = os.environ.get(environment_name)
    selected = value or bound
    if not selected:
        if required:
            raise ValueError(f"{environment_name} or an explicit directory is required")
        return None
    path = Path(selected).resolve()
    if bound and path != Path(bound).resolve():
        raise ValueError(f"directory differs from host binding: {environment_name}")
    if not path.is_dir():
        raise ValueError(f"directory does not exist: {path}")
    return path


async def serve(workspace, artifact_root=None):
    artifacts = ArtifactStore(artifact_root) if artifact_root else None
    try:
        server = create_server(workspace, artifacts)
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        if artifacts:
            artifacts.close()


def main():
    parser = argparse.ArgumentParser(prog="codeswarm-mcp")
    parser.add_argument("--workspace")
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        workspace = _bound_path(args.workspace, "CODESWARM_WORKSPACE", required=True)
        artifact_root = _bound_path(args.artifact_root, "CODESWARM_ARTIFACT_ROOT")
        asyncio.run(serve(workspace, artifact_root))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
