import asyncio
import inspect
from contextlib import AsyncExitStack
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..tools.mcp_tool import MCPTool
from .config import MCPServerConfig


class MCPServerStatus(Enum):
    DISCONNECTED = "disconnected"  # Server is disconnected or experiencing errors
    CONNECTING = "connecting"  # Server is in the process of connecting
    CONNECTED = "connected"  # Server is connected and ready to use


class MCPDiscoveryState(Enum):
    """State of MCP discovery process."""

    NOT_STARTED = "not_started"  # Discovery has not started yet
    IN_PROGRESS = "in_progress"  # Discovery is currently in progress
    # Discovery has completed (with or without errors)
    COMPLETED = "completed"


class MCPClient:
    def __init__(self, cancellation=None):
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stacks: dict[str, AsyncExitStack] = {}
        self.server_timeouts: dict[str, float] = {}
        self.mcp_servers_status: dict[str, MCPServerStatus] = {}
        self.cancellation = cancellation
        self.timeout_seconds: float = 60.0
        self._owners: dict[str, asyncio.Task] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self._cleanup_locks: dict[str, asyncio.Lock] = {}

    def get_mcp_server_status(self, mcp_server_name: str) -> MCPServerStatus:
        return self.mcp_servers_status.get(mcp_server_name, MCPServerStatus.DISCONNECTED)

    def update_mcp_server_status(self, mcp_server_name, status: MCPServerStatus):
        self.mcp_servers_status[mcp_server_name] = status

    async def _bounded(self, operation, timeout):
        if self.cancellation is not None and self.cancellation.cancelled:
            if inspect.iscoroutine(operation):
                operation.close()
            self.cancellation.raise_if_cancelled()
        bounded = asyncio.wait_for(operation, timeout)
        # Timeout belongs to the request; global cancellation stays distinguishable.
        return await self.cancellation.run(bounded) if self.cancellation is not None else await bounded

    async def connect_and_discover(
        self,
        mcp_server_name: str,
        mcp_server_config: MCPServerConfig,
        mcp_tools_container: list,
        model_provider,
    ):
        if mcp_server_name in self._owners:
            raise ValueError(f"MCP server is already managed: {mcp_server_name}")
        timeout = float(mcp_server_config.timeout or 60)
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        self.timeout_seconds = timeout
        self.server_timeouts[mcp_server_name] = timeout
        ready = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        self._stops[mcp_server_name] = stop

        async def own_connection():
            # AnyIO transport/session scopes must enter and exit in the same task.
            stack = AsyncExitStack()
            self.exit_stacks[mcp_server_name] = stack
            try:
                async with stack:
                    if mcp_server_config.http_url:
                        from mcp.client.streamable_http import streamablehttp_client
                        context = streamablehttp_client(mcp_server_config.http_url,
                            headers=mcp_server_config.headers, timeout=timeout)
                    elif mcp_server_config.url:
                        raise NotImplementedError("WebSocket transport is not implemented yet")
                    elif mcp_server_config.command:
                        context = stdio_client(StdioServerParameters(
                            command=mcp_server_config.command, args=mcp_server_config.args or [],
                            env=mcp_server_config.env, cwd=mcp_server_config.cwd))
                    else:
                        raise ValueError(f"Invalid MCP server configuration for {mcp_server_name}")
                    transport = await stack.enter_async_context(context)
                    await self.connect_to_server(mcp_server_name, transport)
                    discovered = await self.list_tools(mcp_server_name)
                    if not ready.done():
                        ready.set_result(discovered)
                    await stop.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
            finally:
                self.exit_stacks.pop(mcp_server_name, None)
                session = self.sessions.pop(mcp_server_name, None)
                if self.session is session:
                    self.session = next(iter(self.sessions.values()), None)
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)

        self._owners[mcp_server_name] = asyncio.create_task(own_connection(), name=f"mcp:{mcp_server_name}")
        try:
            operation = asyncio.shield(ready)
            discovered = await self._bounded(operation, timeout)
            for tool in discovered.tools:
                mcp_tools_container.append(MCPTool(self, tool, model_provider, server_name=mcp_server_name))
        except BaseException:
            ready.cancel()
            await self.cleanup(mcp_server_name)
            raise

    async def connect_to_server(self, mcp_server_name, transport):
        """Connect to an MCP server

        Args:
            server_params: Parameters for connecting to the MCP server.
        """
        if self.get_mcp_server_status(mcp_server_name) != MCPServerStatus.CONNECTED:
            self.update_mcp_server_status(mcp_server_name, MCPServerStatus.CONNECTING)
            try:
                stdio, write = transport[:2]
                stack = self.exit_stacks.get(mcp_server_name, self.exit_stack)
                session = await stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                operation = session.initialize()
                await self._bounded(operation, self.server_timeouts.get(mcp_server_name, 60.0))
                self.sessions[mcp_server_name] = session
                self.session = session  # compatibility for callers using a single server
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.CONNECTED)
            except BaseException as e:
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)
                raise e

    async def call_tool(self, name, args, server_name: str | None = None):
        session = self.sessions.get(server_name) if server_name else self.session
        if session is None:
            raise RuntimeError("MCP session is not connected")
        operation = session.call_tool(name, args)
        try:
            return await self._bounded(
                operation,
                self.server_timeouts.get(server_name or "", self.timeout_seconds),
            )
        except (TimeoutError, asyncio.CancelledError):
            if server_name and asyncio.current_task() is not self._owners.get(server_name):
                await self.cleanup(server_name)
            raise

    async def list_tools(self, server_name: str | None = None):
        session = self.sessions.get(server_name) if server_name else self.session
        if session is None:
            raise RuntimeError("MCP session is not connected")
        operation = session.list_tools()
        try:
            return await self._bounded(
                operation,
                self.server_timeouts.get(server_name or "", self.timeout_seconds),
            )
        except (TimeoutError, asyncio.CancelledError):
            if server_name and asyncio.current_task() is not self._owners.get(server_name):
                await self.cleanup(server_name)
            raise

    async def cleanup(self, mcp_server_name):
        """Clean up resources"""
        async with self._cleanup_locks.setdefault(mcp_server_name, asyncio.Lock()):
            try:
                await self._cleanup_server(mcp_server_name)
            finally:
                self._owners.pop(mcp_server_name, None)
                self._stops.pop(mcp_server_name, None)
                self.server_timeouts.pop(mcp_server_name, None)
                self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)

    async def _cleanup_server(self, mcp_server_name):
        owner = self._owners.get(mcp_server_name)
        if owner is not None:
            self._stops[mcp_server_name].set()
            if self.get_mcp_server_status(mcp_server_name) != MCPServerStatus.CONNECTED:
                owner.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(owner), timeout=10.0)
            except TimeoutError:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            except asyncio.CancelledError:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
                if asyncio.current_task().cancelling():
                    raise
        else:
            stack = self.exit_stacks.pop(mcp_server_name, None)
            if stack is not None:
                await stack.aclose()
            elif not self.exit_stacks:
                await self.exit_stack.aclose()
        session = self.sessions.pop(mcp_server_name, None)
        if self.session is session:
            self.session = next(iter(self.sessions.values()), None)
        self.server_timeouts.pop(mcp_server_name, None)
        self.update_mcp_server_status(mcp_server_name, MCPServerStatus.DISCONNECTED)

    async def cleanup_all(self) -> None:
        for server_name in tuple(dict.fromkeys((*self._owners, *self.exit_stacks))):
            await self.cleanup(server_name)
