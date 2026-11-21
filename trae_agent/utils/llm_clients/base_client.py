# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


import asyncio
from abc import ABC, abstractmethod

from trae_agent.tools.base import Tool
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model_config: ModelConfig):
        self.api_key: str = model_config.model_provider.api_key
        self.base_url: str | None = model_config.model_provider.base_url
        self.api_version: str | None = model_config.model_provider.api_version
        self.trajectory_recorder: TrajectoryRecorder | None = None  # TrajectoryRecorder instance

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Set the trajectory recorder for this client."""
        self.trajectory_recorder = recorder

    @abstractmethod
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM."""
        pass

    async def achat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Run a provider request without blocking the agent event loop.

        Native async providers can override this method.  The compatibility path
        closes its transport when cancelled so the connection is not reused.
        """

        try:
            return await asyncio.to_thread(
                self.chat, messages, model_config, tools, reuse_history
            )
        except asyncio.CancelledError:
            self.cancel_active_request()
            raise

    def cancel_active_request(self) -> None:
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def supports_tool_calling(self, model_config: ModelConfig) -> bool:
        """Check if the current model supports tool calling."""
        return model_config.supports_tool_calling
