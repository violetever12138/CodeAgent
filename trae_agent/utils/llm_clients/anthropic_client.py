# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Anthropic API client wrapper with tool integration."""

import json
from typing import override

import anthropic
from anthropic.types.tool_union_param import TextEditor20250429

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import async_retry_with, retry_with
from trae_agent.utils.request_observer import request, sdk_client


class AnthropicClient(BaseLLMClient):
    """Anthropic client wrapper with tool schema generation."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.client: anthropic.Anthropic = anthropic.Anthropic(
            api_key=self.api_key, base_url=self.base_url
        )
        self.async_client: anthropic.AsyncAnthropic | None = None
        self.message_history: list[anthropic.types.MessageParam] = []
        self.system_message: str | anthropic.NotGiven = anthropic.NOT_GIVEN

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_anthropic_response(
        self,
        model_config: ModelConfig,
        tool_schemas: list[anthropic.types.ToolUnionParam] | anthropic.NotGiven,
    ) -> anthropic.types.Message:
        """Create a response using Anthropic API. This method will be decorated with retry logic."""
        parameters = {
            'model': model_config.model,
            'messages': self.message_history,
            'max_tokens': model_config.max_tokens,
            'system': self.system_message,
            'tools': tool_schemas,
            'temperature': model_config.temperature,
            'top_p': model_config.top_p,
            'top_k': model_config.top_k,
        }
        with request("anthropic", parameters) as observed:
            response = sdk_client(self.client).messages.create(**parameters)
            observed.response(response)
            return response

    async def _create_anthropic_response_async(
        self,
        model_config: ModelConfig,
        tool_schemas: list[anthropic.types.ToolUnionParam] | anthropic.NotGiven,
    ) -> anthropic.types.Message:
        if self.async_client is None:
            self.async_client = anthropic.AsyncAnthropic(
                api_key=self.api_key, base_url=self.base_url
            )
        parameters = {
            'model': model_config.model,
            'messages': self.message_history,
            'max_tokens': model_config.max_tokens,
            'system': self.system_message,
            'tools': tool_schemas,
            'temperature': model_config.temperature,
            'top_p': model_config.top_p,
            'top_k': model_config.top_k,
        }
        with request("anthropic", parameters) as observed:
            response = await sdk_client(self.async_client).messages.create(**parameters)
            observed.response(response)
            return response

    def _prepare_request(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None,
        reuse_history: bool,
    ) -> list[anthropic.types.ToolUnionParam] | anthropic.NotGiven:
        anthropic_messages = self.parse_messages(messages)
        self.message_history = (
            self.message_history + anthropic_messages if reuse_history else anthropic_messages
        )
        tool_schemas: list[anthropic.types.ToolUnionParam] | anthropic.NotGiven = (
            anthropic.NOT_GIVEN
        )
        if tools:
            tool_schemas = []
            for tool in tools:
                if tool.name == "str_replace_based_edit_tool":
                    tool_schemas.append(
                        TextEditor20250429(
                            name="str_replace_based_edit_tool", type="text_editor_20250429"
                        )
                    )
                elif tool.name == "bash":
                    tool_schemas.append(
                        anthropic.types.ToolBash20250124Param(
                            name="bash", type="bash_20250124"
                        )
                    )
                else:
                    tool_schemas.append(
                        anthropic.types.ToolParam(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.get_input_schema(),
                        )
                    )
        return tool_schemas

    def _consume_response(
        self,
        response: anthropic.types.Message,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None,
    ) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCall] = []
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
                self.message_history.append(
                    anthropic.types.MessageParam(role="assistant", content=content_block.text)
                )
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input,  # pyright: ignore[reportArgumentType]
                    )
                )
                self.message_history.append(
                    anthropic.types.MessageParam(role="assistant", content=[content_block])
                )
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls or None,
        )
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="anthropic",
                model=model_config.model,
                tools=tools,
            )
        return llm_response

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to Anthropic with optional tool support."""
        tool_schemas = self._prepare_request(messages, tools, reuse_history)

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_anthropic_response,
            provider_name="Anthropic",
            max_retries=model_config.max_retries,
        )
        response = retry_decorator(model_config, tool_schemas)

        return self._consume_response(response, messages, model_config, tools)

    @override
    async def achat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        tool_schemas = self._prepare_request(messages, tools, reuse_history)
        request = async_retry_with(
            self._create_anthropic_response_async,
            provider_name="Anthropic",
            max_retries=model_config.max_retries,
        )
        response = await request(model_config, tool_schemas)
        return self._consume_response(response, messages, model_config, tools)

    async def aclose(self) -> None:
        if self.async_client is not None:
            await self.async_client.close()
        await super().aclose()

    def parse_messages(self, messages: list[LLMMessage]) -> list[anthropic.types.MessageParam]:
        """Parse the messages to Anthropic format."""
        anthropic_messages: list[anthropic.types.MessageParam] = []
        for msg in messages:
            if msg.role == "system":
                self.system_message = msg.content if msg.content else anthropic.NOT_GIVEN
            elif msg.tool_result:
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="user",
                        content=[self.parse_tool_call_result(msg.tool_result)],
                    )
                )
            elif msg.tool_call:
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="assistant", content=[self.parse_tool_call(msg.tool_call)]
                    )
                )
            else:
                if msg.role == "user":
                    role = "user"
                elif msg.role == "assistant":
                    role = "assistant"
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")

                if not msg.content:
                    raise ValueError("Message content is required")

                anthropic_messages.append(
                    anthropic.types.MessageParam(role=role, content=msg.content)
                )
        return anthropic_messages

    def parse_tool_call(self, tool_call: ToolCall) -> anthropic.types.ToolUseBlockParam:
        """Parse the tool call from the LLM response."""
        return anthropic.types.ToolUseBlockParam(
            type="tool_use",
            id=tool_call.call_id,
            name=tool_call.name,
            input=json.dumps(tool_call.arguments),
        )

    def parse_tool_call_result(
        self, tool_call_result: ToolResult
    ) -> anthropic.types.ToolResultBlockParam:
        """Parse the tool call result from the LLM response."""
        result: str = ""
        if tool_call_result.result:
            result = result + tool_call_result.result + "\n"
        if tool_call_result.error:
            result += "Tool call failed with error:\n"
            result += tool_call_result.error
        result = result.strip()

        # Provide a default error message if the tool failed but didn't provide details
        if not tool_call_result.success and not result:
            result = "Tool execution failed without providing error details."

        return anthropic.types.ToolResultBlockParam(
            tool_use_id=tool_call_result.call_id,
            type="tool_result",
            content=result,
            is_error=not tool_call_result.success,
        )
