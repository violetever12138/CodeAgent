# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import json
from typing import override

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import async_retry_with, retry_with
from trae_agent.utils.request_observer import request, sdk_client


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client: openai.AsyncOpenAI | None = None
        self.message_history: ResponseInputParam = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_openai_response(
        self,
        api_call_input: ResponseInputParam,
        model_config: ModelConfig,
        tool_schemas: list[ToolParam] | None,
    ) -> Response:
        """Create a response using OpenAI API. This method will be decorated with retry logic."""
        parameters = {
            'input': api_call_input,
            'model': model_config.model,
            'tools': tool_schemas if tool_schemas else openai.NOT_GIVEN,
            'temperature': model_config.temperature
            if "o3" not in model_config.model
            and "o4-mini" not in model_config.model
            and "gpt-5" not in model_config.model
            else openai.NOT_GIVEN,
            'top_p': model_config.top_p,
            'max_output_tokens': model_config.max_tokens,
        }
        with request("openai", parameters) as observed:
            response = sdk_client(self.client).responses.create(**parameters)
            observed.response(response)
            return response

    async def _create_openai_response_async(
        self,
        api_call_input: ResponseInputParam,
        model_config: ModelConfig,
        tool_schemas: list[ToolParam] | None,
    ) -> Response:
        if self.async_client is None:
            self.async_client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url
            )
        parameters = {
            'input': api_call_input,
            'model': model_config.model,
            'tools': tool_schemas if tool_schemas else openai.NOT_GIVEN,
            'temperature': model_config.temperature
            if "o3" not in model_config.model
            and "o4-mini" not in model_config.model
            and "gpt-5" not in model_config.model
            else openai.NOT_GIVEN,
            'top_p': model_config.top_p,
            'max_output_tokens': model_config.max_tokens,
        }
        with request("openai", parameters) as observed:
            response = await sdk_client(self.async_client).responses.create(**parameters)
            observed.response(response)
            return response

    def _prepare_request(
        self,
        messages: list[LLMMessage],
        tools: list[Tool] | None,
        reuse_history: bool,
    ) -> tuple[ResponseInputParam, list[ToolParam] | None]:
        openai_messages = self.parse_messages(messages)
        self.message_history = (
            self.message_history + openai_messages if reuse_history else openai_messages
        )
        tool_schemas = None
        if tools:
            tool_schemas = [
                FunctionToolParam(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.get_input_schema(),
                    strict=getattr(tool, "strict_schema", True),
                    type="function",
                )
                for tool in tools
            ]
        return self.message_history, tool_schemas

    def _consume_response(
        self,
        response: Response,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None,
    ) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCall] = []
        for output_block in response.output:
            if output_block.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        call_id=output_block.call_id,
                        name=output_block.name,
                        arguments=json.loads(output_block.arguments)
                        if output_block.arguments
                        else {},
                        id=output_block.id,
                    )
                )
                tool_call_param = ResponseFunctionToolCallParam(
                    arguments=output_block.arguments,
                    call_id=output_block.call_id,
                    name=output_block.name,
                    type="function_call",
                )
                if output_block.status:
                    tool_call_param["status"] = output_block.status
                if output_block.id:
                    tool_call_param["id"] = output_block.id
                self.message_history.append(tool_call_param)
            elif output_block.type == "message":
                content = "".join(
                    block.text for block in output_block.content if block.type == "output_text"
                )
        if content:
            self.message_history.append(
                EasyInputMessageParam(content=content, role="assistant", type="message")
            )
        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                cache_read_input_tokens=response.usage.input_tokens_details.cached_tokens or 0,
                reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens or 0,
            )
        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.status,
            tool_calls=tool_calls or None,
        )
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
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
        """Send chat messages to OpenAI with optional tool support."""
        api_call_input, tool_schemas = self._prepare_request(messages, tools, reuse_history)

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_openai_response,
            provider_name="OpenAI",
            max_retries=model_config.max_retries,
        )
        response = retry_decorator(api_call_input, model_config, tool_schemas)

        return self._consume_response(response, messages, model_config, tools)

    @override
    async def achat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        api_call_input, tool_schemas = self._prepare_request(messages, tools, reuse_history)
        request = async_retry_with(
            self._create_openai_response_async,
            provider_name="OpenAI",
            max_retries=model_config.max_retries,
        )
        response = await request(api_call_input, model_config, tool_schemas)
        return self._consume_response(response, messages, model_config, tools)

    async def aclose(self) -> None:
        if self.async_client is not None:
            await self.async_client.close()
        await super().aclose()

    def parse_messages(self, messages: list[LLMMessage]) -> ResponseInputParam:
        """Parse the messages to OpenAI format."""
        openai_messages: ResponseInputParam = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                openai_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call(self, tool_call: ToolCall) -> ResponseFunctionToolCallParam:
        """Parse the tool call from the LLM response."""
        return ResponseFunctionToolCallParam(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=json.dumps(tool_call.arguments),
            type="function_call",
        )

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> FunctionCallOutput:
        """Parse the tool call result from the LLM response to FunctionCallOutput format."""
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()

        return FunctionCallOutput(
            type="function_call_output",  # Explicitly set the type field
            call_id=tool_call_result.call_id,
            output=result_content,
        )
