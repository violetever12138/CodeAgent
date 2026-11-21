# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Google Gemini API client wrapper with tool integration."""

import json
import os
import traceback
import uuid
from typing import override

from google import genai
from google.genai import types

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import async_retry_with, retry_with
from trae_agent.utils.request_observer import google_config, request, sdk_client


class GoogleClient(BaseLLMClient):
    """Google Gemini client wrapper with tool schema generation."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("Google API key is required")
        self.client = genai.Client(api_key=self.api_key)
        self.message_history: list[types.Content] = []
        self.system_instruction: str | None = None

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history, self.system_instruction = self.parse_messages(messages)

    @override
    def supports_tool_calling(self, model_config: ModelConfig) -> bool:
        """Gemini's current model families support function calling."""
        return "gemini" in model_config.model.lower()

    def _create_google_response(
        self,
        model_config: ModelConfig,
        current_chat_contents: list[types.Content],
        generation_config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Create a response using Google Gemini API. This method will be decorated with retry logic."""
        parameters = {
            'model': model_config.model,
            'contents': current_chat_contents,
            'config': google_config(generation_config),
        }
        with request("google", parameters) as observed:
            response = sdk_client(self.client).models.generate_content(**parameters)
            observed.response(response)
            return response

    async def _create_google_response_async(
        self,
        model_config: ModelConfig,
        current_chat_contents: list[types.Content],
        generation_config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        parameters = {
            'model': model_config.model,
            'contents': current_chat_contents,
            'config': google_config(generation_config),
        }
        with request("google", parameters) as observed:
            response = await sdk_client(self.client).aio.models.generate_content(**parameters)
            observed.response(response)
            return response

    def _prepare_request(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None,
        reuse_history: bool,
    ) -> tuple[list[types.Content], types.GenerateContentConfig, str | None, list[types.Content]]:
        newly_parsed_messages, system_instruction = self.parse_messages(messages)
        current_system_instruction = system_instruction or self.system_instruction
        current_chat_contents = (
            self.message_history + newly_parsed_messages
            if reuse_history
            else newly_parsed_messages
        )
        generation_config = types.GenerateContentConfig(
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            max_output_tokens=model_config.max_tokens,
            candidate_count=model_config.candidate_count,
            stop_sequences=model_config.stop_sequences,
            system_instruction=current_system_instruction,
        )
        if tools:
            generation_config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            **({"parameters_json_schema": tool.get_input_schema()}
                               if getattr(tool, "strict_schema", True) is False
                               else {"parameters": tool.get_input_schema()}),
                        )
                    ]
                )
                for tool in tools
            ]
        return (
            current_chat_contents,
            generation_config,
            current_system_instruction,
            newly_parsed_messages,
        )

    def _consume_response(
        self,
        response: types.GenerateContentResponse,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None,
        reuse_history: bool,
        newly_parsed_messages: list[types.Content],
        current_system_instruction: str | None,
    ) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCall] = []
        assistant_response_content = None
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                assistant_response_content = candidate.content
                for part in candidate.content.parts:
                    if part.text:
                        content += part.text
                    elif part.function_call:
                        tool_calls.append(
                            ToolCall(
                                call_id=str(uuid.uuid4()),
                                name=part.function_call.name or "tool",
                                arguments=dict(part.function_call.args)
                                if part.function_call.args
                                else {},
                            )
                        )
        new_history = (
            self.message_history + newly_parsed_messages
            if reuse_history
            else newly_parsed_messages
        )
        if assistant_response_content:
            new_history.append(assistant_response_content)
        self.message_history = new_history
        if current_system_instruction:
            self.system_instruction = current_system_instruction
        usage = None
        if response.usage_metadata:
            usage = LLMUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                cache_read_input_tokens=response.usage_metadata.cached_content_token_count or 0,
                cache_creation_input_tokens=0,
            )
        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=model_config.model,
            finish_reason=str(
                response.candidates[0].finish_reason.name
                if response.candidates[0].finish_reason
                else "unknown"
            )
            if response.candidates
            else "UNKNOWN",
            tool_calls=tool_calls or None,
        )
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="google",
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
        """Send chat messages to Gemini with optional tool support."""
        (
            current_chat_contents,
            generation_config,
            current_system_instruction,
            newly_parsed_messages,
        ) = self._prepare_request(messages, model_config, tools, reuse_history)

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_google_response,
            provider_name="Google Gemini",
            max_retries=model_config.max_retries,
        )
        response = retry_decorator(model_config, current_chat_contents, generation_config)

        return self._consume_response(
            response,
            messages,
            model_config,
            tools,
            reuse_history,
            newly_parsed_messages,
            current_system_instruction,
        )

    @override
    async def achat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        (
            current_chat_contents,
            generation_config,
            current_system_instruction,
            newly_parsed_messages,
        ) = self._prepare_request(messages, model_config, tools, reuse_history)
        request = async_retry_with(
            self._create_google_response_async,
            provider_name="Google Gemini",
            max_retries=model_config.max_retries,
        )
        response = await request(model_config, current_chat_contents, generation_config)
        return self._consume_response(
            response,
            messages,
            model_config,
            tools,
            reuse_history,
            newly_parsed_messages,
            current_system_instruction,
        )

    async def aclose(self) -> None:
        await self.client.aio.aclose()
        self.client.close()

    def parse_messages(self, messages: list[LLMMessage]) -> tuple[list[types.Content], str | None]:
        """Parse the messages to Gemini format, separating system instructions."""
        gemini_messages: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue
            elif msg.tool_result:
                gemini_messages.append(
                    types.Content(
                        role="tool",
                        parts=[self.parse_tool_call_result(msg.tool_result)],
                    )
                )
            elif msg.tool_call:
                gemini_messages.append(
                    types.Content(role="model", parts=[self.parse_tool_call(msg.tool_call)])
                )
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append(
                    types.Content(role=role, parts=[types.Part(text=msg.content or "")])
                )

        return gemini_messages, system_instruction

    def parse_tool_call(self, tool_call: ToolCall) -> types.Part:
        """Parse a ToolCall into a Gemini FunctionCall Part for history."""
        return types.Part.from_function_call(name=tool_call.name, args=tool_call.arguments)

    def parse_tool_call_result(self, tool_result: ToolResult) -> types.Part:
        """Parse a ToolResult into a Gemini FunctionResponse Part for history."""
        result_content: dict[str, str] = {}
        if tool_result.result is not None:
            try:
                json.dumps(tool_result.result)
                result_content["result"] = tool_result.result
            except (TypeError, OverflowError) as e:
                tb = traceback.format_exc()
                serialization_error = f"JSON serialization failed for tool result: {e}\n{tb}"
                if tool_result.error:
                    result_content["error"] = f"{tool_result.error}\n\n{serialization_error}"
                else:
                    result_content["error"] = serialization_error
                result_content["result"] = str(tool_result.result)

        if tool_result.error and "error" not in result_content:
            result_content["error"] = tool_result.error

        if not result_content:
            result_content["status"] = "Tool executed successfully but returned no output."

        if not hasattr(tool_result, "name") or not tool_result.name:
            raise AttributeError(
                "ToolResult must have a 'name' attribute matching the function that was called."
            )
        return types.Part.from_function_response(name=tool_result.name, response=result_content)
