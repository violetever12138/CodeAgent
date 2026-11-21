import copy
import json
from typing import override

import mcp
from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class MCPTool(Tool):
    strict_schema = False  # Preserve external schemas rather than rewriting their semantics.
    def __init__(
        self, client, tool: mcp.types.Tool, model_provider: str | None = None,
        *, server_name: str = "mcp", policy_effect: str = "read",
    ):
        super().__init__(model_provider)
        self.client = client
        self.tool = tool
        self.server_name = server_name
        self.policy_effect = policy_effect

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return f"mcp__{self.server_name}__{self.tool.name}"

    @override
    def get_description(self) -> str:
        return self.tool.description or ""

    def get_input_schema(self):
        return copy.deepcopy(self.tool.inputSchema)

    def validate_arguments(self, arguments):
        try:
            Draft202012Validator.check_schema(self.tool.inputSchema)
            Draft202012Validator(self.tool.inputSchema).validate(arguments)
        except (SchemaError, ValidationError) as exc:
            return f"MCP input schema: {exc.message}"
        return None

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        def properties_to_parameter():
            parameters = []
            inputSchema = self.tool.inputSchema
            required = inputSchema.get("required", [])
            properties = inputSchema.get("properties", {})
            for name, prop in properties.items():
                tool_para = ToolParameter(
                    name=name,
                    type=prop.get("type", ["string", "number", "boolean", "object", "array", "null"]),
                    items=prop.get("items", None),
                    description=prop.get("description", ""),
                    enum=prop.get("enum"),
                    required=name in required,
                )
                parameters.append(tool_para)
            return parameters

        return properties_to_parameter()

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            error = self.validate_arguments(arguments)
            if error:
                return ToolExecResult(error=error, error_code=-1)
            output = await self.client.call_tool(
                self.tool.name, arguments, server_name=self.server_name
            )
            blocks = output.content or []
            structured = getattr(output, "structuredContent", None)
            if not isinstance(structured, dict):
                structured = None
            if structured is None and all(isinstance(getattr(block, "text", None), str) for block in blocks):
                rendered = "\n\n".join(block.text for block in blocks)
            else:
                rendered = json.dumps({
                    "content": [block.model_dump(mode="json", by_alias=True, exclude_none=True) for block in blocks],
                    "structuredContent": structured,
                }, ensure_ascii=False)
            if output.isError:
                return ToolExecResult(error=rendered or "MCP tool reported an error without content", error_code=-1)
            return ToolExecResult(output=rendered)

        except Exception as e:
            return ToolExecResult(error=f"Error running mcp tool: {e}", error_code=-1)
