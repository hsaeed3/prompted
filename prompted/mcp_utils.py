import functools
import json
from typing import TYPE_CHECKING, Any, List

from .agents.agent_tool import AgentTool, default_error_handler
from .types.mcp import (
    CallToolResult,
    UserError,
)  # MCP UserError, CallToolResult
from .mcp import MCPServer  # The MCPServer base class from mcp.py

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .types.mcp import Tool as MCPTool  # MCP Tool type


# Define prompted-specific exceptions or use generic ones
class PromptedUserError(UserError):  # Inherit from MCP's UserError or a common base
    pass


class PromptedAgentsException(Exception):
    pass


class PromptedModelBehaviorError(Exception):
    pass


class MCPUtil:
    """Set of utilities for interop between MCP and prompted AgentTools."""

    @classmethod
    async def get_all_agent_tools_from_servers(
        cls, servers: List[MCPServer]
    ) -> List[AgentTool]:
        """Get all AgentTools from a list of MCP servers."""
        all_tools: List[AgentTool] = []
        tool_names: set[str] = set()
        for server in servers:
            try:
                # Ensure server is connected before listing tools
                # Connection management should ideally be handled by the Agent's run lifecycle
                # For now, let's assume connect/cleanup is managed outside this direct utility
                # if not server.session:
                #     await server.connect()

                server_tools = await cls.get_agent_tools_from_server(server)
                current_server_tool_names = {tool.name for tool in server_tools}

                overlapping_names = current_server_tool_names & tool_names
                if overlapping_names:
                    # Correctly formatted f-string for the error message
                    error_message = f"Duplicate tool names found across MCP servers: {overlapping_names}"
                    raise PromptedUserError(error_message)

                tool_names.update(current_server_tool_names)
                all_tools.extend(server_tools)
            except (
                Exception
            ) as e:  # Catch exceptions during tool processing for a specific server
                logger.error(
                    f"Error getting tools from MCP server '{server.name}': {e}"
                )
                # Optionally re-raise or handle if a server failing shouldn't stop all tool loading
        return all_tools

    @classmethod
    async def get_agent_tools_from_server(cls, server: MCPServer) -> List[AgentTool]:
        """Get all AgentTools from a single MCP server."""
        try:
            mcp_tools = await server.list_tools()
            logger.debug(
                f"Fetched {len(mcp_tools)} tools from MCP server: {server.name}"
            )
        except Exception as e:
            logger.error(f"Failed to list tools from MCP server '{server.name}': {e}")
            return []  # Return empty list if server communication fails

        return [cls.to_agent_tool(mcp_tool, server) for mcp_tool in mcp_tools]

    @classmethod
    def to_agent_tool(cls, mcp_tool: "MCPTool", server: MCPServer) -> AgentTool:
        """Convert an MCP tool to a prompted AgentTool."""

        # The AgentTool's 'function' will be a partial that calls invoke_mcp_tool
        # It needs to match the signature AgentTool.execute expects for its function attribute
        # which means it might implicitly receive 'context' and 'args' (as JSON string)
        # or just **kwargs if AgentTool's execute method parses and passes them.
        # For AgentTool, the function it stores is typically called with (context, **kwargs)
        # or just (**kwargs).
        # Let's make invoke_mcp_tool a suitable callable for AgentTool's function field.

        async def invoke_func_for_agent_tool(
            context: Any, **kwargs_from_agent_tool_execute: Any
        ) -> str:
            # AgentTool.execute will parse the JSON string args from the LLM
            # and pass them as **kwargs_from_agent_tool_execute.
            # We need to repack these into a JSON string for the original invoke_mcp_tool if it expects that,
            # or pass them directly if invoke_mcp_tool can take them as a dict.
            # The original openai-agents invoke_mcp_tool takes 'input_json: str'.
            # So we re-serialize kwargs.
            input_json_str = json.dumps(kwargs_from_agent_tool_execute)
            return await cls.invoke_mcp_tool(server, mcp_tool, context, input_json_str)

        # Ensure schema has 'properties' if it's missing, as AgentTool might expect it
        # or it's good practice for JSON schema.
        schema = mcp_tool.inputSchema
        if not isinstance(schema, dict):  # Should be a dict
            logger.warning(
                f"MCPTool '{mcp_tool.name}' has non-dict schema: {schema}. Using empty schema."
            )
            schema = {"type": "object", "properties": {}}
        elif "properties" not in schema:
            schema["properties"] = {}

        # The 'strict_schema' for AgentTool is about how it handles input from the LLM.
        # The 'strict' field in MCPTool.function is about how the MCP server itself validates.
        # We can pass the MCP's strictness to AgentTool.
        is_mcp_tool_strict = (
            mcp_tool.model_extra.get("strict", False) if mcp_tool.model_extra else False
        )

        return AgentTool(
            name=mcp_tool.name,
            description=mcp_tool.description
            or f"MCP tool '{mcp_tool.name}' from server '{server.name}'.",
            parameters=schema,  # This should be FunctionParameters type for AgentTool
            function=invoke_func_for_agent_tool,  # This will be called by AgentTool.execute
            strict_schema=is_mcp_tool_strict,  # Use MCP tool's strictness
            error_handler=default_error_handler,  # Or a custom MCP-aware error handler
        )

    @classmethod
    async def invoke_mcp_tool(
        cls,
        server: MCPServer,
        mcp_tool_definition: "MCPTool",
        context: Any,
        input_json: str,
    ) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            # input_json comes from invoke_func_for_agent_tool, already a JSON string
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except json.JSONDecodeError as e:
            logger.debug(
                f"Invalid JSON input for MCP tool {mcp_tool_definition.name}: {input_json}"
            )
            raise PromptedModelBehaviorError(
                f"Invalid JSON input for tool {mcp_tool_definition.name}: {input_json}"
            ) from e

        logger.debug(
            f"Invoking MCP tool {mcp_tool_definition.name} on server '{server.name}' with input: {json.dumps(json_data)}"
        )

        try:
            # Ensure server is connected before calling a tool
            # Again, lifecycle management is ideally external to this specific call
            # if not server.session:
            #     logger.warning(f"Attempting to call tool on disconnected MCP server '{server.name}'. Trying to connect.")
            #     await server.connect() # This might be too late or problematic here.

            result: CallToolResult = await server.call_tool(
                mcp_tool_definition.name, json_data
            )
        except Exception as e:
            logger.error(
                f"Error invoking MCP tool {mcp_tool_definition.name} on server '{server.name}': {e}"
            )
            # Consider how to propagate this error. The AgentTool's error_handler will catch it.
            raise PromptedAgentsException(
                f"Error invoking MCP tool {mcp_tool_definition.name} on server '{server.name}': {e}"
            ) from e

        logger.debug(f"MCP tool {mcp_tool_definition.name} returned {result}")

        # Convert MCP tool result (list of content items) to a single string for AgentTool
        if result.content:
            if len(result.content) == 1:
                # Assuming content items are Pydantic models, use model_dump_json
                tool_output = result.content[0].model_dump_json()
            else:
                tool_output = json.dumps([item.model_dump() for item in result.content])
        else:  # No content or error
            tool_output = (
                result.model_dump_json()
            )  # Send the whole result object if content is empty or it was an error
            if (
                not result.isError
            ):  # No content, but not an error, means empty successful result
                tool_output = "Tool executed successfully with no output."
            else:  # isError is true
                logger.warning(
                    f"MCP tool '{mcp_tool_definition.name}' on server '{server.name}' indicated an error with empty content. Result: {result.model_dump_json()}"
                )
                tool_output = f"Tool '{mcp_tool_definition.name}' execution resulted in an error. Details: {result.model_dump_json()}"

        # Tracing/logging of output would go here if adapted
        return tool_output
