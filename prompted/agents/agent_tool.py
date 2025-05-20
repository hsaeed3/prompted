"""
💭 prompted.agent_tool

Contains the implementation of tools that can be used with chat completions.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Concatenate, ParamSpec, TypedDict

from pydantic import BaseModel, ValidationError

from ..types.chat_completions import Function, Tool, FunctionParameters

# Type variables for generic function types
ToolParams = ParamSpec("ToolParams")
ToolResult = TypeVar("ToolResult")

# Type aliases for tool functions
ToolFunctionWithoutContext = Callable[ToolParams, Any]
ToolFunctionWithContext = Callable[Concatenate[Any, ToolParams], Any]
ToolFunction = Union[
    ToolFunctionWithoutContext[ToolParams],
    ToolFunctionWithContext[ToolParams],
]

# Type alias for error handling function
ToolErrorFunction = Callable[[Any, Exception], Union[str, Awaitable[str]]]


@dataclass
class AgentTool:
    """A tool that can be used with chat completions.

    This class wraps a function and provides the necessary metadata and schema
    for the function to be used as a tool in chat completions.
    """

    name: str
    """The name of the tool, as shown to the LLM."""

    description: str
    """A description of the tool, as shown to the LLM."""

    parameters: FunctionParameters
    """The JSON schema for the tool's parameters."""

    function: ToolFunction
    """The function that implements the tool's behavior."""

    strict_schema: bool = True
    """Whether to enforce strict schema validation."""

    error_handler: Optional[ToolErrorFunction] = None
    """Optional function to handle errors during tool execution."""

    def to_tool(self) -> Tool:
        """Convert this tool to the format expected by chat completions."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict_schema,
            },
        }

    async def execute(self, context: Any, args: str) -> Any:
        """Execute the tool with the given context and arguments.

        Args:
            context: The context object to pass to the function if it accepts one.
            args: The arguments as a JSON string.

        Returns:
            The result of executing the tool.

        Raises:
            ValidationError: If the arguments don't match the schema.
            Exception: If the tool execution fails.
        """
        try:
            # Parse and validate arguments
            json_data = json.loads(args) if args else {}
            if self.strict_schema:
                # TODO: Implement strict schema validation
                pass

            # Execute the function
            if inspect.iscoroutinefunction(self.function):
                if self._takes_context():
                    result = await self.function(context, **json_data)
                else:
                    result = await self.function(**json_data)
            else:
                if self._takes_context():
                    result = self.function(context, **json_data)
                else:
                    result = self.function(**json_data)

            return result

        except Exception as e:
            if self.error_handler:
                error_result = self.error_handler(context, e)
                if inspect.isawaitable(error_result):
                    return await error_result
                return error_result
            raise

    def _takes_context(self) -> bool:
        """Check if the function takes a context parameter."""
        sig = inspect.signature(self.function)
        return len(sig.parameters) > 0 and list(sig.parameters.keys())[0] == "context"


def default_error_handler(context: Any, error: Exception) -> str:
    """Default error handler that returns a generic error message."""
    return f"An error occurred while running the tool: {str(error)}"


@overload
def agent_tool(
    func: ToolFunction[...],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict_schema: bool = True,
    error_handler: Optional[ToolErrorFunction] = None,
) -> AgentTool:
    """Overload for usage as @agent_tool (no parentheses)."""
    ...


@overload
def agent_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict_schema: bool = True,
    error_handler: Optional[ToolErrorFunction] = None,
) -> Callable[[ToolFunction[...]], AgentTool]:
    """Overload for usage as @agent_tool(...)."""
    ...


def agent_tool(
    func: Optional[ToolFunction[...]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strict_schema: bool = True,
    error_handler: Optional[ToolErrorFunction] = default_error_handler,
) -> Union[AgentTool, Callable[[ToolFunction[...]], AgentTool]]:
    """Decorator to create an AgentTool from a function.

    Args:
        func: The function to wrap.
        name: Optional name override for the tool.
        description: Optional description override for the tool.
        strict_schema: Whether to enforce strict schema validation.
        error_handler: Optional function to handle errors during tool execution.

    Returns:
        An AgentTool instance or a decorator function.
    """

    def _create_tool(the_func: ToolFunction[...]) -> AgentTool:
        # Get function metadata
        func_name = name or the_func.__name__
        func_description = description or (the_func.__doc__ or "").strip()

        # TODO: Implement schema generation from function signature and docstring
        # For now, use a basic schema
        schema: FunctionParameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        return AgentTool(
            name=func_name,
            description=func_description,
            parameters=schema,
            function=the_func,
            strict_schema=strict_schema,
            error_handler=error_handler,
        )

    if func is not None:
        return _create_tool(func)

    def decorator(real_func: ToolFunction[...]) -> AgentTool:
        return _create_tool(real_func)

    return decorator


__all__ = [
    "AgentTool",
    "agent_tool",
    "default_error_handler",
]
