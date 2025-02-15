"""
## 💭 chatspec.utils

Contains the utils and helpers within the chatspec library.
These range from helpers for instance checking response/input types,
as well as message formatting / tool conversion, etc.
"""

import logging
import docstring_parser
import hashlib
import msgspec
from cachetools import cached, TTLCache
from dataclasses import is_dataclass, fields as dataclass_fields
from inspect import signature, getdoc
from pydantic import BaseModel, Field, create_model

from typing import (
    Any,
    Union,
    List,
    Iterable,
    Literal,
    Optional,
    Dict,
    Callable,
    Type,
    Sequence,
    Set,
    get_type_hints,
)
from .types import Completion, CompletionChunk, Message, Tool

__all__ = [
    "is_completion",
    "is_stream",
    "is_message",
    "is_tool",
    "has_system_prompt",
    "has_tool_call",
    "was_tool_called",
    "run_tool",
    "create_tool_message",
    "get_tool_calls",
    "dump_stream_to_message",
    "dump_stream_to_completion",
    "parse_model_from_completion",
    "parse_model_from_stream",
    "print_stream",
    "normalize_messages",
    "normalize_system_prompt",
    "create_field_mapping",
    "extract_function_fields",
    "convert_to_pydantic_model",
    "convert_to_tools",
    "convert_to_tool",
    "create_literal_pydantic_model",
    "stream_passthrough",
    "markdownify",
]


# ------------------------------------------------------------------------------
# Configuration && Logging
#
# i was debating not keeping a logger in this lib, but i think its useful
# for debugging
logger = logging.getLogger("chatspec")
#
# cache
_chatspec_cache = TTLCache(maxsize=1000, ttl=3600)


#
# exception
class ChatSpecError(Exception):
    """
    Base exception for all errors raised by the `chatspec` library.
    """

    pass


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Helper Methods
# ------------------------------------------------------------------------------


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """
    Helper function to retrieve a value from an object either as an attribute or as a dictionary key.
    """
    try:
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default
    except Exception as e:
        logger.debug(f"Error getting value for key {key}: {e}")
        return default


def _make_hashable(obj: Any) -> str:
    """
    Helper function to make an object hashable by converting it to a stable hash string.
    Uses SHA-256 to generate a consistent hash representation of any object.
    """
    try:
        # Handle basic types first
        if isinstance(obj, (str, int, float, bool, bytes)):
            return hashlib.sha256(str(obj).encode()).hexdigest()

        if isinstance(obj, (tuple, list)):
            # Recursively handle sequences
            return hashlib.sha256(
                ",".join(_make_hashable(x) for x in obj).encode()
            ).hexdigest()
        
        if isinstance(obj, dict):
            # Sort dict items for consistent hashing
            return hashlib.sha256(
                ",".join(
                    f"{k}:{_make_hashable(v)}"
                    for k, v in sorted(obj.items())
                ).encode()
            ).hexdigest()
            
        if isinstance(obj, type):
            # Handle types (classes)
            return hashlib.sha256(
                f"{obj.__module__}.{obj.__name__}".encode()
            ).hexdigest()
            
        if callable(obj):
            # Handle functions
            return hashlib.sha256(
                f"{obj.__module__}.{obj.__name__}".encode()
            ).hexdigest()
            
        if hasattr(obj, "__dict__"):
            # Use the __dict__ for instance attributes if available
            return _make_hashable(obj.__dict__)
        
        # Fallback for any other types that can be converted to string
        return hashlib.sha256(str(obj).encode()).hexdigest()
        
    except Exception as e:
        logger.debug(f"Error making object hashable: {e}")
        # Fallback to a basic string hash
        return hashlib.sha256(str(type(obj)).encode()).hexdigest()


_TYPE_MAPPING = {
    int: ("integer", int),
    float: ("number", float),
    str: ("string", str),
    bool: ("boolean", bool),
    list: ("array", list),
    dict: ("object", dict),
    tuple: ("array", tuple),
    set: ("array", set),
    Any: ("any", Any),
}


# ------------------------------------------------------------------------------
# Streaming
#
# 'chatspec' builds in a `passthrough` functionality, which caches response chunks
# to allow for multiple uses of the same response.
# this helps for if for example:
# -- you have a method that displays a stream as soon as you get it
# -- but you want to send & display that stream somewhere else immediately
# ------------------------------------------------------------------------------


class _StreamPassthrough:
    """
    Synchronous wrapper for a streamed object wrapped by
    `.passthrough()`.

    Once iterated, all chunks are stored in .chunks, and the full
    object can be 'restreamed' as well as accessed in its entirety.
    """

    def __init__(self, stream: Any):
        self._stream = stream
        self.chunks: Iterable[CompletionChunk] = []
        self._consumed = False

    def __iter__(self):
        if not self._consumed:
            for chunk in self._stream:
                self.chunks.append(chunk)
                yield chunk
            self._consumed = True
        else:
            for chunk in self.chunks:
                yield chunk


class _AsyncStreamPassthrough:
    """
    Asynchronous wrapper for a streamed object wrapped by
    `.passthrough()`.
    """

    def __init__(self, async_stream: Any):
        self._async_stream = async_stream
        self.chunks: List[CompletionChunk] = []
        self._consumed = False

    async def __aiter__(self):
        if not self._consumed:
            async for chunk in self._async_stream:
                self.chunks.append(chunk)
                yield chunk
            self._consumed = True
        else:
            for chunk in self.chunks:
                yield chunk

    async def consume(self) -> List[CompletionChunk]:
        """
        Consume the stream and return all chunks as a list.
        """
        return list(self)


# primary passthrough method
# this is the first 'public' object defined in this script
# it is able to wrap a streamed object, and return a stream that can be
# used multiple times
def stream_passthrough(completion: Any) -> Iterable[CompletionChunk]:
    """
    Wrap a chat completion stream within a cached object that can
    be iterated and consumed over multiple times.

    Supports both synchronous and asynchronous streams.

    Args:
        completion: The chat completion stream to wrap.

    Returns:
        An iterable of completion chunks.
    """
    try:
        if hasattr(completion, "__aiter__"):
            logger.debug("Wrapping an async streamed completion")
            return _AsyncStreamPassthrough(completion)
        if hasattr(completion, "__iter__"):
            logger.debug("Wrapping a streamed completion")
            return _StreamPassthrough(completion)
        return completion
    except Exception as e:
        logger.debug(f"Error in stream_passthrough: {e}")
        return completion


# ------------------------------------------------------------------------------
# 'Core' Methods
# (instance checking & validation methods)
#
# All methods in this block are cached for performance, and are meant to
# be used as 'stdlib' style methods.
# ------------------------------------------------------------------------------


@cached(
    cache=_chatspec_cache,
    key=lambda completion: _make_hashable(completion) if completion else "",
)
def is_completion(completion: Any) -> bool:
    """
    Checks if a given object is a valid chat completion.

    Supports both standard completion objects, as well as
    streamed responses.
    """
    try:
        # Handle passthrough wrapper (sync or async)
        if hasattr(completion, "chunks"):
            return bool(completion.chunks) and any(
                _get_value(chunk, "choices") for chunk in completion.chunks
            )

        # Original logic
        choices = _get_value(completion, "choices")
        if not choices:
            return False
        first_choice = choices[0]
        return bool(
            _get_value(first_choice, "message")
            or _get_value(first_choice, "delta")
        )
    except Exception as e:
        logger.debug(f"Error checking if object is chat completion: {e}")
        return False


@cached(
    cache=_chatspec_cache,
    key=lambda completion: _make_hashable(completion) if completion else "",
)
def is_stream(completion: Any) -> bool:
    """
    Checks if the given object is a valid stream of 'chat completion'
    chunks.

    Args:
        completion: The object to check.

    Returns:
        True if the object is a valid stream, False otherwise.
    """
    try:
        # Handle passthrough wrapper (sync or async)
        if hasattr(completion, "chunks"):
            return bool(completion.chunks) and any(
                _get_value(_get_value(chunk, "choices", [{}])[0], "delta")
                for chunk in completion.chunks
            )

        # Original logic
        choices = _get_value(completion, "choices")
        if not choices:
            return False
        first_choice = choices[0]
        return bool(_get_value(first_choice, "delta"))
    except Exception as e:
        logger.debug(f"Error checking if object is stream: {e}")
        return False


@cached(
    cache=_chatspec_cache,
    key=lambda message: _make_hashable(message) if message else "",
)
def is_message(message: Any) -> bool:
    """Checks if a given object is a valid chat message."""
    try:
        if not isinstance(message, dict):
            return False
        allowed_roles = {"assistant", "user", "system", "tool", "developer"}
        role = message.get("role")
        # First check role validity
        if role not in allowed_roles:
            return False
        # Check content and tool_call_id requirements
        if role == "tool":
            return bool(message.get("content")) and bool(message.get("tool_call_id"))
        elif role == "assistant" and "tool_calls" in message:
            return True
        # For all other roles, just need content
        return message.get("content") is not None
    except Exception as e:
        logger.debug(f"Error validating message: {e}")
        return False


@cached(
    cache=_chatspec_cache,
    key=lambda tool: _make_hashable(tool) if tool else "",
)
def is_tool(tool: Any) -> bool:
    """
    Checks if a given object is a valid tool in the OpenAI API.

    Args:
        tool: The object to check.

    Returns:
        True if the object is a valid tool, False otherwise.
    """
    try:
        if not isinstance(tool, dict):
            return False
        if tool.get("type") != "function":
            return False
        if "function" not in tool:
            return False
        return True
    except Exception as e:
        logger.debug(f"Error validating tool: {e}")
        return False


@cached(
    cache=_chatspec_cache,
    key=lambda messages: _make_hashable(messages) if messages else "",
)
def has_system_prompt(messages: List[Message]) -> bool:
    """
    Checks if the message thread contains at least one system prompt.

    Args:
        messages: The list of messages to check.

    Returns:
        True if the message thread contains at least one system prompt,
        False otherwise.
    """
    try:
        if not isinstance(messages, list):
            raise TypeError("Messages must be a list")
        for msg in messages:
            if not isinstance(msg, dict):
                raise TypeError("Each message must be a dict")
            if (
                msg.get("role") == "system"
                and msg.get("content") is not None
            ):
                return True
        return False
    except Exception as e:
        logger.debug(f"Error checking for system prompt: {e}")
        raise


@cached(
    cache=_chatspec_cache,
    key=lambda completion: _make_hashable(completion) if completion else "",
)
def has_tool_call(completion: Any) -> bool:
    """
    Checks if a given object contains a tool call.
    
    Args:
        completion: The object to check.

    Returns:
        True if the object contains a tool call, False otherwise.
    """
    try:
        if not is_completion(completion):
            return False

        choices = _get_value(completion, "choices", [])
        if not choices:
            return False

        first_choice = choices[0]
        message = _get_value(first_choice, "message", {})
        tool_calls = _get_value(message, "tool_calls", [])
        return bool(tool_calls)
    except Exception as e:
        logger.debug(f"Error checking for tool call: {e}")
        return False


# ------------------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------------------


def dump_stream_to_message(stream: Any) -> Message:
    """
    Aggregates a stream of ChatCompletionChunks into a single Message.

    Args:
        stream: An iterable of ChatCompletionChunk objects.

    Returns:
        A Message containing the complete assistant response.
    """
    try:
        content_parts: List[str] = []
        tool_calls_dict: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            choices = _get_value(chunk, "choices", [])
            if not choices:
                continue

            for choice in choices:
                delta = _get_value(choice, "delta", {})
                content = _get_value(delta, "content")
                if content:
                    content_parts.append(content)

                tool_calls = _get_value(delta, "tool_calls", [])
                for tool_call in tool_calls:
                    index = _get_value(tool_call, "index")
                    if index is None:
                        continue
                    if index not in tool_calls_dict:
                        tool_calls_dict[index] = {
                            "id": _get_value(tool_call, "id", ""),
                            "type": "function",
                            "function": {
                                "name": _get_value(
                                    _get_value(tool_call, "function", {}),
                                    "name",
                                    "",
                                ),
                                "arguments": _get_value(
                                    _get_value(tool_call, "function", {}),
                                    "arguments",
                                    "",
                                ),
                            },
                        }
                    else:
                        func_obj = _get_value(tool_call, "function", {})
                        if _get_value(func_obj, "arguments"):
                            tool_calls_dict[index]["function"][
                                "arguments"
                            ] += _get_value(func_obj, "arguments")
                        if _get_value(func_obj, "name"):
                            tool_calls_dict[index]["function"]["name"] += (
                                _get_value(func_obj, "name")
                            )
                        if _get_value(tool_call, "id"):
                            tool_calls_dict[index]["id"] = _get_value(
                                tool_call, "id"
                            )

        message: Message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls_dict:
            message["tool_calls"] = list(tool_calls_dict.values())
        return message
    except Exception as e:
        logger.debug(f"Error dumping stream to message: {e}")
        raise


def dump_stream_to_completion(stream: Any) -> Completion:
    """
    Aggregates a stream of ChatCompletionChunks into a single Completion.
    """
    try:
        choices = []
        for chunk in stream:
            delta = _get_value(
                _get_value(chunk.choices[0], "delta", {}), "content", ""
            )
            choices.append(
                {"message": {"role": "assistant", "content": delta}}
            )

        return Completion(
            id="stream", choices=choices, created=0, model="stream"
        )
    except Exception as e:
        logger.debug(f"Error dumping stream to completion: {e}")
        raise


def parse_model_from_completion(
    completion: Any, model: type[BaseModel]
) -> BaseModel:
    """
    Extracts the JSON content from a non-streaming chat completion and initializes
    and returns an instance of the provided Pydantic model.
    """
    try:
        choices = getattr(completion, "choices", None) or completion.get(
            "choices"
        )
        if not choices or len(choices) == 0:
            raise ValueError("No choices found in the completion object.")

        first_choice = choices[0]
        message = getattr(
            first_choice, "message", None
        ) or first_choice.get("message", {})
        content = message.get("content")

        if content is None:
            raise ValueError("No content found in the completion message.")

        try:
            data = msgspec.json.decode(content)
        except Exception as e:
            raise ValueError(f"Error parsing JSON content: {e}")

        return model.model_validate(data)
    except Exception as e:
        logger.debug(f"Error parsing model from completion: {e}")
        raise


def parse_model_from_stream(
    stream: Any, model: type[BaseModel]
) -> BaseModel:
    """
    Aggregates a stream of chat completion chunks, extracts the JSON content from the
    aggregated message, and initializes and returns an instance of the provided Pydantic model.
    """
    try:
        message = dump_stream_to_message(stream)
        content = message.get("content")

        if content is None:
            raise ValueError(
                "No content found in the aggregated stream message."
            )

        try:
            data = msgspec.json.decode(content)
        except Exception as e:
            raise ValueError(
                f"Error parsing JSON content from stream: {e}"
            )

        return model.model_validate(data)
    except Exception as e:
        logger.debug(f"Error parsing model from stream: {e}")
        raise


def print_stream(stream: Any) -> None:
    """
    Prints a stream of chat completion chunks in a human-readable format.
    Shows both content and tool calls if present.
    """
    try:
        if is_stream(stream):
            for chunk in stream:
                delta = chunk.choices[0].delta

                # Handle content
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)

                # Handle tool calls
                if hasattr(delta, "tool_calls"):
                    tool_calls = delta.tool_calls
                    if tool_calls:
                        for tool_call in tool_calls:
                            if hasattr(tool_call, "function"):
                                func = tool_call.function
                                print("\n=== Tool Call ===")
                                print(f"Name: {func.name}")
                                print(f"Arguments: {func.arguments}")
                                print("================\n")
    except Exception as e:
        logger.debug(f"Error printing stream: {e}")


# ------------------------------------------------------------------------------
# Tool Calls
# ------------------------------------------------------------------------------


def get_tool_calls(completion: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool calls from a given chat completion object.

    Args:
        completion: A chat completion object (streaming or non-streaming).

    Returns:
        A list of tool call dictionaries (each containing id, type, and function details).
    """
    try:
        if not has_tool_call(completion):
            return []
        choices = _get_value(completion, "choices", [])
        if not choices:
            return []
        message = _get_value(choices[0], "message", {})
        return _get_value(message, "tool_calls", [])
    except Exception as e:
        logger.debug(f"Error getting tool calls: {e}")
        return []


@cached(
    cache=_chatspec_cache,
    key=lambda completion, tool: _make_hashable((completion, tool.__name__ if callable(tool) else tool)) if completion else "",
)
def was_tool_called(
    completion: Any, tool: Union[str, Callable, Dict[str, Any]]
) -> bool:
    """Checks if a given tool was called in a chat completion."""
    try:
        tool_name = ""
        if isinstance(tool, str):
            tool_name = tool
        elif callable(tool):
            tool_name = tool.__name__
        elif isinstance(tool, dict) and "name" in tool:
            tool_name = tool["name"]
        else:
            return False

        tool_calls = get_tool_calls(completion)
        return any(
            _get_value(_get_value(call, "function", {}), "name") == tool_name
            for call in tool_calls
        )
    except Exception as e:
        logger.debug(f"Error checking if tool was called: {e}")
        return False


def run_tool(completion: Any, tool: callable) -> Any:
    """
    Executes a tool based on parameters extracted from a completion object.
    """
    try:
        tool_calls = get_tool_calls(completion)
        tool_name = tool.__name__
        matching_call = next(
            (
                call
                for call in tool_calls
                if call.get("function", {}).get("name") == tool_name
            ),
            None,
        )

        if not matching_call:
            raise ValueError(
                f"Tool '{tool_name}' was not called in this completion"
            )

        try:
            args_str = matching_call["function"]["arguments"]
            args = msgspec.json.decode(args_str)
            if isinstance(args, dict):
                return tool(**args)
            else:
                raise ValueError(
                    f"Invalid arguments format for tool '{tool_name}'"
                )
        except msgspec.DecodeError:
            raise ValueError(
                f"Invalid JSON in arguments for tool '{tool_name}'"
            )
    except Exception as e:
        logger.debug(f"Error running tool: {e}")
        raise


def create_tool_message(completion: Any, output: Any) -> Message:
    """
    Creates a tool message from a given chat completion or stream and tool output.

    Args:
        completion: A chat completion object.
        output: The output from running the tool.

    Returns:
        A Message object with the tool's response.

    Raises:
        ValueError: If no tool calls are found in the completion.
    """
    try:
        tool_calls = get_tool_calls(completion)
        if not tool_calls:
            raise ValueError("No tool calls found in completion")
        tool_call_id = tool_calls[0].get("id")
        if not tool_call_id:
            raise ValueError("Tool call ID not found")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(output),
        }
    except Exception as e:
        logger.debug(f"Error creating tool message: {e}")
        raise


@cached(
    cache=_chatspec_cache,
    key=lambda tool: _make_hashable(tool) if tool else "",
)
def convert_to_tool(
    tool: Union[BaseModel, Callable, Dict[str, Any]],
) -> Tool:
    """
    Converts a given object into a tool.

    This function handles:
      - Pydantic models (using their schema and docstring),
      - Python functions (using type hints and docstring),
      - Existing tool dictionaries.

    Args:
        tool: The object to convert into a tool.

    Returns:
        A Tool dictionary compatible with chat completions.

    Raises:
        TypeError: If the input cannot be converted to a tool.
    """
    try:
        if (
            isinstance(tool, dict)
            and "type" in tool
            and "function" in tool
        ):
            return tool

        if isinstance(tool, type) and issubclass(tool, BaseModel):
            schema = tool.model_json_schema()
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())
                schema["additionalProperties"] = False
            function_def = {"name": tool.__name__, "parameters": schema}
            if tool.__doc__:
                docstring = docstring_parser.parse(tool.__doc__)
                function_def["description"] = docstring.description
            return {"type": "function", "function": function_def}

        if callable(tool):
            import inspect

            sig = inspect.signature(tool)
            properties = {}
            required = []

            # Parse docstring if available
            param_descriptions = {}
            if tool.__doc__:
                docstring = docstring_parser.parse(tool.__doc__)
                description = docstring.description
                for param in docstring.params:
                    param_descriptions[param.arg_name] = param.description

            for name, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                param_schema = {"type": "string"}
                if name in param_descriptions:
                    param_schema["description"] = param_descriptions[name]

                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        param_schema["type"] = "string"
                    elif param.annotation == int:
                        param_schema["type"] = "integer"
                    elif param.annotation == float:
                        param_schema["type"] = "number"
                    elif param.annotation == bool:
                        param_schema["type"] = "boolean"
                    elif param.annotation == list:
                        param_schema["type"] = "array"
                    elif param.annotation == dict:
                        param_schema["type"] = "object"

                properties[name] = param_schema
                if param.default == inspect.Parameter.empty:
                    required.append(name)

            parameters_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

            function_def = {
                "name": tool.__name__,
                "parameters": parameters_schema,
            }

            if tool.__doc__:
                docstring = docstring_parser.parse(tool.__doc__)
                function_def["description"] = docstring.description
                if docstring.returns:
                    function_def["returns"] = docstring.returns.description

            return {"type": "function", "function": function_def}

        raise TypeError(f"Cannot convert {type(tool)} to tool")
    except Exception as e:
        logger.debug(f"Error converting to tool: {e}")
        raise


def convert_to_tools(
    tools: Union[List[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Converts a list of tools (which may be BaseModel, callable, or Tool dict)
    into a dictionary mapping tool names to tool definitions.
    If a tool is not already in Tool format, it is converted via convert_to_tool.
    If the original tool is callable, it is attached as the "callable" key.

    Args:
        tools: A list of tools (which may be BaseModel, callable, or Tool dict)

    Returns:
        A dictionary mapping tool names to tool definitions.
    """
    tools_dict: Dict[str, Any] = {}

    if isinstance(tools, dict):
        # Assume already keyed by tool name
        return tools

    if isinstance(tools, list):
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" in tool
            ):
                # Tool is already in correct format
                name = tool["function"].get("name")
                if name:
                    tools_dict[name] = tool
            else:
                # Convert tool to proper format
                converted = convert_to_tool(tool)
                if (
                    "function" in converted
                    and "name" in converted["function"]
                ):
                    name = converted["function"]["name"]
                    tools_dict[name] = converted
                    # Attach original callable if applicable
                    if callable(tool):
                        tools_dict[name]["callable"] = tool

    return tools_dict


# ------------------------------------------------------------------------------
# Messages
# ------------------------------------------------------------------------------


@cached(
    cache=_chatspec_cache,
    key=lambda messages: _make_hashable(messages) if messages else "",
)
def normalize_messages(messages: Any) -> List[Message]:
    """Formats the input into a list of chat completion messages."""
    try:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list):
            messages = [messages]

        normalized = []
        for message in messages:
            if isinstance(message, dict):
                # Create a new dict to avoid modifying the original
                normalized.append({**message})
            elif hasattr(message, "model_dump"):
                normalized.append(message.model_dump())
            else:
                raise ValueError(f"Invalid message format: {message}")
        return normalized
    except Exception as e:
        logger.debug(f"Error normalizing messages: {e}")
        raise


@cached(
    cache=_chatspec_cache,
    key=lambda messages, system_prompt=None, blank=False: 
        _make_hashable((messages, system_prompt, blank)),
)
def normalize_system_prompt(
    messages: List[Message],
    system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
    blank: bool = False,
) -> List[Message]:
    """
    Normalizes a message thread by gathering all system messages at the start.

    Args:
        messages: List of messages to normalize.
        system_prompt: Optional system prompt to prepend.
        blank: If True, ensures at least one system message exists (even empty).

    Returns:
        A normalized list of messages.
    """
    try:
        system_messages = [
            msg for msg in messages if msg.get("role") == "system"
        ]
        other_messages = [
            msg for msg in messages if msg.get("role") != "system"
        ]

        if system_prompt:
            if isinstance(system_prompt, str):
                new_system = {"role": "system", "content": system_prompt}
            elif isinstance(system_prompt, dict):
                new_system = {**system_prompt, "role": "system"}
                if "content" not in new_system:
                    raise ValueError(
                        "System prompt dict must contain 'content' field"
                    )
            else:
                raise ValueError("System prompt must be string or dict")
            system_messages.insert(0, new_system)

        if not system_messages and blank:
            system_messages = [{"role": "system", "content": ""}]
        elif not system_messages:
            return messages

        if len(system_messages) > 1:
            combined_content = "\n".join(
                msg["content"] for msg in system_messages
            )
            system_messages = [
                {"role": "system", "content": combined_content}
            ]

        return system_messages + other_messages
    except Exception as e:
        logger.debug(f"Error normalizing system prompt: {e}")
        raise


# ------------------------------------------------------------------------------
# pydantic models
#
# i made `chatspec` for my own use, to be in conjunction with `instructor`
# [instructor](https://github.com/instructor-ai/instructor) and OpenAI, which
# is what I use for my own projects. This is why `pydantic` is the an added (and
# only) depedency.
#
# the following methods are specifically for working with pydantic models and
# most useful when in the context of creating structured outputs.
# ------------------------------------------------------------------------------


@cached(
    cache=_chatspec_cache,
    key=lambda type_hint, index=None, description=None, default=...: 
        _make_hashable((type_hint, index, description, default)),
)
def create_field_mapping(
    type_hint: Type,
    index: Optional[int] = None,
    description: Optional[str] = None,
    default: Any = ...,
) -> Dict[str, Any]:
    """
    Creates a Pydantic field mapping from a type hint.

    Args:
        type_hint: The Python type to convert
        index: Optional index to append to field name for uniqueness
        description: Optional field description
        default: Optional default value

    Returns:
        Dictionary mapping field name to (type, Field) tuple
    """
    try:
        base_name, _ = _TYPE_MAPPING.get(type_hint, ("value", type_hint))
        field_name = (
            f"{base_name}_{index}" if index is not None else base_name
        )
        return {
            field_name: (
                type_hint,
                Field(default=default, description=description),
            )
        }
    except Exception as e:
        logger.debug(f"Error creating field mapping: {e}")
        raise


@cached(
    cache=_chatspec_cache,
    key=lambda func: _make_hashable(func),
)
def extract_function_fields(func: Callable) -> Dict[str, Any]:
    """
    Extracts fields from a function's signature and docstring.

    Args:
        func: The function to analyze

    Returns:
        Dictionary of field definitions
    """
    try:
        import docstring_parser

        hints = get_type_hints(func)
        sig = signature(func)
        docstring = docstring_parser.parse(func.__doc__ or "")
        fields = {}

        for name, param in sig.parameters.items():
            field_type = hints.get(name, Any)
            default = (
                ... if param.default is param.empty else param.default
            )
            description = next(
                (
                    p.description
                    for p in docstring.params
                    if p.arg_name == name
                ),
                "",
            )
            fields[name] = (
                field_type,
                Field(default=default, description=description),
            )

        return fields
    except Exception as e:
        logger.debug(f"Error extracting function fields: {e}")
        raise


# ----------------------------------------------------------------------
# Model Creation
# ----------------------------------------------------------------------


@cached(
    cache=_chatspec_cache,
    key=lambda target, init=False, name=None, description=None, default=...: 
        _make_hashable((target, init, name, description, default)),
)
def convert_to_pydantic_model(
    target: Union[Type, Sequence[Type], Dict[str, Any], BaseModel, Callable],
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """
    Converts various input types into a pydantic model class or instance.

    Args:
        target: The target to convert (type, sequence, dict, model, or function)
        init: Whether to initialize the model with values (for dataclasses/dicts)
        name: Optional name for the generated model
        description: Optional description for the model/field
        default: Optional default value for single-type models

    Returns:
        A pydantic model class or instance if init=True
    """
    model_name = name or "GeneratedModel"

    # Handle existing Pydantic models
    if isinstance(target, type) and issubclass(target, BaseModel):
        return target

    # Handle dataclasses
    if is_dataclass(target):
        hints = get_type_hints(target)
        fields = {
            name: (
                hint,
                Field(default=getattr(target, name) if init else ...),
            )
            for name, hint in hints.items()
        }
        model_class = create_model(
            model_name, __doc__=description, **fields
        )
        if init and isinstance(target, type):
            return model_class
        elif init:
            return model_class(
                **{name: getattr(target, name) for name in hints}
            )
        return model_class

    # Handle callable (functions)
    if callable(target) and not isinstance(target, type):
        fields = extract_function_fields(target)
        return create_model(
            name or target.__name__,
            __doc__=description or target.__doc__,
            **fields,
        )

    # Handle single types
    if isinstance(target, type):
        field_mapping = create_field_mapping(
            target, description=description, default=default
        )
        return create_model(
            model_name, __doc__=description, **field_mapping
        )

    # Handle sequences of types
    if isinstance(target, (list, tuple)):
        field_mapping = {}
        for i, type_hint in enumerate(target):
            if not isinstance(type_hint, type):
                raise ValueError("Sequence elements must be types")
            field_mapping.update(create_field_mapping(type_hint, index=i))
        return create_model(
            model_name, __doc__=description, **field_mapping
        )

    # Handle dictionaries
    if isinstance(target, dict):
        if init:
            model_class = create_model(
                model_name,
                __doc__=description,
                **{
                    k: (type(v), Field(default=v))
                    for k, v in target.items()
                },
            )
            return model_class(**target)
        return create_model(model_name, __doc__=description, **target)

    # Handle model instances
    if isinstance(target, BaseModel):
        if init:
            model_class = create_model(
                model_name,
                __doc__=description,
                **{
                    k: (type(v), Field(default=v))
                    for k, v in target.model_dump().items()
                },
            )
            return model_class(**target.model_dump())
        return target.__class__

    raise ValueError(
        f"Unsupported target type: {type(target)}. Must be a type, "
        "sequence of types, dict, dataclass, function, or Pydantic model."
    )


# this one is kinda super specific
@cached(
    cache=_chatspec_cache,
    key=lambda target, name=None: _make_hashable((target, name)) if target else "",
)
def create_literal_pydantic_model(
    target: Union[Type, List[str]],
    name: Optional[str] = "Selection",
    description: Optional[str] = None,
    default: Any = ...,
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for handling selections/literals.

    Args:
        target: Either a Literal type or a list of strings representing allowed values

    Returns:
        A Pydantic model class with a single 'value' field constrained to the allowed values
    """
    if isinstance(target, list):
        # For list of strings, create a Literal type with those values
        literal_type = Literal[tuple(str(v) for v in target)]  # type: ignore
    elif getattr(target, "__origin__", None) is Literal:
        # For existing Literal types, use directly
        literal_type = target
    else:
        raise ValueError(
            "Target must be either a Literal type or a list of strings"
        )

    return create_model(
        name or "Selection",
        value=(
            literal_type,
            Field(
                default=default,
                description=description or "The selected value",
            ),
        ),
    )


# ------------------------------------------------------------------------------
# Markdown Formatting
# this is used to format text, or any other arbitrary 'thing' as a formatted
# markdown string.
# ------------------------------------------------------------------------------


def _get_field_description(field_info: Any) -> Optional[str]:
    """Extract field description from Pydantic field info.

    Args:
        field_info: The Pydantic field info object to extract description from

    Returns:
        The field description if available, None otherwise
    """
    try:
        if hasattr(field_info, "__doc__") and field_info.__doc__:
            doc = docstring_parser.parse(field_info.__doc__)
            if doc.short_description:
                return doc.short_description

        if hasattr(field_info, "description"):
            return field_info.description

        return None
    except Exception:
        return None


def _format_docstring(
    doc_dict: dict, prefix: str = "", compact: bool = False
) -> str:
    """Format parsed docstring into markdown.

    Args:
        doc_dict: Dictionary containing parsed docstring sections
        prefix: String to prepend to each line for indentation
        compact: If True, produces more compact output

    Returns:
        Formatted markdown string
    """
    try:
        if not doc_dict:
            return ""

        if isinstance(doc_dict, str):
            doc = docstring_parser.parse(doc_dict)
        else:
            doc = docstring_parser.parse(str(doc_dict))

        parts = []

        if doc.short_description:
            parts.append(f"{prefix}_{doc.short_description}_")

        if doc.long_description:
            parts.append(f"{prefix}_{doc.long_description}_")

        if doc.params:
            parts.append(f"{prefix}_Parameters:_")
            for param in doc.params:
                type_str = (
                    f": {param.type_name}" if param.type_name else ""
                )
                parts.append(
                    f"{prefix}  - `{param.arg_name}{type_str}` - {param.description}"
                )

        if doc.returns:
            parts.append(f"{prefix}_Returns:_ {doc.returns.description}")

        if doc.raises:
            parts.append(f"{prefix}_Raises:_")
            for exc in doc.raises:
                parts.append(
                    f"{prefix}  - `{exc.type_name}` - {exc.description}"
                )

        return "\n".join(parts)
    except Exception:
        return str(doc_dict)


@cached(
    cache=_chatspec_cache,
    key=lambda cls: _make_hashable(cls),
)
def get_type_name(cls: Any) -> str:
    """Get a clean type name for display"""
    # Handle None type
    if cls is None:
        return "None"
    # Handle basic types with __name__ attribute
    if hasattr(cls, "__name__"):
        return cls.__name__
    # Handle typing types like Optional, List etc
    if hasattr(cls, "__origin__"):
        # Get the base type (List, Optional etc)
        origin = cls.__origin__.__name__
        # Handle special case of Optional which is really Union[T, None]
        if (
            origin == "Union"
            and len(cls.__args__) == 2
            and cls.__args__[1] is type(None)
        ):
            return f"Optional[{get_type_name(cls.__args__[0])}]"
        # For other generic types, recursively get names of type arguments
        args = ", ".join(get_type_name(arg) for arg in cls.__args__)
        return f"{origin}[{args}]"

    # Fallback for any other types
    return str(cls)


def _parse_docstring(obj: Any) -> Optional[dict]:
    """
    Extract and parse docstring from an object using docstring-parser.

    Returns:
        Dictionary containing parsed docstring components:
        - short_description: Brief description
        - long_description: Detailed description
        - params: List of parameters
        - returns: Return value description
        - raises: List of exceptions
    """
    doc = getdoc(obj)
    if not doc:
        return None

    try:
        parsed = docstring_parser.parse(doc)
        result = {
            "short": parsed.short_description,
            "long": parsed.long_description,
            "params": [
                (p.arg_name, p.type_name, p.description)
                for p in parsed.params
            ],
            "returns": parsed.returns.description
            if parsed.returns
            else None,
            "raises": [
                (e.type_name, e.description) for e in parsed.raises
            ],
        }
        return {k: v for k, v in result.items() if v}
    except:
        # Fallback to simple docstring if parsing fails
        return {"short": doc.strip()}


# -----------------------------------------------------------------------------
# Public API: markdownify
# -----------------------------------------------------------------------------


@cached(
    cache=_chatspec_cache,
    key=lambda target, indent=0, code_block=False, compact=False, show_types=True, 
          show_title=True, show_bullets=True, show_docs=True, bullet_style="-", _visited=None:
        _make_hashable((target, indent, code_block, compact, show_types, show_title, 
                       show_bullets, show_docs, bullet_style)),
)
def markdownify(
    target: Any,
    indent: int = 0,
    code_block: bool = False,
    compact: bool = False,
    show_types: bool = True,
    show_title: bool = True,
    show_bullets: bool = True,
    show_docs: bool = True,
    bullet_style: str = "-",
    _visited: set[int] | None = None,
) -> str:
    """
    Formats a target object into markdown optimized for LLM prompts.
    """
    visited = _visited or set()
    obj_id = id(target)
    if obj_id in visited:
        return "<circular>"
    visited.add(obj_id)

    prefix = "  " * indent
    bullet = f"{bullet_style} " if show_bullets else ""

    if target is None or isinstance(target, (str, int, float, bool)):
        return str(target)
    if isinstance(target, bytes):
        return f"b'{target.hex()}'"

    # Handle Pydantic models
    try:
        if isinstance(target, BaseModel) or (
            isinstance(target, type) and issubclass(target, BaseModel)
        ):
            is_class = isinstance(target, type)
            model_name = (
                target.__name__ if is_class else target.__class__.__name__
            )

            if code_block:
                data = (
                    target.model_dump()
                    if not is_class
                    else {
                        field: f"{get_type_name(field_info.annotation)}"
                        if show_types
                        else "..."
                        for field, field_info in target.model_fields.items()
                    }
                )
                json_str = msgspec.json.encode(
                    data, sort_keys=True, indent=2
                ).decode("utf-8")
                return f"```json\n{json_str}\n```"

            header_parts = [f"{prefix}{bullet}**{model_name}**:"]
            if show_docs:
                try:
                    doc_dict = _parse_docstring(target)
                except Exception as e:
                    logger.warning(
                        f"Error parsing docstring for {model_name}: {e}"
                    )
                    doc_dict = None
                if doc_dict:
                    doc_md = _format_docstring(
                        doc_dict, prefix + "  ", compact
                    )
                    if doc_md:
                        header_parts.append(doc_md)

            header = "\n".join(header_parts) if show_title else ""

            fields = (
                target.model_fields.items()
                if is_class
                else target.model_dump().items()
            )
            field_lines = []
            indent_step = 1 if compact else 2
            field_prefix = prefix + ("  " if not compact else "")

            for key, value in fields:
                field_info = target.model_fields[key] if is_class else None
                field_value = (
                    get_type_name(value.annotation) if is_class else value
                )
                field_line = [
                    f"{field_prefix}{bullet}**{key}**: {field_value}"
                ]
                if (
                    show_docs
                    and field_info
                    and (desc := _get_field_description(field_info))
                ):
                    field_line.append(f"{field_prefix}  _{desc}_")
                rendered = markdownify(
                    field_value,
                    indent + indent_step,
                    code_block,
                    compact,
                    show_types,
                    show_title,
                    show_bullets,
                    show_docs,
                    bullet_style,
                    visited.copy(),
                )
                field_lines.extend(field_line)

            return (
                "\n".join(filter(None, [header] + field_lines))
                if show_title
                else "\n".join(field_lines)
        )
    except Exception as e:
        logger.error(
            f"Error formatting pydantic model target {target} to markdown: {e}"
        )
        raise e

    # Handle collections
    if isinstance(target, (list, tuple, set)):
        if not target:
            return (
                "[]"
                if isinstance(target, list)
                else "()"
                if isinstance(target, tuple)
                else "{}"
            )

        if code_block and isinstance(target[0], (dict, BaseModel)):
            json_str = msgspec.json.encode(
                list(target), sort_keys=True, indent=2
            ).decode("utf-8")
            return f"```json\n{json_str}\n```"

        type_name = target.__class__.__name__ if show_types else ""
        header = (
            f"{prefix}{bullet}**{type_name}**:"
            if show_types and show_title
            else f"{prefix}{bullet}"
        )
        indent_step = 1 if compact else 2
        item_prefix = prefix + ("  " if not compact else "")

        items = [
            f"{item_prefix}{bullet}{markdownify(item, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, visited.copy())}"
            for item in target
        ]
        return (
            "\n".join([header] + items)
            if show_types and show_title
            else "\n".join(items)
        )

    # Handle dictionaries
    if isinstance(target, dict):
        if not target:
            return "{}"

        if code_block:
            json_str = msgspec.json.encode(
                target, sort_keys=True, indent=2
            ).decode("utf-8")
            return f"```json\n{json_str}\n```"

        type_name = target.__class__.__name__ if show_types else ""
        header = (
            f"{prefix}{bullet}**{type_name}**:"
            if show_types and show_title
            else f"{prefix}{bullet}"
        )
        indent_step = 1 if compact else 2
        item_prefix = prefix + ("  " if not compact else "")

        items = [
            f"{item_prefix}{bullet}**{key}**: {markdownify(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, visited.copy())}"
            for key, value in target.items()
        ]
        return (
            "\n".join([header] + items)
            if show_types and show_title
            else "\n".join(items)
        )

    # Handle dataclasses
    if is_dataclass(target):
        type_name = target.__class__.__name__ if show_types else ""
        header = (
            f"{prefix}{bullet}**{type_name}**:"
            if show_types and show_title
            else f"{prefix}{bullet}"
        )
        indent_step = 1 if compact else 2
        item_prefix = prefix + ("  " if not compact else "")

        fields_list = [
            (f.name, getattr(target, f.name))
            for f in dataclass_fields(target)
        ]
        items = [
            f"{item_prefix}{bullet}**{name}**: {markdownify(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, visited.copy())}"
            for name, value in fields_list
        ]
        return (
            "\n".join([header] + items)
            if show_types and show_title
            else "\n".join(items)
        )

    return str(target)
