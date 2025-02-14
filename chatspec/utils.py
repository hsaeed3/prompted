"""
💭 chatspec.utils

Contains all utility functions within the chatspec package.
"""

import logging
import hashlib
import docstring_parser
from cachetools import cached, TTLCache
from dataclasses import is_dataclass
from inspect import signature
import json
from pydantic import BaseModel, Field, create_model

from typing import (
    Any,
    Union,
    List,
    Optional,
    Dict,
    Callable,
    Type,
    Sequence,
    get_type_hints,
    Literal,
)
from .types import Message, Tool

__all__ = [
    "is_chat_completion",
    "is_stream",
    "is_message",
    "is_tool",
    "has_system_prompt",
    "has_tool_call",
    "dump_stream_to_message",
    "get_tool_calls",
    "was_tool_called",
    "run_tool",
    "create_tool_message",
    "convert_to_tool",
    "normalize_messages",
    "normalize_system_prompt",
    "create_field_mapping",
    "extract_function_fields",
    "convert_to_pydantic_model",
    "create_literal_pydantic_model",
    "print_stream",
    "passthrough",
]


# logger init
logger = logging.getLogger("chatspec")


# exceptions
class ChatSpecError(Exception):
    """
    Base exception for all ChatSpec errors.
    """

    pass


# ------------------------------------------------------------------------------
# Helper Functions / Type Mappings
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

    Args:
        obj: Any Python object to make hashable

    Returns:
        A hex string hash that uniquely identifies the object content
    """
    if isinstance(obj, dict):
        # Sort dict items for consistent hashing
        obj = {k: obj[k] for k in sorted(obj.keys())}
    # Convert to JSON string with sorted keys for consistent serialization
    json_str = json.dumps(obj, sort_keys=True, default=str)
    # Generate SHA-256 hash
    return hashlib.sha256(json_str.encode()).hexdigest()


TYPE_MAPPING = {
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
# Passthrough for Streaming Responses
# ------------------------------------------------------------------------------


class SyncStreamPassthrough:
    """
    Wraps a synchronous stream (an iterable) to cache chunks as they pass through.

    Once iterated, all chunks are stored in .chunks.
    """

    def __init__(self, stream: Any):
        self._stream = stream
        self.chunks: List[Any] = []
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


class AsyncStreamPassthrough:
    """
    Wraps an asynchronous stream (an async iterable) to cache chunks as they pass through.

    Use the async iterator to consume chunks. You can also call .consume() to fully read the stream.
    """

    def __init__(self, async_stream: Any):
        self._async_stream = async_stream
        self.chunks: List[Any] = []
        self._consumed = False

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        if not self._consumed:
            async for chunk in self._async_stream:
                self.chunks.append(chunk)
                yield chunk
            self._consumed = True
        else:
            for chunk in self.chunks:
                yield chunk

    async def consume(self) -> List[Any]:
        """
        Fully consumes the async stream and stores all chunks.
        """
        if not self._consumed:
            async for chunk in self._async_stream:
                self.chunks.append(chunk)
            self._consumed = True
        return self.chunks


def passthrough(completion: Any) -> Any:
    """
    A utility function that stores completion/stream data while passing it through.
    Allows multiple passes over a completion or stream that would otherwise be consumed.

    If the completion is asynchronous (has __aiter__), it returns an AsyncStreamPassthrough.
    If it is a synchronous iterable (but not a dict or str), it returns a SyncStreamPassthrough.
    Otherwise, returns the completion as-is.

    Can be used as a wrapper:
        completion = passthrough(client.chat.completions.create(...))

    Args:
        completion: A chat completion or stream

    Returns:
        A wrapped completion with stored chunks for multiple passes.
    """
    try:
        if hasattr(completion, "__aiter__"):
            return AsyncStreamPassthrough(completion)
        if hasattr(completion, "__iter__") and not isinstance(
            completion, (dict, str)
        ):
            return SyncStreamPassthrough(completion)
        return completion
    except Exception as e:
        logger.debug(f"Error in passthrough: {e}")
        return completion


# ------------------------------------------------------------------------------
# Instance checking
# ------------------------------------------------------------------------------


@cached(
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda completion: _make_hashable(completion),
)
def is_chat_completion(completion: Any) -> bool:
    """
    Checks if the given object is a chat completion.

    Returns True if the object represents a stream or a standard chat completion.
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda completion: _make_hashable(completion),
)
def is_stream(completion: Any) -> bool:
    """
    Checks if the given object is a streaming chat completion.

    Returns True if the object specifically contains 'delta' in its first choice.
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


# ------------------------------------------------------------------------------
# New Instance Check Methods
# ------------------------------------------------------------------------------


def is_message(message: Any) -> bool:
    """
    Checks if the given object is a valid chat message.

    A valid message is a dict with a 'role' in {'assistant', 'user', 'system', 'tool', 'developer'}.
    For non-assistant messages (or if tool_calls are absent), a 'content' field is required.
    Additionally, tool messages must have a 'tool_call_id'.
    """
    try:
        if not isinstance(message, dict):
            return False
        allowed_roles = {"assistant", "user", "system", "tool", "developer"}
        role = message.get("role")
        if role not in allowed_roles:
            return False
        if role != "assistant" or "tool_calls" not in message:
            if "content" not in message or message["content"] is None:
                return False
        if role == "tool" and "tool_call_id" not in message:
            return False
        return True
    except Exception as e:
        logger.debug(f"Error validating message: {e}")
        return False


def is_tool(tool: Any) -> bool:
    """
    Checks if the given object is a valid tool dictionary.

    A valid tool is a dict with a "type" equal to "function" and a "function" field.
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda messages: _make_hashable(messages),
)
def has_system_prompt(messages: List[Message]) -> bool:
    """
    Checks if the message thread contains at least one system prompt.

    Raises:
        TypeError: If messages is not a list or any message is not a dict.
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda completion: _make_hashable(completion),
)
def has_tool_call(completion: Any) -> bool:
    """
    Checks if a given object contains a tool call.
    """
    try:
        if not is_chat_completion(completion):
            return False

        if is_stream(completion):
            for chunk in completion:
                choices = _get_value(chunk, "choices", [])
                if choices:
                    first_choice = choices[0]
                    delta = _get_value(first_choice, "delta", {})
                    tool_calls = _get_value(delta, "tool_calls", None)
                    if tool_calls:
                        return True
        else:
            choices = _get_value(completion, "choices", [])
            if choices:
                first_choice = choices[0]
                message = _get_value(first_choice, "message", {})
                tool_calls = _get_value(message, "tool_calls", None)
                if tool_calls:
                    return True
        return False
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


def parse_model_from_completion(
    completion: Any, model: Type[BaseModel]
) -> BaseModel:
    """
    Extracts the JSON content from a non-streaming chat completion and initializes
    and returns an instance of the provided Pydantic model.

    Args:
        completion: A chat completion object (non-streaming) that contains a 'choices' field.
        model: A Pydantic model class to instantiate.

    Returns:
        An instance of the provided Pydantic model populated with data parsed from the JSON content.

    Raises:
        ValueError: If the content cannot be found or if the JSON cannot be parsed.
    """
    try:
        # Try to get the choices from the completion
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
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Error parsing JSON content: {e}")

        return model.model_validate(data)
    except Exception as e:
        logger.debug(f"Error parsing model from completion: {e}")
        raise


def parse_model_from_steam(
    stream: Any, model: Type[BaseModel]
) -> BaseModel:
    """
    Aggregates a stream of chat completion chunks, extracts the JSON content from the
    aggregated message, and initializes and returns an instance of the provided Pydantic model.

    Args:
        stream: An iterable (stream) of chat completion chunks.
        model: A Pydantic model class to instantiate.

    Returns:
        An instance of the provided Pydantic model populated with data parsed from the aggregated JSON content.

    Raises:
        ValueError: If no content is found or if the JSON cannot be parsed.
    """
    try:
        # Use the existing helper to aggregate the stream into a complete message
        message = dump_stream_to_message(stream)
        content = message.get("content")

        if content is None:
            raise ValueError(
                "No content found in the aggregated stream message."
            )

        try:
            data = json.loads(content)
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
        if is_stream(completion):
            return _get_value(
                dump_stream_to_message(completion), "tool_calls", []
            )
        choices = _get_value(completion, "choices", [])
        if choices:
            first_choice = choices[0]
            message = _get_value(first_choice, "message", {})
            return _get_value(message, "tool_calls", [])
        return []
    except Exception as e:
        logger.debug(f"Error getting tool calls: {e}")
        return []


@cached(
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda completion: _make_hashable(completion),
)
def was_tool_called(
    completion: Any, tool: Union[str, Callable, Dict[str, Any]]
) -> bool:
    """
    Checks if a given tool was called in a chat completion or stream.

    Args:
        completion: A chat completion object.
        tool: The tool to check for (can be a function, dict, or string name).

    Returns:
        True if the tool was called; otherwise False.
    """
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
            call.get("function", {}).get("name") == tool_name
            for call in tool_calls
        )
    except Exception as e:
        logger.debug(f"Error checking if tool was called: {e}")
        return False


def run_tool(completion: Any, tool: Callable) -> Any:
    """
    Executes a tool based on parameters extracted from a completion object.

    Args:
        completion: A chat completion object (streaming or non-streaming).
        tool: The callable function to run.

    Returns:
        The result of running the tool with the extracted parameters.

    Raises:
        ValueError: If the tool wasn't called or if arguments are invalid.
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

        import json

        try:
            args_str = matching_call["function"]["arguments"]
            args = json.loads(args_str)
            if isinstance(args, dict):
                return tool(**args)
            else:
                raise ValueError(
                    f"Invalid arguments format for tool '{tool_name}'"
                )
        except json.JSONDecodeError:
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda tool: _make_hashable(tool),
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


# ------------------------------------------------------------------------------
# Messages
# ------------------------------------------------------------------------------


@cached(
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda messages: _make_hashable(messages),
)
def normalize_messages(messages: Any) -> List[Message]:
    """
    Formats the input into a list of chat completion messages.

    Args:
        messages: Input messages (string, single message, or list of messages).

    Returns:
        A normalized list of message dictionaries.
    """
    try:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list):
            messages = [messages]

        normalized_thread = []
        base_fields = {"role", "content", "name"}
        assistant_fields = {"function_call", "tool_calls", "refusal"}
        tool_fields = {"tool_call_id"}

        for message in messages:
            normalized_msg = {}
            if hasattr(message, "model_fields"):
                for field in base_fields:
                    value = _get_value(message, field)
                    if value is not None:
                        normalized_msg[field] = value
                role = _get_value(message, "role")
                if role == "assistant":
                    for field in assistant_fields:
                        value = _get_value(message, field)
                        if value is not None:
                            normalized_msg[field] = value
                elif role == "tool":
                    for field in tool_fields:
                        value = _get_value(message, field)
                        if value is not None:
                            normalized_msg[field] = value
            elif isinstance(message, dict):
                for field in base_fields:
                    if field in message and message[field] is not None:
                        normalized_msg[field] = message[field]
                role = message.get("role")
                if role == "assistant":
                    for field in assistant_fields:
                        if field in message and message[field] is not None:
                            normalized_msg[field] = message[field]
                elif role == "tool":
                    for field in tool_fields:
                        if field in message and message[field] is not None:
                            normalized_msg[field] = message[field]
            else:
                raise ValueError(f"Invalid message format: {message}")

            if "role" not in normalized_msg:
                raise ValueError(
                    f"Message missing required 'role' field: {message}"
                )
            if "content" not in normalized_msg and not (
                normalized_msg.get("role") == "assistant"
                and "tool_calls" in normalized_msg
            ):
                raise ValueError(
                    f"Message must have 'content' or tool_calls if assistant: {message}"
                )
            if (
                normalized_msg.get("role") == "tool"
                and "tool_call_id" not in normalized_msg
            ):
                raise ValueError(
                    f"Tool message missing required 'tool_call_id' field: {message}"
                )

            normalized_thread.append(normalized_msg)

        return normalized_thread
    except Exception as e:
        logger.debug(f"Error normalizing messages: {e}")
        raise


@cached(
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda messages: _make_hashable(messages),
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda type_hint: _make_hashable(type_hint),
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
        base_name, _ = TYPE_MAPPING.get(type_hint, ("value", type_hint))
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
    cache=TTLCache(maxsize=1000, ttl=3600),
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda target: _make_hashable(target),
)
def convert_to_pydantic_model(
    target: Union[
        Type, Sequence[Type], Dict[str, Any], BaseModel, Callable
    ],
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
    cache=TTLCache(maxsize=1000, ttl=3600),
    key=lambda target: _make_hashable(target),
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
