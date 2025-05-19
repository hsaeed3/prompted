"""
## 💭 chatspec.utils

Contains the utils and helpers within the chatspec library.
These range from helpers for instance checking response/input types,
as well as message formatting / tool conversion, etc.
"""

import logging
import hashlib
import json
import inspect
from cachetools import TTLCache
from functools import wraps, update_wrapper
from dataclasses import is_dataclass
from docstring_parser import parse
from inspect import signature
from pydantic import BaseModel, Field, create_model
from pathlib import Path

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
    TypeVar,
    Sequence,
    get_type_hints,
    Iterator,
)
from .types import (
    Completion,
    CompletionChunk,
    CompletionMessage,
    Message,
    MessageContentImagePart,
    MessageContentAudioPart,
    MessageContentTextPart,
    Tool,
)

__all__ = (
    "is_completion",
    "is_stream",
    "is_message",
    "is_tool",
    "has_system_prompt",
    "has_tool_call",
    "was_tool_called",
    "run_tool",
    "create_tool_message",
    "create_image_message",
    "create_input_audio_message",
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
    "create_selection_model",
    "create_bool_model",
)


# ------------------------------------------------------------------------------
# Configuration && Logging
#
logger = logging.getLogger(__name__)
#
# cache
_CACHE = TTLCache(maxsize=1000, ttl=3600)


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


T = TypeVar("T")


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


def _cached(
    key_fn,
):
    """More efficient caching decorator that only creates cache entries when needed."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get the original signature
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                # Include function name in cache key to avoid cross-function cache collisions
                func_name = func.__name__
                cache_key = f"{func_name}:{key_fn(*args, **kwargs)}"
                if cache_key not in _CACHE:
                    _CACHE[cache_key] = func(*args, **kwargs)
                return _CACHE[cache_key]
            except Exception:
                # On any error, fall back to uncached function call
                return func(*args, **kwargs)

        return wrapper

    return decorator


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
                # Ensure chunk.choices[0].delta is a CompletionMessage
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta"):
                        content = ""
                        tool_calls = None

                        # Get content and tool_calls from delta
                        if isinstance(choice.delta, dict):
                            content = choice.delta.get("content", "")
                            tool_calls = choice.delta.get("tool_calls")
                        else:
                            content = getattr(choice.delta, "content", "")
                            tool_calls = getattr(
                                choice.delta, "tool_calls", None
                            )

                        # Create a proper CompletionMessage with empty string as default content
                        choice.delta = CompletionMessage(
                            role="assistant",
                            content=""
                            if content is None
                            else content,  # Ensure content is never None
                            name=None,
                            function_call=None,
                            tool_calls=tool_calls,
                            tool_call_id=None,
                        )
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
                # Ensure chunk.choices[0].delta is a CompletionMessage
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta"):
                        content = ""
                        if isinstance(choice.delta, dict):
                            content = choice.delta.get("content", "")
                        else:
                            content = getattr(choice.delta, "content", "")

                        # Create a proper CompletionMessage
                        choice.delta = CompletionMessage(
                            role="assistant",
                            content=content,
                            name=None,
                            function_call=None,
                            tool_calls=None,
                            tool_call_id=None,
                        )
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


def is_completion(completion: Any) -> bool:
    """
    Checks if a given object is a valid chat completion.

    Supports both standard completion objects, as well as
    streamed responses.
    """

    @_cached(
        lambda completion: _make_hashable(completion) if completion else ""
    )
    def _is_completion(completion: Any) -> bool:
        try:
            # Handle passthrough wrapper (sync or async)
            if hasattr(completion, "chunks"):
                return bool(completion.chunks) and any(
                    _get_value(chunk, "choices")
                    for chunk in completion.chunks
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
            logger.debug(
                f"Error checking if object is chat completion: {e}"
            )
            return False

    return _is_completion(completion)


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


def is_message(message: Any) -> bool:
    """Checks if a given object is a valid chat message."""

    @_cached(lambda message: _make_hashable(message) if message else "")
    def _is_message(message: Any) -> bool:
        try:
            if not isinstance(message, dict):
                return False
            allowed_roles = {
                "assistant",
                "user",
                "system",
                "tool",
                "developer",
            }
            role = message.get("role")
            # First check role validity
            if role not in allowed_roles:
                return False
            # Check content and tool_call_id requirements
            if role == "tool":
                return bool(message.get("content")) and bool(
                    message.get("tool_call_id")
                )
            elif role == "assistant" and "tool_calls" in message:
                return True
            # For all other roles, just need content
            return message.get("content") is not None
        except Exception as e:
            logger.debug(f"Error validating message: {e}")
            return False

    return _is_message(message)


def is_tool(tool: Any) -> bool:
    """
    Checks if a given object is a valid tool in the OpenAI API.

    Args:
        tool: The object to check.

    Returns:
        True if the object is a valid tool, False otherwise.
    """

    @_cached(lambda tool: _make_hashable(tool) if tool else "")
    def _is_tool(tool: Any) -> bool:
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

    return _is_tool(tool)


def has_system_prompt(messages: List[Message]) -> bool:
    """
    Checks if the message thread contains at least one system prompt.

    Args:
        messages: The list of messages to check.

    Returns:
        True if the message thread contains at least one system prompt,
        False otherwise.
    """

    @_cached(lambda messages: _make_hashable(messages) if messages else "")
    def _has_system_prompt(messages: Any) -> bool:
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

    return _has_system_prompt(messages)


def has_tool_call(completion: Any) -> bool:
    """
    Checks if a given object contains a tool call.

    Args:
        completion: The object to check.

    Returns:
        True if the object contains a tool call, False otherwise.
    """

    @_cached(
        lambda completion: _make_hashable(completion) if completion else ""
    )
    def _has_tool_call(completion: Any) -> bool:
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

    return _has_tool_call(completion)


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

                # Add null check for tool_calls
                tool_calls = _get_value(delta, "tool_calls", []) or []
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
    Aggregates a stream of ChatCompletionChunks into a single Completion using the standardized Pydantic models.

    Instead of creating a dictionary for each choice, this function now creates a proper
    CompletionMessage (and Completion.Choice) so that the resulting Completion adheres to the
    models and types expected throughout the library (as seen in chatspec/mock.py).

    Returns:
        A Completion object as defined in `chatspec/types.py`.
    """
    try:
        from .types import (
            Completion,
            CompletionMessage,
        )  # using the models directly

        choices = []
        for chunk in stream:
            # Safely extract content from the chunk's delta field.
            delta_content = _get_value(
                _get_value(chunk.choices[0], "delta", {}), "content", ""
            )
            # Create a proper CompletionMessage instance.
            message = CompletionMessage(
                role="assistant",
                content=delta_content if delta_content is not None else "",
                name=None,
                function_call=None,
                tool_calls=None,
                tool_call_id=None,
            )
            # Wrap the message in a Completion.Choice instance.
            choice_obj = Completion.Choice(
                message=message,
                finish_reason="stop",  # default finish_reason; adjust as needed
                index=0,
                logprobs=None,
            )
            choices.append(choice_obj)

        # Construct and return the Completion object using the proper types.
        return Completion(
            id="stream",
            choices=choices,
            created=0,
            model="stream",
            object="chat.completion",
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
            data = json.loads(content)
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
            data = json.loads(content)
        except Exception as e:
            raise ValueError(
                f"Error parsing JSON content from stream: {e}"
            )

        return model.model_validate(data)
    except Exception as e:
        logger.debug(f"Error parsing model from stream: {e}")
        raise


def print_stream(stream: Iterator[CompletionChunk]) -> None:
    """
    Helper method to print a stream of completion chunks.
    """
    try:
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    # Handle content
                    content = ""
                    if isinstance(choice.delta, dict):
                        content = choice.delta.get("content", "")
                    else:
                        content = getattr(choice.delta, "content", "")

                    if content:
                        print(content, end="", flush=True)

                    # Handle tool calls
                    tool_calls = None
                    if isinstance(choice.delta, dict):
                        tool_calls = choice.delta.get("tool_calls")
                    else:
                        tool_calls = getattr(
                            choice.delta, "tool_calls", None
                        )

                    if tool_calls:
                        for tool_call in tool_calls:
                            print("\nTool Call:")
                            print(f"  ID: {tool_call.id}")
                            print(f"  Type: {tool_call.type}")
                            print(f"  Function: {tool_call.function.name}")
                            print(
                                f"  Arguments: {tool_call.function.arguments}"
                            )
        print()  # Add final newline
    except Exception as e:
        logger.error(f"Error printing stream: {e}")
        raise ChatSpecError(f"Failed to print stream: {str(e)}")


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


@_cached(
    lambda completion, tool: _make_hashable(
        (completion, tool.__name__ if callable(tool) else tool)
    )
    if completion
    else ""
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
            _get_value(_get_value(call, "function", {}), "name")
            == tool_name
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
    from typing_inspect import is_literal_type
    from pydantic import BaseModel
    from typing import get_args

    @_cached(lambda tool: _make_hashable(tool) if tool else "")
    def _convert_to_tool(tool: Any) -> Tool:
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
                    for prop_name, prop_schema in schema[
                        "properties"
                    ].items():
                        if "enum" in prop_schema:
                            # Handle enum fields as literals
                            prop_schema["enum"] = list(prop_schema["enum"])
                            prop_schema["title"] = prop_name.capitalize()
                            prop_schema["type"] = "string"
                        elif is_literal_type(prop_schema.get("type")):
                            prop_schema["enum"] = list(
                                get_args(prop_schema["type"])
                            )
                            prop_schema["title"] = prop_name.capitalize()
                            prop_schema["type"] = "string"
                        else:
                            prop_schema["title"] = prop_name.capitalize()
                    schema["required"] = list(schema["properties"].keys())
                    schema["additionalProperties"] = False
                    schema["title"] = tool.__name__
                return {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "parameters": schema,
                        "strict": True,
                    },
                }

            if callable(tool):
                import inspect

                sig = inspect.signature(tool)
                properties = {}
                required = []

                # Parse docstring using docstring_parser instead of inspect
                docstring = tool.__doc__
                doc_info = None
                if docstring:
                    doc_info = parse(docstring)

                for name, param in sig.parameters.items():
                    if param.kind in (
                        param.VAR_POSITIONAL,
                        param.VAR_KEYWORD,
                    ):
                        continue

                    param_schema = {
                        "type": "string",
                        "title": name.capitalize(),
                    }

                    # Add description from docstring if available
                    if doc_info and doc_info.params:
                        for doc_param in doc_info.params:
                            if doc_param.arg_name == name:
                                if doc_param.description:
                                    param_schema["description"] = (
                                        doc_param.description
                                    )
                                # Check if parameter is required from docstring
                                if (
                                    doc_param.description
                                    and "required"
                                    in doc_param.description.lower()
                                ):
                                    if name not in required:
                                        required.append(name)

                    if param.annotation != inspect.Parameter.empty:
                        if is_literal_type(param.annotation):
                            param_schema["enum"] = list(
                                get_args(param.annotation)
                            )
                        else:
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
                    if (
                        param.default == inspect.Parameter.empty
                        and name not in required
                    ):
                        required.append(name)

                parameters_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "title": tool.__name__,
                    "additionalProperties": False,
                }

                # Add function description from docstring
                function_schema = {
                    "name": tool.__name__,
                    "strict": True,
                    "parameters": parameters_schema,
                }

                if doc_info and doc_info.short_description:
                    function_schema["description"] = (
                        doc_info.short_description
                    )
                    if doc_info.long_description:
                        function_schema["description"] += (
                            "\n\n" + doc_info.long_description
                        )

                return {
                    "type": "function",
                    "function": function_schema,
                }

            raise TypeError(f"Cannot convert {type(tool)} to tool")
        except Exception as e:
            logger.debug(f"Error converting to tool: {e}")
            raise

    return _convert_to_tool(tool)


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


def normalize_messages(messages: Any) -> List[Message]:
    """Formats the input into a list of chat completion messages."""

    @_cached(lambda messages: _make_hashable(messages) if messages else "")
    def _normalize_messages(messages: Any) -> List[Message]:
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

    return _normalize_messages(messages)


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

    @_cached(
        lambda messages, system_prompt=None, blank=False: _make_hashable(
            (messages, system_prompt, blank)
        )
    )
    def _normalize_system_prompt(
        messages: Any,
        system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
        blank: bool = False,
    ) -> List[Message]:
        try:
            system_messages = [
                msg for msg in messages if msg.get("role") == "system"
            ]
            other_messages = [
                msg for msg in messages if msg.get("role") != "system"
            ]

            if system_prompt:
                if isinstance(system_prompt, str):
                    new_system = {
                        "role": "system",
                        "content": system_prompt,
                    }
                elif isinstance(system_prompt, dict):
                    new_system = {**system_prompt, "role": "system"}
                    if "content" not in new_system:
                        raise ValueError(
                            "System prompt dict must contain 'content' field"
                        )
                else:
                    raise ValueError(
                        "System prompt must be string or dict"
                    )
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

    return _normalize_system_prompt(messages, system_prompt, blank)


# these are for using the 'contentpart' types specifically
# by setting content to type[list], you can define images & input audio.
def create_image_message(
    image: Union[str, Path, bytes],
    detail: Literal["auto", "low", "high"] = "auto",
    message: Optional[Union[str, Message]] = None,
) -> Message:
    """
    Creates a message with image content from a url, path, or bytes.

    This method is also useful for 'injecting' an image into an existing
    message's content.

    Args:
        image: The image to include - can be a URL string, Path object, or raw bytes
        detail: The detail level for the image - one of "auto", "low", or "high"
        message: Optional existing message to add the image content to

    Returns:
        A Message object with the image content part included
    """
    import base64
    from urllib.parse import urlparse

    # Convert image to base64 if needed
    if isinstance(image, Path):
        with open(image, "rb") as f:
            image_bytes = f.read()
            image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    elif isinstance(image, bytes):
        image = f"data:image/png;base64,{base64.b64encode(image).decode()}"
    elif isinstance(image, str) and not urlparse(image).scheme:
        # Handle string path
        with open(image, "rb") as f:
            image_bytes = f.read()
            image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"

    image_part: MessageContentImagePart = {
        "type": "image_url",
        "image_url": {"url": image, "detail": detail},
    }

    if message is None:
        return {"role": "user", "content": [image_part]}

    if isinstance(message, str):
        text_part: MessageContentTextPart = {
            "type": "text",
            "text": message,
        }
        return {"role": "user", "content": [text_part, image_part]}

    # Handle existing Message dict
    if not message.get("content"):
        message["content"] = [image_part]
    elif isinstance(message["content"], str):
        message["content"] = [
            {"type": "text", "text": message["content"]},
            image_part,
        ]
    elif isinstance(message["content"], (list, tuple)):
        message["content"] = list(message["content"]) + [image_part]

    return message


def create_input_audio_message(
    audio: Union[str, Path, bytes],
    format: Literal["wav", "mp3"] = "wav",
    message: Optional[Union[str, Message]] = None,
) -> Message:
    """
    Creates a message with input audio content from a url, path, or bytes.

    Args:
        audio: The audio to include - can be a URL string, Path object, or raw bytes
        format: The audio format - either "wav" or "mp3"
        message: Optional existing message to add the audio content to

    Returns:
        A Message object with the audio content part included
    """
    import base64
    from urllib.parse import urlparse

    # Convert audio to base64 if needed
    if isinstance(audio, Path):
        with open(audio, "rb") as f:
            audio_bytes = f.read()
            audio = base64.b64encode(audio_bytes).decode()
    elif isinstance(audio, bytes):
        audio = base64.b64encode(audio).decode()
    elif isinstance(audio, str) and not urlparse(audio).scheme:
        # Handle string path
        with open(audio, "rb") as f:
            audio_bytes = f.read()
            audio = base64.b64encode(audio_bytes).decode()

    audio_part: MessageContentAudioPart = {
        "type": "input_audio",
        "input_audio": {"data": audio, "format": format},
    }

    if message is None:
        return {"role": "user", "content": [audio_part]}

    if isinstance(message, str):
        text_part: MessageContentTextPart = {
            "type": "text",
            "text": message,
        }
        return {"role": "user", "content": [text_part, audio_part]}

    # Handle existing Message dict
    if not message.get("content"):
        message["content"] = [audio_part]
    elif isinstance(message["content"], str):
        message["content"] = [
            {"type": "text", "text": message["content"]},
            audio_part,
        ]
    elif isinstance(message["content"], (list, tuple)):
        message["content"] = list(message["content"]) + [audio_part]

    return message


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

    @_cached(
        lambda type_hint,
        index=None,
        description=None,
        default=...: _make_hashable(
            (type_hint, index, description, default)
        )
    )
    def _create_field_mapping(
        type_hint: Type,
        index: Optional[int] = None,
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Dict[str, Any]:
        try:
            base_name, _ = _TYPE_MAPPING.get(
                type_hint, ("value", type_hint)
            )
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

    return _create_field_mapping(type_hint, index, description, default)


def extract_function_fields(func: Callable) -> Dict[str, Any]:
    """
    Extracts fields from a function's signature and docstring.

    Args:
        func: The function to analyze

    Returns:
        Dictionary of field definitions
    """

    @_cached(lambda func: _make_hashable(func))
    def _extract_function_fields(func: Callable) -> Dict[str, Any]:
        try:
            hints = get_type_hints(func)
            sig = signature(func)
            docstring = parse(func.__doc__ or "")
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

    return _extract_function_fields(func)


# ----------------------------------------------------------------------
# Model Creation
# ----------------------------------------------------------------------


def create_selection_model(
    name: str = "Selection",
    description: str | None = None,
    fields: List[str] = [],
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for making a selection from a list of string options.

    The model will have a single field named `selection`. The type of this field
    will be `Literal[*fields]`, meaning its value must be one of the strings
    provided in the `fields` list.

    Args:
        name: The name for the created Pydantic model. Defaults to "Selection".
        description: An optional description for the model (becomes its docstring).
        fields: A list of strings representing the allowed choices for the selection.
                This list cannot be empty.

    Returns:
        A new Pydantic BaseModel class with a 'selection' field.

    Raises:
        ValueError: If the `fields` list is empty, as Literal requires at least one option.
    """
    if not fields:
        raise ValueError(
            "`fields` list cannot be empty for `create_selection_model` "
            "as it defines the possible selections for the Literal type."
        )

    # Create the Literal type from the list of field strings.
    # We can't use unpacking syntax directly with Literal, so we need to handle it differently
    if len(fields) == 1:
        selection_type = Literal[fields[0]]
    else:
        # For multiple fields, we need to use eval to create the Literal type
        # This is because Literal needs to be constructed with the actual string values
        # as separate arguments, not as a list
        literal_str = f"Literal[{', '.join(repr(f) for f in fields)}]"
        selection_type = eval(literal_str)

    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        "selection": (
            selection_type,
            Field(
                ...,
                description="The selected value from the available options.",
            ),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        if fields:
            model_docstring = f"A model for selecting one option from: {', '.join(fields)}."
        else:  # Should not be reached due to the check above, but for completeness
            model_docstring = "A selection model."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel


def create_bool_model(
    name: str = "Confirmation",
    description: str | None = None,
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for boolean confirmation/response.

    The model will have a single field named `confirmed`. The type of this field
    will be `bool`, meaning its value must be either True or False.

    Args:
        name: The name for the created Pydantic model. Defaults to "Confirmation".
        description: An optional description for the model (becomes its docstring).

    Returns:
        A new Pydantic BaseModel class with a 'confirmed' field.
    """
    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        "confirmed": (
            bool,
            Field(..., description="The boolean confirmation value."),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        model_docstring = "A model for boolean confirmation."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
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

    @_cached(
        lambda target,
        init=False,
        name=None,
        description=None,
        default=...: _make_hashable(
            (target, init, name, description, default)
        )
    )
    def _convert_to_pydantic_model(
        target: Union[
            Type, Sequence[Type], Dict[str, Any], BaseModel, Callable
        ],
        init: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Union[Type[BaseModel], BaseModel]:
        model_name = name or "GeneratedModel"

        # Handle existing Pydantic models
        if isinstance(target, type) and issubclass(target, BaseModel):
            return target

        # Handle dataclasses
        if is_dataclass(target):
            hints = get_type_hints(target)
            fields = {}

            # Parse docstring if available
            docstring = target.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            for field_name, hint in hints.items():
                description = ""
                if doc_info and doc_info.params:
                    description = next(
                        (
                            p.description
                            for p in doc_info.params
                            if p.arg_name == field_name
                        ),
                        "",
                    )

                fields[field_name] = (
                    hint,
                    Field(
                        default=getattr(target, field_name)
                        if init
                        else ...,
                        description=description,
                    ),
                )

            model_class = create_model(
                model_name,
                __doc__=description
                or (doc_info.short_description if doc_info else None),
                **fields,
            )

            if init and isinstance(target, type):
                return model_class
            elif init:
                return model_class(
                    **{
                        field_name: getattr(target, field_name)
                        for field_name in hints
                    }
                )
            return model_class

        # Handle callable (functions)
        if callable(target) and not isinstance(target, type):
            fields = extract_function_fields(target)

            # Extract just the short description from the docstring
            doc_info = parse(target.__doc__ or "")
            clean_description = (
                doc_info.short_description if doc_info else None
            )

            return create_model(
                name or target.__name__,
                __doc__=description or clean_description,
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
                field_mapping.update(
                    create_field_mapping(type_hint, index=i)
                )
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
            # Parse docstring from the model's class
            docstring = target.__class__.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            if init:
                fields = {}
                for k, v in target.model_dump().items():
                    description = ""
                    if doc_info and doc_info.params:
                        description = next(
                            (
                                p.description
                                for p in doc_info.params
                                if p.arg_name == k
                            ),
                            "",
                        )
                    fields[k] = (
                        type(v),
                        Field(default=v, description=description),
                    )

                model_class = create_model(
                    model_name,
                    __doc__=description
                    or (doc_info.short_description if doc_info else None),
                    **fields,
                )
                return model_class(**target.model_dump())
            return target.__class__

        raise ValueError(
            f"Unsupported target type: {type(target)}. Must be a type, "
            "sequence of types, dict, dataclass, function, or Pydantic model."
        )

    return _convert_to_pydantic_model(
        target, init, name, description, default
    )


# this one is kinda super specific
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

    @_cached(
        lambda target, name=None: _make_hashable((target, name))
        if target
        else ""
    )
    def _create_literal_pydantic_model(
        target: Union[Type, List[str]],
        name: Optional[str] = "Selection",
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Type[BaseModel]:
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

    return _create_literal_pydantic_model(
        target, name, description, default
    )
