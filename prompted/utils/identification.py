"""
💬 prompted.utils.identification

Contains utilities used for identifying objects.
"""

import logging
from typing import (
    Any,
    List,
)

from ..common.cache import (
    cached,
    make_hashable,
)
from ..types.chat_completions import Message

logger = logging.getLogger(__name__)

__all__ = [
    "is_completion",
    "is_stream",
    "is_message",
    "has_system_prompt",
    "has_tool_call",
]


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
    

# ------------------------------------------------------------------------------
# Identification Methods
# ------------------------------------------------------------------------------


def is_completion(completion: Any) -> bool:
    """
    Checks if a given object is a valid chat completion.

    Supports both standard completion objects, as well as
    streamed responses.
    """

    @cached(
        lambda completion: make_hashable(completion) if completion else ""
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

    @cached(lambda message: make_hashable(message) if message else "")
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
                # ADDED FOR GOOGLE A2A
                "agent"
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


def has_system_prompt(messages: List[Message]) -> bool:
    """
    Checks if the message thread contains at least one system prompt.

    Args:
        messages: The list of messages to check.

    Returns:
        True if the message thread contains at least one system prompt,
        False otherwise.
    """

    @cached(lambda messages: make_hashable(messages) if messages else "")
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

    @cached(
        lambda completion: make_hashable(completion) if completion else ""
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