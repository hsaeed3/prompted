"""
💬 prompted.utils.converters

Contains the various converter functions available within
the `prompted` package
"""

import json
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

from ..common.cache import cached, make_hashable
from ..types.chat_completions import (
    Message,
    MessageRole,
    MessageContentPart,
    MessageContentTextPart,
    Tool,
)
from .formatting import format_to_markdown
from .identification import is_message


def convert_to_message(
    message: Any,
    role: MessageRole | str = "user",
    markdown: bool = False,
    use_parts: bool = False,
    schema: bool = False,
) -> Message:
    """
    Converts a given object into a Chat Completions compatible
    `Message` object.

    Args:
        message : Any
            The object to convert.
        role : MessageRole | str
            The role of the message.
        markdown : bool
            Whether to use markdown.
        use_parts : bool
            Whether to use message content parts.
        schema : bool
            If True, only show schema. If False, show values for initialized objects.

    Returns:
        Message
            The converted message.
    """

    @cached(
        lambda message,
        role="user",
        markdown=False,
        use_parts=False,
        schema=False: make_hashable(
            (message, role, markdown, use_parts, schema)
        )
    )
    def _convert_to_message(
        message: Any,
        role: MessageRole | str = "user",
        markdown: bool = False,
        use_parts: bool = False,
        schema: bool = False,
    ) -> Message:
        if is_message(message):
            return message

        if markdown:
            if not isinstance(message, str):
                try:
                    message = format_to_markdown(message, schema=schema)
                except Exception as e:
                    raise ValueError(
                        f"Error converting object to markdown: {e}"
                    )
        else:
            if not isinstance(message, str):
                try:
                    message = json.dumps(message)
                except Exception as e:
                    raise ValueError(
                        f"Error converting object to JSON: {e}"
                    )

        if use_parts:
            message = MessageContentTextPart(type="text", text=message)
        return Message(
            role=role, content=message if not use_parts else [message]
        )

    return _convert_to_message(message, role, markdown, use_parts, schema)
