"""
💬 prompted.utils.streaming

This module provides a `stream_passthrough` function that wraps a chat completion stream within a cached object that can be iterated and consumed over multiple times.

It supports both synchronous and asynchronous streams.
"""

import logging
from typing import (
    Any,
    Iterable,
    List,
)
from ..types.chat_completions import (
    CompletionChunk,
    CompletionMessage,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Streaming
#
# 'prompted' builds in a `passthrough` functionality, which caches response chunks
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


__all__ = [
    "stream_passthrough",
]
