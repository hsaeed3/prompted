"""
prompted.messages

Contains the primary `Message` and `MessageList` classes used to represent
user -> agent messages within the `prompted` framework.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar, 
    Dict,
    List,
    Literal,
    Optional,
    Union
)
from typing_extensions import TypeAliasType

from ._cache import cached, make_hashable
from .logger import _get_logger

logger = _get_logger(__name__)

__all__ = [
    "Message",
    "MessageList",
]


# ------------------------------------------------------------------------------
# Message Types & Models
# ------------------------------------------------------------------------------


MessageRoleType = TypeAliasType(
    "MessageRoleType",
    Literal["user", "assistant", "system", "tool"] | str
)


@dataclass
class MessageContentImagePart:
    """
    Represents an image part of a message.
    """
    __prompted_origin__ : ClassVar[str] = "MessageContentImagePart"

    data : str | bytes
    """
    String or base64 encoded byte representation of the image.
    """
    detail : Optional[Literal["auto", "low", "high"]] = "auto"
    """
    The detail level of the image.
    """


@dataclass
class MessageContentInputAudioPart:
    """
    Represents an input audio part of a message.
    """
    __prompted_origin__ : ClassVar[str] = "MessageContentInputAudioPart"

    data : str | bytes
    """
    String or base64 encoded byte representation of the audio data."""
    format : Optional[Literal["wav", "mp3", "opus", "flac", "aac", "m4a"]] = "wav"
    """
    The format of the audio data.
    """


@dataclass
class MessageContentAudioUrlPart:
    """
    Represents an audio URL part of a message.
    """
    __prompted_origin__ : ClassVar[str] = "MessageContentAudioUrlPart"

    url : str
    """
    The URL of the audio data.
    """


@dataclass
class MessageContentTextPart:
    """
    Represents a text part of a message.
    """
    __prompted_origin__ : ClassVar[str] = "MessageContentTextPart"

    text : str
    """
    The text of the message.
    """


@dataclass
class MessageContentRefusalPart:
    """
    Represents a refusal part of a message.
    """
    __prompted_origin__ : ClassVar[str] = "MessageContentRefusalPart"

    refusal : str
    """
    The refusal message.
    """


