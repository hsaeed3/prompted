"""
prompted.resources.specifications.openai.types.messages

Contains type definitions used to define messages in the
OpenAI specification within the `prompted` framework.
"""

from typing import Any, List, Union, Literal
from typing_extensions import Required, TypedDict, NotRequired, TypeAliasType

from .tool_calls import OpenAIToolCall

__all__ = [
    "OpenAIMessageRole",
    "OpenAIImageUrl",
    "OpenAIInputAudio",
    "OpenAIAudioUrl",
    "OpenAIMessageContentImagePart",
    "OpenAIMessageContentAudioUrlPart",
    "OpenAIMessageContentInputAudioPart",
    "OpenAIMessageContentTextPart",
    "OpenAIMessageContentRefusalPart",
    "OpenAIMessageContentPart",
    "OpenAISystemMessage",
    "OpenAIUserMessage",
    "OpenAIToolMessage",
    "OpenAIAssistantMessage",
    "OpenAIMessage",
]


# ------------------------------------------------------------------------------
# Generic Message Types
# ------------------------------------------------------------------------------


OpenAIMessageRole = TypeAliasType(
    "OpenAIMessageRole", Literal["user", "assistant", "tool", "system", "developer"]
)
"""
Type definition for accepted message roles within a message in the OpenAI chat completions
specification.
"""


class OpenAIImageUrl(TypedDict):
    """
    Represents the url to an image defined within a message content
    part.
    """

    url: Required[str]
    """The URL of the image."""
    detail: NotRequired[Literal["auto", "low", "high"]]
    """
    The detail level of this image.

    - `auto` lets the model decide.
    - `low` uses fewer tokens.
    - `high` uses more tokens.

    Defaults to `auto` (behaviorally, not enforced by TypedDict).
    """


class OpenAIInputAudio(TypedDict):
    """
    Definition for an audio input usable within chat completion
    message content parts.
    """

    data: Required[str]
    """The base64 encoded audio data."""
    format: NotRequired[Literal["wav", "mp3", "opus", "flac", "aac", "m4a"]]
    """The format of the audio data."""


class OpenAIAudioUrl(TypedDict):
    """
    Definition for an audio input usable within chat completion
    message content parts.
    """

    url: Required[str]
    """The URL of the audio data."""


# ------------------------------------------------------------------------------
# Message Content Parts
# ------------------------------------------------------------------------------


class OpenAIMessageContentImagePart(TypedDict):
    """
    A part within a chat completion message that represents/contains image
    content.
    """

    type: Required[Literal["image_url"]]
    """The type of this content part."""
    image_url: Required[OpenAIImageUrl]
    """
    The image to be sent / displayed. Represented as a
    URL.
    """


class OpenAIMessageContentAudioUrlPart(TypedDict):
    """
    A part within a chat completion message that represents/contains audio
    content.

    NOTE: Only one of `audio_url` or `input_audio` can be used by the producer.
    """

    type: Required[Literal["audio_url"]]
    """The type of this content part."""
    audio_url: Required[OpenAIAudioUrl]
    """
    The URL of the audio data.
    """


class OpenAIMessageContentInputAudioPart(TypedDict):
    """
    A part within a chat completion message that represents/contains audio
    content.

    NOTE: Only one of `audio_url` or `input_audio` can be used by the producer.
    """

    type: Required[Literal["input_audio"]]
    """The type of this content part."""
    input_audio: Required[OpenAIInputAudio]
    """
    The base64 encoded audio data.
    """


class OpenAIMessageContentTextPart(TypedDict):
    """
    A part within a chat completion message that represents/contains text
    content.
    """

    type: Required[Literal["text"]]
    """The type of this content part."""
    text: Required[str]
    """The text of the message."""


class OpenAIMessageContentRefusalPart(TypedDict):
    """
    A part within a chat completion message that represents/contains refusal
    content.

    NOTE: This is a special case, only assistant messages can return
    or use refusal content parts.
    """

    type: Required[Literal["refusal"]]
    """The type of this content part."""
    refusal: Required[str]
    """The refusal message by the assistant."""


OpenAIMessageContentPart = TypeAliasType(
    "OpenAIMessageContentPart",
    Union[
        OpenAIMessageContentImagePart,
        OpenAIMessageContentAudioUrlPart,
        OpenAIMessageContentInputAudioPart,
        OpenAIMessageContentTextPart,
        OpenAIMessageContentRefusalPart,
    ],
)
"""
A part within a chat completion message that represents/contains content.
The 'type' field serves as the discriminator.
"""


# ------------------------------------------------------------------------------
# Message Types
# ------------------------------------------------------------------------------


class OpenAISystemMessage(TypedDict):
    """
    A system message in the OpenAI chat completions specification.
    """

    role: Required[Literal["system"]]
    """The role of this message. (Always `system`)"""
    content: Required[Union[str, List[OpenAIMessageContentPart]]]
    """
    The content of this message.
    """
    name: NotRequired[str]
    """An optional name for the participant."""
    # model_config = {"extra": "allow"} from Pydantic is not directly translatable.
    # TypedDict does not enforce this at runtime in the same way.


class OpenAIUserMessage(TypedDict):
    """
    A user message in the OpenAI chat completions specification.
    """

    role: Required[Literal["user"]]
    """The role of this message. (Always `user`)"""
    content: Required[Union[str, List[OpenAIMessageContentPart]]]
    """
    The content of this message.
    NOTE: User messages can not have 'None' message content.
    """
    name: NotRequired[str]
    """An optional name for the participant."""


class OpenAIToolMessage(TypedDict):
    """
    A tool message in the OpenAI chat completions specification.
    """

    role: Required[Literal["tool"]]
    """The role of this message. (Always `tool`)"""
    content: Required[Union[str, List[OpenAIMessageContentTextPart]]]
    """The content of this message.
    Can be a string or a list of message content **TEXT** parts."""
    tool_call_id: Required[str]
    """The ID of the tool call that this message is responding to."""
    name: NotRequired[str]
    """
    Provides the model information to differentiate between participants of the same
    role.
    """


class OpenAIAssistantMessage(TypedDict):
    """
    An assistant message in the OpenAI chat completions specification.
    """

    role: Required[Literal["assistant"]]
    """The role of this message. (Always `assistant`)"""
    content: Required[
        Union[
            str,
            List[Union[OpenAIMessageContentTextPart, OpenAIMessageContentRefusalPart]],
        ]
    ]
    """The content of this message.
    
    Can be a string or a list of message content parts.
    """
    refusal: NotRequired[str]
    """The refusal message by the assistant."""

    function_call: NotRequired[Any]
    """Function call response.
    NOTE:
    This is deprecated in favor of `tool_calls`.
    """
    tool_calls: NotRequired[List[OpenAIToolCall]]
    """Tool calls created by the assistant."""
    name: NotRequired[str]
    """
    Provides the model information to differentiate between participants of the same
    role.
    """


OpenAIMessage = TypeAliasType(
    "OpenAIMessage",
    Union[
        OpenAISystemMessage,
        OpenAIUserMessage,
        OpenAIToolMessage,
        OpenAIAssistantMessage,
    ],
)
"""
A message in the OpenAI chat completions specification.
The 'role' field serves as the discriminator.
"""
