"""
💭 chatspec.types

Core type definitions for chat completions.
"""

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Literal,
    TypeAlias,
    Any,
    Iterable,
)
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------------


class MessageContentImage(TypedDict):
    """
    A dictionary representing parameters or data referring to an image for
    multi-modal completions.
    """

    class ImageURL(TypedDict):
        url: str
        detail: Literal["auto", "low", "high"]

    image_url: ImageURL
    type: Literal["image_url"]


class MessageContentText(TypedDict):
    """
    A dictionary representing a text message.
    """

    text: str
    type: Literal["text"]


class MessageContentAudio(TypedDict):
    """
    A dictionary representing parameters or data referring to an audio file
    for multi-modal completions.
    """

    class InputAudio(TypedDict):
        data: str
        format: Literal["wav", "mp3"]

    input_audio: InputAudio
    type: Literal["input_audio"]
    
    

MessageContent = Union[MessageContentImage, MessageContentText, MessageContentAudio]


# developer is openai specific and
# only supported by their reasoning models
MessageRole = Literal["assistant", "user", "system", "tool", "developer"]


class Message(TypedDict):
    """Core message type for chat completions."""

    role: MessageRole
    content: Optional[Union[Iterable[MessageContent], str]]
    name: NotRequired[str]
    tool_calls: NotRequired[List[Dict[str, Any]]]
    tool_call_id: NotRequired[str]


Messages = List[Message]


# ----------------------------------------------------------------------------
# Tools & Functions
# ----------------------------------------------------------------------------


class FunctionDefinition(TypedDict):
    """Core function definition for tool calling."""

    name: str
    description: str
    parameters: Dict[str, Any]
    strict: NotRequired[bool]


class Tool(TypedDict):
    """Core tool type for chat completions."""

    type: Literal["function"]
    function: FunctionDefinition


# ----------------------------------------------------------------------------
# Common Parameters
# ----------------------------------------------------------------------------


ResponseFormat = Literal["text", "json_object", "json_schema"]
ServiceTier = Literal["auto", "default", "scale"]
ToolChoice = Literal["auto", "none", "required"]
Modality = Literal["text", "audio"]


# ----------------------------------------------------------------------------
# Helper Type Aliases
# ----------------------------------------------------------------------------


Metadata = Dict[str, str]
Temperature = float  # 0.0 to 2.0
TopP = float  # 0.0 to 1.0
MaxTokens = int
Stop = Union[str, List[str]]


# ----------------------------------------------------------------------------
# Response Types
# ----------------------------------------------------------------------------


class Logprobs(BaseModel):
    """Core logprobs type for chat completions."""

    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]


class Choice(BaseModel):
    """Core choice type for chat completions."""

    message: Message
    finish_reason: Literal["stop", "length", "tool_calls"]
    index: int
    logprobs: Optional[Logprobs]


class ChatCompletion(BaseModel):
    """Core chat completion type."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]


# Streaming


class ChunkChoice(BaseModel):
    """Core chunk choice type for chat completions."""

    delta: Message
    finish_reason: Literal["stop", "length", "tool_calls"]
    index: int
    logprobs: Optional[Logprobs]


class ChatCompletionChunk(BaseModel):
    """Core chat completion chunk type."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]


# ----------------------------------------------------------------------------
# Params
# ----------------------------------------------------------------------------


class Params(TypedDict):
    """
    Common parameter types for chat completions.
    """

    model: str
    messages: Messages
    api_key: Optional[str]
    base_url: Optional[str]
    organization: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_completion_tokens: Optional[int]
    stop: Optional[Stop]
    stream: Optional[bool]


__all__ = [
    # Core Types
    "Message",
    "Messages",
    "MessageContentImage",
    "MessageContentText",
    "MessageContentAudio",
    "MessageContent",
    "MessageRole",
    "Tool",
    "FunctionDefinition",
    # Parameters
    "ResponseFormat",
    "ServiceTier",
    "ToolChoice",
    "Modality",
    # Helpers
    "Metadata",
    "Temperature",
    "TopP",
    "MaxTokens",
    "Stop",
    # Response Types
    "Logprobs",
    "Choice",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChunkChoice",
    # Params
    "Params",
]
