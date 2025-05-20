"""
💭 prompted.types

Contains type definitions, models and light utilities for types 
found within the `Chat Completions`, `Anthropic Model Context Protocol`
and `Google A2A` specifications.
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_completions import *
    from .chat_completions_params import *

__all__ = [
    "FunctionParameters",
    "Function",
    "Tool",
    "FunctionCall",
    "ToolCall",
    "MessageContentImageURL",
    "MessageContentImagePart",
    "MessageContentAudioPart",
    "MessageContentTextPart",
    "MessageContentPart",
    "MessageContent",
    "MessageTextContent",
    "MessageRole",
    "Message",
    "Subscriptable",
    "TopLogprob",
    "TokenLogprob",
    "ChoiceLogprobs",
    "CompletionFunction",
    "CompletionToolCall",
    "CompletionMessage",
    "CompletionUsage",
    "Completion",
    "CompletionChunk",
    "Embedding",
    "EmbeddingsUsage", 

    "InstructorModeParam",
    "MessagesParam",
    "ChatModel",
    "ModelParam",
    "BaseURLParam",
    "FunctionCallParam", 
    "ToolChoiceParam",
    "ToolChoiceNamedTool",
    "ModalitiesParam",
    "PredictionParam",
    "AudioParam",
    "ReasoningEffortParam",
    "ResponseFormatParam",
    "StreamOptionsParam",
    "ClientParams",
    "EmbeddingParams",
    "CompletionParams",
    "Params",   
]

IMPORT_MAP : Dict[str, Tuple[str, str]] = {
    # chat completions
    "FunctionParameters": (".chat_completions", "FunctionParameters"),
    "Function": (".chat_completions", "Function"),
    "Tool": (".chat_completions", "Tool"),
    "FunctionCall": (".chat_completions", "FunctionCall"),
    "ToolCall": (".chat_completions", "ToolCall"),
    "MessageContentImageURL": (".chat_completions", "MessageContentImageURL"),
    "MessageContentImagePart": (".chat_completions", "MessageContentImagePart"),
    "MessageContentAudioPart": (".chat_completions", "MessageContentAudioPart"),
    "MessageContentTextPart": (".chat_completions", "MessageContentTextPart"),
    "MessageContentPart": (".chat_completions", "MessageContentPart"),
    "MessageContent": (".chat_completions", "MessageContent"),
    "MessageTextContent": (".chat_completions", "MessageTextContent"),
    "MessageRole": (".chat_completions", "MessageRole"),
    "Message": (".chat_completions", "Message"),
    "Subscriptable": (".chat_completions", "Subscriptable"),
    "TopLogprob": (".chat_completions", "TopLogprob"),
    "TokenLogprob": (".chat_completions", "TokenLogprob"),
    "ChoiceLogprobs": (".chat_completions", "ChoiceLogprobs"),
    "CompletionFunction": (".chat_completions", "CompletionFunction"),
    "CompletionToolCall": (".chat_completions", "CompletionToolCall"),
    "CompletionMessage": (".chat_completions", "CompletionMessage"),
    "CompletionUsage": (".chat_completions", "CompletionUsage"),
    "Completion": (".chat_completions", "Completion"),
    "CompletionChunk": (".chat_completions", "CompletionChunk"),
    "Embedding": (".chat_completions", "Embedding"),
    "EmbeddingsUsage": (".chat_completions", "EmbeddingsUsage"),

    # chat completions params
    "InstructorModeParam": (".chat_completions_params", "InstructorModeParam"),
    "MessagesParam": (".chat_completions_params", "MessagesParam"), 
    "ChatModel": (".chat_completions_params", "ChatModel"),
    "ModelParam": (".chat_completions_params", "ModelParam"),
    "BaseURLParam": (".chat_completions_params", "BaseURLParam"),
    "FunctionCallParam": (".chat_completions_params", "FunctionCallParam"),
    "ToolChoiceParam": (".chat_completions_params", "ToolChoiceParam"),
    "ToolChoiceNamedTool": (".chat_completions_params", "ToolChoiceNamedTool"),
    "ModalitiesParam": (".chat_completions_params", "ModalitiesParam"),
    "PredictionParam": (".chat_completions_params", "PredictionParam"),
    "AudioParam": (".chat_completions_params", "AudioParam"),
    "ReasoningEffortParam": (".chat_completions_params", "ReasoningEffortParam"),
    "ResponseFormatParam": (".chat_completions_params", "ResponseFormatParam"),
    "StreamOptionsParam": (".chat_completions_params", "StreamOptionsParam"),
    "ClientParams": (".chat_completions_params", "ClientParams"),
    "CompletionParams": (".chat_completions_params", "CompletionParams"),
    "EmbeddingParams": (".chat_completions_params", "EmbeddingParams"),
    "CompletionParams": (".chat_completions_params", "CompletionParams"),
    "Params": (".chat_completions_params", "Params"),
}


def __getattr__(name: str) -> Any:
    """Handle dynamic imports for module attributes."""
    if name in IMPORT_MAP:
        module_path, attr_name = IMPORT_MAP[name]
        module = import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    """Return list of module attributes for auto-completion."""
    return list(__all__)


# Set module attribute for __getattr__ in Python 3.7+
if sys.version_info >= (3, 7):
    __getattr__.__module__ = __name__