"""
## 💭 prompted

[`hammad saeed`](https://github.com/hsaeed3) | 2025
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

from .common.logger import (
    # NOTE: EXPORTED
    verbosity,
    setup_logging as _setup_logging,
)

_setup_logging()


if TYPE_CHECKING:
    # AGENTS! WOOHOOOO!!!
    from .agents.agent import Agent, create_agent

    # FUNCTION TOOLS! WOOOHOOOOOOO!
    from .agents.agent_tool import AgentTool, agent_tool

    from .create import (
        create_from_attributes,
        create_from_function,
        create_from_options,
        create_from_prompt,
        create_from_schema,
        async_create_from_attributes,
        async_create_from_options,
        async_create_from_prompt,
        async_create_from_schema,
    )

    from .utils.identification import (
        is_completion,
        is_stream,
        is_message,
        is_tool,
        has_system_prompt,
        has_tool_call,
        has_specific_tool_call,
    )

    from .utils.mock import (
        mock_completion,
        mock_embedding,
        amock_completion as async_mock_completion,
        amock_embedding as async_mock_embedding,
    )

    from .utils.converters import (
        convert_completion_to_pydantic_model,
        convert_stream_to_completion,
        convert_stream_to_message,
        convert_to_boolean_model,
        convert_to_field,
        convert_to_image_message,
        convert_to_input_audio_message,
        convert_to_message,
        convert_to_pydantic_model,
        convert_to_selection_model,
        convert_to_tool_definition,
        convert_to_tool_definitions,
        convert_completion_to_tool_calls,
    )

    from .utils.formatting import (
        format_to_markdown,
        format_messages,
        format_system_prompt,
        format_docstring,
        get_type_name,
    )

    from .utils.streaming import stream_passthrough

    from .types import (
        MethodType,
        FunctionParameters,
        Function,
        Tool,
        FunctionCall,
        ToolCall,
        MessageContentImageURL,
        MessageContentImagePart,
        MessageContentAudioPart,
        MessageContentTextPart,
        MessageContentPart,
        MessageContent,
        MessageTextContent,
        MessageRole,
        Message,
        Subscriptable,
        TopLogprob,
        TokenLogprob,
        ChoiceLogprobs,
        CompletionFunction,
        CompletionToolCall,
        CompletionMessage,
        CompletionUsage,
        Completion,
        CompletionChunk,
        Embedding,
        EmbeddingsUsage,
        InstructorModeParam,
        MessagesParam,
        ChatModel,
        ModelParam,
        BaseURLParam,
        FunctionCallParam,
        ToolChoiceParam,
        ToolChoiceNamedTool,
        ModalitiesParam,
        PredictionParam,
        AudioParam,
        ReasoningEffortParam,
        ResponseFormatParam,
        StreamOptionsParam,
        ClientParams,
        EmbeddingParams,
        CompletionParams,
        Params,
    )


IMPORT_MAP: Dict[str, Tuple[str, str]] = {
    # AGENTS!
    "Agent": (".agents.agent", "Agent"),
    "create_agent": (".agents.agent", "create_agent"),
    # FUNCTION TOOLS!
    "AgentTool": (".agents.agent_tool", "AgentTool"),
    "agent_tool": (".agents.agent_tool", "agent_tool"),
    # LOGGING
    "verbosity": (".common.logger", "verbosity"),
    # TYPES
    "MethodType": (".types", "MethodType"),
    "FunctionParameters": (
        ".types.chat_completions",
        "FunctionParameters",
    ),
    "Function": (".types.chat_completions", "Function"),
    "Tool": (".types.chat_completions", "Tool"),
    "FunctionCall": (".types.chat_completions", "FunctionCall"),
    "ToolCall": (".types.chat_completions", "ToolCall"),
    "MessageContentImageURL": (
        ".types.chat_completions",
        "MessageContentImageURL",
    ),
    "MessageContentImagePart": (
        ".types.chat_completions",
        "MessageContentImagePart",
    ),
    "MessageContentAudioPart": (
        ".types.chat_completions",
        "MessageContentAudioPart",
    ),
    "MessageContentTextPart": (
        ".types.chat_completions",
        "MessageContentTextPart",
    ),
    "MessageContentPart": (
        ".types.chat_completions",
        "MessageContentPart",
    ),
    "MessageContent": (".types.chat_completions", "MessageContent"),
    "MessageTextContent": (
        ".types.chat_completions",
        "MessageTextContent",
    ),
    "MessageRole": (".types.chat_completions", "MessageRole"),
    "Message": (".types.chat_completions", "Message"),
    "Subscriptable": (".types.chat_completions", "Subscriptable"),
    "TopLogprob": (".types.chat_completions", "TopLogprob"),
    "TokenLogprob": (".types.chat_completions", "TokenLogprob"),
    "ChoiceLogprobs": (".types.chat_completions", "ChoiceLogprobs"),
    "CompletionFunction": (
        ".types.chat_completions",
        "CompletionFunction",
    ),
    "CompletionToolCall": (
        ".types.chat_completions",
        "CompletionToolCall",
    ),
    "Completion": (".types.chat_completions", "Completion"),
    "CompletionChunk": (".types.chat_completions", "CompletionChunk"),
    "Embedding": (".types.chat_completions", "Embedding"),
    "EmbeddingsUsage": (".types.chat_completions", "EmbeddingsUsage"),
    # INSTRUCTOR MODE
    "InstructorModeParam": (".types.instructor", "InstructorModeParam"),
    # PARAMS
    "MessagesParam": (".types.params", "MessagesParam"),
    "ChatModel": (".types.params", "ChatModel"),
    "ModelParam": (".types.params", "ModelParam"),
    "BaseURLParam": (".types.params", "BaseURLParam"),
    "FunctionCallParam": (".types.params", "FunctionCallParam"),
    "ToolChoiceParam": (".types.params", "ToolChoiceParam"),
    "ToolChoiceNamedTool": (".types.params", "ToolChoiceNamedTool"),
    "ModalitiesParam": (".types.params", "ModalitiesParam"),
    "PredictionParam": (".types.params", "PredictionParam"),
    "AudioParam": (".types.params", "AudioParam"),
    "ReasoningEffortParam": (".types.params", "ReasoningEffortParam"),
    "ResponseFormatParam": (".types.params", "ResponseFormatParam"),
    "StreamOptionsParam": (".types.params", "StreamOptionsParam"),
    "ClientParams": (".types.params", "ClientParams"),
    "EmbeddingParams": (".types.params", "EmbeddingParams"),
    "CompletionParams": (".types.params", "CompletionParams"),
    "Params": (".types.params", "Params"),
    # UTILS - STREAMING
    "stream_passthrough": (".utils.streaming", "stream_passthrough"),
    # UTILS - IDENTIFICATION
    "is_completion": (".utils.identification", "is_completion"),
    "is_stream": (".utils.identification", "is_stream"),
    "is_message": (".utils.identification", "is_message"),
    "is_tool": (".utils.identification", "is_tool"),
    "has_system_prompt": (".utils.identification", "has_system_prompt"),
    "has_tool_call": (".utils.identification", "has_tool_call"),
    "has_specific_tool_call": (
        ".utils.identification",
        "has_specific_tool_call",
    ),
    # UTILS - CONVERTERS
    "convert_completion_to_pydantic_model": (
        ".utils.converters",
        "convert_completion_to_pydantic_model",
    ),
    "convert_stream_to_completion": (
        ".utils.converters",
        "convert_stream_to_completion",
    ),
    "convert_stream_to_message": (
        ".utils.converters",
        "convert_stream_to_message",
    ),
    "convert_to_boolean_model": (
        ".utils.converters",
        "convert_to_boolean_model",
    ),
    "convert_to_field": (".utils.converters", "convert_to_field"),
    "convert_to_image_message": (
        ".utils.converters",
        "convert_to_image_message",
    ),
    "convert_to_input_audio_message": (
        ".utils.converters",
        "convert_to_input_audio_message",
    ),
    "convert_to_message": (".utils.converters", "convert_to_message"),
    "convert_to_pydantic_model": (
        ".utils.converters",
        "convert_to_pydantic_model",
    ),
    "convert_to_selection_model": (
        ".utils.converters",
        "convert_to_selection_model",
    ),
    "convert_to_tool_definition": (
        ".utils.converters",
        "convert_to_tool_definition",
    ),
    "convert_to_tool_definitions": (
        ".utils.converters",
        "convert_to_tool_definitions",
    ),
    "convert_completion_to_tool_calls": (
        ".utils.converters",
        "convert_completion_to_tool_calls",
    ),
    # UTILS - FORMATTING
    "format_to_markdown": (".utils.formatting", "format_to_markdown"),
    "format_messages": (".utils.formatting", "format_messages"),
    "format_system_prompt": (".utils.formatting", "format_system_prompt"),
    "format_docstring": (".utils.formatting", "format_docstring"),
    "get_type_name": (".utils.formatting", "get_type_name"),
    # UTILS - MOCK
    "mock_completion": (".utils.mock", "mock_completion"),
    "mock_embedding": (".utils.mock", "mock_embedding"),
    "async_mock_completion": (".utils.mock", "async_mock_completion"),
    "async_mock_embedding": (".utils.mock", "async_mock_embedding"),
    # CREATE
    "create_from_attributes": (".create", "create_from_attributes"),
    "create_from_function": (".create", "create_from_function"),
    "create_from_options": (".create", "create_from_options"),
    "create_from_prompt": (".create", "create_from_prompt"),
    "create_from_schema": (".create", "create_from_schema"),
    "async_create_from_attributes": (
        ".create",
        "async_create_from_attributes",
    ),
    "async_create_from_options": (".create", "async_create_from_options"),
    "async_create_from_prompt": (".create", "async_create_from_prompt"),
    "async_create_from_schema": (".create", "async_create_from_schema"),
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


if sys.version_info >= (3, 7):
    __getattr__.__module__ = __name__


__all__ = [
    "Agent",
    "create_agent",
    "AgentTool",
    "agent_tool",
    "verbosity",
    "MethodType",
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
    "stream_passthrough",
    "is_completion",
    "is_stream",
    "is_message",
    "is_tool",
    "has_system_prompt",
    "has_tool_call",
    "has_specific_tool_call",
    "convert_completion_to_pydantic_model",
    "convert_stream_to_completion",
    "convert_stream_to_message",
    "convert_to_boolean_model",
    "convert_to_field",
    "convert_to_image_message",
    "convert_to_input_audio_message",
    "convert_to_message",
    "convert_to_pydantic_model",
    "convert_to_selection_model",
    "convert_to_tool_definition",
    "convert_to_tool_definitions",
    "convert_completion_to_tool_calls",
    "format_to_markdown",
    "format_messages",
    "format_system_prompt",
    "format_docstring",
    "get_type_name",
    "mock_completion",
    "mock_embedding",
    "async_mock_completion",
    "async_mock_embedding",
    "create_from_attributes",
    "create_from_function",
    "create_from_options",
    "create_from_prompt",
    "create_from_schema",
    "async_create_from_attributes",
    "async_create_from_options",
    "async_create_from_prompt",
    "async_create_from_schema",
]
