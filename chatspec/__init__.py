"""
## 💭 chatspec

Types & utilities built for processing & interacting with objects used in
the chat completions API specification.

### Package Contents:

- `markdown.py`: Utilities for rendering objects as markdown.
- `params.py`: Types for the parameters used in chat completions.
- `types.py`: Types for the responses from chat completions.
- `mock.py`: A mocked implementation of the OpenAI client for chat completions.
- `utils.py`: Utilities for processing & interacting with chat completions.

---

[`hammad saeed`](https://github.com/hsaeed3) | 2025
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Import everything here for type checking and IDE support
    from .markdown import (
        markdownify,
        _format_docstring as format_docstring,
    )
    from .mock import (
        AI,
        AsyncAI,
        mock_completion,
        mock_embedding,
        amock_completion,
        amock_embedding,
        AIError,
    )
    from .types import (
        FunctionParameters,
        Function,
        Tool,
        FunctionCall,
        ToolCall,
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
        Completion,
        CompletionChunk,
        Embedding,
    )
    from .params import (
        InstructorModeParam,
        MessagesParam,
        ChatModel,
        ModelParam,
        BaseURLParam,
        FunctionCallParam,
        ToolChoiceParam,
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
        to_client_params,
        to_completion_params,
    )
    from .utils import (
        is_completion,
        is_stream,
        is_message,
        is_tool,
        has_system_prompt,
        has_tool_call,
        was_tool_called,
        run_tool,
        create_tool_message,
        create_image_message,
        create_input_audio_message,
        get_tool_calls,
        dump_stream_to_message,
        dump_stream_to_completion,
        normalize_messages,
        normalize_system_prompt,
        create_field_mapping,
        extract_function_fields,
        convert_to_pydantic_model,
        convert_to_tools,
        convert_to_tool,
        create_literal_pydantic_model,
        stream_passthrough,
        create_selection_model,
        create_bool_model,
    )

__all__ = (
    # markdown
    "markdownify",
    "format_docstring",
    # mock
    "ChatCompletion",
    "AI",
    "AsyncAI",
    "mock_completion",
    "mock_embedding",
    "amock_completion",
    "amock_embedding",
    "AIError",
    # types
    "Message",
    "MessageRole",
    "MessageContent",
    "MessageTextContent",
    "MessageContentPart",
    "MessageContentTextPart",
    "MessageContentImagePart",
    "MessageContentAudioPart",
    "Completion",
    "CompletionMessage",
    "CompletionChunk",
    "Tool",
    "Function",
    "FunctionCall",
    "ToolCall",
    "FunctionParameters",
    "EmbeddingResponse",
    "EmbeddingData",
    "Embedding",
    "Subscriptable",
    "TopLogprob",
    "TokenLogprob",
    "ChoiceLogprobs",
    "CompletionFunction",
    "CompletionToolCall",
    # params
    "ModelParam",
    "BaseURLParam",
    "MessagesParam",
    "ChatModel",
    "InstructorModeParam",
    "FunctionCallParam",
    "ToolChoiceParam",
    "ModalitiesParam",
    "PredictionParam",
    "AudioParam",
    "ReasoningEffortParam",
    "ResponseFormatParam",
    "StreamOptionsParam",
    "ClientParams",
    "CompletionParams",
    "EmbeddingParams",
    "Params",
    "to_client_params",
    "to_completion_params",
    # utils
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
    "normalize_messages",
    "normalize_system_prompt",
    "create_field_mapping",
    "extract_function_fields",
    "convert_to_pydantic_model",
    "convert_to_tools",
    "convert_to_tool",
    "create_literal_pydantic_model",
    "stream_passthrough",
    # aliases for backward compatibility
    "add_audio_to_message",
    "create_selection_model",
    "get_content",
    "passthrough",
    "create_bool_model",
    "create_selection_model",
)

# Dynamic imports mapping
_dynamic_imports: Dict[str, Tuple[str, str]] = {
    # markdown
    "markdownify": (".markdown", "markdownify"),
    "format_docstring": (".markdown", "_format_docstring"),
    # mock
    "ChatCompletion": (".mock", "AI"),
    "AI": (".mock", "AI"),
    "AsyncAI": (".mock", "AsyncAI"),
    "mock_completion": (".mock", "mock_completion"),
    "mock_embedding": (".mock", "mock_embedding"),
    "amock_completion": (".mock", "amock_completion"),
    "amock_embedding": (".mock", "amock_embedding"),
    "AIError": (".mock", "AIError"),
    # types
    "Message": (".types", "Message"),
    "MessageRole": (".types", "MessageRole"),
    "MessageContent": (".types", "MessageContent"),
    "MessageTextContent": (".types", "MessageTextContent"),
    "MessageContentPart": (".types", "MessageContentPart"),
    "MessageContentTextPart": (".types", "MessageContentTextPart"),
    "MessageContentImagePart": (".types", "MessageContentImagePart"),
    "MessageContentAudioPart": (".types", "MessageContentAudioPart"),
    "Completion": (".types", "Completion"),
    "CompletionMessage": (".types", "CompletionMessage"),
    "CompletionChunk": (".types", "CompletionChunk"),
    "Tool": (".types", "Tool"),
    "Function": (".params", "Function"),
    "FunctionCall": (".types", "FunctionCall"),
    "ToolCall": (".types", "ToolCall"),
    "FunctionParameters": (".types", "FunctionParameters"),
    "EmbeddingResponse": (".types", "Embedding"),
    "EmbeddingData": (".types", "Embedding"),
    "Embedding": (".types", "Embedding"),
    "Subscriptable": (".types", "Subscriptable"),
    "TopLogprob": (".types", "TopLogprob"),
    "TokenLogprob": (".types", "TokenLogprob"),
    "ChoiceLogprobs": (".types", "ChoiceLogprobs"),
    "CompletionFunction": (".types", "CompletionFunction"),
    "CompletionToolCall": (".types", "CompletionToolCall"),
    # params
    "ModelParam": (".params", "ModelParam"),
    "BaseURLParam": (".params", "BaseURLParam"),
    "MessagesParam": (".params", "MessagesParam"),
    "ChatModel": (".params", "ModelParam"),
    "ClientParams": (".params", "Params"),
    "CompletionParams": (".params", "Params"),
    "EmbeddingParams": (".params", "Params"),
    "Params": (".params", "Params"),
    "InstructorModeParam": (".params", "InstructorModeParam"),
    "FunctionCallParam": (".params", "FunctionCallParam"),
    "ToolChoiceParam": (".params", "ToolChoiceParam"),
    "ModalitiesParam": (".params", "ModalitiesParam"),
    "PredictionParam": (".params", "PredictionParam"),
    "AudioParam": (".params", "AudioParam"),
    "ReasoningEffortParam": (".params", "ReasoningEffortParam"),
    "ResponseFormatParam": (".params", "ResponseFormatParam"),
    "StreamOptionsParam": (".params", "StreamOptionsParam"),
    "to_client_params": (".params", "to_client_params"),
    "to_completion_params": (".params", "to_completion_params"),
    # utils
    "is_completion": (".utils", "is_completion"),
    "is_stream": (".utils", "is_stream"),
    "is_message": (".utils", "is_message"),
    "is_tool": (".utils", "is_tool"),
    "has_system_prompt": (".utils", "has_system_prompt"),
    "has_tool_call": (".utils", "has_tool_call"),
    "was_tool_called": (".utils", "was_tool_called"),
    "run_tool": (".utils", "run_tool"),
    "create_tool_message": (".utils", "create_tool_message"),
    "create_image_message": (".utils", "create_image_message"),
    "create_input_audio_message": (".utils", "create_input_audio_message"),
    "get_tool_calls": (".utils", "get_tool_calls"),
    "dump_stream_to_message": (".utils", "dump_stream_to_message"),
    "dump_stream_to_completion": (".utils", "dump_stream_to_completion"),
    "normalize_messages": (".utils", "normalize_messages"),
    "normalize_system_prompt": (".utils", "normalize_system_prompt"),
    "create_field_mapping": (".utils", "create_field_mapping"),
    "extract_function_fields": (".utils", "extract_function_fields"),
    "convert_to_pydantic_model": (".utils", "convert_to_pydantic_model"),
    "convert_to_tools": (".utils", "convert_to_tools"),
    "convert_to_tool": (".utils", "convert_to_tool"),
    "create_literal_pydantic_model": (
        ".utils",
        "create_literal_pydantic_model",
    ),
    "stream_passthrough": (".utils", "stream_passthrough"),
    # aliases for backward compatibility
    "add_audio_to_message": (".utils", "create_input_audio_message"),
    "create_selection_model": (".utils", "create_selection_model"),
    "create_literal_pydantic_model": (
        ".utils",
        "create_literal_pydantic_model",
    ),
    "get_content": (".utils", "get_content"),
    "passthrough": (".utils", "stream_passthrough"),
    "create_bool_model": (".utils", "create_bool_model"),
}


def __getattr__(name: str) -> Any:
    """Handle dynamic imports for module attributes."""
    if name in _dynamic_imports:
        module_path, attr_name = _dynamic_imports[name]
        module = import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    """Return list of module attributes for auto-completion."""
    return list(__all__)


# Set module attribute for __getattr__ in Python 3.7+
if sys.version_info >= (3, 7):
    __getattr__.__module__ = __name__
