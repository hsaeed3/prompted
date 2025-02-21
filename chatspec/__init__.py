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

# [markdown]
from .markdown import markdownify, MarkdownObject

# [mock]
from .mock import mock_completion, mock_embedding, AI

# [params]
from .params import (
    Params,
    ClientParams,
    CompletionParams,
    to_client_params,
    to_completion_params,
    # [general]
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
    CompletionParams,
    EmbeddingParams
)

# [types]
from .types import (
    FunctionParameters,
    Function,
    Tool,
    FunctionCall,
    ToolCall,
    MessageContentImagePart,
    MessageContentAudioPart,
    MessageContentTextPart,
    MessageContent,
    MessageTextContent,
    MessageRole,
    Message,
    TopLogprob,
    TokenLogprob,
    ChoiceLogprobs,
    Completion,
    CompletionChunk,
    CompletionMessage,
    CompletionFunction,
    CompletionToolCall,
    Embedding
)

# [utils]
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
    parse_model_from_completion,
    parse_model_from_stream,
    print_stream,
    normalize_messages,
    normalize_system_prompt,
    create_field_mapping,
    extract_function_fields,
    convert_to_pydantic_model,
    convert_to_tools,
    convert_to_tool,
    create_literal_pydantic_model,
    stream_passthrough,
)
