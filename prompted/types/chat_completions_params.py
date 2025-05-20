"""
💭 prompted.types.chat_completions_params

Contains scoped parameter specific types relative to the Chat 
Completions API specification.
"""

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    Any,
)
from typing_extensions import TypeAlias, TypedDict, Required, NotRequired, TypeAliasType
from .chat_completions import Message, Tool, Function

__all__ = (
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
)


# ----------------------------------------------------------------------------
# General Params
#
# some of these are opinionated (or contain opinionated defaults or options)
# ----------------------------------------------------------------------------


MessagesParam: TypeAlias = Iterable[Message]
"""
The messages to use when creating a chat completion. A list of message objects,
where each object has a role (either "system", "user", or "assistant") and
content (the content of the message).
"""


ChatModel = TypeAliasType(
    'ChatModel',
    Literal[
        'anthropic/claude-3-7-sonnet-latest',
        'anthropic/claude-3-5-haiku-latest',
        'anthropic/claude-3-5-sonnet-latest',
        'anthropic/claude-3-opus-latest',
        'claude-3-7-sonnet-latest',
        'claude-3-5-haiku-latest',
        'bedrock/amazon.titan-tg1-large',
        'bedrock/amazon.titan-text-lite-v1',
        'bedrock/amazon.titan-text-express-v1',
        'bedrock/us.amazon.nova-pro-v1:0',
        'bedrock/us.amazon.nova-lite-v1:0',
        'bedrock/us.amazon.nova-micro-v1:0',
        'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock/anthropic.claude-instant-v1',
        'bedrock/anthropic.claude-v2:1',
        'bedrock/anthropic.claude-v2',
        'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock/anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock/us.anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock/anthropic.claude-3-opus-20240229-v1:0',
        'bedrock/us.anthropic.claude-3-opus-20240229-v1:0',
        'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock/cohere.command-text-v14',
        'bedrock/cohere.command-r-v1:0',
        'bedrock/cohere.command-r-plus-v1:0',
        'bedrock/cohere.command-light-text-v14',
        'bedrock/meta.llama3-8b-instruct-v1:0',
        'bedrock/meta.llama3-70b-instruct-v1:0',
        'bedrock/meta.llama3-1-8b-instruct-v1:0',
        'bedrock/us.meta.llama3-1-8b-instruct-v1:0',
        'bedrock/meta.llama3-1-70b-instruct-v1:0',
        'bedrock/us.meta.llama3-1-70b-instruct-v1:0',
        'bedrock/meta.llama3-1-405b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-11b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-90b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-1b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-3b-instruct-v1:0',
        'bedrock/us.meta.llama3-3-70b-instruct-v1:0',
        'bedrock/mistral.mistral-7b-instruct-v0:2',
        'bedrock/mistral.mixtral-8x7b-instruct-v0:1',
        'bedrock/mistral.mistral-large-2402-v1:0',
        'bedrock/mistral.mistral-large-2407-v1:0',
        'claude-3-5-sonnet-latest',
        'claude-3-opus-latest',
        'cohere/c4ai-aya-expanse-32b',
        'cohere/c4ai-aya-expanse-8b',
        'cohere/command',
        'cohere/command-light',
        'cohere/command-light-nightly',
        'cohere/command-nightly',
        'cohere/command-r',
        'cohere/command-r-03-2024',
        'cohere/command-r-08-2024',
        'cohere/command-r-plus',
        'cohere/command-r-plus-04-2024',
        'cohere/command-r-plus-08-2024',
        'cohere/command-r7b-12-2024',
        'deepseek/deepseek-chat',
        'deepseek/deepseek-reasoner',
        'google-gla/gemini-1.0-pro',
        'google-gla/gemini-1.5-flash',
        'google-gla/gemini-1.5-flash-8b',
        'google-gla/gemini-1.5-pro',
        'google-gla/gemini-2.0-flash-exp',
        'google-gla/gemini-2.0-flash-thinking-exp-01-21',
        'google-gla/gemini-exp-1206',
        'google-gla/gemini-2.0-flash',
        'google-gla/gemini-2.0-flash-lite-preview-02-05',
        'google-gla/gemini-2.0-pro-exp-02-05',
        'google-gla/gemini-2.5-flash-preview-04-17',
        'google-gla/gemini-2.5-pro-exp-03-25',
        'google-gla/gemini-2.5-pro-preview-03-25',
        'google-vertex/gemini-1.0-pro',
        'google-vertex/gemini-1.5-flash',
        'google-vertex/gemini-1.5-flash-8b',
        'google-vertex/gemini-1.5-pro',
        'google-vertex/gemini-2.0-flash-exp',
        'google-vertex/gemini-2.0-flash-thinking-exp-01-21',
        'google-vertex/gemini-exp-1206',
        'google-vertex/gemini-2.0-flash',
        'google-vertex/gemini-2.0-flash-lite-preview-02-05',
        'google-vertex/gemini-2.0-pro-exp-02-05',
        'google-vertex/gemini-2.5-flash-preview-04-17',
        'google-vertex/gemini-2.5-pro-exp-03-25',
        'google-vertex/gemini-2.5-pro-preview-03-25',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-0301',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo-16k',
        'gpt-3.5-turbo-16k-0613',
        'gpt-4',
        'gpt-4-0125-preview',
        'gpt-4-0314',
        'gpt-4-0613',
        'gpt-4-1106-preview',
        'gpt-4-32k',
        'gpt-4-32k-0314',
        'gpt-4-32k-0613',
        'gpt-4-turbo',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-turbo-preview',
        'gpt-4-vision-preview',
        'gpt-4.1',
        'gpt-4.1-2025-04-14',
        'gpt-4.1-mini',
        'gpt-4.1-mini-2025-04-14',
        'gpt-4.1-nano',
        'gpt-4.1-nano-2025-04-14',
        'gpt-4o',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
        'gpt-4o-2024-11-20',
        'gpt-4o-audio-preview',
        'gpt-4o-audio-preview-2024-10-01',
        'gpt-4o-audio-preview-2024-12-17',
        'gpt-4o-mini',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini-audio-preview',
        'gpt-4o-mini-audio-preview-2024-12-17',
        'gpt-4o-mini-search-preview',
        'gpt-4o-mini-search-preview-2025-03-11',
        'gpt-4o-search-preview',
        'gpt-4o-search-preview-2025-03-11',
        'groq/distil-whisper-large-v3-en',
        'groq/gemma2-9b-it',
        'groq/llama-3.3-70b-versatile',
        'groq/llama-3.1-8b-instant',
        'groq/llama-guard-3-8b',
        'groq/llama3-70b-8192',
        'groq/llama3-8b-8192',
        'groq/whisper-large-v3',
        'groq/whisper-large-v3-turbo',
        'groq/playai-tts',
        'groq/playai-tts-arabic',
        'groq/qwen-qwq-32b',
        'groq/mistral-saba-24b',
        'groq/qwen-2.5-coder-32b',
        'groq/qwen-2.5-32b',
        'groq/deepseek-r1-distill-qwen-32b',
        'groq/deepseek-r1-distill-llama-70b',
        'groq/llama-3.3-70b-specdec',
        'groq/llama-3.2-1b-preview',
        'groq/llama-3.2-3b-preview',
        'groq/llama-3.2-11b-vision-preview',
        'groq/llama-3.2-90b-vision-preview',
        'mistral/codestral-latest',
        'mistral/mistral-large-latest',
        'mistral/mistral-moderation-latest',
        'mistral/mistral-small-latest',
        'o1',
        'o1-2024-12-17',
        'o1-mini',
        'o1-mini-2024-09-12',
        'o1-preview',
        'o1-preview-2024-09-12',
        'o3',
        'o3-2025-04-16',
        'o3-mini',
        'o3-mini-2025-01-31',
        'openai/chatgpt-4o-latest',
        'openai/gpt-3.5-turbo',
        'openai/gpt-3.5-turbo-0125',
        'openai/gpt-3.5-turbo-0301',
        'openai/gpt-3.5-turbo-0613',
        'openai/gpt-3.5-turbo-1106',
        'openai/gpt-3.5-turbo-16k',
        'openai/gpt-3.5-turbo-16k-0613',
        'openai/gpt-4',
        'openai/gpt-4-0125-preview',
        'openai/gpt-4-0314',
        'openai/gpt-4-0613',
        'openai/gpt-4-1106-preview',
        'openai/gpt-4-32k',
        'openai/gpt-4-32k-0314',
        'openai/gpt-4-32k-0613',
        'openai/gpt-4-turbo',
        'openai/gpt-4-turbo-2024-04-09',
        'openai/gpt-4-turbo-preview',
        'openai/gpt-4-vision-preview',
        'openai/gpt-4.1',
        'openai/gpt-4.1-2025-04-14',
        'openai/gpt-4.1-mini',
        'openai/gpt-4.1-mini-2025-04-14',
        'openai/gpt-4.1-nano',
        'openai/gpt-4.1-nano-2025-04-14',
        'openai/gpt-4o',
        'openai/gpt-4o-2024-05-13',
        'openai/gpt-4o-2024-08-06',
        'openai/gpt-4o-2024-11-20',
        'openai/gpt-4o-audio-preview',
        'openai/gpt-4o-audio-preview-2024-10-01',
        'openai/gpt-4o-audio-preview-2024-12-17',
        'openai/gpt-4o-mini',
        'openai/gpt-4o-mini-2024-07-18',
        'openai/gpt-4o-mini-audio-preview',
        'openai/gpt-4o-mini-audio-preview-2024-12-17',
        'openai/gpt-4o-mini-search-preview',
        'openai/gpt-4o-mini-search-preview-2025-03-11',
        'openai/gpt-4o-search-preview',
        'openai/gpt-4o-search-preview-2025-03-11',
        'openai/o1',
        'openai/o1-2024-12-17',
        'openai/o1-mini',
        'openai/o1-mini-2024-09-12',
        'openai/o1-preview',
        'openai/o1-preview-2024-09-12',
        'openai/o3',
        'openai/o3-2025-04-16',
        'openai/o3-mini',
        'openai/o3-mini-2025-01-31',
        'openai/o4-mini',
        'openai/o4-mini-2025-04-16',
        'test',
    ],
)
"""
Helper for a bunch of models. Format might vary based on actual API provider.
OpenAI models are listed without a prefix.
For multi-provider scenarios like LiteLLM, prefixes like 'anthropic/', 'openai/', 'gemini/' are common.
"""

ModelParam = Union[str, ChatModel]
"""
The model to use when creating a chat completion. Can be a string for any model
or one of the predefined ChatModel literals.
"""


BaseURLParam: TypeAlias = Union[
    str,
    # OpenAI
    Literal["https://api.openai.com/v1"],
    # Anthropic
    Literal["https://api.anthropic.com/v1"], # Added common alternative
    # Google (Vertex AI / AI Platform) - example, specific endpoint can vary
    Literal["https://us-central1-aiplatform.googleapis.com"], # Example
    # Cohere
    Literal["https://api.cohere.ai/v1"], # Added common alternative
    # DeepSeek
    Literal["https://api.deepseek.com"],
    # Perplexity
    Literal["https://api.perplexity.ai"], # Updated, /chat/completions is usually part of the path
    # Ollama default
    Literal["http://localhost:11434/v1"],
    # LMStudio default
    Literal["http://localhost:1234/v1"],
    # OpenRouter
    Literal["https://openrouter.ai/api/v1"],
]
"""
The base URL to use for the chat completion. Contains opinionated defaults
for a few common providers.
"""


class FunctionCallParam(TypedDict): # Deprecated by OpenAI in favor of tool_choice
    """
    (Deprecated) A dictionary representing a specific function to be called.
    Use `tool_choice` with a specific function name instead.

    Example:
    `{"name": "my_function"}`
    """
    name: Required[str]
    """
    The name of the function to call.
    """
    # arguments: Required[str] # Arguments are not specified by the user when forcing a function call


class ToolChoiceNamedToolFunction(TypedDict):
    """Specifies the function name when choosing a named tool."""
    name: Required[str]
    """The name of the function to call."""

class ToolChoiceNamedTool(TypedDict):
    """Forces the model to call a specific tool (e.g., a function)."""
    type: Required[Literal["function"]]
    """The type of the tool. Currently, only `function` is supported."""
    function: Required[ToolChoiceNamedToolFunction]
    """The function to be called."""


ToolChoiceParam: TypeAlias = Union[
    Literal["none"], # Model will not call any tool and instead generates a message.
    Literal["auto"], # Model can pick between generating a message or calling one or more tools.
    Literal["required"], # Model must call one or more tools.
    ToolChoiceNamedTool, # Model must call the specified tool.
]
"""
Controls which (if any) tool is called by the model.
- `none`: The model will not call any tool and will generate a message.
- `auto`: The model can choose between generating a message or calling one or more tools.
- `required`: The model must call one or more tools.
- `ToolChoiceNamedTool` (object): Forces the model to call a specific function.
  Example: `{"type": "function", "function": {"name": "my_function"}}`
"""


# ModalitiesParam, PredictionParam, AudioParam, ReasoningEffortParam
# These appear to be custom parameters not standard in the OpenAI API for chat completions.
# They might be part of a higher-level library. We will keep their definitions
# as they were in the original script, assuming they serve a purpose in that context.

ModalitiesParam: TypeAlias = Iterable[Literal["text", "image"]]
"""
(Custom Parameter) The modalities to use when creating a chat completion.
This is not a standard OpenAI chat completion parameter. Vision capabilities are
usually indicated by the model choice (e.g., "gpt-4-vision-preview") and
image content provided in the messages.

- `text`: The model will use text input.
- `image`: The model will use image input.
"""

class PredictionParam(TypedDict):
    """
    (Custom Parameter) A dictionary representing a prediction for chat completion content matching.
    This is not a standard OpenAI chat completion parameter.
    """
    content: Required[Union[str, Iterable[Dict[str, str]]]]
    """
    The content that should be matched when generating a model response.
    """
    type: Required[Literal["content"]]
    """
    The type of the predicted content. Always "content" for this type.
    """

class AudioParam(TypedDict): # This is more aligned with Speech-to-Text (STT) or Text-to-Speech (TTS) APIs
    """
    (Custom Parameter or for TTS/STT) A dictionary representing audio properties.
    For OpenAI Chat Completions, audio input/output is not specified this way.
    Audio input for transcription or output for speech generation uses separate APIs/parameters.
    """
    format: Required[Literal["wav", "mp3", "flac", "opus", "pcm16", "aac", "m4a"]] # Added common formats
    """
    The format of the audio input/output.
    """
    voice: Required[ # These are OpenAI TTS voices
        Literal[
            "alloy", "echo", "fable", "onyx", "nova", "shimmer" # Standard TTS voices
            # "ash", "ballad", "coral", "sage", "verse" # These seem like older or different set
        ]
    ]
    """
    The voice to use for the audio.
    """

ReasoningEffortParam: TypeAlias = Literal["low", "medium", "high"]
"""
(Custom Parameter) The reasoning effort to use when creating a chat completion.
This is not a standard OpenAI chat completion parameter.
"""


class ResponseFormatParam(TypedDict):
    """
    An object specifying the format that the model must output.
    Setting to `{"type": "json_object"}` enables JSON mode, which guarantees the message
    the model generates is valid JSON.

    Important: When using JSON mode, you must also instruct the model to produce JSON
    via a system or user message.
    """
    type: Required[Literal["text", "json_object"]] # "json_schema" is not a direct type value, but part of json_object with a schema.
    # json_schema: NotRequired[Dict[str, Any]] # For more advanced JSON schema enforcement, typically done via instructing the model and validating output.
    # The official `response_format` only takes `{"type": "json_object"}` or `{"type": "text"}`.
    # For JSON schema, you use `{"type": "json_object"}` and also include instructions in your prompt
    # or potentially use a tool that enforces a schema.
"""
The type of the response format.
- `text`: Default, model can output arbitrary text.
- `json_object`: Enables JSON mode, guaranteeing valid JSON output. You must instruct the model to produce JSON in the prompt.
"""


class StreamOptionsParam(TypedDict):
    """
    Options for streaming responses. Only set this when `stream: true`.
    """
    include_usage: NotRequired[bool] # Changed to NotRequired as it's optional
    """
    If set, an additional chunk will be streamed before the `data: [DONE]` message.
    The `usage` field on this chunk will contain the token usage statistics for the
    entire request, and the `choices` field will be an empty array.
    """


# ----------------------------------------------------------------------------
# Params Object
# ----------------------------------------------------------------------------


class ClientParams(TypedDict, total=False): # Changed to total=False for more flexibility
    """
    A dictionary representing parameters used to initialize an API client.
    Fields are typically optional at initialization, relying on defaults or environment variables.
    """
    base_url: NotRequired[BaseURLParam]
    api_key: NotRequired[str]
    organization: NotRequired[str] # OpenAI specific
    timeout: NotRequired[float] # In seconds
    max_retries: NotRequired[int]
    # project: NotRequired[str] # OpenAI specific, for project-based billing
    # default_headers: NotRequired[Dict[str, str]] # Common for custom headers


class EmbeddingParams(TypedDict):
    """
    A dictionary representing parameters used when creating an embedding.
    """
    input: Required[Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]]
    """
    Input text to embed, encoded as a string or array of tokens. To embed multiple
    inputs in a single request, pass a list of strings or a list of token arrays.
    """
    model: Required[ModelParam] # Or a dedicated embedding model type, e.g., "text-embedding-3-small"
    """
    ID of the model to use. You can use the `text-embedding-3-small`,
    `text-embedding-3-large`, or `text-embedding-ada-002` model ID.
    """
    dimensions: NotRequired[int]
    """
    The number of dimensions the resulting outputted embedding should have.
    Only supported in `text-embedding-3` and later models.
    """
    encoding_format: NotRequired[Literal["float", "base64"]]
    """
    The format to return the embeddings in. Can be `float` or `base64`.
    Defaults to `float`.
    """
    user: NotRequired[str]
    """
    A unique identifier representing your end-user, which can help OpenAI to
    monitor and detect abuse.
    """
    # timeout: Optional[float] # Timeout is usually a client-level config, not per-request.


class CompletionParams(TypedDict, total=False): # Changed to total=False as most are optional
    """
    A dictionary representing parameters used when creating a chat completion.
    Corresponds to `openai.chat.completions.create()` parameters.
    """
    messages: Required[MessagesParam]
    model: Required[ModelParam]

    # audio: Optional[AudioParam] # Not a standard OpenAI chat completion param in this structure
    frequency_penalty: NotRequired[Optional[float]] # Number between -2.0 and 2.0. Defaults to 0.
    # function_call: Optional[FunctionCallParam] # Deprecated, use tool_choice
    # functions: Optional[Iterable[Function]] # Deprecated, use tools
    logit_bias: NotRequired[Optional[Dict[str, int]]] # Max 100 tokens. Values between -100 and 100.
    logprobs: NotRequired[Optional[bool]] # Defaults to false.
    top_logprobs: NotRequired[Optional[int]] # 0 to 5. Only if logprobs is true.
    # max_completion_tokens: Optional[int] # This is usually 'max_tokens'
    max_tokens: NotRequired[Optional[int]] # Max tokens to generate.
    # metadata: Optional[Dict[str, str]] # Not a standard OpenAI param, maybe for other providers
    # modalities: Optional[ModalitiesParam] # Custom param
    n: NotRequired[Optional[int]] # How many choices to generate. Defaults to 1.
    parallel_tool_calls: NotRequired[Optional[bool]] # Defaults to True. Setting to False disables parallel tool calling.
    # prediction: Optional[PredictionParam] # Custom param
    presence_penalty: NotRequired[Optional[float]] # Number between -2.0 and 2.0. Defaults to 0.
    # reasoning_effort: Optional[ReasoningEffortParam] # Custom param
    response_format: NotRequired[Optional[ResponseFormatParam]]
    seed: NotRequired[Optional[int]] # For reproducibility. Beta.
    # service_tier: Optional[Literal["auto", "default"]] # OpenAI specific, for enterprise. Not commonly exposed for all models.
    stop: NotRequired[Optional[Union[str, List[str]]]] # Up to 4 sequences.
    # store: Optional[bool] # Not a standard OpenAI param, maybe custom for caching
    stream: NotRequired[Optional[bool]] # Defaults to false.
    stream_options: NotRequired[Optional[StreamOptionsParam]] # Only if stream is true.
    temperature: NotRequired[Optional[float]] # Between 0 and 2. Defaults to 1.
    top_p: NotRequired[Optional[float]] # Between 0 and 1. Defaults to 1.
    tools: NotRequired[Optional[Iterable[Tool]]]
    tool_choice: NotRequired[Optional[ToolChoiceParam]]
    user: NotRequired[Optional[str]] # End-user identifier.


class Params(ClientParams, CompletionParams): # Defaulting to total=False behavior from constituent TypedDicts
    """
    A dictionary representing unified parameters for a chat completion, potentially
    for a higher-level library like LiteLLM that combines client and completion params.
    """
    pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def to_client_params(params: Params) -> ClientParams:
    """
    Convert a `Params` object to a `ClientParams` object.
    """
    # Need to handle potential Total=False by checking for key existence
    valid_keys = ClientParams.__annotations__.keys()
    # Ensure that only keys actually present in params and valid for ClientParams are passed
    filtered_params = {k: v for k, v in params.items() if k in valid_keys and k in params}
    return ClientParams(**filtered_params)


def to_completion_params(params: Params) -> CompletionParams:
    """
    Convert a `Params` object to a `CompletionParams` object.
    """
    valid_keys = CompletionParams.__annotations__.keys()
     # Ensure that only keys actually present in params and valid for CompletionParams are passed
    filtered_params = {k: v for k, v in params.items() if k in valid_keys and k in params}
    # Required fields in CompletionParams must be present in params
    if "messages" not in filtered_params:
        raise ValueError("Missing required parameter 'messages' for CompletionParams")
    if "model" not in filtered_params:
        raise ValueError("Missing required parameter 'model' for CompletionParams")
    return CompletionParams(**filtered_params) # type: ignore[arg-type] # if params might not have all required fields initially