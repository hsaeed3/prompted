"""
## 💭 chatspec.params

Contains various types used specifically as parameters when creating chat
completions.
"""

from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)
from typing_extensions import TypeAlias, TypedDict, Required
from .types import Message, Tool, Function


# ----------------------------------------------------------------------------
# General Params
#
# some of these are opinionated (or contain opinionated defaults or options)
# (i made this library for myself, so .... )
# ----------------------------------------------------------------------------


InstructorModeParam : TypeAlias = Literal[
    "function_call",
    "parallel_tool_call",
    "tool_call",
    "mistral_tools",
    "json_mode",
    "json_o1",
    "markdown_json_mode", 
    "json_schema_mode",
    "anthropic_tools",
    "anthropic_json",
    "cohere_tools",
    "vertexai_tools",
    "vertexai_json",
    "vertexai_parallel_tools",
    "gemini_json",
    "gemini_tools",
    "json_object",
    "tools_strict",
    "cerebras_tools",
    "cerebras_json",
    "fireworks_tools",
    "fireworks_json",
    "writer_tools"
]
"""
Domain-specific parameter for the `Instructor` libraries client
module.
"""


MessagesParam: TypeAlias = Iterable[Message]
"""
The messages to use when creating a chat completion
"""


ChatModel = Literal[
    "gpt-4o-mini",
    "gpt-4o",
    "o1-mini",
    "o1-preview",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "gpt-4",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-opus-latest",
    # Anthropic Models
    "anthropic/claude-3-5-haiku-latest",  # Vision
    "anthropic/claude-3-5-sonnet-latest",  # Vision
    "anthropic/claude-3-opus-latest",
    # Cohere Models
    "cohere/command-r-plus",
    "cohere/command-r",
    # Databricks Models
    "databricks/databricks-dbrx-instruct",
    # Deepseek Models
    "deepseek/deepseek-coder",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    # Gemini Models
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro-latest",
    # OpenAI Models
    "openai/gpt-4o-mini",  # Vision
    "openai/gpt-4o",  # Vision
    "openai/o1-mini",
    "openai/o1-preview",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4-turbo",
    "openai/gpt-4",
    "openai/gpt-4-vision",  # Vision
    "openai/gpt-3.5-turbo",
    # Ollama Models
    "ollama/bespoke-minicheck",
    "ollama/llama3",
    "ollama/llama3.1",
    "ollama/llama3.2",
    "ollama/llama3.2-vision",  # Vision
    "ollama/llama-guard3",
    "ollama/llava",  # Vision
    "ollama/llava-llama3",  # Vision
    "ollama/llava-phi3",  # Vision
    "ollama/gemma2",
    "ollama/granite3-dense",
    "ollama/granite3-guardian",
    "ollama/granite3-moe",
    "ollama/minicpm-v",  # Vision
    "ollama/mistral",
    "ollama/mistral-nemo",
    "ollama/mistral-small",
    "ollama/mixtral",
    "ollama/moondream",  # Vision
    "ollama/nemotron",
    "ollama/nuextract",
    "ollama/opencoder",
    "ollama/phi3",
    "ollama/reader-lm",
    "ollama/smollm2",
    "ollama/shieldgemma",
    "ollama/tinyllama",
    "ollama/qwen",
    "ollama/qwen2",
    "ollama/qwen2.5",
    # Perplexity Models
    "perplexity/pplx-7b-chat",
    "perplexity/pplx-70b-chat",
    "perplexity/pplx-7b-online",
    "perplexity/pplx-70b-online",
    # XAI Models
    "xai/grok-beta",
]
"""
Helper for a bunch of models!!!
this is in the litellm format
(opinionation to 100%)
"""

ModelParam = Union[str, ChatModel]
"""
The model to use when creating a chat completion.
"""


BaseURLParam: TypeAlias = Union[
    str,
    # openai
    Literal["https://api.openai.com/v1"],
    # deepseek
    Literal["https://api.deepseek.com"],
    # perplexity
    Literal["https://api.perplexity.ai/chat/completions"],
    # ollama default
    Literal["http://localhost:11434/v1"],
    # lmstudio default
    Literal["http://localhost:1234/v1"],
    # openrouter
    Literal["https://openrouter.ai/api/v1"],
]
"""
The base URL to use for the chat completion. Contains opinionated defaults
for a few common providers.

- OpenAI: `https://api.openai.com/v1`
- DeepSeek: `https://api.deepseek.com`
- Perplexity: `https://api.perplexity.ai/chat/completions`
- Ollama (default): `http://localhost:11434/v1`
- LMStudio (default): `http://localhost:1234/v1`
- OpenRouter: `https://openrouter.ai/api/v1`
"""


class FunctionCallParam(TypedDict):
    """
    A dictionary representing a function call for a chat completion.
    """

    name: Required[str]
    """
    The name of the function to call.
    """
    arguments: Required[str]
    """
    The arguments to call the function with.
    """


ToolChoiceParam: TypeAlias = Literal["auto", "none", "required"]
"""
The tool choice to use when creating a chat completion.

- `auto`: The model will pick the best tool to use.
- `none`: The model will not use any tools.
- `required`: The model must use the tools specified in the `tools` parameter.
"""


ModalitiesParam: TypeAlias = Iterable[Literal["text", "image"]]
"""
The modalities to use when creating a chat completion.

- `text`: The model will use text input.
- `image`: The model will use image input.
"""


class PredictionParam(TypedDict):
    """
    A dictionary representing a prediction for chat completion content matching.
    """

    content: Required[Union[str, Iterable[Dict[str, str]]]]
    """
    The content that should be matched when generating a model response. If 
    generated tokens would match this content, the entire model response can be
    returned much more quickly.
    """
    type: Required[Literal["content"]]
    """
    The type of the predicted content. Always "content" for this type.
    """


class AudioParam(TypedDict):
    """
    A dictionary representing an audio input for a chat completion.
    """

    format: Required[Literal["wav", "mp3", "flac", "opus", "pcm16"]]
    """
    The format of the audio input.
    """
    voice: Required[
        Literal[
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "sage",
            "shimmer",
            "verse",
        ]
    ]
    """
    The voice to use for the audio input.
    """


ReasoningEffortParam: TypeAlias = Literal["low", "medium", "high"]
"""
The reasoning effort to use when creating a chat completion.
"""


class ResponseFormatParam(TypedDict):
    """
    A dictionary representing a response format for a chat completion.
    """

    type: Required[Literal["text", "json_object", "json_schema"]]
    """
    The type of the response format.
    """


class StreamOptionsParam(TypedDict):
    """
    A dictionary representing stream options for a chat completion.
    """

    include_usage: Required[bool]
    """
    Whether to include usage information in the stream.
    """


# ----------------------------------------------------------------------------
# Params Object
# ----------------------------------------------------------------------------


class ClientParams(TypedDict):
    """
    A dictionary representing parameters used to initialize a chat completion
    client ('OpenAI')
    """

    base_url: BaseURLParam
    api_key: str
    organization: str
    timeout: float
    max_retries: int


class EmbeddingParams(TypedDict):
    """
    A dictionary representing parameters used when creating an embedding.
    """

    input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    model: ModelParam
    dimensions: Optional[int]
    encoding_format: Optional[Literal["float", "base64"]]
    user: Optional[str]
    timeout: Optional[float]


class CompletionParams(TypedDict):
    """
    A dictionary representing parameters used when creating a chat completion.

    `openai.chat.completions.create()`
    """

    messages: MessagesParam
    model: ModelParam
    audio: Optional[AudioParam]
    frequency_penalty: Optional[float]
    function_call: Optional[FunctionCallParam]
    functions: Optional[Iterable[Function]]
    logit_bias: Optional[Dict[str, int]]
    logprobs: Optional[bool]
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    metadata: Optional[Dict[str, str]]
    modalities: Optional[ModalitiesParam]
    n: Optional[int]
    parallel_tool_calls: Optional[bool]
    prediction: Optional[PredictionParam]
    presence_penalty: Optional[float]
    reasoning_effort: Optional[ReasoningEffortParam]
    response_format: Optional[ResponseFormatParam]
    seed: Optional[int]
    service_tier: Optional[Literal["auto", "default"]]
    stop: Optional[Union[str, List[str]]]
    store: Optional[bool]
    stream: Optional[bool]
    temperature: Optional[float]
    top_p: Optional[float]
    tools: Optional[Iterable[Tool]]
    tool_choice: Optional[ToolChoiceParam]
    top_logprobs: Optional[int]
    user: Optional[str]


class Params(ClientParams, CompletionParams):
    """
    A dictionary representing unified parameters for a chat completion.

    `litellm.completion()`
    """

    pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def to_client_params(params: Params) -> ClientParams:
    """
    Convert a `Params` object to a `ClientParams` object.
    """
    valid_keys = ClientParams.__annotations__.keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}
    return ClientParams(**filtered_params)


def to_completion_params(params: Params) -> CompletionParams:
    """
    Convert a `Params` object to a `CompletionParams` object.
    """
    valid_keys = CompletionParams.__annotations__.keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}
    return CompletionParams(**filtered_params)
