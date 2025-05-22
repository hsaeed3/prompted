"""
propmted.types.completions.params

Contains specific type definitions used for hinting parameters for various
completion resources available within `prompted`.
"""

from typing import Literal, Union, List
from typing_extensions import TypeAliasType

from .messages import CompletionMessage, CompletionMessageRole, CompletionMessageContentPart

# -----------------------------------------------------------------------------
# Client / Provider Parameters
# -----------------------------------------------------------------------------

CompletionModelParam = TypeAliasType(
    "CompletionModelParam",
    Union[
        str,
        Literal[
            "anthropic/claude-3-7-sonnet-latest",
            "anthropic/claude-3-5-haiku-latest",
            "anthropic/claude-3-5-sonnet-latest",
            "anthropic/claude-3-opus-latest",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
            "bedrock/amazon.titan-tg1-large",
            "bedrock/amazon.titan-text-lite-v1",
            "bedrock/amazon.titan-text-express-v1",
            "bedrock/us.amazon.nova-pro-v1:0",
            "bedrock/us.amazon.nova-lite-v1:0",
            "bedrock/us.amazon.nova-micro-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
            "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "bedrock/anthropic.claude-instant-v1",
            "bedrock/anthropic.claude-v2:1",
            "bedrock/anthropic.claude-v2",
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0",
            "bedrock/us.anthropic.claude-3-opus-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
            "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "bedrock/cohere.command-text-v14",
            "bedrock/cohere.command-r-v1:0",
            "bedrock/cohere.command-r-plus-v1:0",
            "bedrock/cohere.command-light-text-v14",
            "bedrock/meta.llama3-8b-instruct-v1:0",
            "bedrock/meta.llama3-70b-instruct-v1:0",
            "bedrock/meta.llama3-1-8b-instruct-v1:0",
            "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
            "bedrock/meta.llama3-1-70b-instruct-v1:0",
            "bedrock/us.meta.llama3-1-70b-instruct-v1:0",
            "bedrock/meta.llama3-1-405b-instruct-v1:0",
            "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
            "bedrock/us.meta.llama3-2-90b-instruct-v1:0",
            "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
            "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
            "bedrock/us.meta.llama3-3-70b-instruct-v1:0",
            "bedrock/mistral.mistral-7b-instruct-v0:2",
            "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
            "bedrock/mistral.mistral-large-2402-v1:0",
            "bedrock/mistral.mistral-large-2407-v1:0",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-latest",
            "cohere/c4ai-aya-expanse-32b",
            "cohere/c4ai-aya-expanse-8b",
            "cohere/command",
            "cohere/command-light",
            "cohere/command-light-nightly",
            "cohere/command-nightly",
            "cohere/command-r",
            "cohere/command-r-03-2024",
            "cohere/command-r-08-2024",
            "cohere/command-r-plus",
            "cohere/command-r-plus-04-2024",
            "cohere/command-r-plus-08-2024",
            "cohere/command-r7b-12-2024",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
            "google-gla/gemini-1.0-pro",
            "google-gla/gemini-1.5-flash",
            "google-gla/gemini-1.5-flash-8b",
            "google-gla/gemini-1.5-pro",
            "google-gla/gemini-2.0-flash-exp",
            "google-gla/gemini-2.0-flash-thinking-exp-01-21",
            "google-gla/gemini-exp-1206",
            "google-gla/gemini-2.0-flash",
            "google-gla/gemini-2.0-flash-lite-preview-02-05",
            "google-gla/gemini-2.0-pro-exp-02-05",
            "google-gla/gemini-2.5-flash-preview-04-17",
            "google-gla/gemini-2.5-pro-exp-03-25",
            "google-gla/gemini-2.5-pro-preview-03-25",
            "google-vertex/gemini-1.0-pro",
            "google-vertex/gemini-1.5-flash",
            "google-vertex/gemini-1.5-flash-8b",
            "google-vertex/gemini-1.5-pro",
            "google-vertex/gemini-2.0-flash-exp",
            "google-vertex/gemini-2.0-flash-thinking-exp-01-21",
            "google-vertex/gemini-exp-1206",
            "google-vertex/gemini-2.0-flash",
            "google-vertex/gemini-2.0-flash-lite-preview-02-05",
            "google-vertex/gemini-2.0-pro-exp-02-05",
            "google-vertex/gemini-2.5-flash-preview-04-17",
            "google-vertex/gemini-2.5-pro-exp-03-25",
            "google-vertex/gemini-2.5-pro-preview-03-25",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4-vision-preview",
            "gpt-4.1",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-audio-preview",
            "gpt-4o-audio-preview-2024-10-01",
            "gpt-4o-audio-preview-2024-12-17",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini-audio-preview",
            "gpt-4o-mini-audio-preview-2024-12-17",
            "gpt-4o-mini-search-preview",
            "gpt-4o-mini-search-preview-2025-03-11",
            "gpt-4o-search-preview",
            "gpt-4o-search-preview-2025-03-11",
            "groq/distil-whisper-large-v3-en",
            "groq/gemma2-9b-it",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/llama-guard-3-8b",
            "groq/llama3-70b-8192",
            "groq/llama3-8b-8192",
            "groq/whisper-large-v3",
            "groq/whisper-large-v3-turbo",
            "groq/playai-tts",
            "groq/playai-tts-arabic",
            "groq/qwen-qwq-32b",
            "groq/mistral-saba-24b",
            "groq/qwen-2.5-coder-32b",
            "groq/qwen-2.5-32b",
            "groq/deepseek-r1-distill-qwen-32b",
            "groq/deepseek-r1-distill-llama-70b",
            "groq/llama-3.3-70b-specdec",
            "groq/llama-3.2-1b-preview",
            "groq/llama-3.2-3b-preview",
            "groq/llama-3.2-11b-vision-preview",
            "groq/llama-3.2-90b-vision-preview",
            "mistral/codestral-latest",
            "mistral/mistral-large-latest",
            "mistral/mistral-moderation-latest",
            "mistral/mistral-small-latest",
            "o1",
            "o1-2024-12-17",
            "o1-mini",
            "o1-mini-2024-09-12",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o3",
            "o3-2025-04-16",
            "o3-mini",
            "o3-mini-2025-01-31",
            "openai/chatgpt-4o-latest",
            "openai/gpt-3.5-turbo",
            "openai/gpt-3.5-turbo-0125",
            "openai/gpt-3.5-turbo-0301",
            "openai/gpt-3.5-turbo-0613",
            "openai/gpt-3.5-turbo-1106",
            "openai/gpt-3.5-turbo-16k",
            "openai/gpt-3.5-turbo-16k-0613",
            "openai/gpt-4",
            "openai/gpt-4-0125-preview",
            "openai/gpt-4-0314",
            "openai/gpt-4-0613",
            "openai/gpt-4-1106-preview",
            "openai/gpt-4-32k",
            "openai/gpt-4-32k-0314",
            "openai/gpt-4-32k-0613",
            "openai/gpt-4-turbo",
            "openai/gpt-4-turbo-2024-04-09",
            "openai/gpt-4-turbo-preview",
            "openai/gpt-4-vision-preview",
            "openai/gpt-4.1",
            "openai/gpt-4.1-2025-04-14",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-mini-2025-04-14",
            "openai/gpt-4.1-nano",
            "openai/gpt-4.1-nano-2025-04-14",
            "openai/gpt-4o",
            "openai/gpt-4o-2024-05-13",
            "openai/gpt-4o-2024-08-06",
            "openai/gpt-4o-2024-11-20",
            "openai/gpt-4o-audio-preview",
            "openai/gpt-4o-audio-preview-2024-10-01",
            "openai/gpt-4o-audio-preview-2024-12-17",
            "openai/gpt-4o-mini",
            "openai/gpt-4o-mini-2024-07-18",
            "openai/gpt-4o-mini-audio-preview",
            "openai/gpt-4o-mini-audio-preview-2024-12-17",
            "openai/gpt-4o-mini-search-preview",
            "openai/gpt-4o-mini-search-preview-2025-03-11",
            "openai/gpt-4o-search-preview",
            "openai/gpt-4o-search-preview-2025-03-11",
            "openai/o1",
            "openai/o1-2024-12-17",
            "openai/o1-mini",
            "openai/o1-mini-2024-09-12",
            "openai/o1-preview",
            "openai/o1-preview-2024-09-12",
            "openai/o3",
            "openai/o3-2025-04-16",
            "openai/o3-mini",
            "openai/o3-mini-2025-01-31",
            "openai/o4-mini",
            "openai/o4-mini-2025-04-16",
            "test",
        ],
    ],
)
"""
Type definition for the `model` parameter of a completion resource.

Helper for a bunch of models. Format might vary based on actual API provider.
OpenAI models are listed without a prefix.
For multi-provider scenarios like LiteLLM, prefixes like 'anthropic/', 'openai/', 'gemini/' are common.
"""

CompletionBaseUrlParam = TypeAliasType(
    "CompletionBaseUrlParam",
    Union[
        str,
        # OpenAI
        Literal["https://api.openai.com/v1"],
        # Anthropic
        Literal["https://api.anthropic.com/v1"],
        # Google (Vertex AI / AI Platform)
        Literal["https://us-central1-aiplatform.googleapis.com"],
        # Cohere
        Literal["https://api.cohere.ai/v1"],
        # DeepSeek
        Literal["https://api.deepseek.com"],
        # Perplexity
        Literal["https://api.perplexity.ai"],
        # Ollama default
        Literal["http://localhost:11434/v1"],
        # LMStudio default
        Literal["http://localhost:1234/v1"],
        # OpenRouter
        Literal["https://openrouter.ai/api/v1"],
    ],
)
"""Helper type definition that provides a bunch of common LLM provider base URLs.

(NOTE: This library uses `litellm` anyways, so <provider>:<model> spec is recommended through
`litellm`'s default behavior.)"""

# -----------------------------------------------------------------------------
# Primary Completion Parameters
# -----------------------------------------------------------------------------

CompletionMessagesParam = TypeAliasType("CompletionMessagesParam", CompletionMessage | List[CompletionMessage])
"""
Type definition for the `messages` parameter of a completions resource.

This parameter can be a single message, or a list of messages.
"""

CompletionMessageRoleParam = TypeAliasType("CompletionMessageRoleParam", Union[str, CompletionMessageRole])
"""Type definition for the `role` parameter of a completion message."""


__all__ = [
    "CompletionModelParam",
    "CompletionBaseUrlParam",
    "CompletionMessagesParam",
    "CompletionMessageRoleParam",
]
