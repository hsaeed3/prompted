"""
## 💭 chatspec.mock

Contains the `MockAI` class, a barebones mocked implementation of the OpenAI client
for chat completions, as well as the mock_completion() method; similar to litellm's
completion() method.
"""

import time
import uuid
from typing import (
    Any,
    Dict,
    Iterator,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    overload,
)
from .types import Completion, CompletionChunk, Message, Tool
from .params import (
    Params,
    ModelParam,
    MessagesParam,
    BaseURLParam,
    ToolChoiceParam,
    AudioParam,
    FunctionCallParam,
    Function,
    Tool,
    ToolChoiceParam,
    ModalitiesParam,
    ReasoningEffortParam,
    ResponseFormatParam,
    PredictionParam,
)
from .utils import (
    logger,
    ChatSpecError,
    # methods
    normalize_messages,
    stream_passthrough,
    convert_to_tools,
)

__all__ = [
    "MockAI",
    "mock_completion",
]


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------


class MockAIError(ChatSpecError):
    """
    An error raised by the MockAI class.
    """


# ----------------------------------------------------------------------------
# MockAI Class
# ----------------------------------------------------------------------------


class MockAI:
    """
    A mocked implementation of the OpenAI client for chat completions.
    """

    def __init__(
        self,
        base_url: Optional[BaseURLParam] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the MockAI client.

        Args:
            base_url: The base URL of the API.
            api_key: The API key to use for the API.
            organization: The organization to use for the API.
            timeout: The timeout for the API.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.timeout = timeout

    @overload
    @classmethod
    def create(
        cls,
        *,
        messages: MessagesParam,
        model: ModelParam = "gpt-4o-mini",
        stream: Literal[False] = False,
        tools: Optional[Iterable[Tool]] = None,
        **kwargs: Params,
    ) -> Completion: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        messages: MessagesParam,
        model: ModelParam = "gpt-4o-mini",
        stream: Literal[True],
        tools: Optional[Iterable[Tool]] = None,
        **kwargs: Params,
    ) -> Iterator[CompletionChunk]: ...

    @classmethod
    def create(
        cls, **kwargs: Any
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """
        Mocks the OpenAI ChatCompletion.create method.

        Accepts parameters similar to a real client, including:
          - messages: Iterable of Message objects (or as defined in Params)
          - model: the model to use (default "gpt-4o-mini")
          - stream: bool indicating if the response should be streamed
          - tools: optionally, a list (or dict) of tools to use
          - plus other parameters (from Params)

        Returns either a standard Completion response or, if stream=True,
        an iterator over CompletionChunk objects.
        """
        try:
            params: Params = {}
            # Normalize messages
            if not kwargs.get("messages"):
                raise MockAIError("messages are required")
            try:
                params["messages"] = normalize_messages(
                    kwargs.get("messages")
                )
            except Exception as e:
                raise MockAIError(
                    f"Failed to normalize messages: {str(e)}"
                )
            params["model"] = kwargs.get("model", "gpt-4o-mini")
            params["stream"] = kwargs.get("stream", False)

            # Process tools if provided
            tools_input = kwargs.get("tools")
            if tools_input:
                try:
                    params["tools"] = convert_to_tools(tools_input)
                    logger.debug(
                        f"Mock completion tools: {params['tools']}"
                    )
                except Exception as e:
                    raise MockAIError(f"Failed to convert tools: {str(e)}")

            # Process other parameters
            for key, value in kwargs.items():
                if key not in params:
                    params[key] = value

            if params["stream"]:
                # Streaming mode: generate a mock Choice and return a stream
                choice = cls._create_mock_response_choice(params)
                logger.debug(f"Mock completion choice: {choice}")
                return stream_passthrough(
                    cls._stream_response(choice, params)
                )
            else:
                # Non-streaming mode: generate a mock Completion
                choice = cls._create_mock_response_choice(params)
                logger.debug(f"Mock completion choice: {choice}")
                comp: Completion = {
                    "id": str(uuid.uuid4()),
                    "choices": [choice],
                    "created": int(time.time()),
                    "model": params["model"],
                    "object": "chat.completion",
                }
                logger.debug(f"Mock completion: {comp}")
                return comp
        except MockAIError:
            raise
        except Exception as e:
            raise MockAIError(f"Unexpected error in create(): {str(e)}")

    @classmethod
    def _stream_response(
        cls, choice: Dict[str, Any], params: Params
    ) -> Iterator[CompletionChunk]:
        """
        Simulates streaming by splitting a Completion.Choice into multiple CompletionChunk objects.
        Yields each chunk sequentially with a small delay. If the original choice contains tool calls,
        a final chunk is yielded with the tool calls.
        """
        try:
            content = choice["message"]["content"]
            words = content.split()
            num_chunks = min(3, len(words))
            chunk_size = max(1, len(words) // num_chunks)
            created = int(time.time())
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])
                chunk: CompletionChunk = {
                    "id": str(uuid.uuid4()),
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": chunk_text,
                            },
                            "finish_reason": None,
                            "index": 0,
                            "logprobs": None,
                        }
                    ],
                    "created": created,
                    "model": params["model"],
                    "object": "chat.completion",
                    "service_tier": None,
                    "system_fingerprint": None,
                    "usage": None,
                }
                yield chunk
                time.sleep(0.2)
            # If tool calls are present, yield a final chunk.
            if tool_calls := choice["message"].get("tool_calls"):
                final_chunk: CompletionChunk = {
                    "id": str(uuid.uuid4()),
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": tool_calls,
                            },
                            "finish_reason": "tool_calls",
                            "index": 0,
                            "logprobs": None,
                        }
                    ],
                    "created": created,
                    "model": params["model"],
                    "object": "chat.completion",
                    "service_tier": None,
                    "system_fingerprint": None,
                    "usage": None,
                }
                yield final_chunk
        except Exception as e:
            raise MockAIError(f"Error in streaming response: {str(e)}")

    @classmethod
    def _create_mock_response_choice(
        cls, params: Params
    ) -> Completion.Choice:
        """
        Creates a mock Completion.Choice object. If tools are provided, simulates a tool call.
        """
        try:
            messages = params.get("messages", [])
            user_input = (
                messages[-1].get("content", "") if messages else ""
            )
            message: Message = {
                "role": "assistant",
                "content": f"Mock response to: {user_input}",
            }
            finish_reason = "stop"
            if tools := params.get("tools"):
                try:
                    tool_name = next(iter(tools.keys()))
                    tool_calls = [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}",
                            },
                        }
                    ]
                    message["tool_calls"] = tool_calls
                    finish_reason = "tool_calls"
                except Exception as e:
                    raise MockAIError(
                        f"Failed to create tool calls: {str(e)}"
                    )
            return {
                "message": message,
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": None,
            }
        except Exception as e:
            raise MockAIError(
                f"Failed to create mock response choice: {str(e)}"
            )

    @overload
    @classmethod
    async def acreate(
        cls,
        *,
        messages: MessagesParam,
        model: ModelParam = "gpt-4o-mini",
        stream: Literal[False] = False,
        tools: Optional[Iterable[Tool]] = None,
        **kwargs: Params,
    ) -> Completion: ...

    @overload
    @classmethod
    async def acreate(
        cls,
        *,
        messages: MessagesParam,
        model: ModelParam = "gpt-4o-mini",
        stream: Literal[True],
        tools: Optional[Iterable[Tool]] = None,
        **kwargs: Params,
    ) -> Iterator[CompletionChunk]: ...

    @classmethod
    async def acreate(
        cls, **kwargs: Any
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """
        Asynchronous version of create.
        For simplicity, reuses the synchronous implementation.
        """
        return cls.create(**kwargs)


# ----------------------------------------------------------------------------
# Completion Methods
# ----------------------------------------------------------------------------


@overload
def mock_completion(
    messages: MessagesParam,
    model: ModelParam = "gpt-4o-mini",
    *,
    stream: Literal[False] = False,
    audio: Optional[AudioParam] = None,
    frequency_penalty: Optional[float] = None,
    function_call: Optional[FunctionCallParam] = None,
    functions: Optional[Iterable[Function]] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    modalities: Optional[ModalitiesParam] = None,
    n: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    prediction: Optional[PredictionParam] = None,
    presence_penalty: Optional[float] = None,
    reasoning_effort: Optional[ReasoningEffortParam] = None,
    response_format: Optional[ResponseFormatParam] = None,
    seed: Optional[int] = None,
    service_tier: Optional[Literal["auto", "default"]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    store: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[Iterable[Tool]] = None,
    tool_choice: Optional[ToolChoiceParam] = None,
    top_logprobs: Optional[int] = None,
    user: Optional[str] = None,
    **kwargs: Params,
) -> Completion: ...


@overload
def mock_completion(
    messages: MessagesParam,
    model: ModelParam = "gpt-4o-mini",
    *,
    stream: Literal[True],
    audio: Optional[AudioParam] = None,
    frequency_penalty: Optional[float] = None,
    function_call: Optional[FunctionCallParam] = None,
    functions: Optional[Iterable[Function]] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    modalities: Optional[ModalitiesParam] = None,
    n: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    prediction: Optional[PredictionParam] = None,
    presence_penalty: Optional[float] = None,
    reasoning_effort: Optional[ReasoningEffortParam] = None,
    response_format: Optional[ResponseFormatParam] = None,
    seed: Optional[int] = None,
    service_tier: Optional[Literal["auto", "default"]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    store: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[Iterable[Tool]] = None,
    tool_choice: Optional[ToolChoiceParam] = None,
    top_logprobs: Optional[int] = None,
    user: Optional[str] = None,
    **kwargs: Params,
) -> Iterator[CompletionChunk]: ...


def mock_completion(
    messages: MessagesParam,
    model: ModelParam = "gpt-4o-mini",
    *,
    stream: Optional[bool] = False,
    audio: Optional[AudioParam] = None,
    frequency_penalty: Optional[float] = None,
    function_call: Optional[FunctionCallParam] = None,
    functions: Optional[Iterable[Function]] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    logprobs: Optional[bool] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    modalities: Optional[ModalitiesParam] = None,
    n: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    prediction: Optional[PredictionParam] = None,
    presence_penalty: Optional[float] = None,
    reasoning_effort: Optional[ReasoningEffortParam] = None,
    response_format: Optional[ResponseFormatParam] = None,
    seed: Optional[int] = None,
    service_tier: Optional[Literal["auto", "default"]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    store: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    tools: Optional[Iterable[Tool]] = None,
    tool_choice: Optional[ToolChoiceParam] = None,
    top_logprobs: Optional[int] = None,
    user: Optional[str] = None,
    **kwargs: Params,
) -> Union[Completion, Iterator[CompletionChunk]]:
    """
    Mocks the OpenAI ChatCompletion.create method.
    """
    params = {
        "messages": messages,
        "model": model,
        "audio": audio,
        "frequency_penalty": frequency_penalty,
        "function_call": function_call,
        "functions": functions,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": store,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "tools": tools,
        "tool_choice": tool_choice,
        "top_logprobs": top_logprobs,
        "user": user,
    }
    for key, value in kwargs.items():
        if key not in params:
            params[key] = value
    return MockAI.create(**params)
