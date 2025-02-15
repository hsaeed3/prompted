"""
## 💭 chatspec.mock

Contains the `MockAI` class, a barebones mocked implementation of the OpenAI client
for chat completions, as well as the mock_completion() method; similar to litellm's
completion() method.
"""


import time
import uuid
import json
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
    get_args
)
from .types import (
    Completion,
    CompletionChunk,
    CompletionMessage,
    Tool,
    CompletionFunction,
    CompletionToolCall,
)
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
                return Completion(
                    id=str(uuid.uuid4()),
                    choices=[choice],
                    created=int(time.time()),
                    model=params["model"],
                    object="chat.completion",
                )
        except MockAIError:
            raise
        except Exception as e:
            raise MockAIError(f"Unexpected error in create(): {str(e)}")

    @classmethod
    def _stream_response(
        cls, choice: Completion.Choice, params: Params
    ) -> Iterator[CompletionChunk]:
        """
        Simulates streaming by splitting a Completion.Choice into multiple CompletionChunk objects.
        """
        try:
            # Access content through the model's attributes
            content = choice.message.content
            tool_calls = choice.message.tool_calls
            created = int(time.time())

            # If there are tool calls but no content, yield only the tool calls
            if tool_calls and not content:
                chunk = CompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        CompletionChunk.Choice(
                            delta=CompletionMessage(
                                role="assistant",
                                content="",
                                name=None,
                                function_call=None,
                                tool_calls=tool_calls,
                                tool_call_id=None,
                            ),
                            finish_reason="tool_calls",
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=created,
                    model=params["model"],
                    object="chat.completion",
                )
                yield chunk
                return

            # Split content by character groups if there is content
            if content:
                chars = list(content)
                num_chunks = min(3, len(chars))
                chunk_size = max(1, len(chars) // num_chunks)

                # For all chunks except the last one
                for i in range(0, len(chars) - chunk_size, chunk_size):
                    chunk_text = "".join(chars[i : i + chunk_size])
                    chunk = CompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            CompletionChunk.Choice(
                                delta=CompletionMessage(
                                    role="assistant",
                                    content=chunk_text,
                                    name=None,
                                    function_call=None,
                                    tool_calls=None,
                                    tool_call_id=None,
                                ),
                                finish_reason="length",
                                index=0,
                                logprobs=None,
                            )
                        ],
                        created=created,
                        model=params["model"],
                        object="chat.completion",
                    )
                    yield chunk
                    time.sleep(0.2)

                # Last chunk of text
                if chars:
                    remaining_text = "".join(
                        chars[-(len(chars) % chunk_size or chunk_size) :]
                    )
                    chunk = CompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            CompletionChunk.Choice(
                                delta=CompletionMessage(
                                    role="assistant",
                                    content=remaining_text,
                                    name=None,
                                    function_call=None,
                                    tool_calls=None,
                                    tool_call_id=None,
                                ),
                                finish_reason="stop" if not tool_calls else "length",
                                index=0,
                                logprobs=None,
                            )
                        ],
                        created=created,
                        model=params["model"],
                        object="chat.completion",
                    )
                    yield chunk

            # If tool calls are present, yield a final chunk with the tool calls
            if tool_calls:
                final_chunk = CompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        CompletionChunk.Choice(
                            delta=CompletionMessage(
                                role="assistant",
                                content="",
                                name=None,
                                function_call=None,
                                tool_calls=tool_calls,
                                tool_call_id=None,
                            ),
                            finish_reason="tool_calls",
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=created,
                    model=params["model"],
                    object="chat.completion",
                )
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
            
            # Initialize tool_calls
            tool_calls: List[CompletionToolCall] = []
            finish_reason = "stop"
            
            # Process tools if present
            if tools := params.get('tools'):
                logger.debug(f"Raw tools input: {tools}")
                try:
                    # Handle tools dictionary
                    if isinstance(tools, dict):
                        for tool_data in tools.values():
                            logger.debug(f"Processing tool data: {tool_data}")
                            if isinstance(tool_data, dict) and tool_data.get("type") == "function":
                                # Generate mock arguments based on parameters
                                mock_args = {}
                                if "parameters" in tool_data["function"]:
                                    params_schema = tool_data["function"]["parameters"]
                                    if "properties" in params_schema:
                                        for param_name, param_info in params_schema["properties"].items():
                                            # Generate mock values based on type
                                            if param_info.get("type") == "string":
                                                mock_args[param_name] = "mock_string"
                                            elif param_info.get("type") == "number":
                                                mock_args[param_name] = 42
                                            elif param_info.get("type") == "boolean":
                                                mock_args[param_name] = True
                                            else:
                                                mock_args[param_name] = "mock_value"

                                # Create CompletionFunction with mock arguments
                                function = CompletionFunction(
                                    name=tool_data["function"]["name"],
                                    arguments=json.dumps(mock_args)  # Convert mock args to JSON string
                                )
                                logger.debug(f"Created function with args: {function}")
                                
                                # Create CompletionToolCall
                                tool_call = CompletionToolCall(
                                    id=str(uuid.uuid4()),
                                    type="function",
                                    function=function
                                )
                                tool_calls.append(tool_call)
                                logger.debug(f"Created tool call: {tool_call}")
                    
                    if tool_calls:
                        finish_reason = "tool_calls"
                        logger.debug(f"Final tool_calls list: {tool_calls}")
                except Exception as e:
                    logger.error(f"Failed to create tool calls: {str(e)}")
                    raise MockAIError(f"Failed to create tool calls: {str(e)}")

            # Create CompletionMessage with all fields
            message = CompletionMessage(
                role="assistant",
                content=f"Mock response to: {user_input}",
                name=None,
                function_call=None,
                tool_calls=tool_calls,
                tool_call_id=None,
            )

            logger.debug(f"Final message: {message}")

            return Completion.Choice(
                message=message,
                finish_reason=finish_reason,
                index=0,
                logprobs=None,
            )
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
