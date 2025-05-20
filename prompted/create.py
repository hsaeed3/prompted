"""
💬 prompted.create

Contains the `Create` module, which is used for creating
chat completions / structured outputs and other LLM related
operations quickly & easily using the `prompted` package.
"""

from typing import TYPE_CHECKING, Any
import logging
import functools
import inspect
import asyncio
from typing import get_type_hints

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        from litellm import (
            CustomStreamWrapper as LiteLLMStream,
            ModelResponse as LiteLLMResponse,
            EmbeddingResponse as LiteLLMEmbeddingResponse,
            ModelResponse,  # Import ModelResponse for type hinting
        )
        from instructor.client import (
            Instructor as InstructorClient,
            AsyncInstructor as AsyncInstructorClient,
        )
        from instructor import Mode as InstructorModeEnum
    except ImportError:
        pass
else:
    LiteLLMStream = Any
    LiteLLMResponse = Any
    LiteLLMEmbeddingResponse = Any
    InstructorClient = Any
    AsyncInstructorClient = Any
    InstructorModeEnum = Any
    ModelResponse = Any  # Define fallback for ModelResponse
    LitellmUnsupportedInputError = Exception  # Define fallback for the custom exception

import json
from dataclasses import dataclass
from typing import (
    Dict,
    Callable,
    List,
    Optional,
    Union,
    Generic,
    Literal,
    Type,
    TypeVar,
    Iterable,
    Tuple,
    Set,
    overload,
)
from pydantic import (
    BaseModel,
    Field,  # Use pydantic.Field instead of PydanticField
    create_model,
    ValidationError,
)

from .utils.converters import (
    convert_to_pydantic_model,
    convert_to_selection_model,
)
from .utils.identification import is_message
from .utils.formatting import format_messages, format_system_prompt
from .types import MethodType
from .types.instructor import InstructorModeParam
from .types.chat_completions import Message, MessageRole, Tool
from .types.chat_completions_params import (
    Params,
    ModelParam,
    ToolChoiceParam,
)


# ------------------------------------------------------------------------------
# TYPES
# ------------------------------------------------------------------------------


PromptType = TypeVar("PromptType", bound=Any)
SchemaType = TypeVar("SchemaType", bound=Union[Type[BaseModel], Type[Any]])
ContextType = TypeVar("ContextType", bound=Union[BaseModel, Dict[str, Any]])
AttributeType = TypeVar("AttributeType", bound=Union[List[str], Dict[str, float]])
OptionsType = TypeVar("OptionsType", bound=Set[Union[str, int, float, bool]])

# For return types
ReturnType = TypeVar("ReturnType")
AsyncReturnType = TypeVar("AsyncReturnType")
StructuredReturnType = TypeVar("StructuredReturnType", bound=BaseModel)
StreamingStructuredReturnType = TypeVar(
    "StreamingStructuredReturnType", bound=BaseModel
)


# ------------------------------------------------------------------------------
# DEPENDENCY MANAGER
#
# NOTE:
# THIS CLASS IS RESPONSIBLE FOR THE USED RESOURCES FROM `litellm` &
# `instructor`
# ------------------------------------------------------------------------------


class LitellmUnsupportedInputError(Exception):
    """
    An exception raised when an input is not supported by litellm.
    """

    pass


class _CreateClientDeps:
    """
    A class that contains the dependencies for the `_CreateClient`
    class.
    """

    is_litellm_initialized = False
    """Whether `litellm` has been initialized."""
    completion = None
    """`litellm.completion`"""
    completion_async = None
    """`litellm.acompletion`"""
    batch_completion = None
    """`litellm.batch_completion`"""
    batch_completion_models = None
    """`litellm.batch_completion_models_all_responses`"""
    embedding = None
    """`litellm.embedding`"""
    image_generation = None
    """`litellm.image_generation`"""
    audio_generation = None
    """`litellm.speech`"""
    audio_transcription = None
    """`litellm.transcription`"""

    is_instructor_initialized = False
    """Whether `instructor` has been initialized."""
    instructor_sync = None
    """Synchronous instructor client."""
    instructor_async = None
    """Asynchronous instructor client."""
    _instructor_mode_enum = None  # Store the actual Instructor Mode Enum

    def __init__(self):
        pass

    def initialize_litellm(self):
        """
        Initialize the litellm client.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "`litellm` is not installed. Please install it with `pip install litellm`."
                f"Or use `pip install chatspec[client]` to install it."
            )

        litellm.drop_params = True
        litellm.modify_params = True
        litellm.ssl_verify = False

        self.completion = staticmethod(litellm.completion)
        self.completion_async = staticmethod(litellm.acompletion)
        self.batch_completion = staticmethod(litellm.batch_completion)
        self.batch_completion_models = staticmethod(
            litellm.batch_completion_models_all_responses
        )
        self.embedding = staticmethod(litellm.embedding)
        self.image_generation = staticmethod(litellm.image_generation)
        self.audio_generation = staticmethod(litellm.speech)
        self.audio_transcription = staticmethod(litellm.transcription)
        self.is_litellm_initialized = True

    def initialize_instructor(self):
        """
        Initialize the instructor client.
        """
        try:
            import instructor
        except ImportError:
            raise ImportError(
                "`instructor` is not installed. Please install it with `pip install instructor`."
                f"Or use `pip install chatspec[client]` to install it."
            )

        if not self.is_litellm_initialized:
            self.initialize_litellm()

        self.instructor_sync = instructor.from_litellm(self.completion)
        self.instructor_async = instructor.from_litellm(self.completion_async)
        # Store the actual Mode enum for internal use
        self._instructor_mode_enum = instructor.Mode

        self.is_instructor_initialized = True

    @property
    def instructor_mode(self) -> str:
        """
        Get the instructor mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        return self.instructor_sync.mode.value

    @instructor_mode.setter
    def instructor_mode(self, mode: InstructorModeParam = "tool_call"):
        """
        Set the instructor mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        # Use the stored Mode enum
        self.instructor_sync.mode = self._instructor_mode_enum(mode)

    @property
    def instructor_async_mode(self) -> str:
        """
        Get the instructor async mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        return self.instructor_async.mode.value

    @instructor_async_mode.setter
    def instructor_async_mode(self, mode: InstructorModeParam = "tool_call"):
        """
        Set the instructor async mode.
        """
        if not self.is_instructor_initialized:
            self.initialize_instructor()

        # Use the stored Mode enum
        self.instructor_async.mode = self._instructor_mode_enum(mode)


CLIENT_DEPS: _CreateClientDeps = _CreateClientDeps()
"""Singleton instance of `ClientDeps`."""


def get_client_deps() -> _CreateClientDeps:
    """
    Retrieves the singleton instance of `ClientDeps`.
    """
    return CLIENT_DEPS


# ------------------------------------------------------------------------------
# CREATE CLIENT HELPERS
# ------------------------------------------------------------------------------


def _parse_string_schema(
    schema_str: str, model_name: str, model_description: Optional[str]
) -> Type[BaseModel]:
    """
    Parses a string like "name: str, age: int" or "name, age" into a Pydantic model.
    Fields without type annotations default to Optional[str].
    """
    fields_config: Dict[str, Tuple[Type, Any]] = {}
    field_definitions = [f.strip() for f in schema_str.split(",")]

    for i, definition in enumerate(field_definitions):
        if not definition:
            continue
        parts = [p.strip() for p in definition.split(":")]
        field_name = parts[0]
        field_type_str: Optional[str] = None
        if len(parts) > 1:
            field_type_str = parts[1]

        actual_field_type: Type = Optional[str]  # Default type
        if field_type_str:
            # Attempt to evaluate the type string
            try:
                # A simple map for common types, extend as needed
                type_map = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "List[str]": List[str],
                    "List[int]": List[int],  # etc.
                    "list": list,
                    "dict": dict,
                    "any": Any,
                }
                # Check for Literal types manually as they are common in schemas
                if field_type_str.lower().startswith("literal["):
                    # This is a basic attempt, might need more robust parsing
                    try:
                        # Extract content within brackets and evaluate
                        literal_content = field_type_str[len("literal[") : -1]
                        # Safely evaluate the content (e.g., "1, 'a', True")
                        # This is risky, consider a safer approach if possible
                        # For now, let's assume simple comma-separated values
                        options = [val.strip() for val in literal_content.split(",")]
                        # Attempt to infer types or keep as strings
                        evaluated_options = []
                        for opt in options:
                            try:
                                evaluated_options.append(
                                    json.loads(opt)
                                )  # Try parsing as JSON (handles strings, numbers, bools)
                            except json.JSONDecodeError:
                                evaluated_options.append(
                                    opt
                                )  # Fallback to string if not JSON
                        actual_field_type = Literal[tuple(evaluated_options)]  # type: ignore
                        logger.debug(
                            f"Parsed Literal type for field '{field_name}': {actual_field_type}"
                        )
                    except Exception as parse_literal_e:
                        logger.warning(
                            f"Failed to parse Literal type string '{field_type_str}' for field '{field_name}': {parse_literal_e}. Defaulting to Optional[str]."
                        )
                        actual_field_type = Optional[str]  # Fallback on error
                elif field_type_str.lower() in type_map:
                    actual_field_type = type_map[field_type_str.lower()]
                else:
                    # For more complex types, this might need more robust parsing or eval (use with caution)
                    logger.warning(
                        f"Unknown type string '{field_type_str}' for field '{field_name}'. Defaulting to Optional[str]."
                    )
            except Exception as e:
                logger.warning(
                    f"Error parsing type string '{field_type_str}' for field '{field_name}': {e}. Defaulting to Optional[str]."
                )
        else:  # No type specified, default to Optional[str]
            pass

        fields_config[field_name] = (
            actual_field_type,
            Field(default=None, description=f"Field '{field_name}'"),
        )

    if not fields_config:
        raise ValueError(
            f"Could not parse any fields from schema string: '{schema_str}'"
        )

    return create_model(
        model_name,
        __base__=BaseModel,
        __doc__=model_description,
        **fields_config,
    )  # type: ignore


def _prepare_llm_call_params(
    model: Union[ModelParam, List[ModelParam]],  # Allow list of models
    messages: List[Message],
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[Iterable[Tool]] = None,
    tool_choice: Optional[ToolChoiceParam] = None,
    params: Optional[Params] = None,
    stream: Optional[bool] = False,
) -> Dict[str, Any]:
    """Prepares the dictionary of parameters for litellm/instructor calls."""
    logger.debug(f"Preparing LLM call parameters for model: {model}, stream: {stream}")
    call_params: Dict[str, Any] = {
        "model": model,  # litellm handles list of models for specific functions
        "messages": messages,
    }
    if stream is not None:  # Ensure stream is always present for clarity, even if False
        call_params["stream"] = stream

    if response_model:
        call_params["response_model"] = response_model

    # Prioritize direct parameters for tools and tool_choice,
    # then check within the 'params' dictionary.
    if tools:
        call_params["tools"] = list(tools)  # Ensure it's a list
        logger.debug("Using 'tools' from direct _prepare_llm_call_params argument.")
    elif params and params.get("tools") is not None:
        call_params["tools"] = list(params["tools"])  # type: ignore
        logger.debug(
            "Using 'tools' from 'params' dictionary passed to _prepare_llm_call_params."
        )

    if tool_choice:
        call_params["tool_choice"] = tool_choice
        logger.debug(
            "Using 'tool_choice' from direct _prepare_llm_call_params argument."
        )
    elif params and params.get("tool_choice") is not None:
        call_params["tool_choice"] = params["tool_choice"]  # type: ignore
        logger.debug(
            "Using 'tool_choice' from 'params' dictionary passed to _prepare_llm_call_params."
        )

    # Unpack relevant fields from Params
    if params:
        # These are common CompletionParams fields
        completion_param_fields = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "temperature",
            "top_p",
            "user",
            "top_logprobs",
        ]
        # Handle stream_options specifically
        if stream and params.get("stream_options") is not None:
            call_params["stream_options"] = params.get("stream_options")
        elif (
            stream and params.get("include_usage") is True
        ):  # Legacy support from foundry GeneratorParams
            call_params["stream_options"] = {"include_usage": True}

        for field_name in completion_param_fields:
            # Check if key exists in params TypedDict and is not None
            if field_name in params and params.get(field_name) is not None:  # type: ignore
                call_params[field_name] = params.get(field_name)  # type: ignore

        # For instructor, some params might be passed differently or handled by instructor itself
        # e.g. api_key, base_url are client level.
        # Litellm specific params from `Params` might also be passed if not covered.
        # For example, `litellm.completion` can take `api_key`, `base_url` directly.
        client_param_fields = [
            "api_key",
            "base_url",
            "timeout",
            "max_retries",
            "metadata",
        ]
        for field_name in client_param_fields:
            if field_name in params and params.get(field_name) is not None:  # type: ignore
                call_params[field_name] = params.get(field_name)  # type: ignore
    logger.debug(f"Final LLM call parameters prepared: {', '.join(call_params.keys())}")
    return call_params


def _format_compiled_messages(
    prompt: PromptType,
    instructions: Optional[str] = None,  # Simplified to str for clarity
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> List[Message]:
    """
    Formats various inputs into a structured list of messages for an LLM call.
    """
    logger.debug(
        f"Formatting messages with prompt type: {type(prompt)}, instructions: {instructions is not None}, existing_messages: {existing_messages is not None}"
    )
    compiled_messages: List[Message] = []

    # 1. Start with existing messages if provided
    if existing_messages:
        # Handle potential nested message structures
        for msg in existing_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if isinstance(msg["content"], list) and all(
                    is_message(m) for m in msg["content"]
                ):
                    # Content is a list of nested messages - extract and preserve their roles
                    for nested_msg in msg["content"]:
                        compiled_messages.append(nested_msg)
                else:
                    # Regular message
                    compiled_messages.append(msg)
            else:
                # Add as is and rely on format_messages to normalize
                compiled_messages.append(msg)

        # Normalize any remaining non-standard formats
        if compiled_messages:
            compiled_messages = format_messages(compiled_messages)
        else:
            # If nothing was properly extracted, fall back to the old behavior
            compiled_messages.extend(format_messages(existing_messages))

    # 2. Incorporate instructions as a system message
    if instructions is not None:
        # normalize_system_prompt merges or prepends the system prompt
        compiled_messages = format_system_prompt(
            compiled_messages, system_prompt=instructions
        )

    # 3. Add the main prompt
    if (
        prompt is None and not compiled_messages
    ):  # Handle case where prompt is None but instructions might exist
        if any(msg["role"] == "system" for msg in compiled_messages):
            compiled_messages.append(
                {
                    "role": "user",
                    "content": "Generate based on the system instructions.",
                }
            )
        else:
            raise ValueError(
                "Prompt cannot be None if there are no existing messages or instructions."
            )
    elif prompt is not None:
        if isinstance(prompt, str):
            compiled_messages.append({"role": role_for_prompt, "content": prompt})
        elif is_message(prompt):  # Assumes prompt is a single Message dict
            # Check for nested messages
            if isinstance(prompt.get("content"), list) and all(
                is_message(m) for m in prompt["content"]
            ):  # type: ignore
                # Extract nested messages while preserving their roles
                for nested_msg in prompt["content"]:  # type: ignore
                    compiled_messages.append(nested_msg)
            else:
                # Regular message
                compiled_messages.append(prompt)  # type: ignore
        elif isinstance(prompt, list) and all(
            is_message(m) for m in prompt
        ):  # Prompt is already a list of messages
            # Process each message for potential nesting
            for msg in prompt:
                if isinstance(msg.get("content"), list) and all(
                    is_message(m) for m in msg["content"]
                ):  # type: ignore
                    # Extract nested messages
                    for nested_msg in msg["content"]:  # type: ignore
                        compiled_messages.append(nested_msg)
                else:
                    # Regular message
                    compiled_messages.append(msg)  # type: ignore
        else:
            # Attempt to coerce to string if not a recognized message format
            logger.debug(
                f"Prompt type {type(prompt)} not a string or Message dict/list. Coercing to string for role '{role_for_prompt}'."
            )
            compiled_messages.append({"role": role_for_prompt, "content": str(prompt)})

    # 4. Ensure there's at least one message.
    if not compiled_messages:
        # This case should ideally be caught by prompt=None check, but as a safeguard:
        raise ValueError("Cannot format messages: Resulting message list is empty.")

    # 5. If only system messages exist, add a default user message.
    if all(msg.get("role") == "system" for msg in compiled_messages):
        logger.debug(
            "Only system messages found after formatting. Adding a default user 'Proceed' message."
        )
        compiled_messages.append(
            {
                "role": "user",
                "content": "Proceed based on the system instructions.",
            }
        )

    logger.debug(
        f"Formatted {len(compiled_messages)} messages: {[msg.get('role') for msg in compiled_messages]}"
    )
    return compiled_messages


@dataclass
class Create:
    """
    A class for creating chat completions, structured outputs,
    and other LLM related operations quickly & easily using
    the `prompted` package.
    """

    CLIENT_DEPS: _CreateClientDeps = get_client_deps()

    @staticmethod
    def _ensure_litellm_initialized():
        if not Create.CLIENT_DEPS.is_litellm_initialized:
            logger.info("Initializing LiteLLM client")
            Create.CLIENT_DEPS.initialize_litellm()

    @staticmethod
    def _get_sync_instructor_client() -> InstructorClient:
        if not Create.CLIENT_DEPS.is_instructor_initialized:
            logger.info("Initializing Instructor sync client")
            Create.CLIENT_DEPS.initialize_instructor()
        return Create.CLIENT_DEPS.instructor_sync

    @staticmethod
    def _get_async_instructor_client() -> (
        AsyncInstructorClient
    ):  # Use AsyncInstructorClient here
        if not Create.CLIENT_DEPS.is_instructor_initialized:
            logger.info("Initializing Instructor async client")
            Create.CLIENT_DEPS.initialize_instructor()
        return Create.CLIENT_DEPS.instructor_async

    # --- Embedding Method ---
    @staticmethod
    def embedding(
        input: Union[str, List[str]],
        model: ModelParam = "text-embedding-ada-002",  # Common default
        params: Optional[Params] = None,
    ) -> LiteLLMEmbeddingResponse:
        """Generates embeddings for the given input text(s)."""
        logger.info(f"Generating embeddings with model {model}")
        Create._ensure_litellm_initialized()
        call_params = {"input": input, "model": model}
        if params:  # Pass other relevant embedding params if any
            if params.get("dimensions"):
                call_params["dimensions"] = params["dimensions"]  # type: ignore
            if params.get("encoding_format"):
                call_params["encoding_format"] = params["encoding_format"]  # type: ignore
            if params.get("user"):
                call_params["user"] = params["user"]  # type: ignore
            # Include client params if needed by litellm.embedding directly
            client_param_fields = [
                "api_key",
                "base_url",
                "timeout",
                "max_retries",
            ]
            for field_name in client_param_fields:
                if field_name in params and params.get(field_name) is not None:  # type: ignore
                    call_params[field_name] = params.get(field_name)  # type: ignore
        try:
            return Create.CLIENT_DEPS.embedding(**call_params)
        except Exception as e:
            logger.error(
                f"Error during embedding generation with model {model}: {e}",
                exc_info=True,
            )
            raise

    # --- from_prompt ---

    @staticmethod
    def from_prompt(
        prompt: PromptType,
        instructions: Optional[str] = None,
        model: Union[ModelParam, List[ModelParam]] = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
    ) -> Union[str, List[str], LiteLLMStream]:
        """
        Generates a text response or streams text from an LLM.
        If `model` is a list, multiple models are called (streaming not supported for this).
        """
        logger.info(f"Generating text with model(s): {model}, stream: {stream}")
        Create._ensure_litellm_initialized()
        messages = _format_compiled_messages(
            prompt, instructions, existing_messages, role_for_prompt
        )

        if isinstance(model, list):
            if stream:
                raise LitellmUnsupportedInputError(
                    "Streaming is not supported when 'model' is a list for from_prompt. Call one model at a time for streaming."
                )
            call_params = _prepare_llm_call_params(
                model, messages, params=model_params, stream=False
            )  # type: ignore
            try:
                # Use batch_completion_models_all_responses for multiple models
                logger.info(f"Calling batch completion for {len(model)} models")
                responses: List[ModelResponse] = (
                    Create.CLIENT_DEPS.batch_completion_models(**call_params)
                )  # type: ignore
                return [
                    str(r.choices[0].message.content)
                    if r.choices
                    and r.choices[0].message
                    and r.choices[0].message.content
                    else ""
                    for r in responses
                ]
            except Exception as e:
                logger.error(
                    f"Error during batch text generation with models {model}: {e}",
                    exc_info=True,
                )
                raise
        else:  # Single model
            call_params = _prepare_llm_call_params(
                model, messages, params=model_params, stream=stream
            )
            try:
                logger.info(f"Calling completion for model {model}")
                response_or_stream = Create.CLIENT_DEPS.completion(**call_params)
                if stream:
                    logger.debug("Returning stream wrapper")
                    return response_or_stream  # type: ignore
                else:  # LiteLLMModelResponse
                    if (
                        response_or_stream.choices
                        and response_or_stream.choices[0].message
                        and response_or_stream.choices[0].message.content
                    ):
                        return str(response_or_stream.choices[0].message.content)
                    logger.warning(f"No content in LLM response: {response_or_stream}")
                    return ""
            except Exception as e:
                logger.error(
                    f"Error during text generation with model {model}: {e}",
                    exc_info=True,
                )
                raise

    # --- async_from_prompt (mirroring from_prompt) ---

    @staticmethod
    async def async_from_prompt(
        prompt: PromptType,
        instructions: Optional[str] = None,
        model: Union[ModelParam, List[ModelParam]] = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
    ) -> Union[str, List[str], LiteLLMStream]:
        logger.info(
            f"Generating text asynchronously with model(s): {model}, stream: {stream}"
        )
        Create._ensure_litellm_initialized()
        messages = _format_compiled_messages(
            prompt, instructions, existing_messages, role_for_prompt
        )

        if isinstance(model, list):
            if stream:
                raise LitellmUnsupportedInputError(
                    "Streaming is not supported when 'model' is a list for async_from_prompt."
                )
            call_params = _prepare_llm_call_params(
                model, messages, params=model_params, stream=False
            )  # type: ignore
            try:
                # LiteLLM's batch_completion_models is sync. Async version might need separate handling or not be directly available.
                # For now, let's assume if an async version of batch_completion_models exists, it's used.
                # If not, this part would need adjustment or be documented as synchronous.
                # Assuming `Create.CLIENT_DEPS.batch_completion_models_async` if it existed.
                # For now, raising error as litellm's batch functions are typically sync.
                raise NotImplementedError(
                    "Async batch completion for multiple models is not directly supported by litellm's typical async patterns. Call models individually for async operations."
                )

            except Exception as e:
                logger.error(
                    f"Error during async batch text generation with models {model}: {e}",
                    exc_info=True,
                )
                raise
        else:  # Single model
            call_params = _prepare_llm_call_params(
                model, messages, params=model_params, stream=stream
            )
            try:
                logger.info(f"Calling async completion for model {model}")
                response_or_stream = await Create.CLIENT_DEPS.completion_async(
                    **call_params
                )
                if stream:
                    logger.debug("Returning async stream wrapper")
                    return response_or_stream  # type: ignore
                else:  # LiteLLMModelResponse
                    if (
                        response_or_stream.choices
                        and response_or_stream.choices[0].message
                        and response_or_stream.choices[0].message.content
                    ):
                        return str(response_or_stream.choices[0].message.content)
                    logger.warning(
                        f"No content in async LLM response: {response_or_stream}"
                    )
                    return ""
            except Exception as e:
                logger.error(
                    f"Error during async text generation with model {model}: {e}",
                    exc_info=True,
                )
                raise

    # --- from_schema ---

    @staticmethod
    def from_schema(
        schema: SchemaType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        n: int = 1,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[
        StructuredReturnType,
        List[StructuredReturnType],
        Iterable[StructuredReturnType],
    ]:
        """
        Generates structured data matching the given schema.
        Schema can be a Pydantic model, basic type, function, or string definition.
        Supports iterative generation for complex schemas and batch generation (n > 1).
        """
        logger.info(
            f"Generating structured data with model {model}, schema type: {type(schema).__name__}, iterative: {iterative}, n: {n}, stream: {stream}"
        )
        client = Create._get_sync_instructor_client()
        actual_response_model: Type[BaseModel]
        default_model_name = name_override or "GeneratedSchema"
        default_model_desc = description_override or "Schema generated from prompt."

        if isinstance(schema, str):
            actual_response_model = _parse_string_schema(
                schema, default_model_name, default_model_desc
            )
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            actual_response_model = schema
            if name_override:
                actual_response_model.__name__ = name_override
            if description_override:
                actual_response_model.__doc__ = description_override
        elif callable(schema) and not isinstance(schema, type):  # A function
            m = convert_to_pydantic_model(
                schema,
                name=name_override or schema.__name__,
                description=description_override or schema.__doc__,
            )
            if not (isinstance(m, type) and issubclass(m, BaseModel)):
                raise TypeError("Could not convert function to Pydantic Model")
            actual_response_model = m
        elif isinstance(schema, type):  # Basic type like str, int
            m = convert_to_pydantic_model(
                schema,
                name=name_override or f"{schema.__name__}Response",
                description=description_override,
            )
            if not (isinstance(m, type) and issubclass(m, BaseModel)):
                raise TypeError("Could not convert basic type to Pydantic Model")
            actual_response_model = m
        else:
            raise TypeError(
                f"Unsupported schema type: {type(schema).__name__}. Must be Pydantic BaseModel, basic type, function, or parsable string."
            )

        if prompt is None and not instructions:
            prompt = f"Generate a valid instance of the {actual_response_model.__name__} schema."
            if actual_response_model.__doc__:
                prompt += f"\nSchema description: {actual_response_model.__doc__}"
            if n > 1:
                prompt += f" Please generate {n} distinct instances."

        # Handle iterative generation
        if iterative and n == 1:  # Iterative only makes sense for n=1 for now.
            if not actual_response_model.model_fields:
                logger.warning(
                    f"Schema {actual_response_model.__name__} has no fields for iterative generation."
                )
                # Fallback to non-iterative or handle as error
            else:
                logger.info(
                    f"Starting iterative generation for schema {actual_response_model.__name__}"
                )
                final_data: Dict[str, Any] = {}
                current_context_messages = _format_compiled_messages(
                    prompt,
                    instructions,
                    existing_messages,
                    role_for_prompt,
                )

                for (
                    field_name,
                    field_info,
                ) in actual_response_model.model_fields.items():
                    field_prompt = (
                        f"Based on the overall request and previous context, generate the value for the field '{field_name}'.\n"
                        f"Field description: {field_info.description or 'N/A'}.\n"
                        f"Type expected: {field_info.annotation}.\n"
                        f"Current generated context: {json.dumps(final_data) if final_data else 'None'}"
                    )

                    # Create a temporary model for just this field
                    SingleFieldModel = create_model(
                        f"{actual_response_model.__name__}_{field_name.capitalize()}Field",
                        **{
                            field_name: (
                                field_info.annotation,
                                Field(description=field_info.description),
                            )
                        },  # Use pydantic.Field
                    )

                    field_messages = current_context_messages + [
                        {"role": "user", "content": field_prompt}
                    ]
                    call_params = _prepare_llm_call_params(
                        model,
                        field_messages,
                        response_model=SingleFieldModel,
                        params=model_params,
                        stream=False,  # Stream within iterative is complex
                    )
                    try:
                        logger.debug(f"Generating field '{field_name}' iteratively")
                        client.mode = CLIENT_DEPS._instructor_mode_enum(
                            mode
                        )  # Use stored Enum
                        field_response = client.chat.completions.create(**call_params)
                        client.mode = CLIENT_DEPS._instructor_mode_enum(
                            "tool_call"
                        )  # Use stored Enum
                        if hasattr(field_response, field_name):
                            final_data[field_name] = getattr(field_response, field_name)
                            current_context_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Generated value for {field_name}: {final_data[field_name]}",
                                }
                            )
                        else:
                            logger.warning(
                                f"Iterative generation: LLM did not return field '{field_name}'."
                            )
                    except Exception as e_iter:
                        logger.error(
                            f"Error during iterative generation of field '{field_name}': {e_iter}",
                            exc_info=True,
                        )
                        # Decide: stop, or continue with missing field? For now, stop.
                        raise
                try:
                    logger.info(
                        f"Completed iterative generation for schema {actual_response_model.__name__}"
                    )
                    return actual_response_model.model_validate(final_data)
                except ValidationError as ve:
                    logger.error(
                        f"Validation failed for iteratively generated model {actual_response_model.__name__}: {ve}",
                        exc_info=True,
                    )
                    raise

        # Non-iterative or n > 1
        response_model_for_call = (
            List[actual_response_model] if n > 1 else actual_response_model
        )  # type: ignore

        # Adjust prompt for n > 1 if not already handled
        final_prompt_str = str(prompt)
        if n > 1 and f"generate {n} distinct instances" not in final_prompt_str.lower():
            final_prompt_str += f" Please generate {n} distinct instances."

        messages = _format_compiled_messages(
            final_prompt_str,
            instructions,
            existing_messages,
            role_for_prompt,
        )
        call_params = _prepare_llm_call_params(
            model,
            messages,
            response_model=response_model_for_call,
            params=model_params,
            stream=stream,
        )

        try:
            logger.info(
                f"Calling instructor with mode {mode} for schema {actual_response_model.__name__}"
            )
            client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
            response = client.chat.completions.create(**call_params)
            client.mode = CLIENT_DEPS._instructor_mode_enum(
                "tool_call"
            )  # Use stored Enum
            return response  # type: ignore
        except Exception as e:
            logger.error(
                f"Error in from_schema with model {model} for schema {actual_response_model.__name__}: {e}",
                exc_info=True,
            )
            raise

    # --- async_from_schema (mirroring from_schema) ---

    @staticmethod
    async def async_from_schema(
        schema: SchemaType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        n: int = 1,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[
        StructuredReturnType,
        List[StructuredReturnType],
        Iterable[StructuredReturnType],
    ]:
        client = Create._get_async_instructor_client()
        actual_response_model: Type[BaseModel]
        default_model_name = name_override or "GeneratedSchema"
        default_model_desc = description_override or "Schema generated from prompt."

        if isinstance(schema, str):
            actual_response_model = _parse_string_schema(
                schema, default_model_name, default_model_desc
            )
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            actual_response_model = schema
            if name_override:
                actual_response_model.__name__ = name_override
            if description_override:
                actual_response_model.__doc__ = description_override
        elif callable(schema) and not isinstance(schema, type):  # A function
            m = convert_to_pydantic_model(
                schema,
                name=name_override or schema.__name__,
                description=description_override or schema.__doc__,
            )
            if not (isinstance(m, type) and issubclass(m, BaseModel)):
                raise TypeError("Could not convert function to Pydantic Model")
            actual_response_model = m
        elif isinstance(schema, type):  # Basic type like str, int
            m = convert_to_pydantic_model(
                schema,
                name=name_override or f"{schema.__name__}Response",
                description=description_override,
            )
            if not (isinstance(m, type) and issubclass(m, BaseModel)):
                raise TypeError("Could not convert basic type to Pydantic Model")
            actual_response_model = m
        else:
            raise TypeError(
                f"Unsupported schema type: {type(schema).__name__}. Must be Pydantic BaseModel, basic type, function, or parsable string."
            )

        if prompt is None and not instructions:
            prompt = f"Generate a valid instance of the {actual_response_model.__name__} schema."
            if actual_response_model.__doc__:
                prompt += f"\nSchema description: {actual_response_model.__doc__}"
            if n > 1:
                prompt += f" Please generate {n} distinct instances."

        if iterative and n == 1:
            if not actual_response_model.model_fields:
                logger.warning(
                    f"Schema {actual_response_model.__name__} has no fields for iterative generation."
                )
            else:
                final_data: Dict[str, Any] = {}
                current_context_messages = _format_compiled_messages(
                    prompt,
                    instructions,
                    existing_messages,
                    role_for_prompt,
                )
                for (
                    field_name,
                    field_info,
                ) in actual_response_model.model_fields.items():
                    field_prompt = (
                        f"Based on the overall request and previous context, generate the value for the field '{field_name}'.\n"
                        f"Field description: {field_info.description or 'N/A'}.\n"
                        f"Type expected: {field_info.annotation}.\n"
                        f"Current generated context: {json.dumps(final_data) if final_data else 'None'}"
                    )
                    SingleFieldModel = create_model(
                        f"{actual_response_model.__name__}_{field_name.capitalize()}Field",
                        **{
                            field_name: (
                                field_info.annotation,
                                Field(description=field_info.description),
                            )
                        },  # Use pydantic.Field
                    )
                    field_messages = current_context_messages + [
                        {"role": "user", "content": field_prompt}
                    ]
                    call_params = _prepare_llm_call_params(
                        model,
                        field_messages,
                        response_model=SingleFieldModel,
                        params=model_params,
                        stream=False,
                    )
                    try:
                        client.mode = CLIENT_DEPS._instructor_mode_enum(
                            mode
                        )  # Use stored Enum
                        field_response = await client.chat.completions.create(
                            **call_params
                        )
                        client.mode = CLIENT_DEPS._instructor_mode_enum(
                            "tool_call"
                        )  # Use stored Enum
                        if hasattr(field_response, field_name):
                            val = getattr(field_response, field_name)
                            final_data[field_name] = val
                            current_context_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Generated value for {field_name}: {final_data[field_name]}",
                                }
                            )
                        else:
                            logger.warning(
                                f"Iterative generation: LLM did not return field '{field_name}'."
                            )
                    except Exception as e_iter_async:
                        logger.error(
                            f"Error during async iterative generation of field '{field_name}': {e_iter_async}",
                            exc_info=True,
                        )
                        raise
                try:
                    return actual_response_model.model_validate(final_data)
                except ValidationError as ve_async:
                    logger.error(
                        f"Validation failed for async iteratively generated model {actual_response_model.__name__}: {ve_async}",
                        exc_info=True,
                    )
                    raise

        response_model_for_call = (
            List[actual_response_model] if n > 1 else actual_response_model
        )  # type: ignore
        final_prompt_str = str(prompt)
        if n > 1 and f"generate {n} distinct instances" not in final_prompt_str.lower():
            final_prompt_str += f" Please generate {n} distinct instances."

        messages = _format_compiled_messages(
            final_prompt_str,
            instructions,
            existing_messages,
            role_for_prompt,
        )
        call_params = _prepare_llm_call_params(
            model,
            messages,
            response_model=response_model_for_call,
            params=model_params,
            stream=stream,
        )
        try:
            client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
            response = await client.chat.completions.create(**call_params)
            client.mode = CLIENT_DEPS._instructor_mode_enum(
                "tool_call"
            )  # Use stored Enum
            return response  # type: ignore
        except Exception as e_async:
            logger.error(
                f"Error in async_from_schema with model {model} for schema {actual_response_model.__name__}: {e_async}",
                exc_info=True,
            )
            raise

    # --- from_options ---

    @staticmethod
    def from_options(
        options: OptionsType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        n: int = 1,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[BaseModel, List[BaseModel], Iterable[BaseModel]]:
        """
        Prompts the LLM to select one or more options from a given set.
        The 'options' set is converted into a Literal type for schema generation.
        """
        logger.info(
            f"Create.from_options called with n={n}, stream={stream}, model='{model}'"
        )
        logger.debug(
            f"Options provided: {options}, name_override: {name_override}, description_override: {description_override}, instructions: {instructions}"
        )

        if not isinstance(options, set) or not options:
            logger.error("Options must be a non-empty set.")
            raise ValueError(
                "Options must be a non-empty set of strings, ints, floats, or bools."
            )

        options_list_str = sorted([str(opt) for opt in options])
        literal_options_type = Literal[tuple(options_list_str)]  # type: ignore
        logger.debug(f"Converted options to sorted list of strings: {options_list_str}")

        selection_model_name = name_override or "OptionSelection"
        selection_model_desc = (
            description_override
            or f"Select one option from: {', '.join(options_list_str)}"
        )
        logger.info(
            f"Using selection model name: '{selection_model_name}', description: '{selection_model_desc}'"
        )

        selection_schema: Type[BaseModel]
        if all(isinstance(opt, str) for opt in options):
            logger.debug(
                "All options are strings, using create_selection_model utility."
            )
            selection_schema = convert_to_selection_model(
                name=selection_model_name,
                description=selection_model_desc,
                fields=options_list_str,  # create_selection_model expects List[str]
            )
        else:
            logger.debug(
                "Options include non-strings or mixed types, creating custom Pydantic model with Literal field."
            )
            SelectionModelSchema = create_model(
                selection_model_name,
                selection=(
                    literal_options_type,
                    Field(..., description="The selected option."),
                ),
                __base__=BaseModel,
                __doc__=selection_model_desc,
            )
            selection_schema = SelectionModelSchema
        logger.info(f"Generated selection schema: {selection_schema.__name__}")

        final_prompt: PromptType
        if prompt is None:
            final_prompt = f"Please select {n if n > 1 else 'an'} option from the available choices: {', '.join(options_list_str)}."
            logger.info(f"Prompt not provided, generated default prompt.")
            logger.debug(f"Default prompt: '{final_prompt}'")
        else:
            final_prompt = prompt
            logger.debug(f"Using provided prompt: '{final_prompt}'")

        logger.info(
            f"Calling Create.from_schema with schema '{selection_schema.__name__}' for option selection."
        )
        return Create.from_schema(  # type: ignore
            schema=selection_schema,
            prompt=final_prompt,
            instructions=instructions,
            model=model,
            model_params=model_params,
            n=n,
            stream=stream,
            existing_messages=existing_messages,
            role_for_prompt=role_for_prompt,
            mode=mode,
            name_override=selection_model_name,
            description_override=selection_model_desc,
        )

    # --- async_from_options (mirroring from_options) ---

    @staticmethod
    async def async_from_options(
        options: OptionsType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        n: int = 1,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[BaseModel, List[BaseModel], Iterable[BaseModel]]:
        logger.info(
            f"Create.async_from_options called with n={n}, stream={stream}, model='{model}'"
        )
        logger.debug(
            f"Options provided (async): {options}, name_override: {name_override}, description_override: {description_override}, instructions: {instructions}"
        )

        if not isinstance(options, set) or not options:
            logger.error("Options must be a non-empty set (async).")
            raise ValueError(
                "Options must be a non-empty set of strings, ints, floats, or bools."
            )

        options_list_str = sorted([str(opt) for opt in options])
        literal_options_type = Literal[tuple(options_list_str)]  # type: ignore
        logger.debug(
            f"Converted options to sorted list of strings (async): {options_list_str}"
        )

        selection_model_name = name_override or "OptionSelection"
        selection_model_desc = (
            description_override
            or f"Select one option from: {', '.join(options_list_str)}"
        )
        logger.info(
            f"Using selection model name (async): '{selection_model_name}', description: '{selection_model_desc}'"
        )

        selection_schema: Type[BaseModel]
        if all(isinstance(opt, str) for opt in options):
            logger.debug(
                "All options are strings, using create_selection_model utility (async)."
            )
            selection_schema = convert_to_selection_model(
                name=selection_model_name,
                description=selection_model_desc,
                fields=options_list_str,
            )
        else:
            logger.debug(
                "Options include non-strings or mixed types, creating custom Pydantic model with Literal field (async)."
            )
            SelectionModelSchema = create_model(
                selection_model_name,
                selection=(
                    literal_options_type,
                    Field(..., description="The selected option."),
                ),
                __base__=BaseModel,
                __doc__=selection_model_desc,
            )
            selection_schema = SelectionModelSchema
        logger.info(f"Generated selection schema (async): {selection_schema.__name__}")

        final_prompt: PromptType
        if prompt is None:
            final_prompt = f"Please select {n if n > 1 else 'an'} option from the available choices: {', '.join(options_list_str)}."
            logger.info(f"Prompt not provided, generated default prompt (async).")
            logger.debug(f"Default prompt (async): '{final_prompt}'")
        else:
            final_prompt = prompt
            logger.debug(f"Using provided prompt (async): '{final_prompt}'")

        logger.info(
            f"Calling Create.async_from_schema with schema '{selection_schema.__name__}' for option selection (async)."
        )
        return await Create.async_from_schema(  # type: ignore
            schema=selection_schema,
            prompt=final_prompt,
            instructions=instructions,
            model=model,
            model_params=model_params,
            n=n,
            stream=stream,
            existing_messages=existing_messages,
            role_for_prompt=role_for_prompt,
            mode=mode,
            name_override=selection_model_name,
            description_override=selection_model_desc,
        )

    # --- from_context ---
    @overload
    @staticmethod
    def from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: Literal[True] = True,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Iterable[ContextType]: ...  # Type of context

    @overload
    @staticmethod
    def from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: Literal[False] = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> ContextType: ...

    @staticmethod
    def from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[ContextType, Iterable[ContextType]]:
        """
        Updates an existing Pydantic model instance or dictionary based on the prompt.
        First, it determines which fields to update, then generates new values for them.
        """
        logger.info(
            f"Create.from_context called for context type {type(context).__name__}, iterative={iterative}, stream={stream}, update_context={update_context}"
        )
        logger.debug(
            f"Context provided: {type(context)}, name_override: {name_override}, description_override: {description_override}, prompt: {prompt}, instructions: {instructions}"
        )

        client = Create._get_sync_instructor_client()
        context_schema_type: Type[BaseModel]
        context_instance: Optional[BaseModel] = None
        original_data: Dict[str, Any]

        if isinstance(context, BaseModel):
            context_schema_type = type(context)
            context_instance = context
            original_data = context.model_dump()
            logger.debug(f"Context is a BaseModel: {context_schema_type.__name__}")
        elif isinstance(context, dict):
            logger.debug("Context is a dict, attempting to create dynamic model.")
            temp_model_name = name_override or "DynamicContextModel"
            try:
                context_schema_type = create_model(
                    temp_model_name,
                    **{
                        k: (type(v), Field(default=v)) for k, v in context.items()
                    },  # Use pydantic.Field
                )
                context_instance = context_schema_type(**context)  # type: ignore
                original_data = context
                logger.info(
                    f"Dynamically created model '{context_schema_type.__name__}' from dict context."
                )
            except Exception as e_dict_model:
                logger.error(
                    f"Could not dynamically create model from dict context: {e_dict_model}",
                    exc_info=True,
                )
                raise TypeError(
                    "If context is a dict, it must be convertible to a Pydantic model, or use a BaseModel instance directly."
                ) from e_dict_model
        else:
            logger.error(f"Unsupported context type: {type(context)}")
            raise TypeError(
                "Context must be a Pydantic BaseModel instance or a dictionary."
            )

        context_name = name_override or context_schema_type.__name__
        context_doc = (
            description_override
            or context_schema_type.__doc__
            or "Context data structure."
        )
        current_values_str = json.dumps(original_data, indent=2, default=str)
        logger.info(f"Processing context update for '{context_name}'.")
        logger.debug(
            f"Current context values for '{context_name}': {current_values_str}"
        )

        if not context_schema_type.model_fields:
            logger.warning(
                f"Context schema '{context_name}' has no fields. Returning original context."
            )
            return context  # type: ignore

        # Step 1: Select fields to update
        field_names = list(context_schema_type.model_fields.keys())
        selection_model_prompt_str = (
            f"Given the current context data:\n{current_values_str}\n\n"
            f"And the user's request: '{prompt or 'No specific prompt, update based on instructions.'}'\n"
            f"Instructions for update: '{instructions or 'Determine necessary updates.'}'\n\n"
            f"Which fields from '{context_name}' (available: {', '.join(field_names)}) should be updated or set?"
        )
        logger.debug(
            f"Field selection prompt for '{context_name}': {selection_model_prompt_str}"
        )
        FieldSelectionModel = create_model(
            f"{context_name}FieldSelection",
            fields_to_update=(
                List[Literal[tuple(field_names)]],
                Field(default_factory=list),
            ),  # Use pydantic.Field # type: ignore
            __base__=BaseModel,
        )
        logger.debug(f"Created FieldSelectionModel: {FieldSelectionModel.__name__}")

        fields_to_update: List[str] = []
        try:
            client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
            logger.debug(
                f"Calling LLM to select fields for '{context_name}' using model {model}."
            )
            selected_fields_response = client.chat.completions.create(
                model=model,
                messages=_format_compiled_messages(
                    selection_model_prompt_str,
                    instructions="Select fields that need updating based on the prompt and current context.",
                ),
                response_model=FieldSelectionModel,
                **model_params if model_params else {},
            )
            client.mode = CLIENT_DEPS._instructor_mode_enum(
                "tool_call"
            )  # Use stored Enum
            fields_to_update = getattr(selected_fields_response, "fields_to_update", [])
            logger.info(
                f"LLM selected fields to update for '{context_name}': {fields_to_update}"
            )
        except Exception as e_select:
            logger.error(
                f"Error selecting fields to update for context '{context_name}': {e_select}",
                exc_info=True,
            )
            return context  # type: ignore

        if not fields_to_update:
            logger.info(
                f"No fields selected by LLM for updating context '{context_name}'. Returning original context."
            )
            return context  # type: ignore

        updated_values_data: Dict[str, Any] = {}
        fields_to_generate_for = [f for f in fields_to_update if f in field_names]
        logger.debug(
            f"Validated fields to generate values for: {fields_to_generate_for}"
        )

        if not fields_to_generate_for:
            logger.info(
                f"No valid fields to generate after filtering. Returning original context for '{context_name}'."
            )
            return context  # type: ignore

        # Step 2: Generate new values for selected fields
        base_generation_instructions_str = (
            f"Generate new values for the specified fields of '{context_name}' based on the request: '{prompt}'.\n"
            f"Overall instructions: {instructions}\n"
            f"Current full context data for reference:\n{current_values_str}"
        )
        logger.debug(
            f"Base generation instructions for '{context_name}': {base_generation_instructions_str}"
        )

        if iterative:
            logger.info(
                f"Iteratively generating values for fields in '{context_name}': {fields_to_generate_for}"
            )
            current_intermediate_data = original_data.copy()
            for field_name in fields_to_generate_for:
                logger.debug(
                    f"Iteratively generating value for field '{field_name}' in '{context_name}'."
                )
                field_info = context_schema_type.model_fields[field_name]
                SingleFieldUpdateModel = create_model(
                    f"{context_name}{field_name.capitalize()}Update",
                    **{
                        field_name: (
                            field_info.annotation,
                            Field(description=field_info.description),
                        )
                    },  # Use pydantic.Field # type: ignore
                )
                iterative_prompt_str = (
                    f"{base_generation_instructions_str}\n\n"
                    f"Specifically, generate the value for field '{field_name}'.\n"
                    f"Type: {field_info.annotation}. Description: {field_info.description}.\n"
                    f"Current value of '{field_name}': {current_intermediate_data.get(field_name, 'Not set')}.\n"
                    f"Previously updated values in this cycle: {json.dumps({k: v for k, v in updated_values_data.items() if k != field_name}, default=str)}"
                )
                logger.debug(
                    f"Iterative prompt for field '{field_name}': {iterative_prompt_str}"
                )
                try:
                    client.mode = CLIENT_DEPS._instructor_mode_enum(
                        mode
                    )  # Use stored Enum
                    field_val_model = client.chat.completions.create(
                        model=model,
                        messages=_format_compiled_messages(iterative_prompt_str),
                        response_model=SingleFieldUpdateModel,
                        **model_params if model_params else {},
                        stream=False,
                    )
                    client.mode = CLIENT_DEPS._instructor_mode_enum(
                        "tool_call"
                    )  # Use stored Enum
                    if hasattr(field_val_model, field_name):
                        val = getattr(field_val_model, field_name)
                        updated_values_data[field_name] = val
                        current_intermediate_data[field_name] = val
                        logger.info(
                            f"Iteratively generated value for '{field_name}' in '{context_name}': Type {type(val)}"
                        )
                        logger.debug(f"Value for '{field_name}': {val}")
                    else:
                        logger.warning(
                            f"Iterative generation: LLM did not return field '{field_name}' for '{context_name}'."
                        )
                except Exception as e_iter_val:
                    logger.error(
                        f"Error iteratively generating value for field '{field_name}' in '{context_name}': {e_iter_val}",
                        exc_info=True,
                    )
        else:  # Bulk update
            logger.info(
                f"Bulk generating values for fields in '{context_name}': {fields_to_generate_for}"
            )
            BulkUpdateModelSchema = create_model(
                f"{context_name}PartialBulkUpdate",
                **{
                    name: (
                        context_schema_type.model_fields[name].annotation,
                        Field(
                            default=None,
                            description=context_schema_type.model_fields[
                                name
                            ].description,
                        ),
                    )
                    for name in fields_to_generate_for
                },  # Use pydantic.Field # type: ignore
                __base__=BaseModel,
            )
            logger.debug(
                f"Bulk update model schema for '{context_name}': {BulkUpdateModelSchema.__name__}"
            )
            bulk_prompt_str = (
                f"{base_generation_instructions_str}\n\n"
                f"Provide new values for the following fields: {', '.join(fields_to_generate_for)}."
            )
            logger.debug(f"Bulk update prompt for '{context_name}': {bulk_prompt_str}")
            try:
                client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
                bulk_response_model = client.chat.completions.create(
                    model=model,
                    messages=_format_compiled_messages(bulk_prompt_str),
                    response_model=BulkUpdateModelSchema,  # type: ignore
                    **model_params if model_params else {},
                    stream=stream,
                )
                client.mode = CLIENT_DEPS._instructor_mode_enum(
                    "tool_call"
                )  # Use stored Enum
                if stream:
                    logger.info(f"Streaming bulk update response for '{context_name}'.")
                    return bulk_response_model  # type: ignore
                updated_values_data = bulk_response_model.model_dump(
                    exclude_none=True, exclude_unset=True
                )
                logger.info(
                    f"Bulk generated values for '{context_name}': {list(updated_values_data.keys())}"
                )
                logger.debug(f"Bulk values: {updated_values_data}")
            except Exception as e_bulk_val:
                logger.error(
                    f"Error generating bulk values for fields '{fields_to_generate_for}' in '{context_name}': {e_bulk_val}",
                    exc_info=True,
                )

        if stream and not iterative:
            # This path should ideally only be reached if fields_to_generate_for was empty AND stream=True.
            # If bulk generation happened with stream=True, it would have returned the stream already.
            logger.debug(
                f"from_context: stream={stream}, iterative={iterative}. Current updated_values_data: {updated_values_data}. This path implies no stream was returned from bulk, or no bulk call made."
            )
            if update_context:
                logger.warning(
                    "Streaming with update_context=True for from_context will stream the partial update model. Caller needs to merge."
                )
            # If updated_values_data is not a stream here, this is problematic as original comment stated.
            # However, if fields_to_generate_for was empty, updated_values_data is {} which is not a stream.
            return updated_values_data  # type: ignore

        # Final merging
        if not updated_values_data:
            logger.info(
                f"No new values were generated for context '{context_name}'. Returning original context."
            )
            return context  # type: ignore

        logger.info(
            f"Merging updated values with original context for '{context_name}'. Update_context={update_context}"
        )
        if update_context:
            if context_instance:
                final_context = context_instance.model_copy(update=updated_values_data)
            else:
                final_context = context_schema_type(
                    **{**original_data, **updated_values_data}
                )  # type: ignore
            logger.debug(
                f"Returning updated context instance of type {type(final_context).__name__} for '{context_name}'."
            )
            return final_context  # type: ignore
        else:
            logger.debug(
                f"update_context is False. Returning only the updated/new values for '{context_name}'."
            )
            DeltaModel = create_model(
                f"{context_name}DeltaUpdate",
                **{
                    k: (type(v), Field(default=v))
                    for k, v in updated_values_data.items()
                },  # Use pydantic.Field # type: ignore
            )
            logger.debug(
                f"Created delta model '{DeltaModel.__name__}' for '{context_name}'."
            )
            return DeltaModel(**updated_values_data)  # type: ignore

    # --- async_from_context (mirroring from_context) ---
    @overload
    @staticmethod
    async def async_from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: Literal[True] = True,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Iterable[ContextType]: ...

    @overload
    @staticmethod
    async def async_from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: Literal[False] = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> ContextType: ...

    @staticmethod
    async def async_from_context(
        context: ContextType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        update_context: bool = True,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[ContextType, Iterable[ContextType]]:
        logger.info(
            f"Create.async_from_context called for context type {type(context).__name__}, iterative={iterative}, stream={stream}, update_context={update_context}"
        )
        logger.debug(
            f"Context provided (async): {type(context)}, name_override: {name_override}, description_override: {description_override}, prompt: {prompt}, instructions: {instructions}"
        )

        client = Create._get_async_instructor_client()
        context_schema_type: Type[BaseModel]
        context_instance: Optional[BaseModel] = None
        original_data: Dict[str, Any]

        if isinstance(context, BaseModel):
            context_schema_type = type(context)
            context_instance = context
            original_data = context.model_dump()
            logger.debug(
                f"Context is a BaseModel (async): {context_schema_type.__name__}"
            )
        elif isinstance(context, dict):
            logger.debug(
                "Context is a dict, attempting to create dynamic model (async)."
            )
            temp_model_name = name_override or "DynamicContextModel"
            try:
                context_schema_type = create_model(
                    temp_model_name,
                    **{
                        k: (type(v), Field(default=v)) for k, v in context.items()
                    },  # Use pydantic.Field # type: ignore
                )
                context_instance = context_schema_type(**context)  # type: ignore
                original_data = context
                logger.info(
                    f"Dynamically created model '{context_schema_type.__name__}' from dict context (async)."
                )
            except Exception as e_dict_model_async:
                logger.error(
                    f"Could not dynamically create model from dict context (async): {e_dict_model_async}",
                    exc_info=True,
                )
                raise TypeError(
                    "If context is a dict, it must be convertible to a Pydantic model, or use a BaseModel instance directly (async)."
                ) from e_dict_model_async
        else:
            logger.error(f"Unsupported context type (async): {type(context)}")
            raise TypeError(
                "Context must be a Pydantic BaseModel instance or a dictionary (async)."
            )

        context_name = name_override or context_schema_type.__name__
        context_doc = (
            description_override
            or context_schema_type.__doc__
            or "Context data structure."
        )
        current_values_str = json.dumps(original_data, indent=2, default=str)
        logger.info(f"Processing async context update for '{context_name}'.")
        logger.debug(
            f"Current context values for '{context_name}' (async): {current_values_str}"
        )

        if not context_schema_type.model_fields:
            logger.warning(
                f"Context schema '{context_name}' has no fields (async). Returning original context."
            )
            return context  # type: ignore

        field_names = list(context_schema_type.model_fields.keys())
        selection_model_prompt_str = (
            f"Given the current context data:\n{current_values_str}\n\n"
            f"And the user's request: '{prompt or 'No specific prompt, update based on instructions.'}'\n"
            f"Instructions for update: '{instructions or 'Determine necessary updates.'}'\n\n"
            f"Which fields from '{context_name}' (available: {', '.join(field_names)}) should be updated or set?"
        )
        logger.debug(
            f"Field selection prompt for '{context_name}' (async): {selection_model_prompt_str}"
        )
        FieldSelectionModel = create_model(
            f"{context_name}FieldSelectionAsync",
            fields_to_update=(
                List[Literal[tuple(field_names)]],
                Field(default_factory=list),
            ),  # Use pydantic.Field # type: ignore
            __base__=BaseModel,
        )
        logger.debug(
            f"Created FieldSelectionModel (async): {FieldSelectionModel.__name__}"
        )

        fields_to_update: List[str] = []
        try:
            client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
            logger.debug(
                f"Calling LLM to select fields for '{context_name}' using model {model} (async)."
            )
            selected_fields_response = await client.chat.completions.create(
                model=model,
                messages=_format_compiled_messages(
                    selection_model_prompt_str,
                    instructions="Select fields that need updating based on the prompt and current context.",
                ),
                response_model=FieldSelectionModel,
                **model_params if model_params else {},
            )
            client.mode = CLIENT_DEPS._instructor_mode_enum(
                "tool_call"
            )  # Use stored Enum
            fields_to_update = getattr(selected_fields_response, "fields_to_update", [])
            logger.info(
                f"LLM selected fields to update for '{context_name}' (async): {fields_to_update}"
            )
        except Exception as e_select_async:
            logger.error(
                f"Error selecting fields to update for context '{context_name}' (async): {e_select_async}",
                exc_info=True,
            )
            return context  # type: ignore

        if not fields_to_update:
            logger.info(
                f"No fields selected by LLM for updating context '{context_name}' (async). Returning original context."
            )
            return context  # type: ignore

        updated_values_data: Dict[str, Any] = {}
        fields_to_generate_for = [f for f in fields_to_update if f in field_names]
        logger.debug(
            f"Validated fields to generate values for (async): {fields_to_generate_for}"
        )

        if not fields_to_generate_for:
            logger.info(
                f"No valid fields to generate after filtering (async). Returning original context for '{context_name}'."
            )
            return context  # type: ignore

        base_generation_instructions_str = (
            f"Generate new values for the specified fields of '{context_name}' based on the request: '{prompt}'.\n"
            f"Overall instructions: {instructions}\n"
            f"Current full context data for reference:\n{current_values_str}"
        )
        logger.debug(
            f"Base generation instructions for '{context_name}' (async): {base_generation_instructions_str}"
        )

        if iterative:
            logger.info(
                f"Iteratively generating values for fields in '{context_name}' (async): {fields_to_generate_for}"
            )
            current_intermediate_data = original_data.copy()
            for field_name in fields_to_generate_for:
                logger.debug(
                    f"Iteratively generating value for field '{field_name}' in '{context_name}' (async)."
                )
                field_info = context_schema_type.model_fields[field_name]
                SingleFieldUpdateModel = create_model(
                    f"{context_name}{field_name.capitalize()}UpdateAsync",
                    **{
                        field_name: (
                            field_info.annotation,
                            Field(description=field_info.description),
                        )
                    },  # Use pydantic.Field # type: ignore
                )
                iterative_prompt_str = (
                    f"{base_generation_instructions_str}\n\n"
                    f"Specifically, generate the value for field '{field_name}'.\n"
                    f"Type: {field_info.annotation}. Description: {field_info.description}.\n"
                    f"Current value of '{field_name}': {current_intermediate_data.get(field_name, 'Not set')}.\n"
                    f"Previously updated values in this cycle: {json.dumps({k: v for k, v in updated_values_data.items() if k != field_name}, default=str)}"
                )
                logger.debug(
                    f"Iterative prompt for field '{field_name}' (async): {iterative_prompt_str}"
                )
                try:
                    client.mode = CLIENT_DEPS._instructor_mode_enum(
                        mode
                    )  # Use stored Enum
                    field_val_model = await client.chat.completions.create(
                        model=model,
                        messages=_format_compiled_messages(iterative_prompt_str),
                        response_model=SingleFieldUpdateModel,
                        **model_params if model_params else {},
                        stream=False,
                    )
                    client.mode = CLIENT_DEPS._instructor_mode_enum(
                        "tool_call"
                    )  # Use stored Enum
                    if hasattr(field_val_model, field_name):
                        val = getattr(field_val_model, field_name)
                        updated_values_data[field_name] = val
                        current_intermediate_data[field_name] = val
                        logger.info(
                            f"Iteratively generated value for '{field_name}' in '{context_name}' (async): Type {type(val)}"
                        )
                        logger.debug(f"Value for '{field_name}' (async): {val}")
                    else:
                        logger.warning(
                            f"Iterative generation (async): LLM did not return field '{field_name}' for '{context_name}'."
                        )
                except Exception as e_iter_val_async:
                    logger.error(
                        f"Error iteratively generating value for field '{field_name}' (async) in '{context_name}': {e_iter_val_async}",
                        exc_info=True,
                    )
        else:  # Bulk update
            logger.info(
                f"Bulk generating values for fields in '{context_name}' (async): {fields_to_generate_for}"
            )
            BulkUpdateModelSchema = create_model(
                f"{context_name}PartialBulkUpdateAsync",
                **{
                    name: (
                        context_schema_type.model_fields[name].annotation,
                        Field(
                            default=None,
                            description=context_schema_type.model_fields[
                                name
                            ].description,
                        ),
                    )
                    for name in fields_to_generate_for
                },  # Use pydantic.Field # type: ignore
                __base__=BaseModel,
            )
            logger.debug(
                f"Bulk update model schema for '{context_name}' (async): {BulkUpdateModelSchema.__name__}"
            )
            bulk_prompt_str = (
                f"{base_generation_instructions_str}\n\n"
                f"Provide new values for the following fields: {', '.join(fields_to_generate_for)}."
            )
            logger.debug(
                f"Bulk update prompt for '{context_name}' (async): {bulk_prompt_str}"
            )
            try:
                client.mode = CLIENT_DEPS._instructor_mode_enum(mode)  # Use stored Enum
                bulk_response_model = await client.chat.completions.create(
                    model=model,
                    messages=_format_compiled_messages(bulk_prompt_str),
                    response_model=BulkUpdateModelSchema,  # type: ignore
                    params=model_params,  # type: ignore # Pass model_params correctly
                    stream=stream,
                )
                client.mode = CLIENT_DEPS._instructor_mode_enum(
                    "tool_call"
                )  # Use stored Enum
                if stream:
                    logger.info(
                        f"Streaming bulk update response for '{context_name}' (async)."
                    )
                    return bulk_response_model  # type: ignore
                updated_values_data = bulk_response_model.model_dump(
                    exclude_none=True, exclude_unset=True
                )
                logger.info(
                    f"Bulk generated values for '{context_name}' (async): {list(updated_values_data.keys())}"
                )
                logger.debug(f"Bulk values (async): {updated_values_data}")
            except Exception as e_bulk_val_async:
                logger.error(
                    f"Error generating bulk values for fields '{fields_to_generate_for}' (async) in '{context_name}': {e_bulk_val_async}",
                    exc_info=True,
                )

        if stream and not iterative:
            logger.debug(
                f"async_from_context: stream={stream}, iterative={iterative}. Current updated_values_data: {updated_values_data}. This path implies no stream was returned from bulk, or no bulk call made."
            )
            if update_context:
                logger.warning(
                    "Streaming with update_context=True for async_from_context will stream the partial update model. Caller needs to merge."
                )
            return updated_values_data  # type: ignore

        if not updated_values_data:
            logger.info(
                f"No new values were generated for context '{context_name}' (async). Returning original context."
            )
            return context  # type: ignore

        logger.info(
            f"Merging updated values with original context for '{context_name}' (async). Update_context={update_context}"
        )
        if update_context:
            if context_instance:
                final_context = context_instance.model_copy(update=updated_values_data)
            else:
                final_context = context_schema_type(
                    **{**original_data, **updated_values_data}
                )  # type: ignore
            logger.debug(
                f"Returning updated context instance of type {type(final_context).__name__} for '{context_name}' (async)."
            )
            return final_context  # type: ignore
        else:
            logger.debug(
                f"update_context is False. Returning only the updated/new values for '{context_name}' (async)."
            )
            DeltaModel = create_model(
                f"{context_name}DeltaUpdateAsync",
                **{
                    k: (type(v), Field(default=v))
                    for k, v in updated_values_data.items()
                },  # Use pydantic.Field # type: ignore
            )
            logger.debug(
                f"Created delta model '{DeltaModel.__name__}' for '{context_name}' (async)."
            )
            return DeltaModel(**updated_values_data)  # type: ignore

    # --- from_attributes ---

    @staticmethod
    def from_attributes(
        schema: SchemaType,
        attributes: AttributeType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[List[StructuredReturnType], Iterable[StructuredReturnType]]:
        """
        Generates one or more schema instances, each influenced by specified attributes.
        If attributes is a list, one instance per attribute.
        If attributes is a dict (weights), one instance influenced by all (not fully implemented, focuses on list).
        """
        logger.info(
            f"Create.from_attributes called with schema type {type(schema).__name__}, attributes type {type(attributes).__name__}, stream={stream}, iterative={iterative}"
        )
        logger.debug(
            f"Attributes provided: {attributes}, name_override: {name_override}, description_override: {description_override}, prompt: {prompt}, instructions: {instructions}"
        )

        results: List[StructuredReturnType] = []
        attribute_list: List[str]

        if isinstance(attributes, list):
            attribute_list = attributes
            logger.debug(f"Processing list of {len(attribute_list)} attributes.")
        elif isinstance(attributes, dict):
            attribute_list = [
                f"Influenced by attributes: {', '.join(attributes.keys())} with weights {attributes}"
            ]
            logger.debug(
                f"Processing dict of attributes, combined into single attribute string: {attribute_list[0]}"
            )
            if len(attributes) > 1:
                logger.warning(
                    "Using a dict of attributes for from_attributes will generate a single response influenced by all. For distinct responses per attribute, use a list."
                )
        else:
            logger.error(f"Unsupported attributes type: {type(attributes).__name__}")
            raise TypeError(
                "Attributes must be a list of strings or a dictionary of attribute:weight."
            )

        if stream and len(attribute_list) > 1:
            logger.info(
                f"Streaming multiple ({len(attribute_list)}) attribute-based generations as a single list stream."
            )
            actual_schema_for_call: Type[BaseModel]
            default_model_name = name_override or "AttributedResponse"
            default_model_desc = (
                description_override or "Response styled by attributes."
            )

            base_actual_schema: Type[BaseModel]
            if isinstance(schema, str):
                logger.debug(f"Parsing string schema for attributes: '{schema}'")
                base_actual_schema = _parse_string_schema(
                    schema, default_model_name, default_model_desc
                )
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                logger.debug(
                    f"Using provided Pydantic BaseModel for attributes: {schema.__name__}"
                )
                base_actual_schema = schema
            elif callable(schema) and not isinstance(schema, type):
                logger.debug(
                    f"Converting callable to Pydantic model for attributes: {schema.__name__}"
                )
                m_callable = convert_to_pydantic_model(
                    schema,
                    name=name_override or schema.__name__,
                    description=description_override or schema.__doc__,
                )
                if not (
                    isinstance(m_callable, type) and issubclass(m_callable, BaseModel)
                ):
                    logger.error(
                        "Failed to convert callable to Pydantic Model for attributes."
                    )
                    raise TypeError(
                        "Could not convert function to Pydantic Model for attributes"
                    )
                base_actual_schema = m_callable
            elif isinstance(schema, type):
                logger.debug(
                    f"Converting basic type to Pydantic model for attributes: {schema.__name__}"
                )
                m_type = convert_to_pydantic_model(
                    schema,
                    name=name_override or f"{schema.__name__}AttributedResponse",
                    description=description_override,
                )
                if not (isinstance(m_type, type) and issubclass(m_type, BaseModel)):
                    logger.error(
                        "Failed to convert basic type to Pydantic Model for attributes."
                    )
                    raise TypeError(
                        "Could not convert basic type to Pydantic Model for attributes"
                    )
                base_actual_schema = m_type
            else:
                logger.error(
                    f"Unsupported schema type for attributes: {type(schema).__name__}."
                )
                raise TypeError(
                    f"Unsupported schema type for attributes: {type(schema).__name__}."
                )
            logger.info(
                f"Determined base schema for attributes: {base_actual_schema.__name__}"
            )

            list_response_model = List[base_actual_schema]  # type: ignore
            logger.info(
                f"Response model for multi-attribute stream: List[{base_actual_schema.__name__}]"
            )

            combined_prompt_parts = [
                str(
                    prompt
                    if prompt
                    else "Generate data based on the schema and attributes."
                )
            ]
            combined_prompt_parts.append(
                "Generate one response for each of the following attribute styles:"
            )
            for i, attr_str in enumerate(attribute_list):
                combined_prompt_parts.append(f"{i + 1}. Style: {attr_str}")
            final_prompt_for_list = "\n".join(combined_prompt_parts)
            logger.debug(
                f"Combined prompt for list generation: {final_prompt_for_list}"
            )

            final_instructions_for_list = (
                instructions
                or f"Ensure you generate {len(attribute_list)} distinct items, each corresponding to one of the specified attribute styles, matching the schema: {base_actual_schema.__name__}."
            )
            logger.debug(
                f"Final instructions for list generation: {final_instructions_for_list}"
            )

            logger.info(
                f"Calling Create.from_schema to stream List[{base_actual_schema.__name__}]."
            )
            return Create.from_schema(  # type: ignore
                schema=list_response_model,
                prompt=final_prompt_for_list,
                instructions=final_instructions_for_list,
                model=model,
                model_params=model_params,
                iterative=False,
                n=1,
                stream=True,
                existing_messages=existing_messages,
                role_for_prompt=role_for_prompt,
                mode=mode,
                name_override=name_override or f"ListOf{base_actual_schema.__name__}",
                description_override=description_override
                or f"A list of {base_actual_schema.__name__} items, each styled by an attribute.",
            )

        logger.info(
            f"Processing attributes individually. Total: {len(attribute_list)}. Stream per attribute: {stream and len(attribute_list) == 1}"
        )
        for attr_idx, attr in enumerate(attribute_list):
            logger.info(
                f"Processing attribute {attr_idx + 1}/{len(attribute_list)}: '{attr}'"
            )
            current_instructions = (
                str(instructions or "") + f"\nRespond in a style that is: {attr}."
            )
            current_prompt = prompt

            if current_prompt is None and not instructions:
                current_prompt = (
                    f"Generate data for the schema, embodying the attribute: {attr}."
                )
                logger.debug(
                    f"Generated default prompt for attribute '{attr}': {current_prompt}"
                )

            logger.debug(f"Current prompt for attribute '{attr}': {current_prompt}")
            logger.debug(
                f"Current instructions for attribute '{attr}': {current_instructions}"
            )

            try:
                if stream and len(attribute_list) == 1:
                    logger.info(f"Streaming single attribute ('{attr}') response.")
                    yield from Create.from_schema(  # type: ignore
                        schema=schema,
                        prompt=current_prompt,
                        instructions=current_instructions,
                        model=model,
                        model_params=model_params,
                        iterative=iterative,
                        n=1,
                        name_override=name_override,
                        description_override=description_override,
                        stream=True,
                        existing_messages=existing_messages,
                        role_for_prompt=role_for_prompt,
                        mode=mode,
                    )
                    return

                logger.info(
                    f"Calling Create.from_schema for attribute '{attr}' (non-streaming for this attribute call)."
                )
                result = Create.from_schema(  # type: ignore
                    schema=schema,
                    prompt=current_prompt,
                    instructions=current_instructions,
                    model=model,
                    model_params=model_params,
                    iterative=iterative,
                    n=1,
                    name_override=name_override,
                    description_override=description_override,
                    stream=False,
                    existing_messages=existing_messages,
                    role_for_prompt=role_for_prompt,
                    mode=mode,
                )
                results.append(result)  # type: ignore
                logger.debug(f"Successfully generated response for attribute '{attr}'.")
            except Exception as e_attr:
                logger.error(
                    f"Error generating response for attribute '{attr}': {e_attr}",
                    exc_info=True,
                )
                results.append(e_attr)  # type: ignore

        logger.info(
            f"Finished processing all attributes. Returning {len(results)} results."
        )
        return results  # type: ignore

    # --- async_from_attributes (mirroring from_attributes) ---

    @staticmethod
    async def async_from_attributes(
        schema: SchemaType,
        attributes: AttributeType,
        prompt: Optional[PromptType] = None,
        instructions: Optional[str] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        iterative: bool = False,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ) -> Union[List[StructuredReturnType], Iterable[StructuredReturnType]]:
        logger.info(
            f"Create.async_from_attributes called with schema type {type(schema).__name__}, attributes type {type(attributes).__name__}, stream={stream}, iterative={iterative}"
        )
        logger.debug(
            f"Attributes provided (async): {attributes}, name_override: {name_override}, description_override: {description_override}, prompt: {prompt}, instructions: {instructions}"
        )

        results: List[StructuredReturnType] = []
        attribute_list: List[str]

        if isinstance(attributes, list):
            attribute_list = attributes
            logger.debug(
                f"Processing list of {len(attribute_list)} attributes (async)."
            )
        elif isinstance(attributes, dict):
            attribute_list = [
                f"Influenced by attributes: {', '.join(attributes.keys())} with weights {attributes}"
            ]
            logger.debug(
                f"Processing dict of attributes, combined into single attribute string (async): {attribute_list[0]}"
            )
            if len(attributes) > 1:
                logger.warning(
                    "Using a dict of attributes for async_from_attributes will generate a single response. For distinct responses, use a list."
                )
        else:
            logger.error(
                f"Unsupported attributes type (async): {type(attributes).__name__}"
            )
            raise TypeError("Attributes must be a list of strings or a dictionary.")

        if stream and len(attribute_list) > 1:
            logger.info(
                f"Streaming multiple ({len(attribute_list)}) attribute-based generations as a single list stream (async)."
            )
            actual_schema_for_call: Type[BaseModel]
            default_model_name = name_override or "AttributedResponse"
            default_model_desc = (
                description_override or "Response styled by attributes."
            )

            base_actual_schema: Type[BaseModel]
            if isinstance(schema, str):
                logger.debug(
                    f"Parsing string schema for attributes (async): '{schema}'"
                )
                base_actual_schema = _parse_string_schema(
                    schema, default_model_name, default_model_desc
                )
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                logger.debug(
                    f"Using provided Pydantic BaseModel for attributes (async): {schema.__name__}"
                )
                base_actual_schema = schema
            elif callable(schema) and not isinstance(schema, type):
                logger.debug(
                    f"Converting callable to Pydantic model for attributes (async): {schema.__name__}"
                )
                m_callable = convert_to_pydantic_model(
                    schema,
                    name=name_override or schema.__name__,
                    description=description_override or schema.__doc__,
                )
                if not (
                    isinstance(m_callable, type) and issubclass(m_callable, BaseModel)
                ):
                    logger.error(
                        "Failed to convert callable to Pydantic Model for attributes (async)."
                    )
                    raise TypeError(
                        "Could not convert function to Pydantic Model for attributes (async)"
                    )
                base_actual_schema = m_callable
            elif isinstance(schema, type):
                logger.debug(
                    f"Converting basic type to Pydantic model for attributes (async): {schema.__name__}"
                )
                m_type = convert_to_pydantic_model(
                    schema,
                    name=name_override or f"{schema.__name__}AttributedResponse",
                    description=description_override,
                )
                if not (isinstance(m_type, type) and issubclass(m_type, BaseModel)):
                    logger.error(
                        "Failed to convert basic type to Pydantic Model for attributes (async)."
                    )
                    raise TypeError(
                        "Could not convert basic type to Pydantic Model for attributes (async)"
                    )
                base_actual_schema = m_type
            else:
                logger.error(
                    f"Unsupported schema type for attributes (async): {type(schema).__name__}."
                )
                raise TypeError(
                    f"Unsupported schema type for attributes (async): {type(schema).__name__}."
                )
            logger.info(
                f"Determined base schema for attributes (async): {base_actual_schema.__name__}"
            )

            list_response_model = List[base_actual_schema]  # type: ignore
            logger.info(
                f"Response model for multi-attribute stream (async): List[{base_actual_schema.__name__}]"
            )

            combined_prompt_parts = [
                str(
                    prompt
                    if prompt
                    else "Generate data based on the schema and attributes."
                )
            ]
            combined_prompt_parts.append(
                "Generate one response for each of the following attribute styles:"
            )
            for i, attr_str in enumerate(attribute_list):
                combined_prompt_parts.append(f"{i + 1}. Style: {attr_str}")
            final_prompt_for_list = "\n".join(combined_prompt_parts)
            logger.debug(
                f"Combined prompt for list generation (async): {final_prompt_for_list}"
            )

            final_instructions_for_list = (
                instructions
                or f"Ensure you generate {len(attribute_list)} distinct items, each corresponding to one of the specified attribute styles, matching the schema: {base_actual_schema.__name__}."
            )
            logger.debug(
                f"Final instructions for list generation (async): {final_instructions_for_list}"
            )

            logger.info(
                f"Calling Create.async_from_schema to stream List[{base_actual_schema.__name__}] (async)."
            )
            async_iterable_response = Create.async_from_schema(  # type: ignore
                schema=list_response_model,
                prompt=final_prompt_for_list,
                instructions=final_instructions_for_list,
                model=model,
                model_params=model_params,
                iterative=False,
                n=1,
                stream=True,
                existing_messages=existing_messages,
                role_for_prompt=role_for_prompt,
                mode=mode,
                name_override=name_override or f"ListOf{base_actual_schema.__name__}",
                description_override=description_override
                or f"A list of {base_actual_schema.__name__} items, each styled by an attribute.",
            )
            return async_iterable_response  # type: ignore

        logger.info(
            f"Processing attributes individually (async). Total: {len(attribute_list)}. Stream per attribute: {stream and len(attribute_list) == 1}"
        )
        for attr_idx, attr in enumerate(attribute_list):
            logger.info(
                f"Processing attribute {attr_idx + 1}/{len(attribute_list)}: '{attr}' (async)"
            )
            current_instructions = (
                str(instructions or "") + f"\nRespond in a style that is: {attr}."
            )
            current_prompt = prompt
            if current_prompt is None and not instructions:
                current_prompt = (
                    f"Generate data for the schema, embodying the attribute: {attr}."
                )
                logger.debug(
                    f"Generated default prompt for attribute '{attr}' (async): {current_prompt}"
                )

            logger.debug(
                f"Current prompt for attribute '{attr}' (async): {current_prompt}"
            )
            logger.debug(
                f"Current instructions for attribute '{attr}' (async): {current_instructions}"
            )

            try:
                if stream and len(attribute_list) == 1:
                    logger.info(
                        f"Streaming single attribute ('{attr}') response (async)."
                    )
                    async_gen = Create.async_from_schema(  # type: ignore
                        schema=schema,
                        prompt=current_prompt,
                        instructions=current_instructions,
                        model=model,
                        model_params=model_params,
                        iterative=iterative,
                        n=1,
                        name_override=name_override,
                        description_override=description_override,
                        stream=True,
                        existing_messages=existing_messages,
                        role_for_prompt=role_for_prompt,
                        mode=mode,
                    )
                    return async_gen  # type: ignore

                logger.info(
                    f"Calling Create.async_from_schema for attribute '{attr}' (non-streaming for this attribute call) (async)."
                )
                result = await Create.async_from_schema(  # type: ignore
                    schema=schema,
                    prompt=current_prompt,
                    instructions=current_instructions,
                    model=model,
                    model_params=model_params,
                    iterative=iterative,
                    n=1,
                    name_override=name_override,
                    description_override=description_override,
                    stream=False,
                    existing_messages=existing_messages,
                    role_for_prompt=role_for_prompt,
                    mode=mode,
                )
                results.append(result)  # type: ignore
                logger.debug(
                    f"Successfully generated response for attribute '{attr}' (async)."
                )
            except Exception as e_attr_async:
                logger.error(
                    f"Error generating async response for attribute '{attr}': {e_attr_async}",
                    exc_info=True,
                )
                results.append(e_attr_async)  # type: ignore

        logger.info(
            f"Finished processing all attributes (async). Returning {len(results)} results."
        )
        return results  # type: ignore

    @staticmethod
    def from_function(
        model: Optional[ModelParam] = "openai/gpt-4o-mini",  # Allow override
        model_params: Optional[Params] = None,
        iterative: bool = False,
        n: int = 1,  # n > 1 implies List[ReturnType]
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        stream: bool = False,  # Stream implies Iterable[ReturnType]
        existing_messages: Optional[List[Message]] = None,
        role_for_prompt: MessageRole = "user",
        mode: InstructorModeParam = "tool_call",
    ):
        """
        Decorator to turn a function definition into an LLM-powered generator.

        The function signature and docstring are used to guide the LLM.
        Provided arguments are included in the prompt.
        The return type hint determines the expected output schema.
        """

        def decorator(
            func: Callable[..., ReturnType],
        ) -> Callable[
            ...,
            Union[ReturnType, List[ReturnType], Iterable[ReturnType], Any],
        ]:
            @functools.wraps(func)
            async def async_wrapper(
                *args: Any, **kwargs: Any
            ) -> Union[ReturnType, List[ReturnType], Iterable[ReturnType], Any]:
                logger.info(f"Calling LLM-powered function: {func.__name__} (async)")
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                # Use try-except for apply_defaults as it might fail for variadic args without defaults
                try:
                    bound_args.apply_defaults()  # Apply default values from signature
                except TypeError as e:
                    logger.warning(
                        f"Could not apply defaults for function {func.__name__}: {e}. Proceeding without applying defaults."
                    )
                    # If apply_defaults fails, provided_args will only contain explicitly passed args.
                    # This is acceptable; the prompt will just reflect what was passed.

                all_params = sig.parameters
                provided_args = bound_args.arguments
                # Identify missing parameters that don't have defaults
                missing_params = {
                    name: param
                    for name, param in all_params.items()
                    if name not in provided_args
                    or (
                        param.default is not inspect.Parameter.empty
                        and provided_args[name] is param.default
                    )
                }
                # Refined missing_params check: check if argument is *not* in provided_args OR if it's the default value AND the default value is not Parameter.empty
                missing_params = {
                    name: param
                    for name, param in all_params.items()
                    if name not in provided_args
                    or (
                        param.default is not inspect.Parameter.empty
                        and provided_args.get(name) == param.default
                    )
                }

                return_type = get_type_hints(func).get("return", Any)
                docstring = func.__doc__

                # Determine the base prompt
                base_prompt_parts = []
                if docstring:
                    base_prompt_parts.append(docstring)

                # Include provided arguments in the prompt
                if provided_args:
                    # Exclude self/cls if they exist and are the first argument
                    # This is a heuristic for methods
                    args_to_show = provided_args.copy()
                    first_param_name = next(iter(sig.parameters), None)
                    if first_param_name in args_to_show and first_param_name in [
                        "self",
                        "cls",
                    ]:
                        del args_to_show[first_param_name]

                    if args_to_show:
                        provided_args_str = json.dumps(args_to_show, default=str)
                        base_prompt_parts.append(f"Input: {provided_args_str}")
                    else:
                        logger.debug(
                            "No non-self/cls arguments provided to the decorated function."
                        )

                # Add instruction to directly answer the question
                base_prompt_parts.append(
                    "Please provide a direct answer to the question above based on the input. Do not explain your reasoning or provide code."
                )

                final_prompt_str = "\n".join(base_prompt_parts)
                logger.debug(
                    f"Generated prompt for {func.__name__}: {final_prompt_str}"
                )

                # Determine the response schema and method based on return type
                if return_type is str or return_type is Any or return_type is None:
                    # Default to from_prompt for simple string/any/None return types
                    logger.debug(
                        f"Return type is {return_type}, defaulting to async_from_prompt."
                    )
                    # Note: from_prompt doesn't directly handle n > 1 or iterative
                    if n > 1 or iterative:
                        logger.warning(
                            f"from_function with return type {return_type} does not fully support n > 1 or iterative. Ignoring these parameters."
                        )
                    if isinstance(model, list):
                        raise LitellmUnsupportedInputError(
                            "Streaming is not supported when 'model' is a list for async_from_prompt in from_function."
                        )

                    # from_prompt expects a single prompt string, not a list of messages built from parts
                    # Let's rebuild the messages list for from_prompt
                    messages_for_prompt = _format_compiled_messages(
                        final_prompt_str,
                        instructions=None,
                        existing_messages=existing_messages,
                        role_for_prompt=role_for_prompt,
                    )

                    return await Create.async_from_prompt(
                        prompt=messages_for_prompt,  # Pass the formatted messages
                        # instructions=None, # Instructions are part of the docstring/prompt
                        model=model,  # type: ignore
                        model_params=model_params,
                        stream=stream,  # stream is supported for single model from_prompt
                        # existing_messages=existing_messages, # Already included in messages_for_prompt
                        role_for_prompt=role_for_prompt,  # Not used by from_prompt with list messages
                    )
                else:
                    # Use from_schema for structured return types
                    logger.debug(
                        f"Return type is structured ({return_type}), using async_from_schema."
                    )
                    # Convert return type hint to Pydantic model if necessary
                    try:
                        response_schema = convert_to_pydantic_model(
                            return_type,
                            name=name_override or f"{func.__name__}Response",
                            description=description_override
                            or func.__doc__
                            or f"Structured response for function {func.__name__}",
                        )

                        if not (
                            isinstance(response_schema, type)
                            and issubclass(response_schema, BaseModel)
                        ) and not (
                            hasattr(response_schema, "__origin__")
                            and response_schema.__origin__ is list
                            and response_schema.__args__
                            and issubclass(response_schema.__args__[0], BaseModel)
                        ):  # type: ignore
                            raise TypeError(
                                f"Return type hint {return_type} could not be converted to a Pydantic schema."
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to convert return type hint {return_type} to Pydantic schema for function {func.__name__}: {e}",
                            exc_info=True,
                        )
                        raise TypeError(
                            f"Return type hint {return_type} could not be converted to a Pydantic schema."
                        ) from e

                    # from_schema expects a single prompt string, not a list of messages built from parts
                    # Let's rebuild the messages list for from_schema
                    messages_for_schema = _format_compiled_messages(
                        final_prompt_str,
                        instructions=None,
                        existing_messages=existing_messages,
                        role_for_prompt=role_for_prompt,
                    )

                    return await Create.async_from_schema(
                        schema=response_schema,  # Pass the determined schema (could be List[BaseModel] or BaseModel)
                        prompt=messages_for_schema,  # Pass the formatted messages
                        # instructions=None, # Instructions are part of the docstring/prompt
                        model=model,  # type: ignore
                        model_params=model_params,
                        iterative=iterative,
                        n=n,
                        name_override=name_override,
                        description_override=description_override,
                        stream=stream,
                        # existing_messages=existing_messages, # Already included in messages_for_schema
                        role_for_prompt=role_for_prompt,  # Not used by from_schema with list messages
                        mode=mode,
                    )

            # Check if the original function is async
            if asyncio.iscoroutinefunction(func):
                logger.debug(f"Decorating async function: {func.__name__}")
                return async_wrapper
            else:
                logger.debug(f"Decorating sync function: {func.__name__}")

                # Create a sync wrapper that calls the async wrapper using asyncio.run
                @functools.wraps(func)
                def sync_wrapper(
                    *args: Any, **kwargs: Any
                ) -> Union[ReturnType, List[ReturnType], Iterable[ReturnType], Any]:
                    logger.info(f"Calling LLM-powered function: {func.__name__} (sync)")
                    # Note: Running async code from sync might have implications in some environments
                    # Consider using a dedicated async event loop manager if needed.
                    # For simplicity, using asyncio.run here.
                    return asyncio.run(async_wrapper(*args, **kwargs))

                return sync_wrapper

        def decorator_wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                # Decorator called without arguments, first arg is the function
                func = args[0]
                logger.debug(
                    f"from_function decorator called without args on {func.__name__}"
                )
                return decorator(func)
            elif not args:
                # Decorator called with arguments, return the decorator function
                logger.debug(
                    "from_function decorator called with args, returning decorator function"
                )
                return decorator
            else:
                raise TypeError(
                    "from_function decorator accepts only keyword arguments or a single function argument."
                )

        return decorator_wrapper


# ------------------------------------------------------------------------------
# FUNCTION REFERENCES & EXPORTS
# ------------------------------------------------------------------------------


def expose_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Exposes a method from the Create class with proper type annotations.
    """

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return method(*args, **kwargs)

    return wrapper


@overload
def create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: Union[
        ModelParam, List[ModelParam]
    ] = "openai/gpt-4o-mini",  # Default from foundry
    model_params: Optional[Params] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> LiteLLMStream: ...


@overload
def create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> str: ...


@overload
def create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: List[ModelParam] = ["openai/gpt-4o-mini"],  # type: ignore
    model_params: Optional[Params] = None,
    stream: Literal[
        False
    ] = False,  # Streaming not supported for list of models in this way
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> List[str]: ...


def create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: Union[ModelParam, List[ModelParam]] = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> Union[str, List[str], LiteLLMStream]:
    """
    Generate text from a prompt using an LLM.

    Args:
        prompt: The prompt to send to the model. Can be a string, list of messages, or other supported format.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use, either as a string (e.g., "openai/gpt-4o-mini") or a list of models.
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        stream: Whether to stream the response tokens as they're generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").

    Returns:
        A string response, list of responses (if multiple models), or a streaming response object.

    Examples:
        >>> create_from_prompt("Tell me a joke about programming")
        "Why do programmers prefer dark mode? Because light attracts bugs!"

        >>> create_from_prompt(
        ...     prompt="Summarize this article",
        ...     instructions="Keep it under 100 words",
        ...     model="openai/gpt-4o",
        ...     model_params={"temperature": 0.7}
        ... )
    """
    return expose_method(Create.from_prompt)(
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
    )


@overload
async def async_create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: Union[ModelParam, List[ModelParam]] = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> LiteLLMStream: ...


@overload
def async_create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> str: ...


@overload
def async_create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: List[ModelParam] = ["openai/gpt-4o-mini"],  # type: ignore
    model_params: Optional[Params] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> List[str]: ...


async def async_create_from_prompt(
    prompt: PromptType,
    instructions: Optional[str] = None,
    model: Union[ModelParam, List[ModelParam]] = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
) -> Union[str, List[str], LiteLLMStream]:
    """
    Asynchronously generate text from a prompt using an LLM.

    Args:
        prompt: The prompt to send to the model. Can be a string, list of messages, or other supported format.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use, either as a string (e.g., "openai/gpt-4o-mini") or a list of models.
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        stream: Whether to stream the response tokens as they're generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").

    Returns:
        A string response, list of responses (if multiple models), or a streaming response object.

    Examples:
        >>> await async_create_from_prompt("Tell me a joke about programming")
        "Why do programmers prefer dark mode? Because light attracts bugs!"

        >>> await async_create_from_prompt(
        ...     prompt=[{"role": "user", "content": "What's the weather like today?"}],
        ...     model="openai/gpt-4o",
        ...     stream=True
        ... )
    """
    return expose_method(Create.async_from_prompt)(
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
    )


@overload
def create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...  # Type of schema


@overload
def create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> StructuredReturnType: ...  # Type of schema


@overload
def create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,  # stream of List[Schema] yields Schema
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...


@overload
def create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[StructuredReturnType]: ...


def create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[
    StructuredReturnType,
    List[StructuredReturnType],
    Iterable[StructuredReturnType],
]:
    """
    Generate structured data from a prompt using an LLM according to a Pydantic schema.

    Args:
        schema: The Pydantic model class defining the structure of the expected output.
        prompt: Optional prompt to guide the model's response.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        iterative: Whether to generate the response field by field iteratively.
        n: Number of responses to generate.
        name_override: Optional name to use for the schema in the prompt.
        description_override: Optional description to use for the schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A structured response matching the provided schema, a list of responses, or an iterable of responses.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     bio: str
        >>> create_from_schema(
        ...     schema=Person,
        ...     prompt="Generate a profile for a software developer",
        ...     model="openai/gpt-4o"
        ... )
        Person(name='Alex Chen', age=32, bio='Full-stack developer with 8 years experience...')
    """
    return expose_method(Create.from_schema)(
        schema=schema,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        iterative=iterative,
        n=n,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


@overload
async def async_create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...


@overload
async def async_create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> StructuredReturnType: ...


@overload
async def async_create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...


@overload
async def async_create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[StructuredReturnType]: ...


async def async_create_from_schema(
    schema: SchemaType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[
    StructuredReturnType,
    List[StructuredReturnType],
    Iterable[StructuredReturnType],
]:
    """
    Asynchronously generate structured data from a prompt using an LLM according to a Pydantic schema.

    Args:
        schema: The Pydantic model class defining the structure of the expected output.
        prompt: Optional prompt to guide the model's response.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        iterative: Whether to generate the response field by field iteratively.
        n: Number of responses to generate.
        name_override: Optional name to use for the schema in the prompt.
        description_override: Optional description to use for the schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A structured response matching the provided schema, a list of responses, or an iterable of responses.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     bio: str
        >>> await async_create_from_schema(
        ...     schema=Person,
        ...     prompt="Generate a profile for a software developer",
        ...     model="openai/gpt-4o"
        ... )
        Person(name='Alex Chen', age=32, bio='Full-stack developer with 8 years experience...')
    """
    return expose_method(Create.async_from_schema)(
        schema=schema,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        iterative=iterative,
        n=n,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


def create_from_function(
    model: Optional[ModelParam] = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    n: int = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Callable[..., Any]:
    """
    Create a decorator that transforms a function to use an LLM for its implementation.

    Args:
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        iterative: Whether to generate structured responses field by field iteratively.
        n: Number of responses to generate.
        name_override: Optional name to use for the function in the prompt.
        description_override: Optional description to use for the function in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A decorator function that transforms the decorated function to use an LLM.

    Examples:
        >>> @create_from_function(model="openai/gpt-4o")
        ... def summarize_text(text: str, max_words: int = 100) -> str:
        ...     pass
        >>>
        >>> summary = summarize_text("Lorem ipsum dolor sit amet...", max_words=50)
    """
    return expose_method(Create.from_function)(
        model=model,
        model_params=model_params,
        iterative=iterative,
        n=n,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


@overload
def create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[BaseModel]: ...  # Returns Iterable of a model with 'selection' field


@overload
def create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> BaseModel: ...  # Returns a model with 'selection' field


@overload
def create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[BaseModel]: ...  # Iterable of models with 'selection' field


@overload
def create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[BaseModel]: ...  # List of models with 'selection' field


def create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[BaseModel, List[BaseModel], Iterable[BaseModel]]:
    """
    Generate a selection from a set of options using an LLM.

    Args:
        options: A set of options for the model to choose from.
        prompt: Optional prompt to guide the model's selection.
        instructions: Optional additional instructions to guide the model's selection.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        n: Number of selections to generate.
        name_override: Optional name to use for the options schema in the prompt.
        description_override: Optional description to use for the options schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A model containing the selected option, a list of selections, or an iterable of selections.

    Examples:
        >>> options = {"red", "green", "blue", "yellow"}
        >>> create_from_options(
        ...     options=options,
        ...     prompt="What color is a banana?",
        ...     model="openai/gpt-4o"
        ... )
        OptionSelection(selection='yellow')
    """
    return expose_method(Create.from_options)(
        options=options,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        n=n,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


@overload
async def async_create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[BaseModel]: ...


@overload
async def async_create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: Literal[1] = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> BaseModel: ...


@overload
async def async_create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[BaseModel]: ...


@overload
async def async_create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,  # n > 1
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[BaseModel]: ...


async def async_create_from_options(
    options: OptionsType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    n: int = 1,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[BaseModel, List[BaseModel], Iterable[BaseModel]]:
    """
    Asynchronously generate a selection from a set of options using an LLM.

    Args:
        options: A set of options for the model to choose from.
        prompt: Optional prompt to guide the model's selection.
        instructions: Optional additional instructions to guide the model's selection.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        n: Number of selections to generate.
        name_override: Optional name to use for the options schema in the prompt.
        description_override: Optional description to use for the options schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A model containing the selected option, a list of selections, or an iterable of selections.

    Examples:
        >>> options = {"red", "green", "blue", "yellow"}
        >>> await async_create_from_options(
        ...     options=options,
        ...     prompt="What color is a banana?",
        ...     model="openai/gpt-4o"
        ... )
        OptionSelection(selection='yellow')
    """
    return expose_method(Create.async_from_options)(
        options=options,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        n=n,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


@overload
def create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...


@overload
def create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[StructuredReturnType]: ...


def create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[List[StructuredReturnType], Iterable[StructuredReturnType]]:
    """
    Generate structured data focusing on specific attributes using an LLM.

    Args:
        schema: The Pydantic model class defining the structure of the expected output.
        attributes: List of field names or dictionary mapping field names to weights.
        prompt: Optional prompt to guide the model's response.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        iterative: Whether to generate the response field by field iteratively.
        name_override: Optional name to use for the schema in the prompt.
        description_override: Optional description to use for the schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A list of structured responses or an iterable of structured responses.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     description: str
        ...     price: float
        ...     category: str
        >>> create_from_attributes(
        ...     schema=Product,
        ...     attributes=["name", "price"],
        ...     prompt="Generate a product based on these attributes",
        ...     model="openai/gpt-4o"
        ... )
        [Product(name='Wireless Earbuds', description='', price=79.99, category='')]
    """
    return expose_method(Create.from_attributes)(
        schema=schema,
        attributes=attributes,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        iterative=iterative,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


@overload
async def async_create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[True] = True,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Iterable[StructuredReturnType]: ...


@overload
async def async_create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: Literal[False] = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> List[StructuredReturnType]: ...


async def async_create_from_attributes(
    schema: SchemaType,
    attributes: AttributeType,
    prompt: Optional[PromptType] = None,
    instructions: Optional[str] = None,
    model: ModelParam = "openai/gpt-4o-mini",
    model_params: Optional[Params] = None,
    iterative: bool = False,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    stream: bool = False,
    existing_messages: Optional[List[Message]] = None,
    role_for_prompt: MessageRole = "user",
    mode: InstructorModeParam = "tool_call",
) -> Union[List[StructuredReturnType], Iterable[StructuredReturnType]]:
    """
    Asynchronously generate structured data focusing on specific attributes using an LLM.

    Args:
        schema: The Pydantic model class defining the structure of the expected output.
        attributes: List of field names or dictionary mapping field names to weights.
        prompt: Optional prompt to guide the model's response.
        instructions: Optional additional instructions to guide the model's response.
        model: The model to use (e.g., "openai/gpt-4o-mini").
        model_params: Optional parameters to pass to the model (temperature, max_tokens, etc.).
        iterative: Whether to generate the response field by field iteratively.
        name_override: Optional name to use for the schema in the prompt.
        description_override: Optional description to use for the schema in the prompt.
        stream: Whether to stream the response as it's generated.
        existing_messages: Optional list of previous messages for context.
        role_for_prompt: The role to assign to the prompt message (usually "user").
        mode: The instructor mode to use for structured output generation.

    Returns:
        A list of structured responses or an iterable of structured responses.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Product(BaseModel):
        ...     name: str
        ...     description: str
        ...     price: float
        ...     category: str
        >>> await async_create_from_attributes(
        ...     schema=Product,
        ...     attributes={"name": 0.8, "price": 0.2},
        ...     prompt="Generate a product based on these attributes",
        ...     model="openai/gpt-4o"
        ... )
        [Product(name='Smart Water Bottle', description='', price=34.99, category='')]
    """
    return expose_method(Create.async_from_attributes)(
        schema=schema,
        attributes=attributes,
        prompt=prompt,
        instructions=instructions,
        model=model,
        model_params=model_params,
        iterative=iterative,
        name_override=name_override,
        description_override=description_override,
        stream=stream,
        existing_messages=existing_messages,
        role_for_prompt=role_for_prompt,
        mode=mode,
    )


__all__ = [
    "Create",
    "create_from_prompt",
    "create_from_schema",
    "create_from_function",
    "create_from_options",
    "create_from_attributes",
    "async_create_from_prompt",
    "async_create_from_schema",
    "async_create_from_options",
    "async_create_from_attributes",
]
