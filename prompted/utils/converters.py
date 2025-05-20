"""
💬 prompted.utils.converters

Contains the various converter functions available within
the `prompted` package
"""

import json
import logging
from pathlib import Path
from dataclasses import is_dataclass
from docstring_parser import parse
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Sequence,
    Union,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from ..common.cache import cached, make_hashable, TYPE_MAPPING
from ..types.chat_completions import (
    Message,
    MessageContentAudioPart,
    MessageContentImagePart,
    MessageRole,
    MessageContentTextPart,
    Tool,
    Completion
)
from .formatting import format_to_markdown
from .identification import is_message

logger = logging.getLogger(__name__)

__all__ = [
    "convert_to_message",
    "convert_to_tool_definition",
    "convert_to_tool_definitions",
    "convert_to_field",
    "convert_to_pydantic_model",
    "convert_to_selection_model",
    "convert_to_boolean_model",
    "convert_to_input_audio_message",
    "convert_to_image_message",
    "convert_completion_to_pydantic_model",
    "convert_stream_to_completion",
    "convert_stream_to_message",
]


def convert_to_message(
    message: Any,
    role: MessageRole | str = "user",
    markdown: bool = False,
    use_parts: bool = False,
    schema: bool = False,
) -> Message:
    """
    Converts a given object into a Chat Completions compatible
    `Message` object.

    Args:
        message : Any
            The object to convert.
        role : MessageRole | str
            The role of the message.
        markdown : bool
            Whether to use markdown.
        use_parts : bool
            Whether to use message content parts.
        schema : bool
            If True, only show schema. If False, show values for initialized objects.

    Returns:
        Message
            The converted message.
    """

    @cached(
        lambda message,
        role="user",
        markdown=False,
        use_parts=False,
        schema=False: make_hashable(
            (message, role, markdown, use_parts, schema)
        )
    )
    def _convert_to_message(
        message: Any,
        role: MessageRole | str = "user",
        markdown: bool = False,
        use_parts: bool = False,
        schema: bool = False,
    ) -> Message:
        if is_message(message):
            return message

        if markdown:
            if not isinstance(message, str):
                try:
                    message = format_to_markdown(message, schema=schema)
                except Exception as e:
                    raise ValueError(
                        f"Error converting object to markdown: {e}"
                    )
        else:
            if not isinstance(message, str):
                try:
                    message = json.dumps(message)
                except Exception as e:
                    raise ValueError(
                        f"Error converting object to JSON: {e}"
                    )

        if use_parts:
            message = MessageContentTextPart(type="text", text=message)
        return Message(
            role=role, content=message if not use_parts else [message]
        )

    return _convert_to_message(message, role, markdown, use_parts, schema)


def convert_to_tool_definition(
    tool: Union[BaseModel, Callable, Dict[str, Any]],
) -> Tool:
    """
    Converts a given object into a tool.

    This function handles:
      - Pydantic models (using their schema and docstring),
      - Python functions (using type hints and docstring),
      - Existing tool dictionaries.

    Args:
        tool: The object to convert into a tool.

    Returns:
        A Tool dictionary compatible with chat completions.

    Raises:
        TypeError: If the input cannot be converted to a tool.
    """
    from typing_inspect import is_literal_type
    from pydantic import BaseModel
    from typing import get_args

    @cached(lambda tool: make_hashable(tool) if tool else "")
    def _convert_to_tool(tool: Any) -> Tool:
        try:
            if (
                isinstance(tool, dict)
                and "type" in tool
                and "function" in tool
            ):
                return tool

            if isinstance(tool, type) and issubclass(tool, BaseModel):
                schema = tool.model_json_schema()
                if "properties" in schema:
                    for prop_name, prop_schema in schema[
                        "properties"
                    ].items():
                        if "enum" in prop_schema:
                            # Handle enum fields as literals
                            prop_schema["enum"] = list(prop_schema["enum"])
                            prop_schema["title"] = prop_name.capitalize()
                            prop_schema["type"] = "string"
                        elif is_literal_type(prop_schema.get("type")):
                            prop_schema["enum"] = list(
                                get_args(prop_schema["type"])
                            )
                            prop_schema["title"] = prop_name.capitalize()
                            prop_schema["type"] = "string"
                        else:
                            prop_schema["title"] = prop_name.capitalize()
                    schema["required"] = list(schema["properties"].keys())
                    schema["additionalProperties"] = False
                    schema["title"] = tool.__name__
                return {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "parameters": schema,
                        "strict": True,
                    },
                }

            if callable(tool):
                import inspect

                sig = inspect.signature(tool)
                properties = {}
                required = []

                # Parse docstring using docstring_parser instead of inspect
                docstring = tool.__doc__
                doc_info = None
                if docstring:
                    doc_info = parse(docstring)

                for name, param in sig.parameters.items():
                    if param.kind in (
                        param.VAR_POSITIONAL,
                        param.VAR_KEYWORD,
                    ):
                        continue

                    param_schema = {
                        "type": "string",
                        "title": name.capitalize(),
                    }

                    # Add description from docstring if available
                    if doc_info and doc_info.params:
                        for doc_param in doc_info.params:
                            if doc_param.arg_name == name:
                                if doc_param.description:
                                    param_schema["description"] = (
                                        doc_param.description
                                    )
                                # Check if parameter is required from docstring
                                if (
                                    doc_param.description
                                    and "required"
                                    in doc_param.description.lower()
                                ):
                                    if name not in required:
                                        required.append(name)

                    if param.annotation != inspect.Parameter.empty:
                        if is_literal_type(param.annotation):
                            param_schema["enum"] = list(
                                get_args(param.annotation)
                            )
                        else:
                            if param.annotation == str:
                                param_schema["type"] = "string"
                            elif param.annotation == int:
                                param_schema["type"] = "integer"
                            elif param.annotation == float:
                                param_schema["type"] = "number"
                            elif param.annotation == bool:
                                param_schema["type"] = "boolean"
                            elif param.annotation == list:
                                param_schema["type"] = "array"
                            elif param.annotation == dict:
                                param_schema["type"] = "object"

                    properties[name] = param_schema
                    if (
                        param.default == inspect.Parameter.empty
                        and name not in required
                    ):
                        required.append(name)

                parameters_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "title": tool.__name__,
                    "additionalProperties": False,
                }

                # Add function description from docstring
                function_schema = {
                    "name": tool.__name__,
                    "strict": True,
                    "parameters": parameters_schema,
                }

                if doc_info and doc_info.short_description:
                    function_schema["description"] = (
                        doc_info.short_description
                    )
                    if doc_info.long_description:
                        function_schema["description"] += (
                            "\n\n" + doc_info.long_description
                        )

                return {
                    "type": "function",
                    "function": function_schema,
                }

            raise TypeError(f"Cannot convert {type(tool)} to tool")
        except Exception as e:
            logger.debug(f"Error converting to tool: {e}")
            raise

    return _convert_to_tool(tool)


def convert_to_tool_definitions(
    tools: Union[List[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Converts a list of tools (which may be BaseModel, callable, or Tool dict)
    into a dictionary mapping tool names to tool definitions.
    If a tool is not already in Tool format, it is converted via convert_to_tool.
    If the original tool is callable, it is attached as the "callable" key.

    Args:
        tools: A list of tools (which may be BaseModel, callable, or Tool dict)

    Returns:
        A dictionary mapping tool names to tool definitions.
    """
    tools_dict: Dict[str, Any] = {}

    if isinstance(tools, dict):
        # Assume already keyed by tool name
        return tools

    if isinstance(tools, list):
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" in tool
            ):
                # Tool is already in correct format
                name = tool["function"].get("name")
                if name:
                    tools_dict[name] = tool
            else:
                # Convert tool to proper format
                converted = convert_to_tool_definition(tool)
                if (
                    "function" in converted
                    and "name" in converted["function"]
                ):
                    name = converted["function"]["name"]
                    tools_dict[name] = converted
                    # Attach original callable if applicable
                    if callable(tool):
                        tools_dict[name]["callable"] = tool

    return tools_dict


def convert_to_field(
    type_hint: Type,
    index: Optional[int] = None,
    description: Optional[str] = None,
    default: Any = ...,
) -> Dict[str, Any]:
    """
    Creates a Pydantic field mapping from a type hint.

    Args:
        type_hint: The Python type to convert
        index: Optional index to append to field name for uniqueness
        description: Optional field description
        default: Optional default value

    Returns:
        Dictionary mapping field name to (type, Field) tuple
    """

    @cached(
        lambda type_hint,
        index=None,
        description=None,
        default=...: make_hashable(
            (type_hint, index, description, default)
        )
    )
    def _create_field_mapping(
        type_hint: Type,
        index: Optional[int] = None,
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Dict[str, Any]:
        try:
            base_name, _ = TYPE_MAPPING.get(
                type_hint, ("value", type_hint)
            )
            field_name = (
                f"{base_name}_{index}" if index is not None else base_name
            )
            return {
                field_name: (
                    type_hint,
                    Field(default=default, description=description),
                )
            }
        except Exception as e:
            logger.debug(f"Error creating field mapping: {e}")
            raise

    return _create_field_mapping(type_hint, index, description, default)


def convert_to_pydantic_model(
    target: Union[
        Type, Sequence[Type], Dict[str, Any], BaseModel, Callable
    ],
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """
    Converts various input types into a pydantic model class or instance.

    Args:
        target: The target to convert (type, sequence, dict, model, or function)
        init: Whether to initialize the model with values (for dataclasses/dicts)
        name: Optional name for the generated model
        description: Optional description for the model/field
        field_name: Optional field name for the generated model (If the target is a single type)
        default: Optional default value for single-type models

    Returns:
        A pydantic model class or instance if init=True
    """

    @cached(
        lambda target,
        init=False,
        name=None,
        description=None,
        field_name=None,
        default=...: make_hashable(
            (target, init, name, description, field_name, default)
        )
    )
    def _convert_to_pydantic_model(
        target: Union[
            Type, Sequence[Type], Dict[str, Any], BaseModel, Callable
        ],
        init: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
        field_name: Optional[str] = None,
        default: Any = ...,
    ) -> Union[Type[BaseModel], BaseModel]:
        model_name = name or "GeneratedModel"

        # Handle existing Pydantic models
        if isinstance(target, type) and issubclass(target, BaseModel):
            return target

        # Handle dataclasses
        if is_dataclass(target):
            hints = get_type_hints(target)
            fields = {}

            # Parse docstring if available
            docstring = target.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            for field_name, hint in hints.items():
                description = ""
                if doc_info and doc_info.params:
                    description = next(
                        (
                            p.description
                            for p in doc_info.params
                            if p.arg_name == field_name
                        ),
                        "",
                    )

                fields[field_name] = (
                    hint,
                    Field(
                        default=getattr(target, field_name)
                        if init
                        else ...,
                        description=description,
                    ),
                )

            model_class = create_model(
                model_name,
                __doc__=description
                or (doc_info.short_description if doc_info else None),
                **fields,
            )

            if init and isinstance(target, type):
                return model_class
            elif init:
                return model_class(
                    **{
                        field_name: getattr(target, field_name)
                        for field_name in hints
                    }
                )
            return model_class

        # Handle callable (functions)
        if callable(target) and not isinstance(target, type):
            fields = extract_function_fields(target)

            # Extract just the short description from the docstring
            doc_info = parse(target.__doc__ or "")
            clean_description = (
                doc_info.short_description if doc_info else None
            )

            return create_model(
                name or target.__name__,
                __doc__=description or clean_description,
                **fields,
            )

        # Handle single types
        if isinstance(target, type):
            field_mapping = convert_to_field(
                target, description=description, default=default
            )
            # If field_name is provided, override the default field name
            if field_name:
                # Get the first (and only) key-value pair from field_mapping
                _, field_value = next(iter(field_mapping.items()))
                field_mapping = {field_name: field_value}
            return create_model(
                model_name, __doc__=description, **field_mapping
            )

        # Handle sequences of types
        if isinstance(target, (list, tuple)):
            field_mapping = {}
            for i, type_hint in enumerate(target):
                if not isinstance(type_hint, type):
                    raise ValueError("Sequence elements must be types")
                # If field_name is provided and this is the first type, use it
                if field_name and i == 0:
                    field_mapping.update(
                        {field_name: convert_to_field(type_hint, description=description, default=default)[next(iter(convert_to_field(type_hint).keys()))]}
                    )
                else:
                    field_mapping.update(
                        convert_to_field(type_hint, index=i)
                    )
            return create_model(
                model_name, __doc__=description, **field_mapping
            )

        # Handle dictionaries
        if isinstance(target, dict):
            if init:
                model_class = create_model(
                    model_name,
                    __doc__=description,
                    **{
                        k: (type(v), Field(default=v))
                        for k, v in target.items()
                    },
                )
                return model_class(**target)
            return create_model(model_name, __doc__=description, **target)

        # Handle model instances
        if isinstance(target, BaseModel):
            # Parse docstring from the model's class
            docstring = target.__class__.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            if init:
                fields = {}
                for k, v in target.model_dump().items():
                    description = ""
                    if doc_info and doc_info.params:
                        description = next(
                            (
                                p.description
                                for p in doc_info.params
                                if p.arg_name == k
                            ),
                            "",
                        )
                    fields[k] = (
                        type(v),
                        Field(default=v, description=description),
                    )

                model_class = create_model(
                    model_name,
                    __doc__=description
                    or (doc_info.short_description if doc_info else None),
                    **fields,
                )
                return model_class(**target.model_dump())
            return target.__class__

        raise ValueError(
            f"Unsupported target type: {type(target)}. Must be a type, "
            "sequence of types, dict, dataclass, function, or Pydantic model."
        )

    return _convert_to_pydantic_model(
        target, init, name, description, field_name, default
    )


def convert_to_selection_model(
    fields: List[str] = [],
    name: str = "Selection",
    description: str | None = None,
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for making a selection from a list of string options.

    The model will have a single field named `selection`. The type of this field
    will be `Literal[*fields]`, meaning its value must be one of the strings
    provided in the `fields` list.

    Args:
        name: The name for the created Pydantic model. Defaults to "Selection".
        description: An optional description for the model (becomes its docstring).
        fields: A list of strings representing the allowed choices for the selection.
                This list cannot be empty.

    Returns:
        A new Pydantic BaseModel class with a 'selection' field.

    Raises:
        ValueError: If the `fields` list is empty, as Literal requires at least one option.
    """
    if not fields:
        raise ValueError(
            "`fields` list cannot be empty for `create_selection_model` "
            "as it defines the possible selections for the Literal type."
        )

    # Create the Literal type from the list of field strings.
    # We can't use unpacking syntax directly with Literal, so we need to handle it differently
    if len(fields) == 1:
        selection_type = Literal[fields[0]]
    else:
        # For multiple fields, we need to use eval to create the Literal type
        # This is because Literal needs to be constructed with the actual string values
        # as separate arguments, not as a list
        literal_str = f"Literal[{', '.join(repr(f) for f in fields)}]"
        selection_type = eval(literal_str)

    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        "selection": (
            selection_type,
            Field(
                ...,
                description="The selected value from the available options.",
            ),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        if fields:
            model_docstring = f"A model for selecting one option from: {', '.join(fields)}."
        else:  # Should not be reached due to the check above, but for completeness
            model_docstring = "A selection model."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel


def convert_to_boolean_model(
    name : str = "Confirmation",
    description : str | None = None,
    field_name : str = "choice"
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for boolean confirmation/response.

    The model will have a single field named `confirmed`. The type of this field
    will be `bool`, meaning its value must be either True or False.

    Args:
        name: The name for the created Pydantic model. Defaults to "Confirmation".
        description: An optional description for the model (becomes its docstring).

    Returns:
        A new Pydantic BaseModel class with a 'confirmed' field.
    """
    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        field_name: (bool, Field(..., description="The boolean confirmation value."))
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        model_docstring = "A model for boolean confirmation."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel


def convert_to_input_audio_message(
    audio: Union[str, Path, bytes],
    format: Literal["wav", "mp3"] = "wav",
    message: Optional[Union[str, Message]] = None,
    as_part: bool = False,
) -> Union[Message, MessageContentAudioPart]:
    """
    Creates a message with input audio content from a url, path, or bytes.

    Args:
        audio: The audio to include - can be a URL string, Path object, or raw bytes
        format: The audio format - either "wav" or "mp3"
        message: Optional existing message to add the audio content to
        as_part: bool
            If True, return a MessageContentAudioPart object instead of a Message object

    Returns:
        A Message object with the audio content part included or just the MessageContentAudioPart
    """
    import base64
    from urllib.parse import urlparse

    # Convert audio to base64 if needed
    if isinstance(audio, Path):
        with open(audio, "rb") as f:
            audio_bytes = f.read()
            audio = base64.b64encode(audio_bytes).decode()
    elif isinstance(audio, bytes):
        audio = base64.b64encode(audio).decode()
    elif isinstance(audio, str) and not urlparse(audio).scheme:
        # Handle string path
        with open(audio, "rb") as f:
            audio_bytes = f.read()
            audio = base64.b64encode(audio_bytes).decode()

    audio_part: MessageContentAudioPart = {
        "type": "input_audio",
        "input_audio": {"data": audio, "format": format},
    }
    
    if as_part:
        return audio_part

    if message is None:
        return {"role": "user", "content": [audio_part]}

    if isinstance(message, str):
        text_part: MessageContentTextPart = {
            "type": "text",
            "text": message,
        }
        return {"role": "user", "content": [text_part, audio_part]}

    # Handle existing Message dict
    if not message.get("content"):
        message["content"] = [audio_part]
    elif isinstance(message["content"], str):
        message["content"] = [
            {"type": "text", "text": message["content"]},
            audio_part,
        ]
    elif isinstance(message["content"], (list, tuple)):
        message["content"] = list(message["content"]) + [audio_part]

    return message


def convert_to_image_message(
    image: Union[str, Path, bytes],
    detail: Literal["auto", "low", "high"] = "auto",
    message: Optional[Union[str, Message]] = None,
    as_part: bool = False,
) -> Union[Message, MessageContentImagePart]:
    """
    Creates a message with image content from a url, path, or bytes.

    This method is also useful for 'injecting' an image into an existing
    message's content.

    Args:
        image: The image to include - can be a URL string, Path object, or raw bytes
        detail: The detail level for the image - one of "auto", "low", or "high"
        message: Optional existing message to add the image content to
        as_part: bool
            If True, return a MessageContentImagePart object instead of a Message object

    Returns:
        A Message object with the image content part included or just the MessageContentImagePart
    """
    import base64
    from urllib.parse import urlparse

    # Convert image to base64 if needed
    if isinstance(image, Path):
        with open(image, "rb") as f:
            image_bytes = f.read()
            image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
    elif isinstance(image, bytes):
        image = f"data:image/png;base64,{base64.b64encode(image).decode()}"
    elif isinstance(image, str) and not urlparse(image).scheme:
        # Handle string path
        with open(image, "rb") as f:
            image_bytes = f.read()
            image = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"

    image_part: MessageContentImagePart = {
        "type": "image_url",
        "image_url": {"url": image, "detail": detail},
    }
    
    if as_part:
        return image_part

    if message is None:
        return {"role": "user", "content": [image_part]}

    if isinstance(message, str):
        text_part: MessageContentTextPart = {
            "type": "text",
            "text": message,
        }
        return {"role": "user", "content": [text_part, image_part]}

    # Handle existing Message dict
    if not message.get("content"):
        message["content"] = [image_part]
    elif isinstance(message["content"], str):
        message["content"] = [
            {"type": "text", "text": message["content"]},
            image_part,
        ]
    elif isinstance(message["content"], (list, tuple)):
        message["content"] = list(message["content"]) + [image_part]

    return message


def convert_completion_to_pydantic_model(
    completion: Any, model: type[BaseModel]
) -> BaseModel:
    """
    Extracts the JSON content from a non-streaming chat completion and initializes
    and returns an instance of the provided Pydantic model.
    """
    try:
        choices = getattr(completion, "choices", None) or completion.get(
            "choices"
        )
        if not choices or len(choices) == 0:
            raise ValueError("No choices found in the completion object.")

        first_choice = choices[0]
        message = getattr(
            first_choice, "message", None
        ) or first_choice.get("message", {})
        content = message.get("content")

        if content is None:
            raise ValueError("No content found in the completion message.")

        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Error parsing JSON content: {e}")

        return model.model_validate(data)
    except Exception as e:
        logger.debug(f"Error parsing model from completion: {e}")
        raise


def convert_stream_to_completion(stream: Any) -> Completion:
    """
    Aggregates a stream of ChatCompletionChunks into a single Completion using the standardized Pydantic models.

    Instead of creating a dictionary for each choice, this function now creates a proper
    CompletionMessage (and Completion.Choice) so that the resulting Completion adheres to the
    models and types expected throughout the library (as seen in chatspec/mock.py).

    Returns:
        A Completion object as defined in `chatspec/types.py`.
    """
    try:
        from ..types import (
            Completion,
            CompletionMessage,
        )  
        from .identification import _get_value

        choices = []
        for chunk in stream:
            # Safely extract content from the chunk's delta field.
            delta_content = _get_value(
                _get_value(chunk.choices[0], "delta", {}), "content", ""
            )
            # Create a proper CompletionMessage instance.
            message = CompletionMessage(
                role="assistant",
                content=delta_content if delta_content is not None else "",
                name=None,
                function_call=None,
                tool_calls=None,
                tool_call_id=None,
            )
            # Wrap the message in a Completion.Choice instance.
            choice_obj = Completion.Choice(
                message=message,
                finish_reason="stop",  # default finish_reason; adjust as needed
                index=0,
                logprobs=None,
            )
            choices.append(choice_obj)

        # Construct and return the Completion object using the proper types.
        return Completion(
            id="stream",
            choices=choices,
            created=0,
            model="stream",
            object="chat.completion",
        )
    except Exception as e:
        logger.debug(f"Error dumping stream to completion: {e}")
        raise


def convert_stream_to_message(stream: Any) -> Message:
    """
    Aggregates a stream of ChatCompletionChunks into a single Message.

    Args:
        stream: An iterable of ChatCompletionChunk objects.

    Returns:
        A Message containing the complete assistant response.
    """
    from .identification import _get_value

    try:
        content_parts: List[str] = []
        tool_calls_dict: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            choices = _get_value(chunk, "choices", [])
            if not choices:
                continue

            for choice in choices:
                delta = _get_value(choice, "delta", {})
                content = _get_value(delta, "content")
                if content:
                    content_parts.append(content)

                # Add null check for tool_calls
                tool_calls = _get_value(delta, "tool_calls", []) or []
                for tool_call in tool_calls:
                    index = _get_value(tool_call, "index")
                    if index is None:
                        continue
                    if index not in tool_calls_dict:
                        tool_calls_dict[index] = {
                            "id": _get_value(tool_call, "id", ""),
                            "type": "function",
                            "function": {
                                "name": _get_value(
                                    _get_value(tool_call, "function", {}),
                                    "name",
                                    "",
                                ),
                                "arguments": _get_value(
                                    _get_value(tool_call, "function", {}),
                                    "arguments",
                                    "",
                                ),
                            },
                        }
                    else:
                        func_obj = _get_value(tool_call, "function", {})
                        if _get_value(func_obj, "arguments"):
                            tool_calls_dict[index]["function"][
                                "arguments"
                            ] += _get_value(func_obj, "arguments")
                        if _get_value(func_obj, "name"):
                            tool_calls_dict[index]["function"]["name"] += (
                                _get_value(func_obj, "name")
                            )
                        if _get_value(tool_call, "id"):
                            tool_calls_dict[index]["id"] = _get_value(
                                tool_call, "id"
                            )

        message: Message = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls_dict:
            message["tool_calls"] = list(tool_calls_dict.values())
        return message
    except Exception as e:
        logger.debug(f"Error dumping stream to message: {e}")
        raise