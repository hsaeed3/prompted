"""
💬 chatspec.utils.cls

Contains the class export utilities used within the
`chatspec` package. This is used to modularize the
various utilities defined within `chatspec.utils.fn`.
"""

__all__ = [
    "CompletionsUtils",
    "MessagesUtils",
    "ToolsUtils",
    "PydanticUtils",
    "MarkdownUtils",
]


class CompletionsUtils:
    """
    Collection of utility functions in relation to
    chat completion responses & streaming.
    """

    from .fn import (
        is_completion,
        is_stream,
        dump_stream_to_completion,
        dump_stream_to_message,
        parse_model_from_completion,
        parse_model_from_stream,
        print_stream,
        stream_passthrough,
    )

    is_completion = staticmethod(is_completion)
    is_stream = staticmethod(is_stream)
    dump_stream_to_completion = staticmethod(dump_stream_to_completion)
    dump_stream_to_message = staticmethod(dump_stream_to_message)
    parse_model_from_completion = staticmethod(parse_model_from_completion)
    parse_model_from_stream = staticmethod(parse_model_from_stream)
    print_stream = staticmethod(print_stream)
    stream_passthrough = staticmethod(stream_passthrough)


class MessagesUtils:
    """
    Collection of utility functions in relation to
    chat completion messages.
    """

    from .fn import (
        is_message,
        has_system_prompt,
        create_image_message,
        create_input_audio_message,
        normalize_messages,
        normalize_system_prompt,
    )

    is_message = staticmethod(is_message)
    has_system_prompt = staticmethod(has_system_prompt)
    create_image_message = staticmethod(create_image_message)
    create_input_audio_message = staticmethod(create_input_audio_message)
    normalize_messages = staticmethod(normalize_messages)
    normalize_system_prompt = staticmethod(normalize_system_prompt)


class ToolsUtils:
    """
    Collection of utility functions in relation to
    chat completion tools & tool calling.
    """

    from .fn import (
        is_tool,
        has_tool_call,
        was_tool_called,
        run_tool,
        get_tool_calls,
        convert_to_tool,
        convert_to_tools,
    )

    is_tool = staticmethod(is_tool)
    has_tool_call = staticmethod(has_tool_call)
    was_tool_called = staticmethod(was_tool_called)
    run_tool = staticmethod(run_tool)
    get_tool_calls = staticmethod(get_tool_calls)
    convert_to_tool = staticmethod(convert_to_tool)
    convert_to_tools = staticmethod(convert_to_tools)


class PydanticUtils:
    """
    Collection of utility functions in relation to
    pydantic models.
    """

    from .fn import (
        create_field_mapping,
        extract_function_fields,
        convert_to_pydantic_model,
        create_literal_pydantic_model,
        create_selection_model,
        create_bool_model,
    )

    create_field_mapping = staticmethod(create_field_mapping)
    extract_function_fields = staticmethod(extract_function_fields)
    convert_to_pydantic_model = staticmethod(convert_to_pydantic_model)
    create_literal_pydantic_model = staticmethod(
        create_literal_pydantic_model
    )
    create_selection_model = staticmethod(create_selection_model)
    create_bool_model = staticmethod(create_bool_model)


class MarkdownUtils:
    """
    Collection of utility functions in relation to
    markdown formatting.
    """

    from .markdown import (
        markdownify,
        _format_docstring as format_docstring,
    )

    markdownify = staticmethod(markdownify)
    format_docstring = staticmethod(format_docstring)
