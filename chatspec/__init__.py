"""
## 💭 chatspec

Types & utilities built for processing & interacting with objects used in
the chat completions API specification.

### Package Contents:

- `_main.py`: Utilities for processing & interacting with chat completions.
- `params.py`: Types for the parameters used in chat completions.
- `types.py`: Types for the responses from chat completions.
- `mock.py`: A mocked implementation of the OpenAI client for chat completions.
- `state.py`: A class for managing the state of a chat.
"""

from .utils import *
from . import params, types
from .mock import MockAI, mock_completion
from .state import State


# add params class to top level
Params = params.Params


__all__ = [
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
    "get_tool_calls",
    "dump_stream_to_message",
    "dump_stream_to_completion",
    "parse_model_from_completion",
    "parse_model_from_stream",
    "print_stream",
    "normalize_messages",
    "normalize_system_prompt",
    "create_field_mapping",
    "extract_function_fields",
    "convert_to_pydantic_model",
    "convert_to_tools",
    "convert_to_tool",
    "create_literal_pydantic_model",
    "stream_passthrough",
    "markdownify",
    # params
    "params",
    "Params",
    # types
    "types",
    # mock
    "MockAI",
    "mock_completion",
    # state
    "State",
]
