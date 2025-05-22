"""
prompted.types.completions.tools

Contains type definitions for tools used within chat completions.
NOTE:
Tool messages and tool calls are contained within the `.messages` submodule,
in this `prompted.types.completions` module.
"""

from typing import Any, Dict, List, Literal, Union
from typing_extensions import TypeAliasType, TypedDict, Required, NotRequired

# -----------------------------------------------------------------------------
# Function
# -----------------------------------------------------------------------------

FunctionParameters = TypeAliasType("FunctionParameters", Dict[str, object])
"""Generic type variable for the `parameters` field within a function."""


class Function(TypedDict, total=False):
    """
    Type definition for the 'function' field within the definition of a
    completion tool.
    """

    name: str
    """
    The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
    underscores and dashes, with a maximum length of 64.
    """
    description: NotRequired[str]
    """
    A description of what the function does, used by the model to choose when and
    how to call the function.
    """
    parameters: FunctionParameters
    """
    The parameters the functions accepts, described as a JSON Schema object.
    See the OpenAI guide for examples: https://platform.openai.com/docs/guides/function-calling
    and the JSON Schema reference for documentation: https://json-schema.org/understanding-json-schema/
    """
    strict: NotRequired[bool]
    """
    Whether to enable strict schema adherence when generating the function call.
    If set to true, the model will follow the exact schema defined in the parameters field.
    Only a subset of JSON Schema is supported when strict is true.
    """


class CompletionTool(TypedDict):
    """
    A dictionary representing a tool that can be called by an LLM.
    """

    type: Literal["function"]
    """
    The type of the tool. Currently, only `function` is supported.
    """
    function: Function
    """
    The function that the tool calls.
    """


__all__ = ["FunctionParameters", "Function", "CompletionTool"]
