"""
prompted.specifications.openai.types.tools
"""

from typing import Dict, Literal
from typing_extensions import Required, TypedDict, NotRequired, TypeAliasType

__all__ = ["OpenAIFunctionParameters", "OpenAIFunction", "OpenAITool"]


OpenAIFunctionParameters = TypeAliasType("OpenAIFunctionParameters", Dict[str, object])
"""Generic type variable for the `parameters` field within an OpenAI function."""


class OpenAIFunction(TypedDict):
    """
    A function that is defined within a tool's definition within an
    OpenAI Chat Completions tool definition.
    """

    name: Required[str]
    """
    The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
    underscores and dashes, with a maximum length of 64.
    """
    description: NotRequired[str]
    """
    A description of what the function does, used by the model to choose when and
    how to call the function.
    """
    parameters: Required[OpenAIFunctionParameters]
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


class OpenAITool(TypedDict):
    """
    A tool that is defined within a tool's definition in the
    OpenAI format.
    """

    type: Required[Literal["function"]]
    """
    The type of the tool. Currently, only `function` is supported.
    """
    function: Required[OpenAIFunction]
    """The function that the tool calls."""
