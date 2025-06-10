"""prompted.specifications.openai.tools"""

from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional
from typing_extensions import TypeAliasType

__all__ = ["OpenAIFunctionParameters", "OpenAIFunction", "OpenAITool"]


OpenAIFunctionParameters = TypeAliasType("OpenAIFunctionParameters", Dict[str, object])
"""Type alias for the parameters within the definition for a tool/function."""


class OpenAIFunction(BaseModel):
    """
    A function that is defined within a tool's definition in the
    OpenAI format.
    """

    name: str
    """
    The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
    underscores and dashes, with a maximum length of 64.
    """
    description: Optional[str] = None
    """
    A description of what the function does, used by the model to choose when and
    how to call the function.
    """
    parameters: OpenAIFunctionParameters
    """
    The parameters the functions accepts, described as a JSON Schema object.
    See the OpenAI guide for examples: https://platform.openai.com/docs/guides/function-calling
    and the JSON Schema reference for documentation: https://json-schema.org/understanding-json-schema/
    """
    strict: Optional[bool] = None
    """
    Whether to enable strict schema adherence when generating the function call.
    If set to true, the model will follow the exact schema defined in the parameters field.
    Only a subset of JSON Schema is supported when strict is true.
    """


class OpenAITool(BaseModel):
    """
    A tool that is defined within a tool's definition in the
    OpenAI format.
    """

    type: Literal["function"]
    """
    The type of the tool. Currently, only `function` is supported.
    """
    function: OpenAIFunction
    """
    The function that the tool calls."""
