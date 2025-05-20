"""
prompted.types.instructor

Helper types for the `Instructor` library.
"""

from typing import Literal

__all__ = [
    "InstructorModeParam",
]


InstructorModeParam = Literal[
    "function_call",
    "parallel_tool_call",
    "tool_call",
    "tools_strict",
    "json_mode",
    "json_o1",
    "markdown_json_mode",
    "json_schema_mode",
    "anthropic_tools",
    "anthropic_reasoning_tools",
    "anthropic_json",
    "mistral_tools",
    "mistral_structured_outputs",
    "vertexai_tools",
    "vertexai_json",
    "vertexai_parallel_tools",
    "gemini_json",
    "gemini_tools",
    "genai_tools",
    "genai_structured_outputs",
    "cohere_tools",
    "cohere_json_object",
    "cerebras_tools",
    "cerebras_json",
    "fireworks_tools",
    "fireworks_json",
    "writer_tools",
    "bedrock_tools",
    "bedrock_json",
    "perplexity_json",
    "openrouter_structured_outputs",
]
"""Instructor prompt/parsing mode for structured outputs."""