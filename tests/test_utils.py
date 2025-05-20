"""
💭 tests.test_utils

Contains tests for the utilities module within the `chatspec` package.
"""

import pytest
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import List, Optional, Literal

import prompted
from prompted._utils.fn import (
    is_completion,
    is_stream,
    is_message,
    is_tool,
    has_system_prompt,
    has_tool_call,
    was_tool_called,
    run_tool,
    create_tool_message,
    convert_to_tool,
    convert_to_tools,
    normalize_messages,
    normalize_system_prompt,
    convert_to_pydantic_model,
    create_literal_pydantic_model,
    stream_passthrough,
)


# Test Data
class UserProfile(BaseModel):
    """A user profile model."""

    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")
    email: Optional[str] = Field(None, description="The user's email")


@dataclass
class Address:
    """A simple address dataclass."""

    street: str
    city: str
    country: str = "USA"


def example_tool(x: int, y: str) -> str:
    """Example tool function.

    Args:
        x: An integer parameter
        y: A string parameter

    Returns:
        A string result
    """
    return f"{y} {x}"


# Test Fixtures
@pytest.fixture
def completion():
    return {
        "id": "test",
        "choices": [
            {"message": {"role": "assistant", "content": "Hello"}}
        ],
    }


@pytest.fixture
def stream_completion():
    return {
        "id": "test",
        "choices": [{"delta": {"role": "assistant", "content": "Hello"}}],
    }


@pytest.fixture
def tool_completion():
    return {
        "id": "test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "example_tool",
                                "arguments": '{"x": 1, "y": "test"}',
                            },
                        }
                    ],
                }
            }
        ],
    }


# Test Instance Checking
def test_is_completion(completion):
    assert is_completion(completion)
    assert not is_completion({"invalid": "structure"})


def test_is_stream(stream_completion):
    assert is_stream(stream_completion)
    assert not is_stream({"invalid": "structure"})


def test_is_message():
    valid_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "Be helpful"},
        {"role": "tool", "content": "Result", "tool_call_id": "123"},
    ]
    for msg in valid_messages:
        assert is_message(msg)

    invalid_messages = [
        {"role": "invalid", "content": "test"},
        {"role": "user"},
        {"role": "tool", "content": "Missing tool_call_id"},
    ]
    for msg in invalid_messages:
        assert not is_message(msg)


def test_is_tool():
    valid_tool = {
        "type": "function",
        "function": {
            "name": "test",
            "description": "A test function",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert is_tool(valid_tool)
    assert not is_tool({"invalid": "structure"})


# Test System Prompts
def test_has_system_prompt():
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]
    assert has_system_prompt(messages)
    assert not has_system_prompt([{"role": "user", "content": "Hello"}])


def test_normalize_system_prompt():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Be helpful"},
        {"role": "assistant", "content": "Hi"},
    ]
    normalized = normalize_system_prompt(messages)
    assert normalized[0]["role"] == "system"
    assert len(normalized) == len(messages)


# Test Tool Handling
def test_has_tool_call(tool_completion):
    assert has_tool_call(tool_completion)
    assert not has_tool_call(
        {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}
    )


def test_was_tool_called(tool_completion):
    assert was_tool_called(tool_completion, "example_tool")
    assert not was_tool_called(tool_completion, "nonexistent_tool")


def test_run_tool(tool_completion):
    result = run_tool(tool_completion, example_tool)
    assert result == "test 1"


def test_create_tool_message(tool_completion):
    output = "test result"
    message = create_tool_message(tool_completion, output)
    assert message["role"] == "tool"
    assert message["content"] == output
    assert "tool_call_id" in message


# Test Model Conversion
def test_convert_to_tool():
    # Test function conversion
    tool = convert_to_tool(example_tool)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "example_tool"

    # Test pydantic model conversion
    model_tool = convert_to_tool(UserProfile)
    assert model_tool["type"] == "function"
    assert model_tool["function"]["name"] == "UserProfile"


def test_convert_to_tools():
    tools = [example_tool, UserProfile]
    tools_dict = convert_to_tools(tools)
    assert "example_tool" in tools_dict
    assert "UserProfile" in tools_dict


# Test Message Normalization
def test_normalize_messages():
    # Test string input
    assert normalize_messages("Hello") == [
        {"role": "user", "content": "Hello"}
    ]

    # Test message list
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert normalize_messages(messages) == messages


# Test Model Creation
def test_convert_to_pydantic_model():
    # Test dataclass conversion
    address = Address("123 Main St", "City", "USA")
    model = convert_to_pydantic_model(address, init=True)
    assert isinstance(model, BaseModel)

    # Test type conversion
    str_model = convert_to_pydantic_model(str)
    assert issubclass(str_model, BaseModel)


def test_create_literal_pydantic_model():
    choices = ["red", "green", "blue"]
    model = create_literal_pydantic_model(choices)
    assert issubclass(model, BaseModel)
    assert (
        Literal["red", "green", "blue"]
        == model.model_fields["value"].annotation
    )


# Test Stream Handling
def test_stream_passthrough():
    chunks = [{"content": "Hello"}, {"content": "World"}]
    stream = stream_passthrough(chunks)

    # First iteration
    collected = list(stream)
    assert len(collected) == 2

    # Second iteration (from cache)
    cached = list(stream)
    assert cached == collected


if __name__ == "__main__":
    pytest.main(args=["-s", __file__])
