"""
💭 tests.test_mock

Contains tests for the MockAI class and mock_completion function.
"""

import pytest
from chatspec.mock import mock_completion
from chatspec.utils import _StreamPassthrough


def test_mock_completion_basic():
    """Test basic completion without streaming"""
    messages = [{"role": "user", "content": "Hello"}]
    response = mock_completion(messages=messages)

    assert response.id is not None
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert "Mock response to: Hello" in response.choices[0].message.content
    assert response.choices[0].finish_reason == "stop"


def test_mock_completion_streaming():
    """Test streaming completion"""
    messages = [{"role": "user", "content": "Hello"}]
    stream = mock_completion(messages=messages, stream=True)

    assert isinstance(stream, _StreamPassthrough)
    chunks = list(stream)
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.id is not None
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta is not None


def test_mock_completion_with_tools():
    """Test completion with tool calls"""
    messages = [{"role": "user", "content": "Use test tool"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    response = mock_completion(messages=messages, tools=tools)

    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].finish_reason == "tool_calls"
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "test_tool"


def test_mock_completion_streaming_with_tools():
    """Test streaming completion with tool calls"""
    messages = [{"role": "user", "content": "Use test tool"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    stream = mock_completion(messages=messages, tools=tools, stream=True)
    assert isinstance(stream, _StreamPassthrough)
    chunks = list(stream)

    # Last chunk should contain tool calls
    last_chunk = chunks[-1]
    assert last_chunk.choices[0].delta.tool_calls is not None
    assert last_chunk.choices[0].finish_reason == "tool_calls"


def test_mock_completion_parameters():
    """Test mock completion with various parameters"""
    messages = [{"role": "user", "content": "Hello"}]
    response = mock_completion(
        messages=messages,
        model="custom-model",
        temperature=0.7,
        max_tokens=100,
        user="test-user",
    )

    assert response.model == "custom-model"
    assert len(response.choices) == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
