"""
💭 tests.test_mock

Contains tests for the MockAI class and mock_completion function.
"""

import pytest
from chatspec.mock import MockAI, mock_completion, MockAIError
from typing import Dict, List, Iterator
from chatspec.utils import _StreamPassthrough

def test_mock_ai_initialization():
    """Test basic MockAI initialization"""
    client = MockAI(
        base_url="http://test.com",
        api_key="test-key",
        organization="test-org",
        timeout=30.0
    )
    assert client.base_url == "http://test.com"
    assert client.api_key == "test-key"
    assert client.organization == "test-org"
    assert client.timeout == 30.0

def test_mock_completion_basic():
    """Test basic completion without streaming"""
    messages = [{"role": "user", "content": "Hello"}]
    response = mock_completion(messages=messages)
    
    assert "id" in response
    assert "choices" in response
    assert len(response["choices"]) == 1
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert "Mock response to: Hello" in response["choices"][0]["message"]["content"]
    assert response["choices"][0]["finish_reason"] == "stop"

def test_mock_completion_streaming():
    """Test streaming completion"""
    messages = [{"role": "user", "content": "Hello"}]
    stream = mock_completion(messages=messages, stream=True)
    
    assert isinstance(stream, _StreamPassthrough)
    chunks = list(stream)
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert "id" in chunk
        assert "choices" in chunk
        assert len(chunk["choices"]) == 1
        assert "delta" in chunk["choices"][0]

def test_mock_completion_with_tools():
    """Test completion with tool calls"""
    messages = [{"role": "user", "content": "Use test tool"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }]
    
    response = mock_completion(messages=messages, tools=tools)
    
    assert "tool_calls" in response["choices"][0]["message"]
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "test_tool"

def test_mock_completion_streaming_with_tools():
    """Test streaming completion with tool calls"""
    messages = [{"role": "user", "content": "Use test tool"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }]
    
    stream = mock_completion(messages=messages, tools=tools, stream=True)
    assert isinstance(stream, _StreamPassthrough)
    chunks = list(stream)
    
    # Last chunk should contain tool calls
    last_chunk = chunks[-1]
    assert "tool_calls" in last_chunk["choices"][0]["delta"]
    assert last_chunk["choices"][0]["finish_reason"] == "tool_calls"

def test_mock_completion_error_handling():
    """Test error handling in mock completion"""
    with pytest.raises(MockAIError):
        mock_completion(messages=[])  # Empty messages should raise error

def test_mock_ai_create_method():
    """Test MockAI.create class method"""
    messages = [{"role": "user", "content": "Hello"}]
    response = MockAI.create(messages=messages)
    
    assert "id" in response
    assert "choices" in response
    assert len(response["choices"]) == 1
    assert response["model"] == "gpt-4o-mini"  # Default model

def test_mock_completion_parameters():
    """Test mock completion with various parameters"""
    messages = [{"role": "user", "content": "Hello"}]
    response = mock_completion(
        messages=messages,
        model="custom-model",
        temperature=0.7,
        max_tokens=100,
        user="test-user"
    )
    
    assert response["model"] == "custom-model"
    assert "choices" in response
    assert len(response["choices"]) == 1

if __name__ == "__main__":
    pytest.main(["-v", __file__])
