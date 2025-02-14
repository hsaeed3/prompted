import pytest
from typing import List, Dict, Any
from dataclasses import dataclass

from chatspec import State
from chatspec.types import Message

# ------------------------------------------------------------------------------
# Test Data
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]

STREAM_CHUNKS = [
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"content": "Hello"}}]},
    {"choices": [{"delta": {"content": "!"}}]},
]

# ------------------------------------------------------------------------------
# Mock Tools & Agent Functions
# ------------------------------------------------------------------------------

def mock_tool(x: int, y: str = "test") -> str:
    """A mock tool for testing."""
    return f"Tool output: {x}, {y}"

def mock_agent_func(messages: List[Message], **kwargs) -> Dict[str, Any]:
    """Mock agent function that returns a simple response."""
    return {"role": "assistant", "content": "Mock response"}

def mock_streaming_agent_func(messages: List[Message], **kwargs) -> List[Dict[str, Any]]:
    """Mock agent function that returns a stream of chunks."""
    return STREAM_CHUNKS

def mock_tool_using_agent_func(messages: List[Message], **kwargs) -> Dict[str, Any]:
    """Mock agent function that includes a tool call."""
    return {
        "role": "assistant",
        "content": "Let me help with that.",
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "mock_tool",
                "arguments": '{"x": 42, "y": "test"}'
            }
        }]
    }

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_state_initialization():
    """Test basic state initialization and message handling."""
    # Initialize with messages
    state = State(_messages=MESSAGES)
    assert state._system_prompt == SYSTEM_PROMPT
    assert len(state._messages) == 2  # System prompt is extracted
    
    # Initialize empty
    empty_state = State()
    assert empty_state._system_prompt is None
    assert len(empty_state._messages) == 0

def test_message_operations():
    """Test message addition and manipulation."""
    state = State()
    
    # Add single message
    state.add_message("Hello!")
    assert len(state._messages) == 1
    assert state._messages[0]["role"] == "user"
    
    # Add multiple messages
    state.add_messages(MESSAGES[1:])  # Skip system message
    assert len(state._messages) == 3
    
    # Clear messages
    state.clear_messages()
    assert len(state._messages) == 0

def test_context_handling():
    """Test context addition and system prompt generation."""
    state = State(_system_prompt=SYSTEM_PROMPT)
    
    # Add context
    state.add_context({"key": "value"})
    assert "key: value" in state.system_prompt
    
    # Remove context
    state.remove_context(lambda x: x["key"] == "value")
    assert state.system_prompt == SYSTEM_PROMPT

def test_temporary_thread():
    """Test temporary thread creation and isolation."""
    initial_messages = [
        {"role": "user", "content": "Original message"}
    ]
    state = State(_messages=initial_messages.copy())
    
    with state.temporary_thread() as temp:
        temp.add_message("This is temporary")
        # Verify temp thread has the temporary message
        assert "This is temporary" in temp._messages[-1]["content"]
        # Verify original state doesn't have the temporary message
        assert "This is temporary" not in str(state._messages)
    
    # Verify the temporary message doesn't persist after the context
    assert len(state._messages) == 1
    assert "This is temporary" not in str(state._messages)

def test_message_filtering():
    """Test message filtering and subsetting."""
    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    state = State(_messages=test_messages.copy())
    
    # Verify initial state (should have 2 messages after system prompt extraction)
    assert len(state._messages) == 2
    
    # Get subset
    subset = state.last_n(2)
    assert len(subset._messages) == 2
    assert subset._messages[0]["content"] == "Hello!"
    assert subset._messages[1]["content"] == "Hi there!"
    
    # Filter messages
    filtered = state.filter(lambda m: m.get("role") == "user")
    assert all(m.get("role") == "user" for m in filtered._messages)
    assert len(filtered._messages) == 1

def test_tool_registration():
    """Test tool registration and retrieval."""
    state = State()
    
    # Register tool
    state.register_tool("mock_tool", mock_tool)
    assert "mock_tool" in state.tools
    
    # Get tool
    tool = state.get_tool("mock_tool")
    assert tool == mock_tool

def test_agent_integration():
    """Test agent function integration."""
    state = State(agent_func=mock_agent_func)
    
    # Basic prompt
    response = state.prompt("Hello!")
    assert response["role"] == "assistant"
    assert "Mock response" in response["content"]
    
    # Update parameters
    state.update_params(temperature=0.5)
    assert state.temperature == 0.5

def test_streaming_response():
    """Test handling of streaming responses."""
    state = State(agent_func=mock_streaming_agent_func)
    
    # Process streaming response
    response = state.prompt("Hello!")
    assert response["role"] == "assistant"
    assert response["content"] == "Hello!"  # Combined from stream chunks
    
    # Verify message was added
    assert state.last_assistant_message["content"] == "Hello!"

def test_tool_execution():
    """Test tool execution from responses."""
    state = State(agent_func=mock_tool_using_agent_func)
    state.register_tool("mock_tool", mock_tool)
    
    # Prompt and execute tool
    response = state.prompt("Use the tool")
    
    # Verify tool message was added
    tool_message = state.last_message
    assert tool_message["role"] == "tool"
    assert "Tool output: 42, test" in tool_message["content"]

def test_error_handling():
    """Test error handling in various scenarios."""
    state = State()
    
    # Test prompting without agent function
    with pytest.raises(ValueError):
        state.prompt("Hello!")
    
    # Test invalid parameter update
    with pytest.raises(ValueError):
        state.update_params(invalid_param=123)

def test_message_properties():
    """Test convenience properties for message access."""
    state = State(_messages=MESSAGES)
    
    assert state.last_message["content"] == "Hi there!"
    assert state.last_user_message["content"] == "Hello!"
    assert state.last_assistant_message["content"] == "Hi there!"

def test_state_merging():
    """Test merging states."""
    state1 = State(_messages=MESSAGES[:2])
    state2 = State(_messages=MESSAGES[2:])
    
    state1.merge_from(state2)
    assert len(state1._messages) == len(MESSAGES) - 1  # Minus system message

if __name__ == "__main__":
    pytest.main([__file__])
