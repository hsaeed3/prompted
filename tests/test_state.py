"""
💭 tests.test_state

Contains tests for the `State` class.
"""

import pytest
from chatspec import State

def test_state_initialization():
    """Test basic State initialization"""
    state = State(name="TestBot")
    assert state.name == "TestBot"
    assert state.messages == []
    assert state.system_prompt is None

def test_add_message():
    """Test adding messages to state"""
    state = State()
    state.add_message("Hello")
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "user"
    assert state.messages[0]["content"] == "Hello"

def test_context_management():
    """Test context management functionality"""
    state = State()
    state.update_context("test_key", "test_value", shared=True)
    state.update_context("name", "TestBot", identifier=True)
    
    # Test context retrieval through prompt rendering
    result = state.render_prompt("Value is: {test_key}")
    assert result == "Value is: test_value"

def test_system_prompt():
    """Test system prompt handling"""
    state = State(system_prompt="You are {name}")
    state.update_context("name", "TestBot", shared=True)
    
    assert state.system_prompt == "You are TestBot"
    assert state.messages[0]["role"] == "system"
    assert state.messages[0]["content"] == "You are TestBot"

def test_temporary_thread():
    """Test temporary thread context manager"""
    state = State()
    state.add_message("Original message")
    
    with state.temporary_thread() as temp:
        temp.add_message("Temporary message")
        assert len(temp.messages) == 2  # Including original message
        
    assert len(state.messages) == 1  # Original state unchanged

def test_clear_messages():
    """Test clearing messages"""
    state = State(system_prompt="System prompt")
    state.add_message("Test message")
    
    # First clear with keep_system=True
    state.clear_messages(keep_system=True)
    assert len(state._messages) == 0  # Internal messages cleared
    assert len(state.messages) == 1  # System message still visible in property
    
    # Then clear with keep_system=False
    state.clear_messages(keep_system=False)
    assert len(state._messages) == 0
    assert state._system_prompt is None

def test_markdown_context():
    """Test markdown handling in context"""
    state = State()
    state.update_context(
        "code",
        "print('hello')",
        shared=True,
        markdown=True,
        code_block=True
    )
    
    result = state.render_prompt("Here's some code: {code}")
    assert "```" in result  # Check for code block markers
    assert "print('hello')" in result

def test_state_communication():
    """Test communication between states"""
    state1 = State(name="Bot1")
    # Add name to context as identifier
    state1.update_context("name", "Bot1", identifier=True)
    state2 = State(name="Bot2")
    
    state1.send_message_to(state2, "Hello Bot2")
    assert len(state2.messages) == 1
    assert state2.messages[0]["name"] == "Bot1"
    assert state2.messages[0]["content"] == "Hello Bot2"

if __name__ == "__main__":
    pytest.main(args=["-s", __file__])
