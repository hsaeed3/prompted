import pytest
from pydantic import BaseModel
from typing import List, Optional, Literal
from dataclasses import dataclass

import chatspec
from chatspec.utils import (
    _get_value,
    _make_hashable,
    passthrough,
    is_chat_completion,
    is_stream,
    convert_to_pydantic_model,
    create_literal_pydantic_model,
)

# ------------------------------------------------------------------------------
# Test Data
# ------------------------------------------------------------------------------

# Mock chat completion
CHAT_COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
}

# Mock stream chunks
STREAM_CHUNKS = [
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"content": "Hello"}}]},
    {"choices": [{"delta": {"content": "!"}}]},
]


# Helper classes and functions (not tests)
class SampleModel(BaseModel):
    name: str
    age: int
    tags: Optional[List[str]] = None


@dataclass
class SampleDataclass:
    name: str
    value: int


def sample_function(x: int, y: str = "test") -> None:
    """Test function for conversion."""
    pass


# ------------------------------------------------------------------------------
# Helper Function Tests
# ------------------------------------------------------------------------------


def test_get_value():
    obj_dict = {"key": "value"}
    obj_attr = type("Test", (), {"key": "value"})()

    assert _get_value(obj_dict, "key") == "value"
    assert _get_value(obj_attr, "key") == "value"
    assert _get_value({}, "missing", "default") == "default"


def test_make_hashable():
    # Test with different types
    assert isinstance(_make_hashable({"a": 1, "b": 2}), str)
    assert isinstance(_make_hashable([1, 2, 3]), str)
    assert _make_hashable({"b": 1, "a": 2}) == _make_hashable(
        {"a": 2, "b": 1}
    )


# ------------------------------------------------------------------------------
# Stream Tests
# ------------------------------------------------------------------------------


def test_passthrough():
    # Test with sync stream
    stream = passthrough(STREAM_CHUNKS)
    chunks = list(stream)
    assert len(chunks) == 3
    assert stream.chunks == STREAM_CHUNKS

    # Test reuse
    reused_chunks = list(stream)
    assert reused_chunks == STREAM_CHUNKS


def test_is_chat_completion():
    assert is_chat_completion(CHAT_COMPLETION) == True
    assert is_chat_completion({}) == False

    # Test with passthrough
    stream = passthrough(STREAM_CHUNKS)
    list(stream)  # Consume stream
    assert is_chat_completion(stream) == True


def test_is_stream():
    # Test with raw chunks
    assert is_stream(STREAM_CHUNKS[0]) == True

    # Test with passthrough
    stream = passthrough(STREAM_CHUNKS)
    list(stream)  # Consume stream
    assert is_stream(stream) == True


# ------------------------------------------------------------------------------
# Model Tests
# ------------------------------------------------------------------------------


def test_convert_to_pydantic_model():
    # Test with existing model
    assert convert_to_pydantic_model(SampleModel) == SampleModel

    # Test with dataclass
    model = convert_to_pydantic_model(SampleDataclass)
    assert issubclass(model, BaseModel)

    # Test with function
    func_model = convert_to_pydantic_model(sample_function)
    assert issubclass(func_model, BaseModel)
    assert "x" in func_model.model_fields
    assert "y" in func_model.model_fields

    # Test with dict - without init
    dict_model = convert_to_pydantic_model({"name": "test", "value": 123})
    assert issubclass(dict_model, BaseModel)

    # Test with dict - with init
    dict_instance = convert_to_pydantic_model(
        {"name": "test", "value": 123}, init=True
    )
    assert isinstance(dict_instance, BaseModel)
    assert dict_instance.name == "test"
    assert dict_instance.value == 123


def test_create_literal_pydantic_model():
    # Test with list of strings
    choices = ["a", "b", "c"]
    model = create_literal_pydantic_model(choices)
    assert issubclass(model, BaseModel)

    # Test with existing Literal type
    literal_type = Literal["x", "y", "z"]
    model = create_literal_pydantic_model(literal_type)
    assert issubclass(model, BaseModel)

    # Test invalid input
    with pytest.raises(ValueError):
        create_literal_pydantic_model(123)


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------


def test_stream_to_model_conversion():
    """Test converting a stream to a Pydantic model."""
    stream = passthrough(STREAM_CHUNKS)

    # First pass - consume stream
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in stream
    )
    assert content == "Hello!"

    # Second pass - should work from cache
    assert is_stream(stream)
    assert len(stream.chunks) == 3


def test_error_handling():
    """Test error handling in utility functions."""
    with pytest.raises(ValueError):
        convert_to_pydantic_model(123)  # Invalid type

    with pytest.raises(ValueError):
        create_literal_pydantic_model(None)  # Invalid target


if __name__ == "__main__":
    pytest.main([__file__])
