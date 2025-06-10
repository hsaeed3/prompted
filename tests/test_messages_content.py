import pytest
from prompted.messages.content import (
    TextContent,
    ImageContent,
    AudioContent,
    ToolContent,
    RefusalContent,
)


def test_text_content_creation():
    content = TextContent(text="Hello, world!")
    assert content.type == "text"
    assert content.text == "Hello, world!"


def test_text_content_to_openai():
    content = TextContent(text="Hello, world!")
    openai_format = content.to_openai()
    assert openai_format == {"type": "text", "text": "Hello, world!"}


def test_text_content_from_openai():
    data = {"type": "text", "text": "Hello, world!"}
    content = TextContent.from_openai(data)
    assert content.type == "text"
    assert content.text == "Hello, world!"


def test_image_content_creation():
    content = ImageContent(image="data:image/png;base64,abc123")
    assert content.type == "image"
    assert content.image == "data:image/png;base64,abc123"
    assert content.detail == "auto"


def test_image_content_with_custom_detail():
    content = ImageContent(image="data:image/png;base64,abc123", detail="high")
    assert content.detail == "high"


def test_image_content_file_processing():
    """Test that image content can process actual image files"""
    content = ImageContent(image="tests/assets/example.jpg")
    assert content.type == "image"
    assert content.image.startswith("data:image/")
    assert "base64," in content.image


def test_image_content_to_openai():
    content = ImageContent(image="data:image/png;base64,abc123")
    openai_format = content.to_openai()
    assert openai_format == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123", "detail": "auto"},
    }


def test_image_content_from_openai():
    data = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123", "detail": "high"},
    }
    content = ImageContent.from_openai(data)
    assert content.type == "image"
    assert content.image == "data:image/png;base64,abc123"
    assert content.detail == "high"


def test_audio_content_creation():
    # Using valid base64 data
    content = AudioContent(audio="dGVzdCBhdWRpbyBkYXRh", format="mp3")
    assert content.type == "audio"
    assert content.audio == "dGVzdCBhdWRpbyBkYXRh"
    assert content.format == "mp3"


def test_audio_content_format_inference():
    """Test format inference from file extension"""
    content = AudioContent(audio="tests/assets/example.mp3")
    assert content.format == "mp3"
    assert content.type == "audio"
    # Audio should be base64 encoded after processing
    assert len(content.audio) > 100  # Should be much longer after encoding


def test_audio_content_file_processing():
    """Test that audio content can process actual audio files"""
    content = AudioContent(audio="tests/assets/example.mp3")
    assert content.type == "audio"
    assert content.format == "mp3"
    # Should be base64 encoded
    import base64

    try:
        base64.b64decode(content.audio, validate=True)
        assert True  # If no exception, it's valid base64
    except Exception:
        assert False, "Audio should be valid base64 after processing"


def test_audio_content_to_openai():
    content = AudioContent(audio="dGVzdCBhdWRpbyBkYXRh", format="mp3")
    openai_format = content.to_openai()
    assert openai_format == {
        "type": "input_audio",
        "input_audio": {"data": "dGVzdCBhdWRpbyBkYXRh", "format": "mp3"},
    }


def test_audio_content_from_openai():
    data = {
        "type": "input_audio",
        "input_audio": {"data": "dGVzdCBhdWRpbyBkYXRh", "format": "mp3"},
    }
    content = AudioContent.from_openai(data)
    assert content.type == "audio"
    assert content.audio == "dGVzdCBhdWRpbyBkYXRh"
    assert content.format == "mp3"


def test_refusal_content_creation():
    content = RefusalContent(refusal="I cannot do that.")
    assert content.type == "refusal"
    assert content.refusal == "I cannot do that."


def test_refusal_content_to_openai():
    content = RefusalContent(refusal="I cannot do that.")
    openai_format = content.to_openai()
    assert openai_format == {"type": "refusal", "refusal": "I cannot do that."}


def test_refusal_content_from_openai():
    data = {"type": "refusal", "refusal": "I cannot do that."}
    content = RefusalContent.from_openai(data)
    assert content.type == "refusal"
    assert content.refusal == "I cannot do that."


def test_tool_content_creation():
    content = ToolContent(
        id="call_123",
        name="test_tool",
        parameters={"param1": "value1"},
        result="success",
    )
    assert content.type == "tool"
    assert content.id == "call_123"
    assert content.name == "test_tool"
    assert content.parameters == {"param1": "value1"}
    assert content.result == "success"


def test_tool_content_to_openai_tool_call():
    content = ToolContent(
        id="call_123", name="test_tool", parameters={"param1": "value1"}
    )
    tool_call = content.to_openai_tool_call(as_message=False)
    assert tool_call == {
        "id": "call_123",
        "type": "function",
        "function": {"name": "test_tool", "arguments": '{"param1": "value1"}'},
    }


def test_tool_content_to_openai_tool_message():
    content = ToolContent(id="call_123", name="test_tool", result="success")
    tool_message = content.to_openai_tool_message()
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "success",
    }


def test_tool_content_to_openai():
    content = ToolContent(
        id="call_123",
        name="test_tool",
        parameters={"param1": "value1"},
        result="success",
    )
    messages = content.to_openai()
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "tool"


def test_tool_content_to_openai_no_result():
    """Test that only tool call message is returned when no result exists"""
    content = ToolContent(
        id="call_123",
        name="test_tool",
        parameters={"param1": "value1"},
        # No result
    )
    messages = content.to_openai()
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "tool_calls" in messages[0]


def test_tool_content_from_openai_tool_call():
    data = {
        "id": "call_123",
        "type": "function",
        "function": {"name": "test_tool", "arguments": '{"param1": "value1"}'},
    }
    content = ToolContent.from_openai_tool_call(data, result="success")
    assert content.id == "call_123"
    assert content.name == "test_tool"
    assert content.parameters == {"param1": "value1"}
    assert content.result == "success"


def test_tool_content_from_openai_tool_message():
    data = {"role": "tool", "tool_call_id": "call_123", "content": "success"}
    content = ToolContent.from_openai_tool_message(
        data, name="test_tool", parameters={"param1": "value1"}
    )
    assert content.id == "call_123"
    assert content.name == "test_tool"
    assert content.parameters == {"param1": "value1"}
    assert content.result == "success"


def test_tool_content_from_openai():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": '{"param1": "value1"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "success"},
    ]
    content = ToolContent.from_openai(messages)
    assert content.id == "call_123"
    assert content.name == "test_tool"
    assert content.parameters == {"param1": "value1"}
    assert content.result == "success"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
