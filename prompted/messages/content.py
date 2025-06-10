"""prompted.messages.content

Contains the `prompted` implementation of Message Content Parts,
which are used to represent blocks of a single chat completion message
that represent different formats of data, or different purposes of data."""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import TypeAliasType
from pydantic import BaseModel, Field, RootModel, field_validator

__all__ = (
    "TextContent",
    "ImageContent",
    "AudioContent",
    "ToolContent",
    "RefusalContent",
    "Content",
)


class BaseContent(BaseModel):
    """Base representation of a content part within a mesasge."""

    type: str


class TextContent(BaseContent):
    """Message content part that represents text content."""

    type: Literal["text"] = "text"
    """The type of content part. (Always "text")"""
    text: str
    """The text content of the message."""

    def to_openai(self) -> Dict[str, Any]:
        """Converts this text content part into a valid OpenAI Chat Completions
        object."""
        return self.model_dump(mode="json")

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "TextContent":
        """Creates a TextContent instance from OpenAI Chat Completions format."""
        return cls(text=data["text"])


class ImageContent(BaseContent):
    """Combined representation of an image content part within a message. This
    extends the image content part definition within the OpenAI specification
    to allow for an image to be given as a path to a file, along with it's standard
    input of being a base64 encoded string or a URL."""

    type: Literal["image"] = "image"
    """The type of content part. (Always "image")"""
    image: str
    """The image data - can be a file path, base64 encoded string, or URL."""
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    """Specifies the detail level of the image processing."""

    @field_validator("image")
    @classmethod
    def process_image_input(cls, v: str) -> str:
        """Process the image input - convert file paths to base64, leave URLs and base64 as-is."""
        # Check if it's already a base64 data URL
        if v.startswith("data:image/"):
            return v

        # Check if it's a URL
        if v.startswith(("http://", "https://")):
            return v

        # Check if it's already base64 encoded (without data URI prefix)
        try:
            # Try to decode as base64 - if successful, assume it's already encoded
            base64.b64decode(v, validate=True)
            # If we get here, it's valid base64 - add data URI prefix if missing
            if not v.startswith("data:"):
                return f"data:image/png;base64,{v}"
            return v
        except Exception:
            pass

        # Check if it looks like a file path (has extension or exists as file)
        path = Path(v)
        if path.exists() or (
            path.suffix and len(path.suffix) <= 5
        ):  # Check for file extension
            try:
                if not path.exists():
                    raise ValueError(f"Image file not found: {v}")

                # Get MIME type
                mime_type, _ = mimetypes.guess_type(str(path))
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = "image/png"  # Default fallback

                # Read and encode the file
                with open(path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    return f"data:{mime_type};base64,{encoded_string}"

            except Exception as e:
                raise ValueError(f"Failed to process image input '{v}': {str(e)}")

        # If it's not a URL, base64, or file path, just return as-is
        # This allows for custom image data formats or pre-processed strings
        return v

    def to_openai(self) -> Dict[str, Any]:
        """Converts this image content part into a valid OpenAI Chat Completions object."""
        return {
            "type": "image_url",
            "image_url": {"url": self.image, "detail": self.detail},
        }

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "ImageContent":
        """Creates an ImageContent instance from OpenAI Chat Completions format."""
        image_url_data = data["image_url"]
        return cls(
            image=image_url_data["url"], detail=image_url_data.get("detail", "auto")
        )


class AudioContent(BaseContent):
    """Combined representation of an audio content part within a message. This
    extends the audio content part definition within the OpenAI specification
    to allow for an audio file to be given as a path to a file, along with it's standard
    input of being a base64 encoded string."""

    type: Literal["audio"] = "audio"
    """The type of content part. (Always "audio")"""
    audio: str
    """The audio data - can be a file path or base64 encoded string."""
    format: Optional[Literal["wav", "mp3"]] = None
    """The format of the audio. If not specified, will be inferred from file extension."""

    @field_validator("audio")
    @classmethod
    def process_audio_input(cls, v: str) -> str:
        """Process the audio input - convert file paths to base64, leave base64 as-is."""
        # Check if it's already base64 encoded (without data URI prefix)
        try:
            # Try to decode as base64 - if successful, assume it's already encoded
            base64.b64decode(v, validate=True)
            # If we get here, it's valid base64
            return v
        except Exception:
            pass

        # Check if it looks like a file path (has extension or exists as file)
        path = Path(v)
        if path.exists() or (
            path.suffix and len(path.suffix) <= 5
        ):  # Check for file extension
            try:
                if not path.exists():
                    raise ValueError(f"Audio file not found: {v}")

                # Read and encode the file
                with open(path, "rb") as audio_file:
                    encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
                    return encoded_string

            except Exception as e:
                raise ValueError(f"Failed to process audio input '{v}': {str(e)}")

        # If it's not valid base64 and not a file path, just return as-is
        # This allows for custom audio data formats or pre-processed strings
        return v

    def __init__(self, **data):
        """Initialize AudioContent and infer format from file extension if not provided."""
        # Store original audio input before processing for format inference
        original_audio = data.get("audio", "")

        super().__init__(**data)

        # Infer format if not provided
        if self.format is None:
            try:
                # Check if the original input was a file path (not base64)
                base64.b64decode(original_audio, validate=True)
                # If we get here, it's already base64, use default
                self.format = "mp3"
            except Exception:
                # Not base64, likely a file path - infer from extension
                path = Path(original_audio)
                extension = path.suffix.lower()
                if extension == ".wav":
                    self.format = "wav"
                elif extension == ".mp3":
                    self.format = "mp3"
                else:
                    self.format = "mp3"  # Default fallback

    def to_openai(self) -> Dict[str, Any]:
        """Converts this audio content part into a valid OpenAI Chat Completions object."""
        return {
            "type": "input_audio",
            "input_audio": {"data": self.audio, "format": self.format or "mp3"},
        }

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "AudioContent":
        """Creates an AudioContent instance from OpenAI Chat Completions format."""
        input_audio_data = data["input_audio"]
        return cls(
            audio=input_audio_data["data"], format=input_audio_data.get("format", "mp3")
        )


class RefusalContent(BaseContent):
    """Message content part that represents a refusal response from the model."""

    type: Literal["refusal"] = "refusal"
    """The type of content part. (Always "refusal")"""
    refusal: str
    """The refusal message generated by the model."""

    def to_openai(self) -> Dict[str, Any]:
        """Converts this refusal content part into a valid OpenAI Chat Completions object."""
        return {"type": "refusal", "refusal": self.refusal}

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> "RefusalContent":
        """Creates a RefusalContent instance from OpenAI Chat Completions format."""
        return cls(refusal=data["refusal"])


class ToolContent(BaseContent):
    """Custom `prompted` content part that represents both a tool call and the
    result of that tool call. This is converted internally as both a tool call
    and a tool result on runtime.

    NOTE:
    THIS IS ONE OF THE MOST OPINIONATED TYPES IN THE `prompted` FRAMEWORK, IT'S
    RECOMMENDED TO GIVE THESE ATTRIBUTES A QUICK LOOK.

    Attributes:
        type : Literal['tool']
            The type of content part. (Always "tool")
        name : str
            The name of the tool that was called.
        id : str
            The ID of the original tool call that this content part represents.
        parameters : Dict[str, Any]
            The parameters of the original tool call that this content part represents.
            NOTE:
            These are in a dictionary format, not a JSON string.
        result : Any
            The function's return content.
    """

    type: Literal["tool"] = "tool"
    """The type of content part. (Always "tool")"""
    id: str
    """The ID of the original tool call that this content part represents."""
    name: str
    """The name of the tool that was called."""
    parameters: Dict[str, Any] | None = None
    """The parameters of the original tool call that this content part represents.
    NOTE:
    These are in a dictionary format, not a JSON string."""
    result: Any | None = None
    """The function's return content."""

    def to_openai_tool_call(self, as_message: bool = True) -> Dict[str, Any]:
        """Converts this tool content part into a valid OpenAI tool call format.

        Args:
            as_message: If True, returns it within an assistant message dict.
                       If False, returns just the tool call object.
        """
        import json

        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.parameters or {}),
            },
        }

        if as_message:
            return {"role": "assistant", "tool_calls": [tool_call]}
        else:
            return tool_call

    def to_openai_tool_message(self) -> Dict[str, Any]:
        """Converts this tool content part into a valid OpenAI tool message format."""
        # Convert result to string if it's not already
        content = str(self.result) if self.result is not None else ""

        return {"role": "tool", "tool_call_id": self.id, "content": content}

    @classmethod
    def from_openai_tool_call(
        cls, data: Dict[str, Any], result: Any = None
    ) -> "ToolContent":
        """Creates a ToolContent instance from OpenAI tool call format."""
        import json

        function_data = data["function"]
        parameters = (
            json.loads(function_data["arguments"]) if function_data["arguments"] else {}
        )

        return cls(
            id=data["id"],
            name=function_data["name"],
            parameters=parameters,
            result=result,
        )

    @classmethod
    def from_openai_tool_message(
        cls, data: Dict[str, Any], name: str, parameters: Dict[str, Any] = None
    ) -> "ToolContent":
        """Creates a ToolContent instance from OpenAI tool message format."""
        return cls(
            id=data["tool_call_id"],
            name=name,
            parameters=parameters,
            result=data["content"],
        )

    def to_openai(self) -> List[Dict[str, Any]]:
        """Converts this tool content to a list of OpenAI messages.

        Returns:
            List containing:
            - Tool call message (assistant role)
            - Tool result message (tool role) - only if result exists
        """
        messages = [self.to_openai_tool_call(as_message=True)]

        # Add tool result message if result exists
        if self.result is not None:
            messages.append(self.to_openai_tool_message())

        return messages

    @classmethod
    def from_openai(cls, messages: List[Dict[str, Any]]) -> "ToolContent":
        """Creates a ToolContent instance from a list of OpenAI messages.

        Args:
            messages: List containing tool call message and optionally tool result message
        """
        if not messages:
            raise ValueError("At least one message is required")

        # Find the tool call message (assistant role with tool_calls)
        tool_call_msg = None
        tool_result_msg = None

        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_call_msg = msg
            elif msg.get("role") == "tool":
                tool_result_msg = msg

        if not tool_call_msg:
            raise ValueError("No assistant message with tool_calls found")

        # Extract data from the first tool call
        tool_call = tool_call_msg["tool_calls"][0]
        result = None

        # Extract result from tool message if present
        if tool_result_msg:
            result = tool_result_msg["content"]

        return cls.from_openai_tool_call(tool_call, result=result)


class Content(
    RootModel[
        Union[TextContent, ImageContent, AudioContent, ToolContent, RefusalContent]
    ]
):
    """Model representation for a single content part within the
    `content` key of a chat message."""

    root: Union[
        TextContent, ImageContent, AudioContent, ToolContent, RefusalContent
    ] = Field(discriminator="type")
