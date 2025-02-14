"""
💭 chatspec.types

Core type definitions for chat completions.
"""

from typing import Dict, List, Optional, Union, Literal, TypeAlias, Any
from typing_extensions import TypedDict, NotRequired

# ----------------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------------

class Message(TypedDict):
    """Core message type for chat completions."""
    role: Literal["assistant", "user", "system", "tool"]
    content: Optional[str]
    name: NotRequired[str]
    tool_calls: NotRequired[List[Dict[str, Any]]]
    tool_call_id: NotRequired[str]

Messages = List[Message]

# ----------------------------------------------------------------------------
# Tools & Functions
# ----------------------------------------------------------------------------

class FunctionDefinition(TypedDict):
    """Core function definition for tool calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    strict: NotRequired[bool]

class Tool(TypedDict):
    """Core tool type for chat completions."""
    type: Literal["function"]
    function: FunctionDefinition

# ----------------------------------------------------------------------------
# Common Parameters
# ----------------------------------------------------------------------------

ResponseFormat = Literal["text", "json_object", "json_schema"]
ServiceTier = Literal["auto", "default", "scale"]
ToolChoice = Literal["auto", "none", "required"]
Modality = Literal["text", "audio"]

# ----------------------------------------------------------------------------
# Helper Type Aliases
# ----------------------------------------------------------------------------

Metadata = Dict[str, str]
Temperature = float  # 0.0 to 2.0
TopP = float        # 0.0 to 1.0
MaxTokens = int
Stop = Union[str, List[str]]

__all__ = [
    # Core Types
    "Message",
    "Messages",
    "Tool",
    "FunctionDefinition",
    
    # Parameters
    "ResponseFormat",
    "ServiceTier",
    "ToolChoice",
    "Modality",
    
    # Helpers
    "Metadata",
    "Temperature",
    "TopP",
    "MaxTokens",
    "Stop"
]
