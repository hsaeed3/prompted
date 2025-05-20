"""
💭 prompted.agent_types

Dedicated module for types related to agents.
"""

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    Type,
    TypeVar,
    Generic,
)
from typing_extensions import (
    TypeAliasType,
    TypedDict,
    Required,
    NotRequired,
)

from pydantic import BaseModel, Field

from ..types.chat_completions_params import (
    ModelParam,
    MessagesParam,
    Params,
)


# ---------------------------------------------------------------------------------------------
# MODELS / TYPES
# ---------------------------------------------------------------------------------------------


class AgentMessageSchema(BaseModel):
    """
    The schema if an agent returns messages.

    THIS DEFAULTS TO `role == "agent"` TO BE "future" COMPATIBLE WITH THE GOOGLE A2A PROTOCOL.

    You can subclass this to change the role of the message, and set it as the output type
    of an agent.
    """

    role: Literal["agent", "user"] | str = "agent"
    """
    The role of the message.
    """
    content: str
    """
    The content of this message.
    """


class AgentPromptTemplatePart(BaseModel):
    """
    A section within a prompt template. (These are user defined.)

    If a section has a variable that can be formatted, it will be formatted if it is available
    within a passed context object. Otherwise, the section will be left as is.
    """

    name: str | None = None
    """
    The name (header) of this section.
    """
    content: str | None = None
    """
    The content of this section.
    """
    important: bool = False
    """
    Whether this section is important. (this will captialize the full section name
    if given and add extra prompting indicating that the section is important)
    """
    markdown: bool = False
    """
    Whether this section should be formatted as markdown.
    """


class AgentPromptTemplate(BaseModel):
    """
    The prompt template used by agents within the `prompted` library.
    """

    start_part: list[str] | str = Field(default_factory=list)
    """
    The start of the prompt.
    """
    parts: List[AgentPromptTemplatePart] = Field(default_factory=list)
    """
    A list of sections that can be formatted.
    """
    context: str | None = None
    """
    The current context of the agent.
    """
    end_part: list[str] | str = Field(default_factory=list)
    """
    The end of the prompt.
    """


# ---------------------------------------------------------------------------------------------
# SETTINGS & VARS
# ---------------------------------------------------------------------------------------------


AgentOutputType = TypeVar(
    "AgentOutputType", bound=BaseModel | Type | AgentMessageSchema
)
"""
The type of the output of an agent.
"""


AgentContextType = TypeVar("AgentContextType", bound=BaseModel | Dict[str, Any])
"""
If the agent has internal 'state' or context variables (*that can be auto-updated*) by an agent
"""


AgentContextStrategy = TypeAliasType(
    name="AgentContextStrategy", value=Literal["selective", "full"]
)
"""The strategy the agent will use to update it's context."""


AgentEndStrategy = TypeAliasType(
    name="AgentEndStrategy", value=Literal["selective", "full"]
)
"""The strategy the agent will use to end it's response."""
