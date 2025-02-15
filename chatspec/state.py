"""
## 💭 chatspec.state

Contains the `State` class, a manager for messages, tools and other
contextual information used in chat completions in a conversational
context.
"""

import msgspec
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from cachetools import cached, TTLCache
from pydantic import BaseModel
import uuid

from typing_extensions import TypedDict, Required, NotRequired
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
    TypeVar,
)
from .types import (
    Message,
    Tool,
    CompletionChunk,
    Completion,
    CompletionMessage,
    CompletionToolCall,
    CompletionFunction,
)
from .params import Params
from .utils import (
    ChatSpecError,
    logger,
    # methods
    is_completion,
    is_stream,
    dump_stream_to_message,
    normalize_messages,
    markdownify,
)


# ----------------------------------------------------------------------
# TypeVars
# ----------------------------------------------------------------------


_ContextValueT = TypeVar("_ContextValueT")
"""
TypeVar for values with a `State`'s context entries.
"""


# ----------------------------------------------------------------------
# Context Entry
# ----------------------------------------------------------------------


class _ContextEntry(TypedDict):
    """
    A dictionary representing a single entry within
    a `State`'s context entries.
    """

    value: _ContextValueT
    """
    The value of the context entry.
    """
    markdown: bool
    """
    If this context entry is marked as markdown, it will
    be rendered as markdown in prompts.
    """
    code_block: bool
    """
    If this context entry is marked as a code block, it will
    be rendered as a code block in prompts.
    """
    shared: bool
    """
    If this context entry is automatically shared with other
    state instances.
    """
    identifier: bool
    """
    If this context entry is marked as an identifier, it will
    be automatically added to the system prompt of a `State` 
    instance as one of the values in this format:
    
    ```python
    {
        "context" : {
            "name" : {
                "value" : "steve",
                "identifier" : True,
            },
            "role" : {
                "value" : "A genius expert.",
                "identifier" : True,
            },
         }
    }
    ```
    
    ```markdown
    # Context
    
    You are:
        - Name: steve
        - Role: A genius expert.
    ```
    """
    metadata: Dict[str, Any]
    """
    Optional metadata attached to the context entry.
    
    You can use this to store any information you want about the
    context entry.
    """


class _Context(TypedDict):
    """
    A dictionary representing a collection of context entries.
    """

    context: Dict[str, _ContextEntry]
    """
    A dictionary of context entries.
    """
    internal_prompt: Optional[str]
    """
    An optional internal prompt for the context entry, this is 
    used dynamically by a `State` to build its own system prompt.
    
    ```python
    {
        "context" : {
            "internal_prompt" : "You are {name}. A {role}.",
            "name" : {
                "value" : "steve",
                },
            "role" : {
                "value" : "genius expert.",
            },
        }
    }
    ```
    
    ```markdown
    # Context
    
    You are steve. A genius expert.
    ```
    """
    external_prompt: Optional[str]
    """
    An optional external prompt for the context entry, 
    this is what an outside state object would see, if this
    instance creates a message for that state object.
    
    ```python
    {
        "context" : {
            "external_prompt" : "You have recieved information from {name}, a {role}.",
            "name" : {
                "value" : "steve",
                },
            "role" : {
                "value" : "genius expert.",
            },
        }
    }
    ```
    
    ### This would be what another state object would see:
    
    ```markdown
    # Context
    
    You have recieved information from steve, a genius expert.
    ```
    """


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


@dataclass
class State:
    """
    A manager class for messages, tools and other contextual
    information used in chat completions in a conversational
    context.
    """

    name: str = "Assistant"
    """
    The name of the state. This is an important identifier,
    used directly in the `name` key of a `Message` object, and
    helps the model understand what is going on, when an 
    external state object sends it a message.
    """

    _messages: List[Message] = field(default_factory=list)
    """
    A list of messages to be used in the chat completion.
    """
    _system_prompt: Optional[str] = None
    """
    An optional system prompt for the chat completion.
    """
    _context: _Context = field(default_factory=dict)
    """
    A dictionary of context entries.
    """
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """
    A unique identifier for the state.
    """
    params: Params = field(
        default_factory=lambda: {
            "model": "gpt-4o-mini",
        }
    )
    """
    A dictionary of parameters for the chat completion.
    """
    tools: Optional[Dict[str, Callable]] = None
    """
    A dictionary of tools to be used in the chat completion.
    """
    completion_fn: Optional[Callable] = None
    """
    A function that can be called to generate a chat completion.
    The output types must match the OpenAI spec to be valid.
    """

    def __init__(
        self,
        name: str = "Assistant",
        messages: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        context: Optional[
            Union[Dict[str, Any], Dict[str, _ContextEntry]]
        ] = None,
        state_id: Optional[str] = None,
        params: Optional[Params] = None,
        tools: Optional[Dict[str, Callable]] = None,
        completion_fn: Optional[Callable] = None,
    ):
        self.name = name
        self._messages = normalize_messages(messages) if messages else []
        self._system_prompt = system_prompt
        if context:
            converted: Dict[str, _ContextEntry] = {}
            for key, val in context.items():
                if isinstance(val, dict) and "value" in val:
                    converted[key] = {
                        "value": val.get("value"),
                        "shared": bool(val.get("shared", False)),
                        "identifier": bool(val.get("identifier", False)),
                        "metadata": dict(val.get("metadata", {})),
                        "markdown": bool(val.get("markdown", False)),
                        "code_block": bool(val.get("code_block", False)),
                    }
                else:
                    converted[key] = {
                        "value": val,
                        "shared": False,
                        "identifier": False,
                        "metadata": {},
                        "markdown": False,
                        "code_block": False,
                    }
            self._context = {
                "context": converted,
                "internal_prompt": None,
                "external_prompt": None,
            }
        else:
            self._context = {
                "context": {},
                "internal_prompt": None,
                "external_prompt": None,
            }
        self.state_id = state_id if state_id else str(uuid.uuid4())
        self.params = (
            params if params is not None else {"model": "gpt-4o-mini"}
        )
        self.tools = tools or {}
        self.completion_fn = completion_fn

    @cached_property
    def system_prompt(self) -> Optional[str]:
        """
        Builds and returns the system prompt.

        If an internal_prompt is defined in the context, it is rendered using this state's
        shared context; otherwise, the _system_prompt attribute is used.
        """
        if self._context.get("internal_prompt"):
            return self.render_prompt(self._context["internal_prompt"])
        if self._system_prompt:
            return self.render_prompt(self._system_prompt)
        return None

    @property
    def messages(self) -> List[Message]:
        """
        Returns the conversation thread with the rendered system prompt (if any) prepended.
        The system message includes the state's "name" (from context if marked as an identifier)
        or the state_id.
        """
        if self.system_prompt:
            sys_msg: Message = {
                "role": "system",
                "content": self.system_prompt,
                "name": self._context["context"]
                .get("name", {})
                .get("value", self.state_id),
            }
            return [sys_msg] + self._messages
        return self._messages

    def add_message(
        self, message: Union[str, Message], role: Optional[str] = None
    ) -> None:
        """
        Adds a message to the conversation. If a string is provided, it is rendered using this state's
        shared context. If the message does not include a "name", the state's "name" (from context if marked
        as an identifier) or the state_id is attached.
        """
        if isinstance(message, str):
            rendered = self.render_prompt(message)
            msg: Message = {"role": role or "user", "content": rendered}
        else:
            msg = message
            if "content" in msg and isinstance(msg["content"], str):
                msg["content"] = self.render_prompt(msg["content"])
        if "name" not in msg:
            if "name" in self._context["context"] and self._context[
                "context"
            ]["name"].get("identifier", False):
                msg["name"] = self._context["context"]["name"]["value"]
            else:
                msg["name"] = self.state_id
        self._messages.append(msg)

    def clear_messages(self, keep_system: bool = True) -> None:
        """
        Clears the conversation history. If keep_system is False, also clears the system prompt.
        """
        self._messages = []
        if not keep_system:
            self._system_prompt = None

    def update_context(
        self,
        key: str,
        value: Any,
        shared: bool = True,
        identifier: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        markdown: bool = False,
        code_block: bool = False,
    ) -> None:
        """
        Updates (or adds) a context entry.

        Args:
          - key: The context key.
          - value: The value to set.
          - shared: If True, this entry is automatically injected into prompts.
          - identifier: If True, this entry is used as part of agent identification.
          - metadata: Optional extra metadata.
          - markdown: If True, the value is passed through markdownify.
          - code_block: If True and markdown is enabled, renders the value in a code block.
        """
        self._context["context"][key] = {
            "value": value,
            "shared": shared,
            "identifier": identifier,
            "metadata": metadata or {},
            "markdown": markdown,
            "code_block": code_block,
        }

    def remove_context(self, key: str) -> None:
        """
        Removes a context entry.
        """
        if key in self._context["context"]:
            del self._context["context"][key]

    def render_prompt(self, prompt: str, *extra: Any) -> str:
        """
        Renders a prompt string using this state's shared context.

        For each context entry marked as shared, if its "markdown" flag is True then its value is passed
        through markdownify (with code_block as specified). Extra context sources (State objects or dicts)
        can also be provided.
        """
        merged: Dict[str, Any] = {}
        for k, entry in self._context["context"].items():
            if entry.get("shared", False):
                val = entry["value"]
                if entry.get("markdown", False):
                    val = markdownify(
                        val, code_block=entry.get("code_block", False)
                    )
                merged[k] = val
        for src in extra:
            if isinstance(src, State):
                for k, entry in src._context["context"].items():
                    if entry.get("shared", False):
                        val = entry["value"]
                        if entry.get("markdown", False):
                            val = markdownify(
                                val,
                                code_block=entry.get("code_block", False),
                            )
                        merged[k] = val
            elif isinstance(src, dict):
                for k, v in src.items():
                    if isinstance(v, dict) and v.get("shared", False):
                        val = v["value"]
                        if v.get("markdown", False):
                            val = markdownify(
                                val, code_block=v.get("code_block", False)
                            )
                        merged[k] = val
        try:
            return prompt.format(**merged)
        except Exception as e:
            logger.warning(
                f"Error rendering prompt '{prompt}' with context {merged}: {e}"
            )
            return prompt

    @contextmanager
    def temporary_thread(
        self, inherit: bool = True
    ) -> Generator["State", None, None]:
        """
        Creates a temporary conversation thread (scratchpad) that does not affect the main conversation.
        """
        original = deepcopy(self._messages)
        temp = State(
            name=self.name,
            messages=deepcopy(self._messages) if inherit else [],
            system_prompt=self._system_prompt,
            context=deepcopy(self._context["context"]),
            state_id=self.state_id,
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            completion_fn=self.completion_fn,
        )
        temp._context["internal_prompt"] = self._context.get(
            "internal_prompt"
        )
        temp._context["external_prompt"] = self._context.get(
            "external_prompt"
        )
        try:
            yield temp
        finally:
            self._messages = original

    def merge_from(
        self,
        other: "State",
        selector: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        """
        Merges messages from another State into this one.
        An optional selector can filter which messages to merge.
        """
        msgs = other._messages
        if selector:
            msgs = [m for m in msgs if selector(m)]
        self._messages.extend(deepcopy(msgs))

    def subset(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> "State":
        """
        Returns a new State containing a subset of the conversation messages.
        """
        return State(
            name=self.name,
            messages=deepcopy(self._messages[start:end]),
            system_prompt=self._system_prompt,
            context=deepcopy(self._context["context"]),
            state_id=self.state_id,
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            completion_fn=self.completion_fn,
        )

    def last_n(self, n: int) -> "State":
        """
        Returns a new State with the last n messages.
        """
        return self.subset(-n if n else None)

    def filter(self, predicate: Callable[[Message], bool]) -> "State":
        """
        Returns a new State containing only messages that satisfy the predicate.
        """
        filtered = [m for m in self._messages if predicate(m)]
        return State(
            name=self.name,
            messages=filtered,
            system_prompt=self._system_prompt,
            context=self._context["context"],
            state_id=self.state_id,
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            completion_fn=self.completion_fn,
        )

    def register_tool(self, name: str, tool: Callable) -> None:
        """
        Registers a tool (callable) under the given name.
        """
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Retrieves a registered tool by name.
        """
        return self.tools.get(name)

    def execute_tool_calls(self, completion: Union[Dict[str, Any], CompletionMessage]) -> None:
        """
        Executes any tool calls contained in a completion object.
        Uses msgspec for JSON decoding and ensures proper typing with CompletionToolCall.
        """
        # Extract tool_calls, handling both dict and CompletionMessage cases
        tool_calls = []
        if isinstance(completion, dict):
            tool_calls = completion.get("tool_calls", [])
        else:
            tool_calls = completion.tool_calls or []

        for call in tool_calls:
            # Convert dict to CompletionToolCall if needed
            tool_call = (
                call if isinstance(call, CompletionToolCall)
                else CompletionToolCall(
                    id=call.get("id", str(uuid.uuid4())),
                    type="function",
                    function=CompletionFunction(
                        name=call.get("function", {}).get("name", ""),
                        arguments=call.get("function", {}).get("arguments", "{}")
                    )
                )
            )
            
            if tool_call.function.name in self.tools:
                try:
                    args = msgspec.json.decode(tool_call.function.arguments)
                    output = self.tools[tool_call.function.name](**args)
                    self.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(output),
                        }
                    )
                except Exception as e:
                    self.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": "error",
                            "content": f"Error executing tool '{tool_call.function.name}': {e}",
                        }
                    )

    def process_response(self, response: Union[Dict[str, Any], CompletionMessage, Message]) -> None:
        """
        Processes an agent response by executing any contained tool calls.
        Now handles both raw dictionaries and proper CompletionMessage objects.
        """
        if isinstance(response, (dict, CompletionMessage)):
            self.execute_tool_calls(response)

    def prompt(self, user_input: str) -> Union[Message, CompletionMessage]:
        """
        Renders the user input (injecting this state's shared context), adds it as a message,
        and passes the full conversation to the completion function.
        
        Returns:
            Either a Message or CompletionMessage object, depending on the completion function's output.
        """
        rendered = self.render_prompt(user_input)
        self.add_message({"role": "user", "content": rendered})
        
        if not self.completion_fn:
            raise ChatSpecError("No completion function defined.")
            
        response = self.completion_fn(messages=self.messages, **self.params)
        
        # Handle different response types
        if is_completion(response):
            # Extract the CompletionMessage from the Completion
            if isinstance(response, Completion) and response.choices:
                message = response.choices[0].message
                self.add_message(message)
                self.process_response(message)
                return message
                
        elif is_stream(response):
            # Convert stream to a proper Message
            message = dump_stream_to_message(response)
            if isinstance(message, (dict, CompletionMessage)):
                self.add_message(message)
                self.process_response(message)
                return message
                
        # Fallback for other response types
        if isinstance(response, (dict, CompletionMessage)):
            self.add_message(response)
            self.process_response(response)
            return response
            
        raise ChatSpecError(f"Unexpected response type: {type(response)}")

    def send_message_to(
        self, target: "State", message: Union[str, Message, CompletionMessage]
    ) -> None:
        """
        Sends a message to another State.
        Now handles CompletionMessage objects as well.
        """
        if isinstance(message, str):
            rendered = self.render_prompt(message)
            msg: Message = {
                "role": "user",
                "content": rendered,
                "name": self._context["context"]
                .get("name", {})
                .get("value", self.state_id),
            }
        elif isinstance(message, CompletionMessage):
            # Convert CompletionMessage to Message
            msg = {
                "role": message.role,
                "content": self.render_prompt(message.content) if isinstance(message.content, str) else message.content,
                "name": message.name or self._context["context"].get("name", {}).get("value", self.state_id),
            }
        else:
            msg = message
            if "content" in msg and isinstance(msg["content"], str):
                msg["content"] = self.render_prompt(msg["content"])
            if "name" not in msg:
                msg["name"] = (
                    self._context["context"]
                    .get("name", {})
                    .get("value", self.state_id)
                )
        target.receive_message_from(self, msg)

    def receive_message_from(
        self, sender: "State", message: Union[str, Message, CompletionMessage]
    ) -> None:
        """
        Receives a message from another State.
        Now handles CompletionMessage objects properly.
        """
        sender_name = (
            sender._context["context"]
            .get("name", {})
            .get("value", sender.state_id or "unknown")
        )
        
        if isinstance(message, str):
            message = {
                "role": "user",
                "content": message,
                "name": sender_name,
            }
        elif isinstance(message, CompletionMessage):
            # Convert CompletionMessage to Message
            message = {
                "role": message.role,
                "content": sender.render_prompt(message.content) if isinstance(message.content, str) else message.content,
                "name": message.name or sender_name,
            }
        else:
            if "content" in message and isinstance(message["content"], str):
                message["content"] = sender.render_prompt(message["content"])
            if "name" not in message:
                message["name"] = sender_name
                
        self.add_message(message)
