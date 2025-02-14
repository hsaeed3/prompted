"""
💭 chatspec.state

Contains a 'State' class used as a message manager & tool registry.
"""
# you can use this to make agents!!!
# wowwww ✨✨✨✨

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Union, Generator, Callable

from .types import Message
from .utils import (
    normalize_messages,
    normalize_system_prompt,
    has_system_prompt,
    was_tool_called,
    run_tool,
    create_tool_message,
    passthrough,
)

__all__ = [
    "State",
]


# ------------------------------------------------------------------------------
# State Class
# ------------------------------------------------------------------------------


@dataclass
class State:
    """
    A manager class for maintaining and augmenting a list of messages.
    Extended to hold agent parameters, register and run tools, and
    support agent prompting.
    """

    _messages: List[Message] = field(default_factory=list)
    _system_prompt: Optional[str] = None
    context: List[Dict[str, Any]] = field(default_factory=list)

    # Agent parameters
    model: Optional[str] = "gpt-4"
    api_key: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None

    # Tools registry: maps tool names to callables
    tools: Dict[str, Callable] = field(default_factory=dict)

    # Optional agent function to generate responses (e.g., an API call)
    agent_func: Optional[Callable] = None

    @cached_property
    def system_prompt(self) -> Optional[str]:
        """
        Combines the stored system prompt with any context.
        """
        if self._system_prompt is not None:
            if not self.context:
                return self._system_prompt
            context_str = "\n\n## Context\n" + "\n".join(
                f"- {k}: {v}" for item in self.context 
                for k, v in item.items()
            )
            return f"{self._system_prompt}{context_str}"
        return None

    def __post_init__(self):
        """
        Normalizes messages upon initialization and extracts the system prompt.
        """
        if self._messages:
            self._messages = normalize_messages(self._messages)
            if has_system_prompt(self._messages):
                self._messages = normalize_system_prompt(self._messages)
                self._system_prompt = self._messages[0].get("content")
                # Remove the system message from internal messages
                self._messages = self._messages[1:]

    @property
    def messages(self) -> List[Message]:
        """
        Returns the full message thread, including a dynamic system prompt.
        """
        if self._system_prompt is not None:
            system_msg = {"role": "system", "content": self.system_prompt}
            return [system_msg] + self._messages
        return self._messages

    def add_message(self, message: Union[str, Message]) -> None:
        """
        Adds a single message to the thread.
        """
        if isinstance(message, str):
            message = {"role": "user", "content": message}
        normalized = normalize_messages([message])
        self._messages.extend(normalized)

    def add_messages(self, messages: List[Message]) -> None:
        """
        Adds multiple messages to the thread.
        """
        normalized = normalize_messages(messages)
        self._messages.extend(normalized)

    def clear_messages(self, keep_system: bool = True) -> None:
        """
        Clears all messages. Optionally keeps the system prompt.
        """
        self._messages = []  # Always clear the conversation
        if not keep_system:
            self._system_prompt = None
            if "system_prompt" in self.__dict__:
                del self.__dict__["system_prompt"]

    @contextmanager
    def temporary_thread(
        self, inherit: bool = True
    ) -> Generator["State", None, None]:
        """
        Creates a temporary message thread (scratchpad) that can be used without
        affecting the main thread.
        """
        # Store original messages
        original_messages = deepcopy(self._messages)
        
        temp_state = State(
            _messages=deepcopy(self._messages) if inherit else [],
            _system_prompt=deepcopy(self._system_prompt),
            context=deepcopy(self.context),
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
        )
        try:
            yield temp_state
        finally:
            # Restore original messages
            self._messages = original_messages

    def merge_from(
        self,
        other: "State",
        selector: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        """
        Merges messages from another State instance. An optional selector function
        can filter which messages are merged.
        """
        messages_to_merge = other._messages
        if selector:
            messages_to_merge = [
                msg for msg in messages_to_merge if selector(msg)
            ]
        self._messages.extend(deepcopy(messages_to_merge))

    def subset(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> "State":
        """
        Returns a new State with a subset of the current messages.
        """
        messages = self._messages[start:end]
        return State(
            _messages=deepcopy(messages),
            _system_prompt=self._system_prompt,
            context=deepcopy(self.context),
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
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
        return State(
            _messages=[m for m in self._messages if predicate(m)],
            _system_prompt=self._system_prompt,
            context=deepcopy(self.context),
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
        )

    def add_context(self, context: Dict[str, Any]) -> None:
        """
        Adds a context object and clears the cached system prompt.
        """
        self.context.append(context)
        if "system_prompt" in self.__dict__:
            del self.__dict__["system_prompt"]

    def remove_context(
        self, predicate: Callable[[Dict[str, Any]], bool]
    ) -> None:
        """
        Removes context objects that match the predicate.
        """
        self.context = [c for c in self.context if not predicate(c)]
        if "system_prompt" in self.__dict__:
            del self.__dict__["system_prompt"]

    @property
    def last_message(self) -> Optional[Message]:
        """Returns the last message in the thread."""
        return self._messages[-1] if self._messages else None

    @property
    def last_user_message(self) -> Optional[Message]:
        """Returns the last user message in the thread."""
        for msg in reversed(self._messages):
            if msg.get("role") == "user":
                return msg
        return None

    @property
    def last_assistant_message(self) -> Optional[Message]:
        """Returns the last assistant message in the thread."""
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant":
                return msg
        return None

    # --- Extended Functionality for Agents and Tools ---

    def register_tool(self, name: str, tool: Callable) -> None:
        """
        Registers a tool under a given name.

        Args:
            name: The unique name for the tool.
            tool: The callable that implements the tool.
        """
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Retrieves a registered tool by name.

        Args:
            name: The name of the tool.

        Returns:
            The tool callable if found, otherwise None.
        """
        return self.tools.get(name)

    def execute_tool_calls(self, completion: Any) -> None:
        """
        Inspects the given completion for any tool calls and executes the
        registered tools accordingly. The output of each tool is added to the
        conversation as a tool message.

        Args:
            completion: A chat completion object (streaming or non-streaming).
        """
        for tool_name, tool_func in self.tools.items():
            tool_calls = completion.get("tool_calls", [])
            for call in tool_calls:
                if (call.get("type") == "function" and 
                    call.get("function", {}).get("name") == tool_name):
                    try:
                        import json
                        args = json.loads(call["function"]["arguments"])
                        output = tool_func(**args)
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call.get("id", "unknown"),
                            "content": str(output)
                        }
                        self.add_message(tool_msg)
                    except Exception as e:
                        error_msg = {
                            "role": "tool",
                            "tool_call_id": "error",
                            "content": f"Error executing tool '{tool_name}': {e}",
                        }
                        self.add_message(error_msg)

    def process_response(self, response: Any) -> None:
        """
        Processes a response from the agent. This method checks for any tool
        calls within the response and executes them.

        Args:
            response: The response object from the agent.
        """
        self.execute_tool_calls(response)

    def prompt(self, user_input: str) -> Any:
        """
        Adds a user message, invokes the agent function to generate a response,
        and processes any tool calls present in that response.
        
        This version has been upgraded to detect streaming responses. If the agent
        function returns a stream, we automatically use the passthrough wrapper to
        cache incoming message chunks and then combine them into a full message.
        
        Args:
            user_input: The text input from the user.
            
        Returns:
            The processed response message.
            
        Raises:
            ValueError: If no agent function is defined.
        """
        self.add_message({"role": "user", "content": user_input})
        if not self.agent_func:
            raise ValueError("Agent function is not defined for generating responses.")

        # Generate the response using the agent function with current parameters
        response = self.agent_func(
            messages=self.messages,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # Handle streaming responses by combining chunks
        if isinstance(response, (list, tuple)) and response and isinstance(response[0], dict):
            try:
                stream_content = "".join(
                    chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    for chunk in response
                )
                response_final = {"role": "assistant", "content": stream_content}
            except Exception as e:
                # Fallback: if stream processing fails, use the raw response
                response_final = response
        else:
            response_final = response

        self.add_message(response_final)
        self.process_response(response_final)
        return response_final

    def update_params(self, **kwargs) -> None:
        """
        Updates agent parameters such as model, api_key, temperature, top_p, and max_tokens.

        Args:
            **kwargs: Parameter names and their new values.

        Raises:
            ValueError: If an unsupported parameter is provided.
        """
        allowed_params = {
            "model",
            "api_key",
            "temperature",
            "top_p",
            "max_tokens",
        }
        for key, value in kwargs.items():
            if key in allowed_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unsupported parameter: {key}")

    def test_temporary_thread():
        """Test temporary thread creation and isolation."""
        initial_messages = [
            {"role": "user", "content": "Original message"}
        ]
        state = State(_messages=initial_messages.copy())
        
        with state.temporary_thread(inherit=False) as temp:  # Set inherit=False
            temp.add_message("This is temporary")
            # Verify temp thread has the temporary message
            assert "This is temporary" in temp._messages[-1]["content"]
            # Verify original state doesn't have the temporary message
            assert "This is temporary" not in str(state._messages)
        
        # Verify the temporary message doesn't persist after the context
        assert len(state._messages) == 1
        assert "This is temporary" not in str(state._messages)
