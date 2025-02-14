"""
💭 chatspec.state

Contains a 'State' class used as a message manager & tool registry.
Now extended with identification_context logic for inter-agent communications.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Union, Generator, Callable
from typing_extensions import TypedDict

from .types import Message, Params
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


class IdentificationContext(TypedDict):
    """
    A dictionary representing the identification context of an agent.
    """
    name: Optional[str]
    role: Optional[str]
    description: Optional[str]
    capabilities: Optional[str]
    

# ------------------------------------------------------------------------------
# State Class
# ------------------------------------------------------------------------------

@dataclass
class State:
    """
    A manager class for maintaining and augmenting a list of messages.
    
    Can be used to manage a chat history, tools, parameters, context, and now
    identification context for inter-agent communication.
    
    Args:
        messages: A list of messages to initialize the state with.
        system_prompt: A system prompt to initialize the state with.
        context: A list of context objects to initialize the state with.
        params: A dictionary of parameters to initialize the state with.
        tools: A dictionary of tools to initialize the state with.
        agent_func: A function to initialize the state with.
        identification_context: A short string representing the sender's identity.
            This is used when sending messages between states to scope the context.
    """

    _messages: List[Message] = field(default_factory=list)
    _system_prompt: Optional[str] = None
    context: List[Dict[str, Any]] = field(default_factory=list)

    # Parameters for LLM
    params: Params = field(default_factory=lambda: {
        "model": "gpt-4",
        "api_key": None,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_completion_tokens": None,
        "stop": None,
        "stream": None,
        "base_url": None,
        "organization": None,
    })

    # Tools registry: maps tool names to callables
    tools: Dict[str, Callable] = field(default_factory=dict)

    # Optional agent function to generate responses (e.g., an API call)
    agent_func: Optional[Callable] = None

    # New: identification_context used for inter-agent context scoping
    identification_context: IdentificationContext = field(default_factory=lambda: {
        "name": None,
        "role": None,
        "description": None,
        "capabilities": None,
    })

    # Add new field for task completion
    _task_mode: bool = False
    _completion_buffer: List[str] = field(default_factory=list)

    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        _messages: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None,
        params: Optional[Params] = None,
        tools: Optional[Dict[str, Callable]] = None,
        agent_func: Optional[Callable] = None,
        identification_context: Optional[IdentificationContext] = None,
    ):
        # Allow for _messages alias (used internally in subset/temporary thread)
        if _messages is not None:
            self._messages = _messages
        else:
            self._messages = messages or []
        self._system_prompt = system_prompt
        self.context = context or []
        self.params = params or {}
        self.tools = tools or {}
        self.agent_func = agent_func
        self.identification_context = identification_context or {}
        self._task_mode = False
        self._completion_buffer = []
        self.__post_init__()

    def __post_init__(self):
        """
        Normalizes messages upon initialization and extracts the system prompt.
        Also adds identity to system prompt if available.
        """
        if self._messages:
            self._messages = normalize_messages(self._messages)
            if has_system_prompt(self._messages):
                self._messages = normalize_system_prompt(self._messages)
                self._system_prompt = self._messages[0].get("content")
                # Remove the system message from internal messages
                self._messages = self._messages[1:]
        
        # Add identity to system prompt if available
        self._update_system_prompt_with_identity()

    @cached_property
    def system_prompt(self) -> Optional[str]:
        """
        Combines the stored system prompt with any context.
        """
        if self._system_prompt is not None:
            if not self.context:
                return self._system_prompt
            context_str = "\n\n## Context\n" + "\n".join(
                f"- {k}: {v}" for item in self.context for k, v in item.items()
            )
            return f"{self._system_prompt}{context_str}"
        return None

    def _update_system_prompt_with_identity(self) -> None:
        """
        Updates system prompt with identity information if available.
        Called both during initialization and when identification_context changes.
        """
        identity_str = self._format_system_identity()
        if identity_str:
            if self._system_prompt:
                self._system_prompt = f"{identity_str}\n\n{self._system_prompt}"
            else:
                self._system_prompt = identity_str
            # Clear cached system prompt
            if "system_prompt" in self.__dict__:
                del self.__dict__["system_prompt"]

    @property
    def messages(self) -> List[Message]:
        """
        Returns the full message thread, including a dynamic system prompt.
        """
        if self.system_prompt is not None:
            system_msg = {"role": "system", "content": self.system_prompt}
            return [system_msg] + self._messages
        return self._messages

    def add_message(
        self,
        message: Union[str, Message],
        role: Optional[str] = None,
    ) -> None:
        """
        Adds a single message to the thread.
        """
        if isinstance(message, str):
            message = {"role": role or "user", "content": message}
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
            system_prompt=deepcopy(self._system_prompt),
            context=deepcopy(self.context),
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
            identification_context=deepcopy(self.identification_context),
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
            system_prompt=self._system_prompt,
            context=deepcopy(self.context),
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
            identification_context=deepcopy(self.identification_context),
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
            system_prompt=self._system_prompt,
            context=deepcopy(self.context),
            params=deepcopy(self.params),
            tools=deepcopy(self.tools),
            agent_func=self.agent_func,
            identification_context=deepcopy(self.identification_context),
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

    # --- Extended Functionality for Agents, Tools, and Inter-State Communication ---

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
        Enhanced prompt method that enforces task completion in task mode.
        """
        response = super().prompt(user_input)
        
        # In task mode, verify that complete() was called
        if self._task_mode and not self._completion_buffer:
            # If no completion was provided, prompt again for explicit completion
            completion_prompt = {
                "role": "user",
                "content": "Please provide your final answer using the complete() tool."
            }
            self.add_message(completion_prompt)
            response = self.agent_func(
                messages=self.messages,
                **self.params
            )
            self.add_message(response)
            self.process_response(response)
            
        return response

    def update_params(self, **kwargs) -> None:
        """
        Updates agent parameters stored in the params dictionary.

        Args:
            **kwargs: Parameter names and their new values.

        Raises:
            ValueError: If an unsupported parameter is provided.
        """
        allowed_params = set(self.params.keys())
        for key, value in kwargs.items():
            if key in allowed_params:
                self.params[key] = value
            else:
                raise ValueError(f"Unsupported parameter: {key}")

    # --- New Inter-State Communication Methods ---

    def send_message_to(
        self, target_state: "State", message: Union[str, Message]
    ) -> None:
        """
        Sends a message from this state to another state (agent), automatically managing
        context and communication protocols between agents.

        The method:
        1. Adds context about the sender's identity and role
        2. Includes any relevant conversation history
        3. Formats the message appropriately for inter-agent communication

        Args:
            target_state: The receiving State instance.
            message: The message content (either as a string or a Message dict).
        """
        if isinstance(message, str):
            message = {"role": "user", "content": message}

        # Add sender context
        context_prompt = []
        if self.identification_context.get("name"):
            context_prompt.append(
                f"You are communicating as {self.identification_context['name']}, a {self.identification_context['role']}. {self.identification_context['description']}. Your capabilities include: {self.identification_context['capabilities']}"
            )
        if hasattr(self, "capabilities"):
            context_prompt.append(
                f"Your capabilities include: {', '.join(self.capabilities)}"
            )
        if context_prompt:
            self.add_message({
                "role": "system",
                "content": " ".join(context_prompt)
            })

        # Add communication protocol prompt
        protocol_prompt = {
            "role": "system",
            "content": (
                "This is an inter-agent communication. Please:\n"
                "1. Maintain context awareness of your role and the conversation\n"
                "2. Be explicit about any assumptions or requirements\n"
                "3. Format responses in a way that's clear for other agents to process"
            )
        }
        self.add_message(protocol_prompt)
        
        target_state.receive_message_from(self, message)

    def receive_message_from(
        self, sender: "State", message: Union[str, Message]
    ) -> None:
        """
        Receives a message from another state (agent), handling context management
        and communication protocols.

        The method:
        1. Establishes sender context and identity
        2. Sets up appropriate response protocols
        3. Maintains conversation coherence between agents

        Args:
            sender: The sending State instance.
            message: The message content (either as a string or a Message dict).
        """
        # Add the external message note to system prompt if not already present
        external_note = "NOTE: any blocks wrapped in [EXTERNAL] were sent by a source other than the user"
        if self._system_prompt:
            if external_note not in self._system_prompt:
                self._system_prompt = f"{self._system_prompt}\n\n{external_note}"
        else:
            self._system_prompt = external_note
        
        # Clear cached system prompt to ensure it's regenerated
        if "system_prompt" in self.__dict__:
            del self.__dict__["system_prompt"]

        # Add sender context only if identification information exists
        context_prompts = []
        if any(sender.identification_context.values()):
            # Build context string only with available information
            sender_info = []
            if sender.identification_context.get("name"):
                sender_info.append(f"{sender.identification_context['name']}")
            if sender.identification_context.get("role"):
                sender_info.append(f"a {sender.identification_context['role']}")
            if sender.identification_context.get("description"):
                sender_info.append(f"{sender.identification_context['description']}")
            if sender.identification_context.get("capabilities"):
                sender_info.append(f"with capabilities: {sender.identification_context['capabilities']}")
            
            if sender_info:
                context_prompts.append(f"You are receiving a message from: {', '.join(sender_info)}")
        
        # Add communication context if we have any context prompts
        if context_prompts:
            self.add_message({
                "role": "system",
                "content": " ".join(context_prompts)
            })

        # Format the message to include sender's identification context
        if isinstance(message, str):
            message = {"role": "user", "content": message}
        
        # Use the formatted identification for the sender
        sender_id = sender._format_identification()
        original_content = message["content"]
        message["content"] = f"[EXTERNAL from {sender_id}]: {original_content}"

        self.add_message(message)

    @staticmethod
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

    @contextmanager
    def task(self, task_prompt: str) -> Generator["State", None, None]:
        """
        Creates a task-oriented context where the LLM uses chain-of-thought reasoning
        to solve a problem and must provide a final answer using the complete() tool.

        Args:
            task_prompt: The task description or question to be solved
        """
        # Store original state
        original_tools = deepcopy(self.tools)
        original_system = deepcopy(self._system_prompt)
        original_task_mode = self._task_mode
        
        try:
            # Enable task mode
            self._task_mode = True
            self._completion_buffer = []
            
            # Add the complete tool
            def complete(value: Any) -> str:
                """Tool to submit the final answer for the task."""
                self._completion_buffer.append(str(value))
                return "Task completed successfully."
            
            self.register_tool("complete", complete)
            
            # Enhance system prompt with task-specific instructions
            task_system_prompt = """You are a problem-solving assistant that uses chain-of-thought reasoning.
            
1. Break down the problem into steps
2. Think through each step carefully
3. Show your work and reasoning
4. When you reach the final answer, use the complete() tool to submit it
5. You MUST use the complete() tool to provide your final answer

Example thought process:
1. First, I'll...
2. Then, I'll...
3. Finally, I can conclude...
complete(final_answer)"""

            if self._system_prompt:
                self._system_prompt = f"{self._system_prompt}\n\n{task_system_prompt}"
            else:
                self._system_prompt = task_system_prompt

            # Add the task prompt
            self.add_message({
                "role": "user",
                "content": f"Task: {task_prompt}\n\nPlease solve this step by step and provide your final answer using the complete() tool."
            })
            
            yield self

        finally:
            # Restore original state
            self.tools = original_tools
            self._system_prompt = original_system
            self._task_mode = original_task_mode

    def get_task_result(self) -> Optional[str]:
        """
        Returns the final result submitted through the complete() tool.
        """
        return self._completion_buffer[-1] if self._completion_buffer else None

    def _format_identification(self) -> str:
        """
        Formats the identification context into a human-readable string.
        Only includes fields that are set (not None).
        """
        parts = []
        if self.identification_context.get("name"):
            parts.append(f"{self.identification_context['name']}")
        if self.identification_context.get("role"):
            parts.append(f"a {self.identification_context['role']}")
        if self.identification_context.get("description"):
            parts.append(f"({self.identification_context['description']})")
        
        return " ".join(parts) if parts else "unknown agent"

    def _format_system_identity(self) -> str:
        """
        Formats the identification context for system prompts.
        """
        parts = []
        if self.identification_context.get("name"):
            parts.append(f"You are {self.identification_context['name']}")
        if self.identification_context.get("role"):
            if not parts:
                parts.append(f"You are a {self.identification_context['role']}")
            else:
                parts[-1] += f", a {self.identification_context['role']}"
        if self.identification_context.get("description"):
            parts.append(self.identification_context["description"])
        if self.identification_context.get("capabilities"):
            parts.append(f"Your capabilities include: {self.identification_context['capabilities']}")
        
        return ". ".join(parts) if parts else ""

    @property
    def identification_context(self) -> Dict[str, str]:
        return self._identification_context

    @identification_context.setter
    def identification_context(self, value: Dict[str, str]) -> None:
        self._identification_context = value
        self._update_system_prompt_with_identity()
