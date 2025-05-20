"""
💭 prompted.agent

cool fun super duper agent
"""

import asyncio
import json
import logging
import uuid
import re
from dataclasses import (
    dataclass,
    field,
    is_dataclass,
    asdict as dataclass_asdict,
)
from typing import (
    Any,
    Dict,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    TypeVar,
    Generic,
    AsyncIterable,
    cast,
    Literal as TypingLiteral,  # Renamed to avoid conflict
)
from copy import deepcopy

from pydantic import (
    BaseModel,
    Field as PydanticField,
    create_model,
    ValidationError,
    ConfigDict,
)

from ..create import Create, PromptType, SchemaType
from ..create import (
    _format_compiled_messages,
    _prepare_llm_call_params,
)  # Assuming internal framework access
from ..utils.formatting import (
    format_to_markdown,
    format_messages,
    format_system_prompt,
)
from ..utils.identification import is_message
from ..utils.converters import convert_to_pydantic_model
from ..types.chat_completions import (
    Completion,
    Message,
    MessageRole,
    Tool,
    ToolCall,
    CompletionChunk,
)
from ..types.chat_completions_params import ModelParam, Params
from .agent_types import (
    AgentEndStrategy,
    AgentContextStrategy,
    AgentOutputType,
    AgentContextType,
    AgentPromptTemplate,
    AgentPromptTemplatePart,
    AgentMessageSchema,
)
from .agent_tool import (
    AgentTool,
    agent_tool as create_agent_tool_decorator,
    default_error_handler,
)

logger = logging.getLogger(__name__)

# Generic type variables for Agent context and output
CtxType = TypeVar("CtxType", bound=Union[BaseModel, Dict[str, Any], None])
OutType = TypeVar("OutType", bound=Union[BaseModel, Type, AgentMessageSchema, str])


@dataclass
class AgentResponse(Generic[OutType]):
    """
    Represents the complete response from an agent's execution.

    Attributes:
        parts (list[Any]): A list of intermediate parts generated during the agent's run.
                           This can include planning thoughts, reflection steps, tool call details,
                           and other operational messages.
        output (Optional[OutType]): The final, primary output of the agent, structured
                                    according to the agent's `output_type` setting.
        history (List[Message]): The complete conversation history of the agent's run,
                                 including system prompts, user inputs, assistant responses,
                                 and tool interactions.
    """

    parts: list[Any] = field(default_factory=list)
    output: Optional[OutType] = None
    history: List[Message] = field(default_factory=list)


@dataclass
class AgentSettings:
    """
    Configuration settings that dictate the runtime behavior, strategies,
    and operational parameters of an Agent.
    """

    output_type: Type[OutType] = field(
        default_factory=lambda: cast(Type[OutType], AgentMessageSchema)
    )
    """
    The expected Pydantic model, basic Python type, or `AgentMessageSchema` for the agent's final output.
    Defaults to `AgentMessageSchema`, producing a message-like dictionary.
    """
    planning: bool = False
    """
    If `True`, the agent will perform a planning step before each main generation,
    outlining its intended actions. This can improve coherence for complex tasks.
    """
    reflection: bool = False
    """
    If `True`, the agent will reflect on its generated content after each main step,
    allowing for self-correction or confirmation before proceeding.
    """
    model: ModelParam = "openai/gpt-4o-mini"
    """The primary language model used by the agent for its internal reasoning and generation steps."""
    model_params: Params = field(default_factory=dict)
    """Additional parameters (e.g., temperature, top_p) to pass to the language model during API calls."""
    max_steps: int = 10
    """
    The maximum number of main generation steps (excluding planning, reflection, or initial context updates)
    the agent will take before stopping. Prevents runaway execution.
    """
    iterative_output: bool = False
    """
    If `True` and the `output_type` is a Pydantic model, the agent will attempt to generate
    each field of the output model sequentially. Useful for complex, multi-field outputs.
    (Note: Current implementation primarily uses this for the final output generation stage).
    """
    iterative_context: bool = False
    """
    If `True` and `context_strategy` is "selective", the agent will generate updates
    for each selected context field iteratively, one by one.
    """
    force_tools: bool = False
    """
    If `True`, the agent will attempt to generate arguments for and execute all its available tools
    at the beginning of its run, based on the initial user query.
    """
    end_strategy: AgentEndStrategy = "full"
    """
    Determines how the agent decides to conclude its operation:
    - `"full"`: The agent is provided with a special "finalize_response" tool. It must explicitly
                call this tool to end its response.
    - `"selective"`: After each step, the agent uses an LLM call to decide whether to continue or end.
    """
    end_instructions: str = "When you are certain the request is fully addressed, use the 'finalize_response' tool. If unsure, continue."
    """
    Additional instructions appended to the agent's internal prompt when it's deciding to end
    (for "selective" strategy) or when describing the "finalize_response" tool (for "full" strategy).
    This allows customization of the ending behavior.
    """
    context_strategy: AgentContextStrategy = "selective"
    """
    Defines how the agent updates its internal context (if any):
    - `"selective"`: The agent first decides which context fields to update, then generates values
                     only for those fields.
    - `"full"`: The agent generates a complete, new version of the entire context object.
    """
    context_instructions: str = (
        "Consider if context updates are needed for clarity or tracking information."
    )
    """
    Additional instructions appended to the agent's internal prompt when performing context updates.
    This allows users to guide how the agent thinks about modifying its context.
    """
    update_context_before_response: bool = True
    """
    If `True` and a context object is present, the agent will automatically attempt to update
    its context based on the user's query *before* the main response generation loop.
    """
    update_context_after_response: bool = False
    """
    If `True` and a context object is present, the agent will automatically attempt to update
    its context based on its own generated response *after* the main response generation loop.
    """
    add_context_to_prompt: bool = True
    """If `True`, the agent's current context (if any) is formatted and included in its system prompt."""
    add_tools_to_prompt: bool = True
    """If `True`, descriptions of available tools are included in the agent's system prompt."""
    keep_intermediate_steps: bool = True
    """
    If `True`, all intermediate messages (planning, reflection, internal tool calls for context/ending)
    are kept in the message history for subsequent LLM calls and included in `AgentResponse.parts`.
    If `False`, these are hidden from the LLM in later steps and omitted from final `parts`.
    """
    persona_override_intermediate_steps: bool = True
    """
    If `True`, instructs the LLM to adopt the agent's primary persona/style (derived from main
    instructions and prompt parts) when generating content for planning, reflection, and context
    update descriptions. If `False`, these steps use more neutral, functional language.
    """
    save_history: bool = True
    """
    If `True`, the complete message history of the agent's run will be included in the `AgentResponse.history`.
    If `False`, `AgentResponse.history` will be an empty list. This can be useful for reducing memory footprint
    or simplifying the response object when the full history is not needed by the caller.
    """
    show_context: bool = True
    """
    If `True` and `keep_intermediate_steps` is also `True`, messages detailing context updates 
    (e.g., '[Context Update Thought]: ...') will be included in the history passed to the LLM for 
    main generation steps. If `False`, they are excluded from the LLM's view for generation, 
    even if `keep_intermediate_steps` is `True` (they would still be in `AgentResponse.parts` 
    and full history).
    """


# --- Internal Pydantic Models for Agent Logic ---
class SelectContextFields(BaseModel):
    """Schema for LLM to select context fields for updating."""

    fields_to_update: List[str] = PydanticField(
        default_factory=list,
        description="A list of field names from the agent's context that need to be updated based on the recent interaction and overall goal.",
    )


class DecideContinuation(BaseModel):
    """Schema for LLM to decide if the agent should end its operation (selective strategy)."""

    should_end: bool = PydanticField(
        ...,
        description="Set to true if the agent has fully addressed the user's request and no further actions are needed. Set to false if more steps, tool uses, or clarifications are required.",
    )
    reason: Optional[str] = PydanticField(
        None,
        description="A brief justification for the decision to end or continue.",
    )


def _map_json_schema_type_to_python_type(
    schema_prop: Dict[str, Any],
) -> Type:
    """Helper to map JSON schema property types to Python type hints for dynamic Pydantic model creation."""
    schema_type = schema_prop.get("type")
    if isinstance(schema_type, str):
        type_str = schema_type.lower()
        if type_str == "string":
            if (
                "enum" in schema_prop
                and isinstance(schema_prop["enum"], list)
                and schema_prop["enum"]
            ):
                return TypingLiteral[tuple(schema_prop["enum"])]  # type: ignore
            return str
        if type_str == "integer":
            return int
        if type_str == "number":
            return float
        if type_str == "boolean":
            return bool
        if type_str == "array":
            items_schema = schema_prop.get("items")
            if items_schema and isinstance(items_schema, dict):
                item_type = _map_json_schema_type_to_python_type(items_schema)
                return List[item_type]  # type: ignore
            return List[Any]
        if type_str == "object":
            # For 'object' type, ideally we'd recursively create a Pydantic model if 'properties' is defined.
            # For simplicity here, defaulting to Dict. A more advanced version could handle nesting.
            return Dict[str, Any]
    elif isinstance(schema_type, list):  # Handle union types like ["string", "null"]
        non_null_types = [t for t in schema_type if t is not None and t != "null"]
        if len(non_null_types) == 1:
            return Optional[
                _map_json_schema_type_to_python_type({"type": non_null_types[0]})
            ]
        # More complex union type mapping could be added if needed
    return Any


@dataclass
class Agent(Generic[CtxType, OutType]):
    """
    super duper fun and fast agent implementation.

    ```python
    # Although agents are stateful, parameters are not required
    # this is a prototyping library who said we're taking this to prod
    # i do what i want
    agent = Agent.create()
    ```

    Agents are designed to be stateful and can be used as tools for other agents as well.
    Agents implement the following patterns and strategies:
        - Autonomous Context Updating
          - This is kinda cool actually give it a try
          - LLM is able to decide on automatically updating any user defined context object
            before or after it's main response generation loop
        - Autonomous Ending
          - The LLM is also able to decide if the conversation should end or not based on a determined
            strategy.
          - If the conversation should end, the LLM will use the `finalize_response` tool
            to generate a response
        - Planning and Reflection
          - The LLM is able to plan for the future and reflect on the past
          - This is useful for long running conversations and complex tasks
        - Multi-step reasoning
          - The LLM is able to plan for the future and reflect on the past
        - Tool Usage
          - The LLM is able to use tools to help it achieve it's goal
        - Iterative Output
          - The LLM is able to generate an output in a way that can be iterated on
          - Objects like pydantic models can have each field generated sequentially
    """

    name: str = "Agent"
    """The unique name identifying this agent."""
    instructions: str = ""
    """Core instructions defining the agent's purpose, behavior, and persona."""
    context: Optional[CtxType] = None
    """
    An optional mutable object (Pydantic model, dataclass, or dict) representing the
    agent's internal state or memory. It can be read and updated during its execution.
    """
    prompt_template: AgentPromptTemplate = field(default_factory=AgentPromptTemplate)
    """The template used to construct the system prompt for the agent's LLM calls."""
    settings: AgentSettings = field(default_factory=AgentSettings)
    """Configuration settings controlling the agent's runtime behavior."""
    tools: list[AgentTool] = field(default_factory=list)
    """A list of `AgentTool` instances that this agent is capable of using."""

    _create_api: Create = field(default_factory=Create, init=False, repr=False)
    """Internal instance of the `Create` API for LLM interactions."""

    @staticmethod
    def _is_thought_message_content(content: Optional[Any]) -> bool:
        if not isinstance(content, str):
            return False
        known_thought_prefixes = (
            "[Context Update Thought]:",
            "[Planning Phase]:",
            "[Reflection Phase]:",
            "[End Decision Thought]:",
        )
        return any(content.startswith(prefix) for prefix in known_thought_prefixes)

    def __post_init__(self):
        """Validates context type after initialization."""
        if self.context is not None:
            is_pydantic_model_class = isinstance(self.context, type) and issubclass(
                self.context, BaseModel
            )
            is_valid_instance = isinstance(
                self.context, (BaseModel, dict)
            ) or is_dataclass(self.context)

            if is_pydantic_model_class:
                raise TypeError(
                    f"Agent '{self.name}' context was provided the Pydantic model class '{self.context.__name__}' instead of an instance. "
                    f"Please instantiate it, e.g., context={self.context.__name__}()"
                )
            elif not is_valid_instance:
                raise TypeError(
                    f"Agent '{self.name}' context must be a Pydantic BaseModel instance, dataclass instance, or dictionary, "
                    f"got {type(self.context)}."
                )

    @classmethod
    def create(
        cls,
        name: str = "Agent",
        instructions: str = "",
        planning: bool = False,
        reflection: bool = False,
        context: Optional[CtxType] = None,
        output_type: Optional[Type[OutType]] = None,
        model: ModelParam = "openai/gpt-4o-mini",
        model_params: Optional[Params] = None,
        max_steps: int = 10,
        iterative_output: bool = False,
        iterative_context: bool = False,
        force_tools: bool = False,
        end_strategy: AgentEndStrategy = "full",
        end_instructions: Optional[
            str
        ] = None,  # Intentionally allowing None to use class default
        context_strategy: AgentContextStrategy = "selective",
        context_instructions: Optional[str] = None,  # Intentionally allowing None
        update_context_before_response: Optional[bool] = None,
        update_context_after_response: bool = False,
        add_context_to_prompt: bool = True,
        add_tools_to_prompt: bool = True,
        keep_intermediate_steps: bool = True,
        persona_override_intermediate_steps: bool = True,
        prompt_template: Optional[AgentPromptTemplate] = None,
        save_history: Optional[bool] = None,
        show_context: Optional[bool] = None,
    ) -> "Agent[CtxType, OutType]":
        """
        Factory method to create and configure an Agent instance.

        Args:
            - name (str): The name of the agent.
            - instructions (str): Base instructions for the agent.
            - planning (bool): Enable planning steps.
            - reflection (bool): Enable reflection steps.
            - context (Optional[CtxType]): Initial context object for the agent.
            - output_type (Optional[Type[OutType]]): Expected type of the agent's final output.
            - model (str): Language model to use.
            - model_params (Optional[Params]): Additional parameters for the language model.
            - max_steps (int): Maximum main generation steps.
            - iterative_output (bool): Enable iterative generation of final output fields.
            - iterative_context (bool): Enable iterative generation for selective context updates.
            - force_tools (bool): Force execution of all tools at the start.
            - end_strategy (str): Strategy for response termination ('full' or 'selective').
            - end_instructions (Optional[str]): Additional instructions for the ending strategy.
            - context_strategy (str): Strategy for context updates ('selective' or 'full').
            - context_instructions (Optional[str]): Additional instructions for context updates.
            - update_context_before_response (Optional[bool]): Update context before main loop. Defaults to True if context is provided.
            - update_context_after_response (bool): Update context after main loop.
            - add_context_to_prompt (bool): Include context in the system prompt.
            - add_tools_to_prompt (bool): Include tool descriptions in the system prompt.
            - keep_intermediate_steps (bool): Retain planning/reflection messages in history.
            - persona_override_intermediate_steps (bool): Apply agent persona to intermediate step generation.
            - prompt_template (Optional[AgentPromptTemplate]): Custom prompt template for the agent.
            - save_history (Optional[bool]): Whether to save the full message history in the response. Defaults to True.
            - show_context (Optional[bool]): If True and keep_intermediate_steps is True, context update thought messages are passed to the LLM for main generation. Defaults to AgentSettings default (True).

        Returns:
            A new Agent instance.
        """
        actual_output_type = output_type or cast(Type[OutType], AgentMessageSchema)

        # Default for update_context_before_response: True if context is provided, else False
        actual_update_cbr = (
            context is not None
            if update_context_before_response is None
            else update_context_before_response
        )

        # Use sensible defaults for end_strategy based on whether tools will be used
        # For a no-tool agent, selective is more efficient than the full strategy that requires a finalize_response tool
        actual_end_strategy = end_strategy
        if end_strategy == "full":
            logger.debug(
                f"Note: Agent '{name}' created with 'full' end strategy. If no tools are added, it will auto-terminate after one step."
            )

        settings = AgentSettings(
            output_type=actual_output_type,
            planning=planning,
            reflection=reflection,
            model=model,
            model_params=(model_params or {}),
            max_steps=max_steps,
            iterative_output=iterative_output,
            iterative_context=iterative_context,
            force_tools=force_tools,
            end_strategy=actual_end_strategy,
            end_instructions=(
                end_instructions
                if end_instructions is not None
                else AgentSettings().end_instructions
            ),
            context_strategy=context_strategy,
            context_instructions=(
                context_instructions
                if context_instructions is not None
                else AgentSettings().context_instructions
            ),
            update_context_before_response=actual_update_cbr,
            update_context_after_response=update_context_after_response,
            add_context_to_prompt=add_context_to_prompt,
            add_tools_to_prompt=add_tools_to_prompt,
            keep_intermediate_steps=keep_intermediate_steps,
            persona_override_intermediate_steps=persona_override_intermediate_steps,
            save_history=save_history
            if save_history is not None
            else AgentSettings().save_history,  # type: ignore
            show_context=show_context
            if show_context is not None
            else AgentSettings().show_context,
        )
        return cls(
            name=name,
            instructions=instructions,
            context=context,
            prompt_template=(prompt_template or AgentPromptTemplate()),
            settings=settings,
            tools=[],  # Tools are added via add_tools method
        )

    def _get_context_as_dict(
        self, current_context: Optional[CtxType] = None
    ) -> Dict[str, Any]:
        """Serializes the agent's context to a dictionary."""
        ctx_to_use = current_context if current_context is not None else self.context
        if ctx_to_use is None:
            return {}
        if isinstance(ctx_to_use, BaseModel):
            return ctx_to_use.model_dump(
                mode="json"
            )  # Use mode='json' for better serialization
        if is_dataclass(ctx_to_use) and not isinstance(ctx_to_use, type):
            return dataclass_asdict(ctx_to_use)
        if isinstance(ctx_to_use, dict):
            return ctx_to_use
        logger.warning(
            f"Context type {type(ctx_to_use)} for agent '{self.name}' is not directly serializable to dict. Returning empty dict."
        )
        return {}

    def _format_string_with_context(
        self,
        template_string: str,
        current_context: Optional[CtxType] = None,
    ) -> str:
        """Formats a template string by replacing placeholders with values from the context."""
        if not template_string:
            return ""
        context_dict = self._get_context_as_dict(current_context)

        # More robust placeholder replacement to handle missing keys gracefully
        def replace_match(match: re.Match) -> str:
            placeholder = match.group(1)
            if (
                placeholder == "self.context"
            ):  # Special placeholder for the whole context object
                return (
                    format_to_markdown(
                        context_dict,
                        compact=True,
                        show_title=False,
                        show_bullets=False,
                    )
                    if context_dict
                    else "No context available."
                )

            value = context_dict
            try:
                for key_part in placeholder.split(
                    "."
                ):  # Allows for nested access like {user.name}
                    if isinstance(value, dict):
                        value = value.get(key_part)
                    elif hasattr(
                        value, key_part
                    ):  # For Pydantic models/dataclasses accessed as dicts
                        value = getattr(value, key_part)
                    else:
                        value = None  # Key not found
                        break
                return (
                    str(value) if value is not None else match.group(0)
                )  # Return original placeholder if value is None or not found
            except Exception:
                return match.group(
                    0
                )  # Return original placeholder on any error during resolution

        return re.sub(r"\{([\w\.]+)\}", replace_match, template_string)

    def _format_compiled_messages(
        self,
        current_context: Optional[CtxType] = None,
        user_query_for_context_instructions: Optional[str] = None,
    ) -> List[Message]:
        """
        Constructs the system prompt messages for the agent by assembling various parts
        like instructions, context, tool descriptions, and other template sections.
        """
        system_prompt_parts: List[str] = []

        # 1. Add main agent instructions (from self.instructions and prompt_template.start_part)
        start_content = self.prompt_template.start_part
        if isinstance(start_content, str):
            system_prompt_parts.append(
                self._format_string_with_context(start_content, current_context)
            )
        elif isinstance(start_content, list):
            system_prompt_parts.extend(
                [
                    self._format_string_with_context(p_str, current_context)
                    for p_str in start_content
                ]
            )

        if self.instructions:  # Agent's primary instructions
            system_prompt_parts.append(
                self._format_string_with_context(self.instructions, current_context)
            )

        # 2. Add dynamic prompt parts from the template
        for part_template in self.prompt_template.parts:
            content = self._format_string_with_context(
                part_template.content or "", current_context
            )
            if content:  # Only add if there's resolved content
                section_header = ""
                if part_template.name:
                    name_display = (
                        part_template.name.upper()
                        if part_template.important
                        else part_template.name
                    )
                    section_header = f"--- {name_display} ---\n"

                section_footer = ""
                if part_template.name and part_template.important:
                    section_footer = (
                        f"\n--- END OF IMPORTANT {part_template.name.upper()} ---"
                    )

                formatted_section_content = f"{section_header}{content}{section_footer}"
                system_prompt_parts.append(
                    format_to_markdown(formatted_section_content)
                    if part_template.markdown
                    else formatted_section_content
                )

        # 3. Add current context information if enabled
        if self.settings.add_context_to_prompt and (
            current_context is not None or self.context is not None
        ):
            context_to_display = (
                current_context if current_context is not None else self.context
            )
            context_dict = self._get_context_as_dict(context_to_display)
            if context_dict:  # Only add if context is not empty
                context_md = format_to_markdown(
                    context_dict,
                    show_title=True,
                    title_name="Current Context",
                    compact=False,
                )
                system_prompt_parts.append(
                    f"\n--- CURRENT CONTEXT ---\n{context_md}\n--- END OF CURRENT CONTEXT ---"
                )
            elif (
                self.prompt_template.context
            ):  # Fallback to static context string in template if dynamic one is empty
                system_prompt_parts.append(
                    self._format_string_with_context(
                        self.prompt_template.context, current_context
                    )
                )

        # 4. Add tool descriptions if enabled
        active_tools = (
            self._get_current_tools()
        )  # Get current tools to list them accurately
        if self.settings.add_tools_to_prompt and active_tools:
            tool_descriptions_list = ["\n--- AVAILABLE TOOLS ---"]
            for tool_item in active_tools:
                tool_info = f"Tool Name: {tool_item.name}\nDescription: {tool_item.description}\nParameters (JSON Schema): {json.dumps(tool_item.parameters, indent=2)}\n"
                tool_descriptions_list.append(tool_info)
            tool_descriptions_list.append("--- END OF AVAILABLE TOOLS ---")
            system_prompt_parts.append("\n".join(tool_descriptions_list))

        # 5. Add guidelines for context updates
        if (
            self.settings.update_context_before_response
            or self.settings.update_context_after_response
        ):
            # Base prompt for context update logic is usually internal to _handle_context_update
            # self.settings.context_instructions provides *additional* user guidance
            guideline_prompt = "When considering context updates:"
            if self.settings.context_instructions:
                guideline_prompt += f"\n{self._format_string_with_context(self.settings.context_instructions, current_context)}"
            if user_query_for_context_instructions:
                guideline_prompt = f"Regarding the user's query: '{user_query_for_context_instructions}'. {guideline_prompt}"
            system_prompt_parts.append(
                f"\n--- GUIDELINES FOR CONTEXT MANAGEMENT ---\n{guideline_prompt}\n--- END OF GUIDELINES FOR CONTEXT MANAGEMENT ---"
            )

        # 6. Add guidelines for ending the response (relevant for both strategies)
        # Base prompt for ending is internal to _should_agent_end or _get_end_tool
        # self.settings.end_instructions provides *additional* user guidance
        ending_guideline_prompt = "When deciding to conclude your response:"
        if self.settings.end_instructions:
            ending_guideline_prompt += f"\n{self._format_string_with_context(self.settings.end_instructions, current_context)}"
        system_prompt_parts.append(
            f"\n--- GUIDELINES FOR RESPONSE COMPLETION ---\n{ending_guideline_prompt}\n--- END OF GUIDELINES FOR RESPONSE COMPLETION ---"
        )

        # 7. Add end part of the prompt template
        end_content = self.prompt_template.end_part
        if isinstance(end_content, str):
            system_prompt_parts.append(
                self._format_string_with_context(end_content, current_context)
            )
        elif isinstance(end_content, list):
            system_prompt_parts.extend(
                [
                    self._format_string_with_context(p_str, current_context)
                    for p_str in end_content
                ]
            )

        # Combine all parts into a single system message
        final_system_prompt_str = "\n\n".join(filter(None, system_prompt_parts)).strip()
        return (
            [Message(role="system", content=final_system_prompt_str)]
            if final_system_prompt_str
            else []
        )

    def as_tool(
        self,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        parameters_override: Optional[Type[BaseModel]] = None,
    ) -> AgentTool:
        """
        Converts this agent into an `AgentTool` instance.

        This allows the agent to be used as a tool by another agent, facilitating
        hierarchical or chained agentic behaviors. The resulting tool typically
        expects a "query" parameter.

        Args:
            name_override: Optional name for the generated tool. Defaults to `"{agent_name}_agent_tool"`.
            description_override: Optional description for the tool. Defaults to a summary of the agent's instructions.
            parameters_override: Optional Pydantic model defining the parameters for this agent when called as a tool.
                                 Defaults to a model with a single "query: str" field.

        Returns:
            An `AgentTool` instance that can execute this agent.
        """
        tool_name = name_override or f"{self.name}_agent_tool"
        tool_description = (
            description_override
            or f"Invokes the '{self.name}' agent. Purpose: {self.instructions[:150] + '...' if self.instructions else 'No specific instructions.'}"
        )

        param_schema: Dict[str, Any]
        if parameters_override and issubclass(parameters_override, BaseModel):
            param_schema = parameters_override.model_json_schema()
        else:

            class DefaultAgentToolParams(BaseModel):
                query: str = PydanticField(
                    ...,
                    description="The primary query, task, or input for the agent.",
                )
                # Potentially add:
                # initial_history: Optional[List[Message]] = PydanticField(None, description="Optional conversation history to seed the agent's run.")
                # context_overrides: Optional[Dict[str, Any]] = PydanticField(None, description="Optional overrides for the agent's initial context fields.")

            param_schema = DefaultAgentToolParams.model_json_schema()

        async def _agent_as_tool_function(
            context_of_calling_agent: Any, **kwargs: Any
        ) -> AgentResponse[OutType]:
            # context_of_calling_agent is the context of the agent *using* this tool.
            # This agent (self) will run with its own internal context, possibly initialized/overridden.
            query = kwargs.get("query")
            if query is None:
                raise ValueError(
                    f"Agent tool '{tool_name}' called without a 'query' argument."
                )

            # Handle potential overrides from kwargs if AgentToolParams is extended
            # initial_history_override = kwargs.get("initial_history")
            # context_values_override = kwargs.get("context_overrides")

            # For now, a simple run with the query. More complex state passing could be added.
            return await self.run(
                prompt=query,
                # existing_messages=initial_history_override,
                # context=context_values_override # This would require merging logic if self.context exists
            )

        return AgentTool(
            name=tool_name,
            description=tool_description,
            parameters=param_schema,
            function=_agent_as_tool_function,  # type: ignore
            error_handler=default_error_handler,
        )

    def add_tools(
        self,
        tools_to_add: Union[
            Callable[..., Any],
            AgentTool,
            List[Union[Callable[..., Any], AgentTool]],
        ],
    ) -> None:
        """
        Adds one or more tools to the agent's list of usable tools.

        Callables will be automatically converted to `AgentTool` instances.

        Args:
            tools_to_add: A single tool (callable or `AgentTool`) or a list of tools.
        """
        if not isinstance(tools_to_add, list):
            tools_to_add = [tools_to_add]

        for tool_item in tools_to_add:
            if isinstance(tool_item, AgentTool):
                # Avoid adding duplicate tools by name
                if not any(t.name == tool_item.name for t in self.tools):
                    self.tools.append(tool_item)
                else:
                    logger.warning(
                        f"Tool with name '{tool_item.name}' already exists. Skipping duplicate."
                    )
            elif callable(tool_item):
                try:
                    # Convert callable to AgentTool using the decorator
                    converted_tool = create_agent_tool_decorator(tool_item)
                    if not any(t.name == converted_tool.name for t in self.tools):
                        self.tools.append(converted_tool)
                    else:
                        logger.warning(
                            f"Tool (from callable) with name '{converted_tool.name}' already exists. Skipping duplicate."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to convert callable {getattr(tool_item, '__name__', 'Unnamed callable')} to AgentTool: {e}. Skipping."
                    )
            else:
                logger.warning(
                    f"Item {tool_item} is not a valid AgentTool or callable. Skipping."
                )

    def add_prompt_part(
        self,
        name: Optional[str] = None,
        content: Any = None,  # Will be converted to string
        important: bool = False,
        markdown: bool = False,
        position: Optional[int] = None,
    ) -> None:
        """
        Adds a new section (part) to the agent's prompt template.

        Args:
            name: Optional name/header for this prompt section.
            content: The content of the prompt section. Will be stringified if not already a string.
            important: If `True`, the section name is emphasized in the prompt.
            markdown: If `True`, the content is treated as Markdown.
            position: Optional index at which to insert the part. If None, appends to the end.
        """
        if content is None:
            logger.warning(
                f"Agent '{self.name}': Attempted to add a prompt part with no content. Skipping."
            )
            return

        content_str = str(content) if not isinstance(content, str) else content

        part = AgentPromptTemplatePart(
            name=name,
            content=content_str,
            important=important,
            markdown=markdown,
        )
        if position is not None and 0 <= position <= len(self.prompt_template.parts):
            self.prompt_template.parts.insert(position, part)
        else:
            self.prompt_template.parts.append(part)

    async def generate_prompt_parts(
        self,
        names_of_parts: Union[str, list[str]],
        generation_instructions: str = "Generate concise and effective content for the following prompt section(s). Emphasize clarity and directness.",
        instruct_to_use_context_variables: bool = True,
        generation_model: Optional[ModelParam] = None,
        generation_model_params: Optional[Params] = None,
    ) -> None:
        """
        Uses an LLM to generate content for specified prompt template parts and adds them to the agent.

        Args:
            names_of_parts: A name or list of names for the prompt parts to generate.
            generation_instructions: Instructions for the LLM on how to generate the content.
            instruct_to_use_context_variables: If `True`, provides the LLM with available context variables.
            generation_model: Optional model to use for this generation task, defaults to agent's model.
            generation_model_params: Optional parameters for the generation model.
        """
        part_names_list = (
            [names_of_parts] if isinstance(names_of_parts, str) else names_of_parts
        )

        for part_name in part_names_list:

            class GeneratedPromptPartSchema(BaseModel):
                generated_content: str = PydanticField(
                    ...,
                    description=f"The generated textual content for the prompt section named '{part_name}'.",
                )

            request_prompt_str = f'{generation_instructions}\n\nSection Name To Generate: "{part_name}"\n'
            if instruct_to_use_context_variables and self.context:
                context_schema_md = format_to_markdown(
                    self._get_context_as_dict(), schema=True
                )
                request_prompt_str += f"\nConsider using these context variables if relevant:\n{context_schema_md}\n"

            request_prompt_str += "\nPlease generate the content for this section now:"

            try:
                logger.info(
                    f"Agent '{self.name}': Generating prompt part for section: {part_name}"
                )
                generated_part_model = await self._create_api.async_from_schema(
                    schema=GeneratedPromptPartSchema,
                    prompt=request_prompt_str,
                    model=generation_model or self.settings.model,
                    model_params=generation_model_params or self.settings.model_params,
                )
                if generated_part_model and hasattr(
                    generated_part_model, "generated_content"
                ):
                    self.add_prompt_part(
                        name=part_name,
                        content=generated_part_model.generated_content,
                    )  # type: ignore
                    logger.info(
                        f"Agent '{self.name}': Successfully generated and added prompt part: {part_name}"
                    )
                else:
                    logger.warning(
                        f"Agent '{self.name}': LLM did not return valid content for prompt part: {part_name}"
                    )
            except Exception as e:
                logger.error(
                    f"Agent '{self.name}': Error generating prompt part '{part_name}': {e}",
                    exc_info=True,
                )

    async def _handle_context_update(
        self,
        current_messages: List[Message],
        run_time_context: CtxType,
        user_query: Optional[str] = None,
    ) -> Tuple[CtxType, List[Message]]:
        """
        Manages updates to the agent's context based on the configured strategy.

        Args:
            current_messages: The current list of messages in the conversation.
            run_time_context: The agent's current context object for this run.
            user_query: The user query that might influence the context update.

        Returns:
            A tuple containing the (potentially) modified context object and a list of
            any messages generated during the context update process (e.g., thoughts).
        """
        if run_time_context is None:
            logger.debug(
                f"Agent '{self.name}': Context update skipped as no context object is available for this run."
            )
            return run_time_context, []

        is_pydantic_ctx = isinstance(run_time_context, BaseModel)
        is_dataclass_ctx = is_dataclass(run_time_context) and not isinstance(
            run_time_context, type
        )
        original_ctx_type = (
            type(run_time_context) if (is_pydantic_ctx or is_dataclass_ctx) else dict
        )

        working_context_dict: Dict[str, Any]
        if is_pydantic_ctx:
            working_context_dict = cast(BaseModel, run_time_context).model_dump(
                mode="json"
            )
        elif is_dataclass_ctx:
            working_context_dict = dataclass_asdict(run_time_context)
        elif isinstance(run_time_context, dict):
            working_context_dict = deepcopy(run_time_context)  # Ensure mutable copy
        else:  # Should be caught by __post_init__ or create, but as a safeguard
            logger.warning(
                f"Agent '{self.name}': Context update skipped due to unsupported context type {type(run_time_context)}."
            )
            return run_time_context, []

        if not working_context_dict and not (is_pydantic_ctx or is_dataclass_ctx):
            logger.debug(
                f"Agent '{self.name}': Context is an empty dictionary and not a predefined structured type, skipping update logic."
            )
            return run_time_context, []

        context_schema_for_llm: Type[BaseModel]
        if is_pydantic_ctx:
            context_schema_for_llm = original_ctx_type  # type: ignore
        elif is_dataclass_ctx:
            context_schema_for_llm = convert_to_pydantic_model(original_ctx_type)  # type: ignore
        else:  # dict context; create a dynamic Pydantic model
            try:
                field_definitions = {
                    k: (
                        type(v),
                        PydanticField(default=v, description=f"Current value: {v}"),
                    )
                    for k, v in working_context_dict.items()
                }
                if (
                    not field_definitions
                ):  # Handle empty dict that might have been passed
                    logger.debug(
                        f"Agent '{self.name}': Cannot create dynamic Pydantic model from empty dictionary context. Skipping update."
                    )
                    return run_time_context, []
                context_schema_for_llm = create_model(
                    f"{self.name.replace('_', '')}DynamicContext",
                    **field_definitions,
                )  # type: ignore
            except Exception as e_dyn_model:
                logger.error(
                    f"Agent '{self.name}': Failed to create dynamic Pydantic model from dict context: {e_dyn_model}. Skipping context update."
                )
                return run_time_context, []

        additional_messages_from_update: List[Message] = []
        strategy = self.settings.context_strategy
        current_context_snapshot_md = format_to_markdown(
            working_context_dict, compact=True
        )

        persona_guidance = ""
        if self.settings.persona_override_intermediate_steps and self.instructions:
            persona_guidance = f"When evaluating context and describing updates, maintain the agent's persona: {self.instructions[:150]}..."

        # Base prompt for the LLM to think about context updates
        llm_prompt_for_context_task = (
            f"Task: Evaluate and update the agent's internal context.\n"
            f"User Query: {user_query or 'N/A'}\n"
            f"Current Context State:\n{current_context_snapshot_md}\n"
            f"{persona_guidance}\n"
            f"Additional Guidelines for Context Management: {self._format_string_with_context(self.settings.context_instructions, run_time_context)}\n\n"  # User's additional instructions
        )

        updated_fields_from_llm: Dict[str, Any] = {}

        if strategy == "selective":
            selective_prompt = (
                llm_prompt_for_context_task
                + "First, identify which specific fields in the context (if any) need to be updated based on the user query and current state. Then, provide the new values for ONLY those selected fields."
            )
            try:
                # Step 1: Select fields
                field_selection_prompt = llm_prompt_for_context_task + (
                    "Based on the interaction and guidelines, determine which fields in the agent's context require updating. "
                    "List only the names of the fields that should be changed or set."
                )
                field_selection_model = await self._create_api.async_from_schema(
                    schema=SelectContextFields,  # Changed from _ContextFieldSelection
                    prompt=field_selection_prompt,
                    model=self.settings.model,
                    model_params=self.settings.model_params,
                )
                fields_to_update_names = (
                    field_selection_model.fields_to_update
                    if field_selection_model
                    else []
                )  # type: ignore

                if not fields_to_update_names:
                    logger.debug(
                        f"Agent '{self.name}' (Selective Context): No fields identified for update."
                    )
                    return (
                        run_time_context,
                        additional_messages_from_update,
                    )

                if self.settings.keep_intermediate_steps:
                    additional_messages_from_update.append(
                        Message(
                            role="assistant",
                            content=f"[Context Update Thought]: Identified fields for potential update: {', '.join(fields_to_update_names)}",
                        )
                    )

                # Step 2: Generate updates for selected fields
                if self.settings.iterative_context:
                    for field_name in fields_to_update_names:
                        if field_name not in context_schema_for_llm.model_fields:
                            logger.warning(
                                f"Agent '{self.name}': Field '{field_name}' selected for update but not in context schema. Skipping this field."
                            )
                            continue

                        field_info = context_schema_for_llm.model_fields[field_name]
                        field_type_hint = field_info.annotation

                        SingleFieldUpdateModel = create_model(
                            f"{self.name.replace('_', '')}{field_name.capitalize()}ContextUpdate",
                            **{
                                field_name: (
                                    field_type_hint,
                                    PydanticField(
                                        ...,
                                        description=field_info.description
                                        or f"New value for the field '{field_name}'.",
                                    ),
                                )
                            },  # type: ignore
                        )

                        iterative_update_prompt = (
                            llm_prompt_for_context_task
                            + f"Now, generate the new value specifically for the field: '{field_name}' (expected type: {field_type_hint}).\nExisting value: {working_context_dict.get(field_name, 'Not set')}."
                        )
                        try:
                            updated_field_model_instance = (
                                await self._create_api.async_from_schema(
                                    schema=SingleFieldUpdateModel,
                                    prompt=iterative_update_prompt,
                                    model=self.settings.model,
                                    model_params=self.settings.model_params,
                                )
                            )
                            if updated_field_model_instance and hasattr(
                                updated_field_model_instance, field_name
                            ):
                                new_value = getattr(
                                    updated_field_model_instance,
                                    field_name,
                                )
                                updated_fields_from_llm[field_name] = new_value
                                working_context_dict[field_name] = (
                                    new_value  # Update working dict for next iteration's snapshot
                                )
                                current_context_snapshot_md = format_to_markdown(
                                    working_context_dict, compact=True
                                )  # Update for next prompt
                                if self.settings.keep_intermediate_steps:
                                    additional_messages_from_update.append(
                                        Message(
                                            role="assistant",
                                            content=f"[Context Update Thought]: Iteratively updated field '{field_name}' to: {new_value}",
                                        )
                                    )
                        except Exception as e_iter_field:
                            logger.error(
                                f"Agent '{self.name}': Error during iterative context update for field '{field_name}': {e_iter_field}"
                            )
                else:  # Batch update for selected fields
                    valid_fields_for_batch_schema = {
                        fname: (
                            context_schema_for_llm.model_fields[fname].annotation,
                            PydanticField(
                                ...,
                                description=context_schema_for_llm.model_fields[
                                    fname
                                ].description
                                or f"New value for {fname}",
                            ),
                        )
                        for fname in fields_to_update_names
                        if fname in context_schema_for_llm.model_fields
                    }
                    if not valid_fields_for_batch_schema:
                        logger.debug(
                            f"Agent '{self.name}' (Selective Context): No valid fields remaining after schema check for batch update."
                        )
                        return (
                            run_time_context,
                            additional_messages_from_update,
                        )

                    BatchUpdateModel = create_model(
                        f"{self.name.replace('_', '')}SelectiveContextBatchUpdate",
                        **valid_fields_for_batch_schema,
                    )  # type: ignore
                    batch_update_prompt = (
                        llm_prompt_for_context_task
                        + f"Now, provide new values for the following selected fields: {', '.join(valid_fields_for_batch_schema.keys())}."
                    )

                    try:
                        batch_updated_model_instance = (
                            await self._create_api.async_from_schema(
                                schema=BatchUpdateModel,
                                prompt=batch_update_prompt,
                                model=self.settings.model,
                                model_params=self.settings.model_params,
                            )
                        )
                        if batch_updated_model_instance:
                            updated_fields_from_llm.update(
                                batch_updated_model_instance.model_dump(
                                    exclude_unset=True
                                )
                            )
                            if (
                                self.settings.keep_intermediate_steps
                                and updated_fields_from_llm
                            ):
                                additional_messages_from_update.append(
                                    Message(
                                        role="assistant",
                                        content=f"[Context Update Thought]: Batch updated selected fields: {updated_fields_from_llm}",
                                    )
                                )
                    except Exception as e_batch_sel:
                        logger.error(
                            f"Agent '{self.name}': Error during batch selective context update: {e_batch_sel}"
                        )

            except Exception as e_select_fields:
                logger.error(
                    f"Agent '{self.name}': Error during selective context field selection phase: {e_select_fields}"
                )

        elif strategy == "full":
            full_update_prompt = (
                llm_prompt_for_context_task
                + "Provide the complete, updated context object reflecting any necessary changes based on the interaction."
            )
            try:
                updated_context_model_instance = (
                    await self._create_api.async_from_schema(
                        schema=context_schema_for_llm,  # Use the full context schema
                        prompt=full_update_prompt,
                        model=self.settings.model,
                        model_params=self.settings.model_params,
                    )
                )
                if updated_context_model_instance:
                    updated_fields_from_llm = updated_context_model_instance.model_dump(
                        exclude_unset=True
                    )  # This will be the entire new context
                    if (
                        self.settings.keep_intermediate_steps
                        and updated_fields_from_llm
                    ):
                        additional_messages_from_update.append(
                            Message(
                                role="assistant",
                                content=f"[Context Update Thought]: Performed full context update. New state: {format_to_markdown(updated_fields_from_llm, compact=True)}",
                            )
                        )
            except Exception as e_full_update:
                logger.error(
                    f"Agent '{self.name}': Error during full context update: {e_full_update}"
                )

        # Apply updates to the actual run_time_context object
        if updated_fields_from_llm:
            if strategy == "full":  # Replace the entire working dict
                working_context_dict = updated_fields_from_llm
            else:  # Merge updates for selective
                working_context_dict.update(updated_fields_from_llm)

            # Convert back to original type if necessary
            if is_pydantic_ctx:
                try:
                    run_time_context = original_ctx_type.model_validate(
                        working_context_dict
                    )  # type: ignore
                except ValidationError as ve:
                    logger.error(
                        f"Agent '{self.name}': Validation error applying context updates to Pydantic model {original_ctx_type.__name__}: {ve}. Context may be partially updated or unchanged."
                    )
                    run_time_context = cast(
                        CtxType, working_context_dict
                    )  # Fallback to dict if validation fails
            elif is_dataclass_ctx:
                try:
                    run_time_context = original_ctx_type(**working_context_dict)  # type: ignore
                except Exception as e_dc_reconstruct:
                    logger.error(
                        f"Agent '{self.name}': Error reconstructing dataclass {original_ctx_type.__name__} from updated context: {e_dc_reconstruct}. Context may be partially updated."
                    )
                    run_time_context = cast(CtxType, working_context_dict)  # Fallback
            else:  # Original was a dict, already updated
                run_time_context = cast(CtxType, working_context_dict)

        return run_time_context, additional_messages_from_update

    async def _handle_planning_step(
        self,
        current_messages: List[Message],
        run_time_context: Optional[CtxType],
    ) -> Tuple[Optional[str], List[Message]]:
        """
        Performs a planning step if enabled in settings.

        The agent outlines its next actions based on the conversation history and its goal.

        Args:
            current_messages: The current conversation history.
            run_time_context: The agent's current context.

        Returns:
            A tuple containing the generated plan (str) and any messages (e.g., thoughts)
            produced during planning.
        """
        if not self.settings.planning:
            return None, []

        persona_guidance = ""
        if self.settings.persona_override_intermediate_steps and self.instructions:
            persona_guidance = f"Remember to plan in the agent's designated style: {self.instructions[:150]}..."

        planning_prompt_str = (
            f"You are currently in a planning phase. Based on the conversation history and your primary objective, "
            f"critically evaluate the situation and formulate a concise plan for your next one or two immediate actions. "
            f"Do not generate the actual response or execute tools yet, only describe your plan. {persona_guidance}"
        )

        planning_messages_for_llm = current_messages + [
            Message(role="user", content=planning_prompt_str)
        ]

        class PlanSchema(BaseModel):
            plan_description: str = PydanticField(
                ...,
                description="A clear and concise description of the next one or two steps to be taken to address the user's request or achieve the current goal.",
            )

        try:
            plan_model_instance = await self._create_api.async_from_schema(
                schema=PlanSchema,
                prompt=planning_messages_for_llm,
                model=self.settings.model,
                model_params=self.settings.model_params,
            )
            plan_text_content = (
                plan_model_instance.plan_description
                if plan_model_instance
                else "No specific plan was generated."
            )  # type: ignore

            planning_step_messages: List[Message] = []
            if self.settings.keep_intermediate_steps:
                planning_step_messages.append(
                    Message(
                        role="assistant",
                        content=f"[Planning Phase]: {plan_text_content}",
                    )
                )
            return plan_text_content, planning_step_messages
        except Exception as e_planning:
            logger.error(
                f"Agent '{self.name}': Error during planning step: {e_planning}",
                exc_info=True,
            )
            return f"Error occurred during planning: {e_planning}", []

    async def _handle_reflection_step(
        self,
        current_messages: List[Message],
        last_assistant_response_text: str,
        run_time_context: Optional[CtxType],
    ) -> Tuple[Optional[str], List[Message]]:
        """
        Performs a reflection step if enabled in settings.

        The agent reviews its last generated content in the context of the conversation
        and decides on the quality or next immediate action.

        Args:
            current_messages: The current conversation history.
            last_assistant_response_text: The text of the agent's most recent response.
            run_time_context: The agent's current context.

        Returns:
            A tuple containing the reflection text (str) and any messages produced
            during reflection.
        """
        if not self.settings.reflection:
            return None, []

        persona_guidance = ""
        if self.settings.persona_override_intermediate_steps and self.instructions:
            persona_guidance = f"Remember to reflect in the agent's designated style: {self.instructions[:150]}..."

        reflection_prompt_str = (
            f'You are in a reflection phase. You just produced the following content: "{last_assistant_response_text}".\n'
            f"Critically review this content. Is it accurate, complete, aligned with the user's request, your plan, and overall goals? "
            f"What should be your immediate next thought or action based on this reflection? {persona_guidance}"
        )
        reflection_messages_for_llm = current_messages + [
            Message(role="user", content=reflection_prompt_str)
        ]

        class ReflectionSchema(BaseModel):
            critical_reflection: str = PydanticField(
                ...,
                description="Your critical analysis of the last generated content and your thoughts on the next immediate action or adjustment needed.",
            )

        try:
            reflection_model_instance = await self._create_api.async_from_schema(
                schema=ReflectionSchema,
                prompt=reflection_messages_for_llm,
                model=self.settings.model,
                model_params=self.settings.model_params,
            )
            reflection_text_content = (
                reflection_model_instance.critical_reflection
                if reflection_model_instance
                else "No specific reflection was generated."
            )  # type: ignore

            reflection_step_messages: List[Message] = []
            if self.settings.keep_intermediate_steps:
                reflection_step_messages.append(
                    Message(
                        role="assistant",
                        content=f"[Reflection Phase]: {reflection_text_content}",
                    )
                )
            return reflection_text_content, reflection_step_messages
        except Exception as e_reflection:
            logger.error(
                f"Agent '{self.name}': Error during reflection step: {e_reflection}",
                exc_info=True,
            )
            return f"Error occurred during reflection: {e_reflection}", []

    def _get_current_tools(
        self,
        run_tools_override: Optional[List[Union[Callable[..., Any], AgentTool]]] = None,
        exclude_tools_override: Optional[List[str]] = None,
    ) -> List[AgentTool]:
        """
        Determines the active set of tools for the current execution step,
        considering base tools and any runtime overrides.
        """
        current_agent_tools = list(
            self.tools
        )  # Start with a copy of the agent's base tools

        if run_tools_override:
            for tool_item_override in run_tools_override:
                converted_tool_override: Optional[AgentTool] = None
                if isinstance(tool_item_override, AgentTool):
                    converted_tool_override = tool_item_override
                elif callable(tool_item_override):
                    try:
                        converted_tool_override = create_agent_tool_decorator(
                            tool_item_override
                        )
                    except Exception as e_conv:
                        logger.warning(
                            f"Agent '{self.name}': Could not convert callable tool '{getattr(tool_item_override, '__name__', 'Unnamed callable')}' from override list: {e_conv}"
                        )

                if converted_tool_override:
                    # Replace if name exists, else add
                    existing_indices = [
                        i
                        for i, t in enumerate(current_agent_tools)
                        if t.name == converted_tool_override.name
                    ]  # type: ignore
                    if existing_indices:
                        current_agent_tools[existing_indices[0]] = (
                            converted_tool_override
                        )
                    else:
                        current_agent_tools.append(converted_tool_override)

        if exclude_tools_override:
            current_agent_tools = [
                t for t in current_agent_tools if t.name not in exclude_tools_override
            ]

        return current_agent_tools

    async def _handle_tool_calls(
        self,
        tool_calls_from_llm: List[ToolCall],
        run_time_context: Optional[CtxType],
        available_tools_for_step: List[AgentTool],
    ) -> List[Message]:
        """
        Executes the tool calls requested by the LLM and returns messages with their results.
        """
        tool_result_messages: List[Message] = []
        for tool_call_data in tool_calls_from_llm:
            tool_name = tool_call_data["function"]["name"]
            tool_arguments_str = tool_call_data["function"]["arguments"]
            tool_call_id = tool_call_data["id"]

            target_tool = next(
                (t for t in available_tools_for_step if t.name == tool_name),
                None,
            )
            tool_result_content_str = f"Error: Tool '{tool_name}' was called by the model, but it is not available or configured for this agent."

            if target_tool:
                try:
                    logger.info(
                        f"Agent '{self.name}': Executing tool: {tool_name} with args: {tool_arguments_str}"
                    )
                    tool_execution_result = await target_tool.execute(
                        context=run_time_context, args=tool_arguments_str
                    )  # type: ignore

                    if isinstance(tool_execution_result, (BaseModel, dict, list)):
                        tool_result_content_str = json.dumps(
                            tool_execution_result.model_dump(mode="json")
                            if isinstance(tool_execution_result, BaseModel)
                            else tool_execution_result,
                            default=str,
                        )
                    else:
                        tool_result_content_str = str(tool_execution_result)
                except Exception as e_tool_exec:
                    logger.error(
                        f"Agent '{self.name}': Error executing tool {tool_name}: {e_tool_exec}",
                        exc_info=True,
                    )
                    tool_result_content_str = f"Error during execution of tool '{tool_name}': {str(e_tool_exec)}"
            else:
                logger.warning(
                    f"Agent '{self.name}': Tool '{tool_name}' called by LLM but not found in agent's available tools for this step."
                )

            tool_result_messages.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=tool_result_content_str,
                )
            )
        return tool_result_messages

    def _get_end_tool(self, parameterless_finalize: bool = False) -> AgentTool:
        """
        Creates and returns the special "finalize_response" tool used when `end_strategy` is "full".
        Can be configured to be parameter-less if the final output is a structured type that will be generated separately.
        """
        tool_description: str
        params_schema: Type[BaseModel]

        if parameterless_finalize:

            class ParameterlessFinalizeResponseParams(BaseModel):
                model_config = ConfigDict(extra="forbid")
                # No fields needed, this tool is just a signal.
                # OpenAI requires an empty object for parameters if no params are defined.

            params_schema = ParameterlessFinalizeResponseParams
            tool_description = (
                f"IMPORTANT: Use this tool ONLY when you have gathered all necessary information and are ready for the system "
                f"to generate the final structured output. Do not provide any arguments. {self.settings.end_instructions}"
            )
        else:

            class FinalizeResponseParamsWithSummary(BaseModel):
                model_config = ConfigDict(extra="forbid")
                summary_and_final_thoughts: str = PydanticField(
                    ...,
                    description=f"A comprehensive summary of the entire interaction, including key findings, decisions, and any final information to be conveyed to the user. This should encapsulate the agent's work. {self.settings.end_instructions}",
                )

            params_schema = FinalizeResponseParamsWithSummary
            tool_description = (
                f"IMPORTANT: Use this tool ONLY when you are absolutely certain that the user's request has been fully addressed, "
                f"all necessary information has been provided or gathered, all required actions (including other tool uses) are complete, "
                f"and no further steps are needed. {self.settings.end_instructions}"
            )

        def _finalize_response_function(**kwargs: Any) -> Dict[str, str]:
            # This function's primary role is to be called, signaling the end.
            # The actual final output is constructed by the agent from its history and accumulated content.
            if parameterless_finalize:
                return {
                    "status": "Response finalization signaled for structured output generation."
                }
            else:
                return {
                    "status": "Response finalized successfully by agent.",
                    "summary_provided": kwargs.get(
                        "summary_and_final_thoughts",
                        "No summary provided.",
                    ),
                }

        return AgentTool(
            name="finalize_response",
            description=tool_description,
            parameters=params_schema.model_json_schema(),
            function=_finalize_response_function,
        )

    async def _should_agent_end(
        self,
        current_messages: List[Message],
        run_time_context: Optional[CtxType],
        current_end_strategy: AgentEndStrategy,
        accumulated_response_content: str,
    ) -> bool:
        """
        Determines if the agent should end its operation based on the configured strategy.
        For "full" strategy, this method doesn't decide; the LLM calling `finalize_response` does.
        """
        if current_end_strategy == "selective":
            persona_guidance = ""
            if self.settings.persona_override_intermediate_steps and self.instructions:
                persona_guidance = f"Remember to make this decision in the agent's designated style: {self.instructions[:150]}..."

            # Base prompt for the decision
            decision_prompt_str = (
                f"Task: Decide whether to end the current operation or continue.\n"
                f"Review the conversation history, your last action, and the overall goal.\n"
                f"Current accumulated response content for the user (summary): '{accumulated_response_content[:300]}...'\n"
                f"{persona_guidance}\n"
                # User's additional instructions for ending
                f"Additional Guidelines for Ending: {self._format_string_with_context(self.settings.end_instructions, run_time_context)}\n\n"
                f"Based on all this, should you end now, or do you need to continue (e.g., use more tools, gather more info, generate more content, or refine your response)?"
            )

            decision_messages_for_llm = current_messages + [
                Message(role="user", content=decision_prompt_str)
            ]
            try:
                decision_model_instance = await self._create_api.async_from_schema(
                    schema=DecideContinuation,  # Changed from _ShouldEndDecision
                    prompt=decision_messages_for_llm,
                    model=self.settings.model,
                    model_params=self.settings.model_params,
                )
                if decision_model_instance:
                    if self.settings.keep_intermediate_steps:
                        current_messages.append(
                            Message(
                                role="assistant",
                                content=f"[End Decision Thought]: Decided to {'end' if decision_model_instance.should_end else 'continue'}. Reason: {decision_model_instance.reason or 'N/A'}",
                            )
                        )  # type: ignore
                    return decision_model_instance.should_end  # type: ignore
                return False  # Default to continue if no valid decision model
            except Exception as e_selective_end:
                logger.error(
                    f"Agent '{self.name}': Error during selective end decision process: {e_selective_end}",
                    exc_info=True,
                )
                return False  # Default to continue if an error occurs

        # For "full" strategy, this method itself doesn't determine the end.
        # The end is triggered if the LLM calls the "finalize_response" tool,
        # which is checked in the main execution loop.
        return False

    async def _generate_final_output(
        self,
        final_message_history: List[Message],
        run_time_context: Optional[CtxType],
        target_output_type: Type[OutType],
        accumulated_text_content: str,
    ) -> OutType:
        """
        Generates the agent's final output in the specified `target_output_type`,
        based on the complete conversation history and accumulated content.
        """
        # Handle simple output types directly
        if target_output_type == str:
            return cast(OutType, accumulated_text_content)
        if target_output_type == AgentMessageSchema or (
            isinstance(target_output_type, type)
            and issubclass(target_output_type, AgentMessageSchema)
        ):
            return cast(
                OutType,
                AgentMessageSchema(role="assistant", content=accumulated_text_content),
            )

        # For Pydantic models (excluding AgentMessageSchema already handled)
        if isinstance(target_output_type, type) and issubclass(
            target_output_type, BaseModel
        ):  # type: ignore
            # Extract relevant content from the conversation history if needed
            if (
                not accumulated_text_content 
                or accumulated_text_content == "[Agent concluded its operation without generating explicit textual output for the user.]"
            ):
                # Extract the actual user-agent conversation, ignoring system and tool messages
                conversation_extract = []
                for msg in final_message_history:
                    role = msg.get("role")
                    if role in ("user", "assistant") and isinstance(msg.get("content"), str):
                        content = msg.get("content")
                        if content and not self._is_thought_message_content(content):
                            conversation_extract.append(f"{role.capitalize()}: {content}")
                
                if conversation_extract:
                    accumulated_text_content = "\n".join(conversation_extract)
                else:
                    # Fallback if we couldn't extract meaningful conversation
                    for msg in reversed(final_message_history):
                        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                            accumulated_text_content = f"User requested: {msg.get('content')}"
                            break
            
            # Get the agent's core instructions for context but without formatting
            agent_instructions = self.instructions.strip() if self.instructions else ""
            
            # Get context as clean dictionary
            context_dict = self._get_context_as_dict(run_time_context)
            context_str = ""
            if context_dict:
                context_items = [f"{k}: {v}" for k, v in context_dict.items()]
                context_str = "Context: " + ", ".join(context_items)

            # Construct clean, minimal instructions for the schema generation
            # Let instructor handle the schema formatting - we just provide the essential context
            try:
                structured_output_instance = await self._create_api.async_from_schema(
                    schema=target_output_type,  # type: ignore
                    instructions=f"Generate a structured response based on the following conversation. {agent_instructions if agent_instructions else ''}",
                    prompt=f"{context_str}\n\n{accumulated_text_content}",
                    model=self.settings.model,
                    model_params=self.settings.model_params,
                    iterative=self.settings.iterative_output,
                )
                return cast(OutType, structured_output_instance)
            except Exception as e_final_struct:
                logger.error(
                    f"Agent '{self.name}': Error generating final structured output of type {target_output_type.__name__}: {e_final_struct}. Falling back to string representation of accumulated content.",
                    exc_info=True,
                )
                # Fallback if structuring fails; might not match OutType but provides some output
                return cast(
                    OutType,
                    f"Error creating structured output. Raw content: {accumulated_text_content}",
                )

        logger.warning(
            f"Agent '{self.name}': Unsupported output_type '{target_output_type}' for final generation. Returning raw accumulated content as string."
        )
        return cast(OutType, accumulated_text_content)

    async def _run_or_stream(
        self,
        prompt_input: PromptType,
        model_override: Optional[ModelParam] = None,
        model_params_override: Optional[Params] = None,
        context_strategy_override: Optional[AgentContextStrategy] = None,
        end_strategy_override: Optional[AgentEndStrategy] = None,
        tools_override: Optional[List[Union[Callable[..., Any], AgentTool]]] = None,
        exclude_tools_override: Optional[List[str]] = None,
        force_tool_names_override: Optional[List[str]] = None,
        output_type_override: Optional[Type[OutType]] = None,
        context_override: Optional[CtxType] = None,
        existing_messages_override: Optional[List[Message]] = None,
        is_streaming_run: bool = False,
        save_history_override: Optional[bool] = None,
    ) -> AsyncIterable[
        Union[
            AgentResponse[OutType],
            str,
            Message,
            Dict[str, Any],
            CompletionChunk,
        ]
    ]:
        """
        Core internal method that handles both streaming and non-streaming execution logic.
        """

        # --- Configuration for this specific run ---
        run_config = self.settings
        current_run_model = model_override or run_config.model
        current_run_model_params = {
            **run_config.model_params,
            **(model_params_override or {}),
        }
        current_run_output_type = output_type_override or run_config.output_type
        current_run_end_strategy = end_strategy_override or run_config.end_strategy
        current_run_save_history = (
            save_history_override
            if save_history_override is not None
            else run_config.save_history
        )
        # current_run_context_strategy = context_strategy_override or run_config.context_strategy # Used by _handle_context_update

        # Initialize mutable context for this run (deep copy to avoid modifying agent's base context)
        current_run_context = (
            deepcopy(context_override)
            if context_override is not None
            else deepcopy(self.context)
        )  # type: ignore

        # --- Message History & Response Accumulation Setup ---
        message_history: List[Message] = []
        if existing_messages_override:
            # Handle the case where existing_messages contains nested message structures
            for msg in existing_messages_override:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if isinstance(msg["content"], str):
                        # Normal message format
                        message_history.append(format_messages([msg])[0])
                    elif isinstance(msg["content"], list) and all(
                        is_message(m) for m in msg["content"]
                    ):
                        # Handle the case where content is a list of Message objects
                        # This is the fix for the issue where all messages are combined in one user message
                        for nested_msg in msg["content"]:
                            message_history.append(format_messages([nested_msg])[0])
                    else:
                        # Default case
                        message_history.append(format_messages([msg])[0])
                else:
                    # Default case
                    message_history.append(format_messages([msg])[0])

        # Determine the primary user query string for initial prompt building
        primary_user_query_str: str
        if isinstance(prompt_input, str):
            primary_user_query_str = prompt_input
        elif (
            isinstance(prompt_input, list)
            and prompt_input
            and isinstance(prompt_input[-1], dict)
            and "content" in prompt_input[-1]
        ):  # type: ignore
            last_prompt_content = prompt_input[-1]["content"]  # type: ignore
            primary_user_query_str = (
                last_prompt_content
                if isinstance(last_prompt_content, str)
                else next(
                    (
                        part.get("text", "")
                        for part in last_prompt_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ),
                    "Complex user input",
                )
            )  # type: ignore
        elif isinstance(prompt_input, dict) and "content" in prompt_input:  # type: ignore
            prompt_content = prompt_input["content"]  # type: ignore
            primary_user_query_str = (
                prompt_content
                if isinstance(prompt_content, str)
                else next(
                    (
                        part.get("text", "")
                        for part in prompt_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ),
                    "Complex user input",
                )
            )  # type: ignore
        else:
            primary_user_query_str = (
                "User initiated interaction without a clear text query."
            )

        # Build and add system prompt messages
        system_prompt_messages = self._format_compiled_messages(
            current_run_context, primary_user_query_str
        )  # type: ignore
        message_history.extend(system_prompt_messages)

        # Add the main user prompt input to the message history
        if isinstance(prompt_input, str):
            message_history.append(Message(role="user", content=prompt_input))
        elif (
            isinstance(prompt_input, dict)
            and "role" in prompt_input
            and "content" in prompt_input
        ):  # Single Message dict
            if isinstance(prompt_input["content"], list) and all(
                is_message(m) for m in prompt_input["content"]
            ):
                # If content is a list of Message objects, extract and add them preserving roles
                for nested_msg in prompt_input["content"]:
                    message_history.append(cast(Message, nested_msg))
            else:
                # Normal Message format
                message_history.append(cast(Message, prompt_input))
        elif isinstance(prompt_input, list):  # List of Message dicts
            message_history.extend(cast(List[Message], prompt_input))
        else:  # Coerce any other type to string for user content
            message_history.append(Message(role="user", content=str(prompt_input)))

        intermediate_response_parts: List[Any] = []
        accumulated_text_for_final_output = (
            ""  # Accumulates assistant's textual content
        )

        # --- Initial Context Update (Before Main Loop) ---
        if (
            run_config.update_context_before_response
            and current_run_context is not None
        ):
            logger.info(f"Agent '{self.name}': Performing initial context update.")
            (
                current_run_context,
                context_update_messages,
            ) = await self._handle_context_update(  # type: ignore
                message_history,
                current_run_context,
                user_query=primary_user_query_str,
            )
            if context_update_messages:
                message_history.extend(context_update_messages)
                intermediate_response_parts.extend(context_update_messages)
                if is_streaming_run and run_config.keep_intermediate_steps:
                    for msg_part in context_update_messages:
                        yield msg_part

        # --- Forced Tool Execution ---
        current_active_tools = self._get_current_tools(
            tools_override, exclude_tools_override
        )
        tool_names_to_force_execute = force_tool_names_override or (
            [t.name for t in current_active_tools] if run_config.force_tools else []
        )

        if tool_names_to_force_execute:
            logger.info(
                f"Agent '{self.name}': Forcing execution of tools: {', '.join(tool_names_to_force_execute)}"
            )
            initial_messages_for_tool_forcing = [
                msg for msg in message_history if msg["role"] in ("system", "user")
            ]

            for tool_name_to_force in tool_names_to_force_execute:
                tool_instance = next(
                    (t for t in current_active_tools if t.name == tool_name_to_force),
                    None,
                )
                if not tool_instance:
                    logger.warning(
                        f"Agent '{self.name}': Cannot force tool '{tool_name_to_force}', not found in active tools."
                    )
                    continue

                # Dynamically create Pydantic model for tool arguments
                param_fields_for_model = {
                    prop_name: (
                        _map_json_schema_type_to_python_type(prop_schema),
                        PydanticField(
                            None,
                            description=prop_schema.get("description"),
                        ),
                    )
                    for prop_name, prop_schema in tool_instance.parameters.get(
                        "properties", {}
                    ).items()
                }
                ForcedToolArgsModel = (
                    create_model(
                        f"{tool_instance.name.capitalize()}ForcedArgs",
                        **param_fields_for_model,
                    )
                    if param_fields_for_model
                    else BaseModel
                )  # type: ignore

                forcing_tool_prompt = (
                    f"Based on the user's request and current context, generate the necessary arguments to call the tool: '{tool_instance.name}'.\n"
                    f"Tool Description: {tool_instance.description}\n"
                    # Parameter schema is implicitly handled by `async_from_schema` with `ForcedToolArgsModel`
                )
                tool_arg_generation_messages = initial_messages_for_tool_forcing + [
                    Message(role="user", content=forcing_tool_prompt)
                ]

                try:
                    generated_args_model_instance = (
                        await self._create_api.async_from_schema(
                            schema=ForcedToolArgsModel,
                            prompt=tool_arg_generation_messages,
                            model=current_run_model,
                            model_params=current_run_model_params,
                        )
                    )

                    args_dictionary = (
                        generated_args_model_instance.model_dump(exclude_none=True)
                        if generated_args_model_instance and param_fields_for_model
                        else {}
                    )
                    args_json_str = json.dumps(args_dictionary)

                    forced_tool_call_object: ToolCall = {
                        "id": f"forced_tool_{tool_instance.name}_{uuid.uuid4().hex[:6]}",
                        "type": "function",  # Assuming all agent tools are function types
                        "function": {
                            "name": tool_instance.name,
                            "arguments": args_json_str,
                        },
                    }
                    # Add assistant message indicating the forced tool call
                    if run_config.keep_intermediate_steps:
                        message_history.append(
                            Message(
                                role="assistant",
                                tool_calls=[forced_tool_call_object],
                            )
                        )  # type: ignore
                        forced_tool_part = {
                            "type": "forced_tool_call_generated_args",
                            "tool_name": tool_instance.name,
                            "args": args_dictionary,
                        }
                        intermediate_response_parts.append(forced_tool_part)
                        if is_streaming_run:
                            yield forced_tool_part

                    tool_result_messages = await self._handle_tool_calls(
                        [forced_tool_call_object],
                        current_run_context,
                        [tool_instance],
                    )  # type: ignore
                    message_history.extend(tool_result_messages)
                    intermediate_response_parts.extend(tool_result_messages)
                    if is_streaming_run and run_config.keep_intermediate_steps:
                        for tool_res_msg in tool_result_messages:
                            yield tool_res_msg
                except Exception as e_force_tool_exec:
                    logger.error(
                        f"Agent '{self.name}': Error during forced execution of tool {tool_instance.name}: {e_force_tool_exec}",
                        exc_info=True,
                    )
                    error_message_content = f"[Error during forced execution of tool {tool_instance.name}: {e_force_tool_exec}]"
                    message_history.append(
                        Message(role="assistant", content=error_message_content)
                    )  # Or a tool error message
                    error_part = {
                        "type": "error",
                        "source": "forced_tool_execution",
                        "content": error_message_content,
                    }
                    intermediate_response_parts.append(error_part)
                    if is_streaming_run:
                        yield error_part

        # --- Main Agent Execution Loop ---
        current_step_count = 0
        agent_operation_ended = False

        # Recalculate active tools for the loop, in case forced execution changed context or requirements
        loop_active_tools = self._get_current_tools(
            tools_override, exclude_tools_override
        )

        # Determine if finalize_response tool should be parameter-less for this run
        parameterless_finalize_for_run = False
        if (
            isinstance(current_run_output_type, type)
            and issubclass(current_run_output_type, BaseModel)
            and current_run_output_type not in [str, AgentMessageSchema]
        ):  # type: ignore
            parameterless_finalize_for_run = True
            logger.info(
                f"Agent '{self.name}': Output type is BaseModel ({current_run_output_type.__name__}), configuring finalize_response to be parameter-less."
            )

        # If using "full" end strategy, always add the finalize_response tool
        if current_run_end_strategy == "full":
            finalize_response_tool = self._get_end_tool(
                parameterless_finalize=parameterless_finalize_for_run
            )
            # Check if a tool with this name already exists
            if not any(
                t.name == finalize_response_tool.name for t in loop_active_tools
            ):
                logger.debug(
                    f"Agent '{self.name}': Adding finalize_response tool for full end strategy"
                )
                loop_active_tools.append(finalize_response_tool)

        while current_step_count < run_config.max_steps and not agent_operation_ended:
            current_step_count += 1
            logger.info(
                f"Agent '{self.name}' starting main execution step {current_step_count}"
            )

            # Filter messages for LLM for this iteration
            # This snapshot will be used as the base for planning and the main LLM call for this step.
            # messages_for_llm_this_step = [msg for msg in message_history if run_config.keep_intermediate_steps or msg["role"] in ("system", "user", "assistant", "tool")] # OLD LINE

            # New logic for filtering messages for LLM based on keep_intermediate_steps and show_context
            messages_for_llm_this_step = []
            for msg_from_history in message_history:
                msg_content = msg_from_history.get("content")
                is_any_thought = Agent._is_thought_message_content(msg_content)

                include_this_msg = False
                if not is_any_thought:
                    # Always include non-thought messages (system, user, regular assistant, tool results).
                    # Assumes tool calls/results are not themselves prefixed as thoughts.
                    include_this_msg = True
                else:  # It IS a thought message
                    if run_config.keep_intermediate_steps:
                        # If we are keeping intermediate steps, then all thoughts are candidates to be shown to LLM.
                        # Now, apply the specific filter for context update thoughts if it's that type of thought.
                        is_context_update_thought = isinstance(
                            msg_content, str
                        ) and msg_content.startswith("[Context Update Thought]:")

                        if is_context_update_thought:
                            if run_config.show_context:
                                include_this_msg = True  # Include context thought
                            # else: False, Exclude context thought due to the new flag being False
                        else:
                            # For other thoughts (Planning, Reflection, End Decision), include them if keep_intermediate_steps is True
                            include_this_msg = True
                    # else: False, If not keeping intermediate steps, EXCLUDE ALL thoughts from LLM history

                if include_this_msg:
                    messages_for_llm_this_step.append(msg_from_history)

            # Add special instruction for first step with 'full' end strategy for potential early exit
            if current_step_count == 1 and current_run_end_strategy == "full":
                early_exit_instruction = "System Note (First Step Only): If the initial query is fully addressable now, you may use the 'finalize_response' tool immediately. Otherwise, proceed with planning and normal execution."
                # Check if this exact instruction isn't already the last system message to avoid duplicates in case of re-runs/complex scenarios
                already_present = False
                if (
                    messages_for_llm_this_step
                    and messages_for_llm_this_step[-1]["role"] == "system"
                    and messages_for_llm_this_step[-1]["content"]
                    == early_exit_instruction
                ):
                    already_present = True
                if not already_present:
                    messages_for_llm_this_step.append(
                        Message(role="system", content=early_exit_instruction)
                    )
                    logger.debug(
                        f"Agent '{self.name}': Added early exit instruction for step 1."
                    )

            # 1. Planning Step
            # Planning uses the messages prepared for this step, including the potential early exit instruction.
            (
                plan_text_content,
                planning_messages,
            ) = await self._handle_planning_step(
                messages_for_llm_this_step, current_run_context
            )  # type: ignore
            if plan_text_content:
                intermediate_response_parts.append(
                    {"type": "plan", "content": plan_text_content}
                )
            if planning_messages:
                message_history.extend(planning_messages)
                if is_streaming_run and run_config.keep_intermediate_steps:
                    for msg_part in planning_messages:
                        yield msg_part

            # 2. Main LLM Call for response or tool usage
            llm_call_parameters: Dict[str, Any] = {
                "model": current_run_model,
                "messages": messages_for_llm_this_step,  # Use the (potentially modified) messages for this step
                "params": current_run_model_params,  # type: ignore
                "stream": is_streaming_run,
            }
            if loop_active_tools:  # Only add tools if there are any
                llm_call_parameters["tools"] = [t.to_tool() for t in loop_active_tools]
                llm_call_parameters["tool_choice"] = "auto"

            tool_calls_generated_this_step: List[ToolCall] = []
            assistant_content_generated_this_step = ""

            try:
                if is_streaming_run:
                    # Extract messages from call parameters and convert other parameters
                    # to be compatible with Create.from_prompt
                    messages = llm_call_parameters.pop("messages")
                    model = llm_call_parameters.pop("model")
                    model_params = (
                        llm_call_parameters.pop("params", {})
                        if "params" in llm_call_parameters
                        else {}
                    )

                    # Add tools to model_params if they exist
                    if "tools" in llm_call_parameters:
                        tools = llm_call_parameters.pop("tools")
                        tool_choice = llm_call_parameters.pop("tool_choice", "auto")
                        model_params["tools"] = tools
                        model_params["tool_choice"] = tool_choice

                    # Create the proper parameter structure for from_prompt
                    prompt_params = {
                        "prompt": messages,
                        "model": model,
                        "model_params": model_params,
                        "stream": True,
                    }

                    # Add any remaining parameters
                    for key, value in llm_call_parameters.items():
                        if key not in prompt_params:
                            prompt_params[key] = value

                    llm_stream_chunks = self._create_api.from_prompt(**prompt_params)  # type: ignore
                    streamed_tool_calls_buffer: Dict[
                        int, Dict[str, Any]
                    ] = {}  # Maps index to partial tool call data

                    # Handle different stream response types
                    if isinstance(llm_stream_chunks, str):
                        # If we get a string directly instead of a stream
                        assistant_content_generated_this_step = llm_stream_chunks
                        accumulated_text_for_final_output += (
                            assistant_content_generated_this_step
                        )
                        message_history.append(
                            Message(
                                role="assistant",
                                content=assistant_content_generated_this_step,
                            )
                        )
                        yield assistant_content_generated_this_step  # Yield the string directly to caller
                    else:
                        # Normal streaming chunks case
                        async for stream_chunk in llm_stream_chunks:  # type: ignore
                            yield stream_chunk  # Yield raw LLM chunk to the caller
                            if not isinstance(stream_chunk, CompletionChunk):
                                continue

                            choice_data = (
                                stream_chunk.choices[0]
                                if stream_chunk.choices
                                else None
                            )
                            if choice_data and choice_data.delta:
                                delta = choice_data.delta
                                if delta.content:
                                    assistant_content_generated_this_step += (
                                        delta.content
                                    )
                                    accumulated_text_for_final_output += (
                                        delta.content
                                    )  # Accumulate for final output construction
                                if delta.tool_calls:
                                    for tc_delta_item in delta.tool_calls:
                                        tool_call_idx = tc_delta_item.index
                                        if (
                                            tool_call_idx
                                            not in streamed_tool_calls_buffer
                                        ):
                                            streamed_tool_calls_buffer[
                                                tool_call_idx
                                            ] = {
                                                "id": None,
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }

                                        current_buffered_tc = (
                                            streamed_tool_calls_buffer[tool_call_idx]
                                        )
                                        if tc_delta_item.id:
                                            current_buffered_tc["id"] = tc_delta_item.id
                                        if tc_delta_item.type:
                                            current_buffered_tc["type"] = (
                                                tc_delta_item.type
                                            )  # type: ignore
                                        if tc_delta_item.function:
                                            if tc_delta_item.function.name:
                                                current_buffered_tc["function"][
                                                    "name"
                                                ] += tc_delta_item.function.name
                                            if tc_delta_item.function.arguments:
                                                current_buffered_tc["function"][
                                                    "arguments"
                                                ] += tc_delta_item.function.arguments

                            if (
                                choice_data and choice_data.finish_reason
                            ):  # A choice in the stream has finished
                                if (
                                    choice_data.finish_reason == "tool_calls"
                                    or streamed_tool_calls_buffer
                                ):
                                    tool_calls_generated_this_step.extend(
                                        cast(ToolCall, tc_data)
                                        for tc_data in streamed_tool_calls_buffer.values()
                                        if tc_data.get("id")
                                    )
                                    streamed_tool_calls_buffer = {}  # Reset buffer for this choice

                    # After stream, if text content was generated, add it as an assistant message
                    if assistant_content_generated_this_step:
                        message_history.append(
                            Message(
                                role="assistant",
                                content=assistant_content_generated_this_step,
                            )
                        )
                        intermediate_response_parts.append(
                            {
                                "type": "llm_response_fragment",
                                "content": assistant_content_generated_this_step,
                            }
                        )

                else:  # Non-streaming LLM call
                    # Extract messages from call parameters and convert other parameters
                    # to be compatible with Create.async_from_prompt
                    messages = llm_call_parameters.pop("messages")
                    model = llm_call_parameters.pop("model")
                    model_params = (
                        llm_call_parameters.pop("params", {})
                        if "params" in llm_call_parameters
                        else {}
                    )

                    # Add tools to model_params if they exist
                    if "tools" in llm_call_parameters:
                        tools = llm_call_parameters.pop("tools")
                        tool_choice = llm_call_parameters.pop("tool_choice", "auto")
                        model_params["tools"] = tools
                        model_params["tool_choice"] = tool_choice

                    # Create the proper parameter structure for async_from_prompt
                    prompt_params = {
                        "prompt": messages,
                        "model": model,
                        "model_params": model_params,
                        "stream": False,
                    }

                    # Add any remaining parameters
                    for key, value in llm_call_parameters.items():
                        if key not in prompt_params:
                            prompt_params[key] = value

                    # Direct call to LiteLLM for non-streaming to get full Completion object
                    if not self._create_api.CLIENT_DEPS.is_litellm_initialized:
                        self._create_api.CLIENT_DEPS.initialize_litellm()

                    direct_llm_call_args: Dict[str, Any] = {
                        "model": current_run_model,
                        "messages": messages_for_llm_this_step,  # Use the (potentially modified) messages for this step
                        "stream": False,  # Explicitly False for this path
                    }
                    if loop_active_tools:
                        direct_llm_call_args["tools"] = [
                            t.to_tool() for t in loop_active_tools
                        ]
                        # Set tool_choice, default to "auto" if not specified deeper in params
                        # For simplicity, using "auto". A more complex setup might pull tool_choice from current_run_model_params.
                        direct_llm_call_args["tool_choice"] = (
                            current_run_model_params.get("tool_choice", "auto")
                        )

                    # Add other parameters from current_run_model_params, excluding those handled above
                    # LiteLLM will drop unknown params if litellm.drop_params = True
                    additional_params = {
                        k: v
                        for k, v in current_run_model_params.items()
                        if k not in ["tools", "tool_choice"] and v is not None
                    }
                    direct_llm_call_args.update(additional_params)

                    logger.info(
                        f"Agent '{self.name}': Calling LiteLLM directly for non-streaming completion. Args: {list(direct_llm_call_args.keys())}"
                    )
                    llm_completion_response: Completion = (
                        await self._create_api.CLIENT_DEPS.completion_async(
                            **direct_llm_call_args
                        )
                    )  # type: ignore
                    intermediate_response_parts.append(
                        llm_completion_response
                    )  # Store raw completion

                    # Handle the case where the response is a string (direct content)
                    if isinstance(llm_completion_response, str):
                        assistant_content_generated_this_step = llm_completion_response
                        accumulated_text_for_final_output += (
                            "\n" if accumulated_text_for_final_output else ""
                        ) + assistant_content_generated_this_step
                        message_history.append(
                            Message(
                                role="assistant",
                                content=assistant_content_generated_this_step,
                            )
                        )
                    # Handle the case where it's a Completion object with choices
                    elif (
                        hasattr(llm_completion_response, "choices")
                        and llm_completion_response.choices
                    ):
                        assistant_msg_from_completion = (
                            llm_completion_response.choices[0].message
                            if llm_completion_response.choices
                            else None
                        )
                        if assistant_msg_from_completion:
                            message_history.append(
                                cast(
                                    Message,
                                    assistant_msg_from_completion.model_dump(
                                        exclude_none=True
                                    ),
                                )
                            )

                            # Prioritize content from finalize_response tool call arguments if present
                            finalize_summary = None
                            if assistant_msg_from_completion.tool_calls:
                                tool_calls_generated_this_step = [
                                    cast(ToolCall, tc.model_dump())
                                    for tc in assistant_msg_from_completion.tool_calls
                                ]
                                for tc in tool_calls_generated_this_step:
                                    if tc["function"]["name"] == "finalize_response":
                                        # Only try to parse args if not parameter-less
                                        should_parse_args_for_finalize = not (
                                            parameterless_finalize_for_run
                                        )
                                        if should_parse_args_for_finalize:
                                            try:
                                                args = json.loads(
                                                    tc["function"]["arguments"]
                                                )
                                                finalize_summary = args.get(
                                                    "summary_and_final_thoughts"
                                                )
                                            except json.JSONDecodeError:
                                                logger.warning(
                                                    f"Agent '{self.name}': Could not parse arguments for finalize_response (summary expected): {tc['function']['arguments']}"
                                                )
                                        else:
                                            logger.info(
                                                f"Agent '{self.name}': finalize_response called (parameter-less). No summary to extract from args."
                                            )
                                        break  # Found finalize_response

                            if finalize_summary:  # This will only be true if it was not parameter-less and summary was found
                                assistant_content_generated_this_step = finalize_summary
                                accumulated_text_for_final_output += (
                                    "\n" if accumulated_text_for_final_output else ""
                                ) + assistant_content_generated_this_step
                            elif assistant_msg_from_completion.content:  # Fallback to direct content if no finalize_summary or other content
                                assistant_content_generated_this_step = str(
                                    assistant_msg_from_completion.content
                                )
                                accumulated_text_for_final_output += (
                                    "\n" if accumulated_text_for_final_output else ""
                                ) + assistant_content_generated_this_step
                            # Note: tool_calls_generated_this_step is already populated if tool_calls exist
                    else:
                        logger.warning(
                            f"Agent '{self.name}': LLM call in step {current_step_count} returned an unexpected response type: {type(llm_completion_response)}"
                        )
                        message_history.append(
                            Message(
                                role="assistant",
                                content="[LLM generated no message content in this step]",
                            )
                        )

                # Process any tool calls generated in this step (applies to both streaming and non-streaming)
                if tool_calls_generated_this_step:
                    # For non-streaming, ensure the assistant message includes these tool_calls if keep_intermediate_steps
                    if (
                        not is_streaming_run
                        and run_config.keep_intermediate_steps
                        and message_history[-1]["role"] == "assistant"
                    ):
                        message_history[-1]["tool_calls"] = (
                            tool_calls_generated_this_step
                        )

                    tool_result_messages = await self._handle_tool_calls(
                        tool_calls_generated_this_step,
                        current_run_context,
                        loop_active_tools,
                    )  # type: ignore
                    message_history.extend(tool_result_messages)
                    intermediate_response_parts.extend(tool_result_messages)
                    if is_streaming_run and run_config.keep_intermediate_steps:
                        for tool_res_msg in tool_result_messages:
                            yield tool_res_msg

                    # Check for "finalize_response" tool call for 'full' end strategy
                    if current_run_end_strategy == "full" and any(
                        tc["function"]["name"] == "finalize_response"
                        for tc in tool_calls_generated_this_step
                    ):
                        agent_operation_ended = True
                        logger.info(
                            f"Agent '{self.name}' ending operation via 'finalize_response' tool call."
                        )

                    # If we're using full end strategy but have no tools or no finalize_response tool,
                    # end after first complete response to avoid unnecessary steps
                    if (
                        current_run_end_strategy == "full"
                        and (
                            not loop_active_tools
                            or not any(
                                t.name == "finalize_response" for t in loop_active_tools
                            )
                        )
                        and assistant_content_generated_this_step
                        and current_step_count == 1
                    ):
                        logger.info(
                            f"Agent '{self.name}' ending operation after first response since no finalize_response tool is available"
                        )
                        agent_operation_ended = True

            except Exception as e_llm_step:
                logger.error(
                    f"Agent '{self.name}': Error during LLM interaction in step {current_step_count}: {e_llm_step}",
                    exc_info=True,
                )
                message_history.append(
                    Message(
                        role="assistant",
                        content=f"[Error in LLM step: {e_llm_step}]",
                    )
                )
                agent_operation_ended = True  # Stop agent on critical LLM error
                if is_streaming_run:
                    yield {"error": f"LLM step failed: {e_llm_step}"}

            logger.debug(
                f"Agent '{self.name}' at step {current_step_count}, agent_operation_ended status before break check: {agent_operation_ended}"
            )
            if agent_operation_ended:
                break  # Exit main loop if error or finalize_response called

            # 3. Reflection Step
            content_for_reflection = assistant_content_generated_this_step or (
                message_history[-1]["content"]
                if message_history and message_history[-1]["role"] == "tool"
                else ""
            )  # type: ignore
            if isinstance(
                content_for_reflection, list
            ):  # Handle complex content (e.g. from tool)
                content_for_reflection = next(
                    (
                        p["text"]
                        for p in content_for_reflection
                        if isinstance(p, dict) and p.get("type") == "text"
                    ),
                    "Complex content reflected.",
                )  # type: ignore

            (
                reflection_text_content,
                reflection_messages,
            ) = await self._handle_reflection_step(
                message_history,
                str(content_for_reflection),
                current_run_context,
            )  # type: ignore
            if reflection_text_content:
                intermediate_response_parts.append(
                    {
                        "type": "reflection",
                        "content": reflection_text_content,
                    }
                )
            if reflection_messages:
                message_history.extend(reflection_messages)
                if is_streaming_run and run_config.keep_intermediate_steps:
                    for msg_part in reflection_messages:
                        yield msg_part

            # 4. Check End Conditions (for "selective" strategy)
            if not agent_operation_ended and current_run_end_strategy == "selective":
                if await self._should_agent_end(
                    message_history,
                    current_run_context,
                    current_run_end_strategy,
                    accumulated_text_for_final_output,
                ):  # type: ignore
                    agent_operation_ended = True
                    logger.info(
                        f"Agent '{self.name}' ending operation based on selective decision."
                    )

            if agent_operation_ended:
                break  # Exit main loop if ended by selective strategy

        # --- End of Main Agent Loop ---

        # --- Final Context Update (After Main Loop) ---
        if run_config.update_context_after_response and current_run_context is not None:
            logger.info(f"Agent '{self.name}': Performing final context update.")
            final_content_for_context_update = accumulated_text_for_final_output or (
                message_history[-1].get("content", "") if message_history else ""
            )  # type: ignore
            if isinstance(
                final_content_for_context_update, list
            ):  # Handle complex content
                final_content_for_context_update = next(
                    (
                        p["text"]
                        for p in final_content_for_context_update
                        if isinstance(p, dict) and p.get("type") == "text"
                    ),
                    "Final complex content",
                )  # type: ignore

            (
                current_run_context,
                final_context_update_messages,
            ) = await self._handle_context_update(  # type: ignore
                message_history,
                current_run_context,
                user_query=f"Based on final agent response: {str(final_content_for_context_update)[:200]}...",
            )
            if (
                final_context_update_messages
            ):  # Typically not streamed post-loop but recorded
                message_history.extend(final_context_update_messages)
                intermediate_response_parts.extend(final_context_update_messages)

        # --- Construct and Yield Final AgentResponse ---
        final_output_object: Optional[OutType] = None
        if agent_operation_ended or current_step_count >= run_config.max_steps:
            # Ensure there's some textual content if none was explicitly accumulated (e.g., if agent only used tools and planning)
            if not accumulated_text_for_final_output and message_history:
                for msg_item in reversed(
                    message_history
                ):  # Find last meaningful assistant text
                    if (
                        msg_item["role"] == "assistant"
                        and msg_item.get("content")
                        and not str(msg_item.get("content", "")).startswith("[")
                    ):  # Avoid internal thoughts
                        accumulated_text_for_final_output = str(
                            msg_item.get("content", "")
                        )
                        break
            if not accumulated_text_for_final_output:  # Still nothing?
                accumulated_text_for_final_output = "[Agent concluded its operation without generating explicit textual output for the user.]"

            final_output_object = await self._generate_final_output(
                message_history,
                current_run_context,
                current_run_output_type,
                accumulated_text_for_final_output,
            )  # type: ignore

            if is_streaming_run and final_output_object is not None:
                # If streaming, and the final structured output is different from the raw text stream, yield it.
                if (
                    isinstance(final_output_object, str)
                    and current_run_output_type == str
                    and final_output_object == accumulated_text_for_final_output
                ):
                    pass  # String content was already streamed as chunks
                elif (
                    isinstance(final_output_object, AgentMessageSchema)
                    and final_output_object.content != accumulated_text_for_final_output
                ):
                    # If AgentMessageSchema re-processed/summarized content
                    yield {
                        "final_structured_message_content": final_output_object.content
                    }
                elif isinstance(final_output_object, BaseModel) and not isinstance(
                    final_output_object, AgentMessageSchema
                ):
                    # Any other Pydantic model
                    yield {
                        "final_structured_object_json": final_output_object.model_dump_json()
                    }

        final_agent_response_obj = AgentResponse(
            parts=intermediate_response_parts,
            output=final_output_object,
            history=message_history if current_run_save_history else [],
        )

        if is_streaming_run:
            yield final_agent_response_obj  # Yield the comprehensive AgentResponse object at the very end of the stream
        else:
            # For non-streaming, this generator yields the single AgentResponse object and then stops.
            yield final_agent_response_obj

    async def async_run(
        self,
        prompt: PromptType,
        model: Optional[ModelParam] = None,
        model_params: Optional[Params] = None,
        context_strategy: Optional[AgentContextStrategy] = None,
        end_strategy: Optional[AgentEndStrategy] = None,
        tools: Optional[List[Union[Callable[..., Any], AgentTool]]] = None,
        exclude_tools: Optional[List[str]] = None,
        force_tool: Optional[List[str]] = None,
        output_type: Optional[Type[OutType]] = None,
        context: Optional[CtxType] = None,
        existing_messages: Optional[List[Message]] = None,
        save_history: Optional[bool] = None,
    ) -> AgentResponse[OutType]:
        """
        Executes the agent with the given prompt and settings, returning a single comprehensive response.

        This method orchestrates the agent's lifecycle including context updates,
        planning, tool execution, reflection, and final output generation in a non-streaming manner.

        Args:
            prompt: The primary input/query for the agent. Can be a string, a `Message` dict, or a list of `Message` dicts.
            model: (Optional) Override the agent's default language model for this run.
            model_params: (Optional) Override or supplement the agent's default model parameters.
            context_strategy: (Optional) Override the agent's context update strategy for this run.
            end_strategy: (Optional) Override the agent's response termination strategy.
            tools: (Optional) A list of tools to make available for this run, potentially overriding agent's defaults.
            exclude_tools: (Optional) A list of tool names to exclude from use during this run.
            force_tool: (Optional) A list of tool names to force execution of at the beginning of the run.
            output_type: (Optional) Override the agent's expected final output type for this run.
            context: (Optional) Provide or override the agent's context for this specific run.
            existing_messages: (Optional) A list of messages to prepend to the conversation history for this run.
            save_history: (Optional) Override the agent's save_history setting for this run.

        Returns:
            An `AgentResponse` object containing all intermediate parts, the final output,
            and the complete message history of the run.
        """
        final_response_from_run: Optional[AgentResponse[OutType]] = None
        async for item_from_core_logic in self._run_or_stream(
            prompt_input=prompt,
            model_override=model,
            model_params_override=model_params,
            context_strategy_override=context_strategy,
            end_strategy_override=end_strategy,
            tools_override=tools,
            exclude_tools_override=exclude_tools,
            force_tool_names_override=force_tool,
            output_type_override=output_type,
            context_override=context,
            existing_messages_override=existing_messages,
            is_streaming_run=False,  # Key difference for non-streaming behavior
            save_history_override=save_history,
        ):
            if isinstance(item_from_core_logic, AgentResponse):
                final_response_from_run = item_from_core_logic  # type: ignore
                break  # For non-streaming, _run_or_stream yields AgentResponse once and finishes

        if final_response_from_run is None:
            # This path should ideally not be hit if _run_or_stream is correct for non-streaming
            logger.error(
                f"Agent '{self.name}' non-streaming run unexpectedly did not produce a final AgentResponse."
            )
            # Provide a minimal empty response

            message_history = existing_messages if existing_messages else []

            if not existing_messages:
                existing_messages = (
                    message_history if "message_history" in locals() else message_history
                )

            return AgentResponse(
                output=None,
                history=existing_messages if existing_messages else [],
            )  # type: ignore
        return final_response_from_run

    def run(
        self,
        prompt: PromptType,
        model: Optional[ModelParam] = None,
        model_params: Optional[Params] = None,
        context_strategy: Optional[AgentContextStrategy] = None,
        end_strategy: Optional[AgentEndStrategy] = None,
        tools: Optional[List[Union[Callable[..., Any], AgentTool]]] = None,
        exclude_tools: Optional[List[str]] = None,
        force_tool: Optional[List[str]] = None,
        output_type: Optional[Type[OutType]] = None,
        context: Optional[CtxType] = None,
        existing_messages: Optional[List[Message]] = None,
        save_history: Optional[bool] = None,
    ) -> AgentResponse[OutType]:
        """
        Executes the agent with the given prompt and settings, returning a single comprehensive response.

        This method orchestrates the agent's lifecycle including context updates,
        planning, tool execution, reflection, and final output generation in a non-streaming manner.

        Args:
            prompt: The primary input/query for the agent. Can be a string, a `Message` dict, or a list of `Message` dicts.
            model: (Optional) Override the agent's default language model for this run.
            model_params: (Optional) Override or supplement the agent's default model parameters.
            context_strategy: (Optional) Override the agent's context update strategy for this run.
            end_strategy: (Optional) Override the agent's response termination strategy.
            tools: (Optional) A list of tools to make available for this run, potentially overriding agent's defaults.
            exclude_tools: (Optional) A list of tool names to exclude from use during this run.
            force_tool: (Optional) A list of tool names to force execution of at the beginning of the run.
            output_type: (Optional) Override the agent's expected final output type for this run.
            context: (Optional) Provide or override the agent's context for this specific run.
            existing_messages: (Optional) A list of messages to prepend to the conversation history for this run.
            save_history: (Optional) Override the agent's save_history setting for this run.

        Returns:
            An `AgentResponse` object containing all intermediate parts, the final output,
            and the complete message history of the run.
        """
        return asyncio.run(
            self.async_run(
                prompt=prompt,
                model=model,
                model_params=model_params,
                context_strategy=context_strategy,
                end_strategy=end_strategy,
                tools=tools,
                exclude_tools=exclude_tools,
                force_tool=force_tool,
                output_type=output_type,
                context=context,
                existing_messages=existing_messages,
                save_history=save_history,
            )
        )

    async def stream(
        self,
        prompt: PromptType,
        model: Optional[ModelParam] = None,
        model_params: Optional[Params] = None,
        context_strategy: Optional[AgentContextStrategy] = None,
        end_strategy: Optional[AgentEndStrategy] = None,
        tools: Optional[List[Union[Callable[..., Any], AgentTool]]] = None,
        exclude_tools: Optional[List[str]] = None,
        force_tool: Optional[List[str]] = None,
        context: Optional[CtxType] = None,
        existing_messages: Optional[List[Message]] = None,
        output_type_override: Optional[Type[OutType]] = None,
        save_history: Optional[bool] = None,
    ) -> AsyncIterable[
        Union[
            AgentResponse[OutType],
            str,
            Message,
            Dict[str, Any],
            CompletionChunk,
        ]
    ]:
        """
        Executes the agent and streams its responses and intermediate processing steps.

        Yields various types of information as the agent processes:
        - `CompletionChunk`: Raw chunks from the language model during text generation.
        - `Message`: Intermediate messages (e.g., planning, reflection, tool calls/results)
                     if `AgentSettings.keep_intermediate_steps` is True.
        - `Dict[str, Any]`: Structured information about certain events (e.g., forced tool calls, final structured object).
        - `AgentResponse[OutType]`: The final, comprehensive `AgentResponse` object yielded at the very end of the stream.

        Args:
            prompt: The primary input/query for the agent.
            model: (Optional) Override model for this run.
            model_params: (Optional) Override model parameters.
            context_strategy: (Optional) Override context strategy.
            end_strategy: (Optional) Override end strategy.
            tools: (Optional) Override tools for this run.
            exclude_tools: (Optional) Exclude specific tools.
            force_tool: (Optional) Force specific tools to run.
            context: (Optional) Override context for this run.
            existing_messages: (Optional) Prepend messages to history.
            output_type_override: (Optional) Specify the type for the `output` field in the final `AgentResponse`
                                     object that is yielded at the end of the stream. If not provided,
                                     the agent's default `output_type` is used.
            save_history: (Optional) Override the agent's save_history setting for this run.

        Returns:
            An asynchronous iterable yielding parts of the agent's execution process.
        """
        # The `output_type_override` here primarily influences the `output` field of the
        # *final* AgentResponse object yielded by the stream.
        # The intermediate LLM calls for text generation within the stream will still produce text chunks.
        final_object_output_type = output_type_override or self.settings.output_type

        async for item_from_core_logic in self._run_or_stream(
            prompt_input=prompt,
            model_override=model,
            model_params_override=model_params,
            context_strategy_override=context_strategy,
            end_strategy_override=end_strategy,
            tools_override=tools,
            exclude_tools_override=exclude_tools,
            force_tool_names_override=force_tool,
            output_type_override=final_object_output_type,  # For the final AgentResponse.output
            context_override=context,
            existing_messages_override=existing_messages,
            is_streaming_run=True,  # Key for streaming behavior
            save_history_override=save_history,
        ):
            yield item_from_core_logic  # type: ignore

    def __str__(self):
        return f"Agent(name={self.name}, instructions={self.instructions}, context={self.context}, output_type={self.settings.output_type})"

    def __repr__(self):
        return f"Agent(name={self.name}, instructions={self.instructions}, context={self.context}, settings={self.settings})"

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.instructions == other.instructions
            and self.context == other.context
            and self.settings == other.settings
        )

    def __hash__(self):
        return hash((self.name, self.instructions, self.context, self.settings))


# -----------------------------------------------------------------------------
