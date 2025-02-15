# 💭 chatspec

Simple types &amp; utilities built for the OpenAI Chat Completions API specification.

[![](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 📦 Installation

```bash
pip install chatspec
```

#  📚 Documentation & Examples

`promptspec` provides a 'prethora' (as many as would actually be useful) of types, models & methods for validating, converting and augmenting objects used in the OpenAI chat completions API specification, a `State` class for managing messages threads for agentic application, as well as a `MockAI` client & `mock_completion()` method for creating mock llm responses quickly. I use [Instructor](https://github.com/instructor-ai/instructor) for all of my structured outputs, so `Pydantic` is a core part of this library. The point of this library is to provide a common interface for methods that I have found myself needing to replicate across multiple projects.

---

### 📝 Table of Contents

- [Mock Completions](#-mock-completions)
- [Chat Messages](#-chat-messages)
  - [Instance Checking & Validation](#instance-checking--validation-of-messages)
  - [Validation & Normalization](#validation--normalization-of-messages--system-prompts)
  - [Message Type Creation](#convert-or-create-specific-message-types)
- [Tools & Tool Calling](#-tools--tool-calling)
  - [Instance Checking & Validation](#instance-checking--validation-of-tools)
  - [Function Conversion](#convert-python-functions-pydantic-models-dataclasses--more-to-tools)
  - [Tool Call Interaction](#interacting-with-tool-calls-in-completions--executing-tools)
- [Completion Responses & Streams](#-completion-responses--streams)
  - [Instance Checking & Validation](#instance-checking--validation-of-completions--streams)
  - [Stream Passthrough & Methods](#the-stream-passthrough--stream-specific-methods)
- [Types & Parameters](#-types--parameters)
- [State Manager (*For Chatbots & Agentic Applications*)](#-state-manager-for-chatbots--agentic-applications)
- [Pydantic Models & Structured Outputs](#-pydantic-models--structured-outputs)
- [Markdown Formatting](#-markdown-formatting)

---

## 🥸 Mock Completions

`chatspec` provides both `async` & `synchronous` mock completion methods with support for streaming,
simulated tool calls, all with proper typing and overloading for response types.

```python
# create a mock streamed completion
stream = mock_completion(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    model="gpt-4o-mini",
    stream=True,
)
# chatspec provides a helper method to easily print streams
# everything is typed & overloaded properly for streamed & non-streamed responses
# its like its a real client!
chatspec.print_stream(stream)
# >>> Mock response to: Hello, how are you?

# you can also simulate tool calls
# this also works both for streamed & non-streamed responses
mock_completion(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="gpt-4o-mini",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Get the capital of France",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                },
            },
        }
    ],
)
```

<details>
<summary>Output</summary>

```python
Completion(
    id='85aa7221-54db-4ee1-90a4-8b467c90bd02',
    choices=[
        Choice(
            message=CompletionMessage(
                role='assistant',
                content='Mock response to: What is the capital of France?',
                name=None,
                function_call=None,
                tool_calls=[
                    CompletionToolCall(
                        id='17825e39-a2eb-430f-9f2a-7db467d1ec16',
                        type='function',
                        function=CompletionFunction(name='get_capital', arguments='{"city": "mock_string"}')
                    )
                ],
                tool_call_id=None
            ),
            finish_reason='tool_calls',
            index=0,
            logprobs=None
        )
    ],
    created=1739599399,
    model='gpt-4o-mini',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=None
)
```

</details>

## 💬 Chat Messages

`chatspec` provides a variety of utility when working with `Message` objects. These methods can be used for validation, conversion, creation of 
specific message types & more.

#### Instance Checking & Validation of `Messages`

```python
import chatspec

# easily check if an object is a valid message
chatspec.is_message(
    {
        "role" : "assistant",
        "content" : "Hello, how are you?",
        "tool_calls" : [
            {
                "id" : "123",
                "function" : {"name" : "my_function", "arguments" : "{}"}
            }
        ]
    }
)
# >>> True

chatspec.is_message(
    # 'context' key is invalid
    {"role": "user", "context": "Hello, how are you?"}
)
# >>> False
```

#### Validation & Normalization of `Messages` & `System Prompts`

```python
import chatspec

# easily validate & normalize into chat message threads
chatspec.normalize_messages("Hello!")
# >>> [{"role": "user", "content": "Hello!"}]
chatspec.normalize_messages({
    "role" : "system",
    "content" : "You are a helpful assistant."
})
# >>> [{"role": "system", "content": "You are a helpful assistant."}]

# use the `normalize_system_prompt` method to 'normalize' a thread for use with a singular system
# prompt.
# this method automatically formats the entire thread so the system prompt is always the first message.
chatspec.normalize_system_prompt(
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello!"}
    ],
    system_prompt = "You are a helpful assistant."
)
# >>> [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello!"}]

chatspec.normalize_system_prompt(
    [
        {"role": "user", "content": "Hello!"},
        {"role": "system", "content": "You are a helpful"},
        {"role": "system", "content": "assistant."}
    ],
)
# >>> [[{'role': 'system', 'content': 'You are a helpful\nassistant.'}, {'role': 'user', 'content': 'Hello!'}]
```

#### Convert or Create Specific `Message` Types

Using one of the various `create_*_message` methods, you can easily convert to or create specific `Message` types.

```python
import chatspec

# create a tool message from a completion response
# and a function's output
chatspec.create_tool_message()

# create a message with image content
chatspec.create_image_message()

# create a message with input audio content
chatspec.create_input_audio_message()
```

## 🔧 Tools & Tool Calling

#### Instance Checking & Validation of `Tools`

Same as the `Message` types, tools can be validated using the `is_tool` method.

```python
import chatspec

my_tool = {
    "type": "function",
    "function": {
        "name": "my_function",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Some properties"
                }
            }
        }
    }
}

chatspec.is_tool(my_tool)
# >>> True

chatspec.is_tool({})
# >>> False
```

#### Convert `Python Functions`, `Pydantic Models`, `Dataclasses` & more to `Tools`

```python
import chatspec

# you can be super minimal
def my_tool(x : str) -> str:
    return x

chatspec.convert_to_tool(my_tool)
# >>> {
#     "type": "function",
#     "function": {
#         "name": "my_tool",
#         "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
#     }
# }

# or fully define docstrings/annotations
def my_tool(x : str) -> str:
    """
    A tool with some glorious purpose.

    Args:
        x (str): The input to the tool.

    Returns:
        str: The output of the tool.
    """
    return x

chatspec.convert_to_tool(my_tool)
# >>> {
#     'type': 'function',
#     'function': {
#         'name': 'my_tool',
#        'parameters': {'type': 'object', 'properties': {'x': {'type': 'string', 'description': 'The input to the tool.'}}, 'required': ['x'], 'additionalProperties': False},
#        'description': 'A tool with some glorious purpose.\n',
#        'returns': 'The output of the tool.'
#    }
# }
```

#### Interacting with `Tool Calls` in `Completions` & Executing `Tools`

```python
{
    'type': 'function',
    'function': {
        'name': 'my_better_web_tool',
        'parameters': {
            'type': 'object',
            'properties': {'url': {'type': 'string', 'description': 'The URL of the website to get the title of.'}},
            'required': ['url'],
            'additionalProperties': False
        },
        'description': 'This is a tool that can be used to get the title of a website.\n'
    }
}
```
</details>

</br>

## Manage message threads easily with `State`

```python
import chatspec

# create a message store (state)
state = chatspec.State()

# we can add messages to the state normally
state.add_messages(
    [
        {"role": "user", "content": "hello i am steve"},
        {"role": "assistant", "content": "hello steve i am yu"},
    ]
)

# or directly
state.add_message("no i am me who are you?")
# define a role too
state.add_message("i just told you i that i am yu", role="assistant")
# lets see our messages
print(state.messages)
# >>> [
# >>>    {'content': 'hello i am steve', 'role': 'user'},
# >>>    {'content': 'hello steve i am yu', 'role': 'assistant'},
# >>>    {'content': 'no i am me who are you?', 'role': 'user'},
# >>>    {'content': 'i just told you i that i am yu', 'role': 'assistant'}
# >>> ]

# we can have different states interact
# states also have their own system context, with a special
# 'identification_context' attribute
state2 = chatspec.State()
state2.identification_context['name'] = "yu"

# we can send messages between states
state.receive_message_from(state2, "i am me")
# lets see what yu sent
print(state.last_message)
# >>> {'content': '[EXTERNAL from yu]: i am me', 'role': 'user'}
```