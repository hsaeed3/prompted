# 💭 chatspec

Simple types &amp; utilities built for the OpenAI Chat Completions API specification.

[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📦 Installation

```bash
pip install chatspec
```

#  📚 Documentation & Examples

`promptspec` provides a 'prethora' (as many as would actually be useful) of types, models & methods for validating, converting and augmenting objects used in the OpenAI chat completions API specification, a `State` class for managing messages threads for agentic application, as well as a `MockAI` client & `mock_completion()` method for creating mock llm responses quickly. I use [Instructor](https://github.com/instructor-ai/instructor) for all of my structured outputs, so `Pydantic` is a core part of this library. The point of this library is to provide a common interface for methods that I have found myself needing to replicate across multiple projects.

## ✨ Quickstart

```python
# just import the package
import chatspec

# create some tool for the llm to call
def get_capital(city : str) -> str:
    return "Paris"

# create a mock completion with a simulated tool call & stream it!
stream = chatspec.mock_completion(
    # format threads easily
    # this creates a proper list of messages from a:
    # - string
    # - list of messages
    # - single message (pydantic or dict)
    messages = chatspec.normalize_messages("What is the capital of France?"),
    # convert tools pydantic functions / pydantic models / and more to tools easily
    tools = [chatspec.convert_to_tool(get_capital)],
    # streaming is supported & properly typed / handled
    stream = True
)

# chatspec provides a plethora of 'ease of use' methods
chatspec.print_stream(stream)
# >>> Mock response to: What is the capital of France?
# >>> Tool Call:
# >>>   ID: 79d46439-3df2-49de-b978-8be0ef54dccf
# >>>   Type: function
# >>>   Function: get_capital
# >>>   Arguments: {"city": "mock_string"}
```

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
import chatspec

# easily check if a completion or stream has a tool call
chatspec.has_tool_call()

# get the tool calls from a completion or a stream
chatspec.get_tool_calls(completion)

# run a tool using a completion response
# this will only run the function if the tool call was present in the completion
chatspec.run_tool(completion, my_tool)

# create a tool message from a completion response
# and a function's output
chatspec.create_tool_message(completion, my_tool_output)
```

## ✨ Completion Responses & Streams

#### Instance Checking & Validation of `Completions` & `Streams`

```python
import chatspec

# check if an object is a valid chat completion or stream
chatspec.is_completion(completion)

# check if an object is a valid stream
chatspec.is_stream(stream)
```

#### The `Stream Passthrough` & `Stream` Specific Methods

`chatspec` provides an internal system for caching & storing stream responses from chat completions, for use & reuse for any of the methods
within this library for streams. This is helpful, as the user is able to send/display the initial stream response to the client, while still
being able to use it internally for any other use case.

```python
import chatspec
# `openai` is not included in the package, so you'll need to install it separately
from openai import OpenAI

client = OpenAI()

# run the stream through the passthrough
stream = chatspec.stream_passthrough(client.chat.completions.create(
    messages = [{"role": "user", "content": "Hello, how are you?"}],
    model = "gpt-4o-mini",
    stream = True,
))

# print the stream
chatspec.is_stream(stream)
# >>> True

# run any number of other methods over the stream
chatspec.dump_stream_to_message(stream)
# >>> {"role": "assistant", "content": "Hello! I'm just a program, so I don't have feelings, but I'm here and ready to help you. How can I 
# >>> assist you today?"}

# chatspec.dump_stream_to_completion(stream)
# chatspec.is_completion(stream)
# chatspec.print_stream(stream)
```

## 🆉 Types & Parameters

Use any of the provided types & models for schema reference, as well as quick parameter collection & validation with the `Params` model,
as well as a few specific parameter types for quick use.

```python
from chatspec import Params
# or get specific parameter types
from chatspec.params import MessagesParam, StreamOptionsParam

# easily define a collection of parameters for a chat completion
params = Params(
    messages = [{"role": "user", "content": "Hello, how are you?"}],
    model = "gpt-4o-mini",
    temperature = 0.5,
)

# use any of the provided types from `chatspec.types`
from chatspec.types import Message, Tool, ToolCall, Completion
# ...

# create objects directly from the types easily
message = Message(role = "user", content = "Hello, how are you?")
# >>> {"role": "user", "content": "Hello, how are you?"}
```

## 🧠 State Manager (*For Chatbots & Agentic Applications*)

> Documentation coming soon!

## 𝌭 Pydantic Models & Structured Outputs

> Documentation coming soon!

## 📕 Markdown Formatting

> Documentation coming soon!