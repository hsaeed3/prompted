# 💭 chatspec

Simple types &amp; utilities built for the OpenAI Chat Completions API specification.

## 📦 Installation

```bash
pip install chatspec
```

`chatspec` provides (*very*) quick utilities for working with objects used in the OpenAI chat
completions API specification. I use [Instructor](https://github.com/instructor-ai/instructor) 
for all of my structured outputs, so `Pydantic` is a core part of this library. The point
of this library is to provide a common interface for methods that I have found myself
needing to replicate across various projects.

---

# 📚 Examples

## Use `chatspec` for easy conversion or formatting of messages, tools, and more!

### Convert `python functions`, `pydantic models`, `dataclasses`, and more to **tool calling format**

```python
import chatspec

# dont define anything...
def my_web_tool(url: str) -> str:
    return "Hello, world!"

tool = chatspec.convert_to_tool(my_web_tool)
print(tool)
```

<details>
<summary>Output</summary>

```python
{
    'type': 'function',
    'function': {'name': 'my_web_tool', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url'], 'additionalProperties': False}}
}
```

</details>

</br>

```python
# or write a full docstring
def my_better_web_tool(url: str) -> str:
    """
    This is a tool that can be used to get the title of a website.
    
    Args:
        url: The URL of the website to get the title of.
    """
    return "Hello, world!"

tool = chatspec.convert_to_tool(my_better_web_tool)
print(tool)
```

<details>
<summary>Output</summary>

</br>

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

