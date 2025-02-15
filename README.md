# 💭 chatspec

Simple types &amp; utilities built for the OpenAI Chat Completions API specification.

[![](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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
