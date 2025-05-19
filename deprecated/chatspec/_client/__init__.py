"""
💬 chatspec._client

Contains the `client` module, which is used for creating
chat completions / structured outputs and other LLM related
operations quickly & easily using the `chatspec` package.

This module requires `litellm` and `instructor` to be installed &
can be installed using either:

```bash
pip install litellm instructor
```

or

```bash
pip install chatspec[client]
```
"""


class Client:
    """
    Primary client class used for creating chat completions,
    structured outputs, agent runs, and other LLM related
    operations quickly & easily using the `chatspec` package.
    """