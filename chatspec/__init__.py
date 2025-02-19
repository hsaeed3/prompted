"""
## 💭 chatspec

Types & utilities built for processing & interacting with objects used in
the chat completions API specification.

### Package Contents:

- `markdown.py`: Utilities for rendering objects as markdown.
- `params.py`: Types for the parameters used in chat completions.
- `types.py`: Types for the responses from chat completions.
- `mock.py`: A mocked implementation of the OpenAI client for chat completions.
- `utils.py`: Utilities for processing & interacting with chat completions.

---

[`hammad saeed`](https://github.com/hsaeed3) | 2025
"""

from .markdown import *
from .params import *
from .types import *
from .mock import *
from .utils import *