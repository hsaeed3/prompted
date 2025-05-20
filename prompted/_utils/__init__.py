"""
💬 prompted.utils

Contains the various utility functions & resources defined throughout
the `prompted` package, with scoped classes for a simpler & modular
namespace & usage.
"""

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Dict, Any, Tuple, List

if TYPE_CHECKING:
    from .cls import (
        CompletionsUtils,
        MessagesUtils,
        ToolsUtils,
        PydanticUtils,
        MarkdownUtils,
    )


_IMPORT_SPEC: Dict[str, Tuple[str, str]] = {
    "MarkdownUtils": (".markdown", "MarkdownUtils"),
    "CompletionsUtils": (".cls", "CompletionsUtils"),
    "MessagesUtils": (".cls", "MessagesUtils"),
    "ToolsUtils": (".cls", "ToolsUtils"),
    "PydanticUtils": (".cls", "PydanticUtils"),
}


def __getattr__(name: str) -> Any:
    """
    Dynamically import & return attributes from the `_IMPORT_SPEC` dictionary.
    """
    if name in _IMPORT_SPEC:
        module_name, attr_name = _IMPORT_SPEC[name]
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> List[str]:
    """
    Return a list of all attributes in the `_IMPORT_SPEC` dictionary.
    """
    return list(_IMPORT_SPEC.keys())


def __all__() -> List[str]:
    """
    Return a list of all attributes in the `_IMPORT_SPEC` dictionary.
    """
    return list(_IMPORT_SPEC.keys())


class Utils:
    """
    Contains the various utility functions & resources defined throughout
    the `prompted` package, with scoped classes for a simpler & modular
    namespace & usage.
    """

    from .cls import (
        CompletionsUtils,
        MessagesUtils,
        ToolsUtils,
        PydanticUtils,
        MarkdownUtils,
    )

    completions = CompletionsUtils
    messages = MessagesUtils
    tools = ToolsUtils
    pydantic = PydanticUtils
    markdown = MarkdownUtils


__all__ = [
    "Utils",
    "CompletionsUtils",
    "MessagesUtils",
    "ToolsUtils",
    "PydanticUtils",
    "MarkdownUtils",
]
