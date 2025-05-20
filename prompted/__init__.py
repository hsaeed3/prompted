"""
## 💭 prompted

[`hammad saeed`](https://github.com/hsaeed3) | 2025
"""

import sys
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

from .core.logger import setup_logging as _setup_logging
_setup_logging()


if TYPE_CHECKING:
    from ._create.create import Create as create
    from . import types
    from ._utils import Utils as utils


IMPORT_MAP : Dict[str, Tuple[str, str]] = {
    "create": ("._create.create", "Create"),
    "types": (".types", "types"),
    "utils": ("._utils", "Utils"),
}


__all__ = [
    "create",
    "types",
    "utils",
]


def __getattr__(name: str) -> Any:
    """Handle dynamic imports for module attributes."""
    if name in IMPORT_MAP:
        module_path, attr_name = IMPORT_MAP[name]
        module = import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> "list[str]":
    """Return list of module attributes for auto-completion."""
    return list(__all__)


# Set module attribute for __getattr__ in Python 3.7+
if sys.version_info >= (3, 7):
    __getattr__.__module__ = __name__