"""
💬 prompted._common.cache

Contains resources that are used within `prompted` for caching
purposes.
"""

import inspect
import hashlib
import logging
from functools import wraps
from cachetools import TTLCache
from typing import (
    Any,
    Callable,
    TypeVar,
)


logger = logging.getLogger("prompted.common.cache")

__all__ = [
    "get_value",
    "make_hashable",
    "cached",
    "CACHE",
    "TYPE_MAPPING",
    "CACHE_T",
]


# ------------------------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------------------------


CACHE_T = TypeVar("CACHE_T")
"""
Type variable for the cache.
"""


CACHE = TTLCache(maxsize=1000, ttl=3600)
"""
Singleton cache instance for use within the
`prompted` package.
"""


TYPE_MAPPING = {
    int: ("integer", int),
    float: ("number", float),
    str: ("string", str),
    bool: ("boolean", bool),
    list: ("array", list),
    dict: ("object", dict),
    tuple: ("array", tuple),
    set: ("array", set),
    Any: ("any", Any),
}
"""
Type mapping helper for cache keys.
"""


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    """
    Helper function to retrieve a value from an object either as an attribute or as a dictionary key.
    """
    try:
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default
    except Exception as e:
        logger.debug(f"Error getting value for key {key}: {e}")
        return default


def make_hashable(obj: Any) -> str:
    """
    Helper function to make an object hashable by converting it to a stable hash string.
    Uses SHA-256 to generate a consistent hash representation of any object.
    """
    try:
        # Handle basic types first
        if isinstance(obj, (str, int, float, bool, bytes)):
            return hashlib.sha256(str(obj).encode()).hexdigest()

        if isinstance(obj, (tuple, list)):
            # Recursively handle sequences
            return hashlib.sha256(
                ",".join(make_hashable(x) for x in obj).encode()
            ).hexdigest()

        if isinstance(obj, dict):
            # Sort dict items for consistent hashing
            return hashlib.sha256(
                ",".join(
                    f"{k}:{make_hashable(v)}" for k, v in sorted(obj.items())
                ).encode()
            ).hexdigest()

        if isinstance(obj, type):
            # Handle types (classes)
            return hashlib.sha256(
                f"{obj.__module__}.{obj.__name__}".encode()
            ).hexdigest()

        if callable(obj):
            # Handle functions
            return hashlib.sha256(
                f"{obj.__module__}.{obj.__name__}".encode()
            ).hexdigest()

        if hasattr(obj, "__dict__"):
            # Use the __dict__ for instance attributes if available
            return make_hashable(obj.__dict__)

        # Fallback for any other types that can be converted to string
        return hashlib.sha256(str(obj).encode()).hexdigest()

    except Exception as e:
        logger.debug(f"Error making object hashable: {e}")
        # Fallback to a basic string hash
        return hashlib.sha256(str(type(obj)).encode()).hexdigest()


def cached(
    key_fn,
):
    """Caching decorator that only creates cache entries when needed."""

    def decorator(func: Callable[..., CACHE_T]) -> Callable[..., CACHE_T]:
        # Get the original signature
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> CACHE_T:
            try:
                # Include function name in cache key to avoid cross-function cache collisions
                func_name = func.__name__
                cache_key = f"{func_name}:{key_fn(*args, **kwargs)}"
                if cache_key not in CACHE:
                    CACHE[cache_key] = func(*args, **kwargs)
                return CACHE[cache_key]
            except Exception:
                # On any error, fall back to uncached function call
                return func(*args, **kwargs)

        return wrapper

    return decorator
