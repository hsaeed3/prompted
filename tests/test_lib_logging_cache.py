# tests.test_internal_lib_logging_cache

# test modules found within `prompted._lib`
# all of these are internally utilized modules and are not
# meant for any public usage

import pytest
from prompted._lib import (
    logger,
    cache,
    service
)


def test_internal_lib_logger():
    """
    Test the logger module.
    """
    import logging

    l = logger.setup_logging()
    assert isinstance (l, logging.Logger)


def test_internal_lib_cache():
    """
    Tests the @cache decorator available within
    `prompted._lib.cache`
    """
    from prompted._lib.cache import cached, make_hashable, CACHE

    # Test the cached decorator
    @cached(key_fn=lambda x: make_hashable(x))
    def sample_function(value):
        """Sample function that returns the input value."""
        return value * 2

    # Clear cache before testing
    CACHE.clear()
    
    # Test basic caching functionality
    assert sample_function(5) == 10
    assert sample_function(5) == 10  # Should use cached value
    
    # Test with different input
    assert sample_function(10) == 20
    
    # Test make_hashable with different types
    assert isinstance(make_hashable("test_string"), str)
    assert isinstance(make_hashable(123), str)
    assert isinstance(make_hashable({"key": "value"}), str)
    assert isinstance(make_hashable([1, 2, 3]), str)
    
    # Test that different inputs produce different hashes
    assert make_hashable("test1") != make_hashable("test2")
    assert make_hashable({"a": 1}) != make_hashable({"a": 2})
    
    # Test that same inputs produce same hashes
    assert make_hashable("test") == make_hashable("test")
    assert make_hashable([1, 2, 3]) == make_hashable([1, 2, 3])



