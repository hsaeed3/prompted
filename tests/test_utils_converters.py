"""
💬 tests.test_utils_converters
"""

import pytest
from prompted.utils import converters
from pydantic import BaseModel
from typing import Optional


def test_convert_to_message():
    # Test with a simple string
    message = "Hello, world!"
    converted = converters.convert_to_message(message)

    class UserProfile(BaseModel):
        name: str
        age: int
        email: Optional[str]

    user = UserProfile(
        name="John Doe", age=30, email="john.doe@example.com"
    )

    # Test with schema=True (default behavior)
    converted_schema = converters.convert_to_message(
        user, markdown=True, schema=True
    )
    print("With schema=True (default):")
    print(converted_schema)

    # Test with schema=False to show values
    converted_values = converters.convert_to_message(
        user, markdown=True, schema=False
    )
    print("\nWith schema=False:")
    print(converted_values)


if __name__ == "__main__":
    test_convert_to_message()
