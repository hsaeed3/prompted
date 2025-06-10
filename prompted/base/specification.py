"""prompted.base.specification

Contains the lowest level objects are implemented to build the
base types and interfaces for the `prompted` framework."""

from dataclasses import dataclass
from warnings import warn
from typing import (
    Any,
    Self,
)

__all__ = (
    "Specification",
    "SpecificationError",
)


# -----------------------------------------------------------------------------
# Specification
# 
# this is used to define the interface for converting to and from the 
# openai specification
# -----------------------------------------------------------------------------


class SpecificationError(Exception):
    """Base exception for errors within the `Specification` object."""


@dataclass
class Specification:
    """Base mixin defining the interface objects within `prompted` use for converting
    to and from the openai, anthropic & google Agent2Agent specifications.
    """

    def to_openai(self) -> Any:
        """Convert the object to the openai specification."""
        warn(f"The object {self.__class__.__name__} can not be converted to the OpenAI specification.")
        return None
    
    def to_anthropic(self) -> Any:
        """Convert the object to the anthropic specification."""
        warn(f"The object {self.__class__.__name__} can not be converted to the Anthropic specification.")
        return None
    
    def to_a2a(self) -> Any:
        """Convert the object to the google Agent2Agent specification."""
        warn(f"The object {self.__class__.__name__} can not be converted to the Google Agent2Agent specification.")
        return None
    
    @classmethod
    def from_openai(cls, data: Any) -> Self:
        """Convert the object from the openai specification."""
        warn(f"The object {cls.__name__} can not be created from the OpenAI specification.")
        return None
    
    @classmethod
    def from_anthropic(cls, data: Any) -> Self:
        """Convert the object from the anthropic specification."""
        warn(f"The object {cls.__name__} can not be created from the Anthropic specification.")
        return None
    
    @classmethod
    def from_a2a(cls, data: Any) -> Self:
        """Convert the object from the google Agent2Agent specification."""
        warn(f"The object {cls.__name__} can not be created from the Google Agent2Agent specification.")
        return None
    

if __name__ == "__main__":

    class Test(Specification):
        """Test object for the `BaseSpecification` object."""

        def some_method(self) -> str:
            """Some method for the `Test` object."""
            return "some_method"
        

    test = Test()
    print(test.some_method())
    print(test.to_openai())