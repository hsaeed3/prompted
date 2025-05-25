"""
prompted.types.instructions

Contains type definitions for the prompting module and components
of the `prompted` framework.

"Instructions" within `prompted` are used to define the behavior
of an agent or completion, with the ability to provide a list
of instructions for steps, as well as dynamic content that
is present only for specific scenarios.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


