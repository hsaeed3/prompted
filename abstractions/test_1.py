import dataclasses as d
import pydantic as p
import typing as t


# -----------------------------------------------------------------------------
# PROMPTS --- THE CORE ITEM OF `PROMPTED`
# -----------------------------------------------------------------------------


@d.dataclass
class PromptFieldInfo(p.BaseModel):
    """
    Information / configuration about a field within a `Prompt` model.
    """
    default : t.Any = None
    required : bool = False
    strategy : t.Literal["sequential", "merge", "split"] = "merge"
    max_steps : int = 3
    tools : t.List = []


