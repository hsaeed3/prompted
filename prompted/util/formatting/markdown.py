"""
prompted.resources.formatting.markdown

Contains various resources for converting different
filetypes and data types into markdown strings.
"""

from typing import (
    Any,
    List,
    Union,
    Literal,
    Callable,
    Optional,
    get_args,
)
from inspect import getdoc
from dataclasses import fields as dataclass_fields, is_dataclass
from typing_extensions import TypedDict

import typing_inspect as ti
from pydantic import BaseModel

from ..._cache import cached, make_hashable
from ...logger import _get_logger

logger = _get_logger(__name__)

__all__ = [
    "MarkdownSettings",
    "convert_to_markdown",
    "convert_function_to_markdown",
    "convert_dataclass_to_markdown",
    "convert_pydantic_model_to_markdown",
    "convert_object_to_markdown",
    "convert_type_to_markdown",
]


# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------


class MarkdownSettings(TypedDict):
    """
    Settings configuration for markdown formatting.
    """
    indent : int = 0
    split : bool = False
    exclude : Optional[List[str]] = None
    as_code_block : bool = False
    as_natural_language : bool = False
    show_schema : bool = False
    show_types : bool = True
    show_field_descriptions : bool = True
    show_values : bool = True
    show_defaults : bool = True
    show_title : bool = True
    show_bullets : bool = True
    show_docs : bool = True
    bullet_style : str = "-"
    title_level : Literal["h1", "h2", "h3", "bold"] = "h1"
    code_block_language : str | None = None
    show_header : bool = True
    override_title : str | None = None
    override_description : str | None = None
    

# ------------------------------------------------------------------------
# Helper Functions (ported from old formatting.py)
# ------------------------------------------------------------------------


def _get_field_description(field_info: Any) -> Optional[str]:
    """Extract field description from Pydantic field info.

    Args:
        field_info: The Pydantic field info object to extract description from

    Returns:
        The field description if available, None otherwise
    """
    try:
        import docstring_parser
    except ImportError:
        docstring_parser = None

    try:
        if docstring_parser and hasattr(field_info, "__doc__") and field_info.__doc__:
            doc = docstring_parser.parse(field_info.__doc__)
            if doc.short_description:
                return doc.short_description

        if hasattr(field_info, "description"):
            return field_info.description

        return None
    except Exception:
        return None


def _get_dataclass_field_description(field: Any) -> Optional[str]:
    """Extract field description from dataclass field info.

    Args:
        field: The dataclass field object to extract description from

    Returns:
        The field description if available, None otherwise
    """
    try:
        # Check field metadata for description
        if hasattr(field, "metadata") and field.metadata:
            desc = field.metadata.get("description")
            if desc:
                return desc
        
        # Check field docstring if available
        if hasattr(field, "__doc__") and field.__doc__:
            return field.__doc__.strip()
        
        return None
    except Exception:
        return None


def _format_docstring(doc_dict: dict, prefix: str = "") -> str:
    """Format parsed docstring into markdown.

    Args:
        doc_dict: Dictionary containing parsed docstring sections or a raw docstring string.
        prefix: String to prepend to each line for indentation

    Returns:
        Formatted markdown string
    """
    try:
        import docstring_parser
    except ImportError:
        docstring_parser = None

    try:
        if not doc_dict:
            return ""

        if not docstring_parser:
            if isinstance(doc_dict, dict) and doc_dict.get("short"):
                return f"{prefix}_{doc_dict['short']}_"
            elif isinstance(doc_dict, str):
                return f"{prefix}_{doc_dict.strip()}_"
            return ""

        if isinstance(doc_dict, str):
            doc = docstring_parser.parse(doc_dict)
        elif isinstance(doc_dict, dict):
            parts = []
            if doc_dict.get("short"):
                parts.append(f"{prefix}_{doc_dict['short']}_")
            if doc_dict.get("long"):
                parts.append(f"{prefix}_{doc_dict['long']}_")
            if doc_dict.get("params"):
                parts.append(f"{prefix}_Parameters:_")
                for name, type_name, desc in doc_dict["params"]:
                    type_str = f": {type_name}" if type_name else ""
                    parts.append(f"{prefix}  - `{name}{type_str}` - {desc}")
            if doc_dict.get("returns"):
                parts.append(f"{prefix}_Returns:_ {doc_dict['returns']}")
            if doc_dict.get("raises"):
                parts.append(f"{prefix}_Raises:_")
                for type_name, desc in doc_dict["raises"]:
                    parts.append(f"{prefix}  - `{type_name}` - {desc}")
            return "\n".join(parts)
        else:
            return str(doc_dict)

        parts = []
        if doc.short_description:
            parts.append(f"{prefix}_{doc.short_description}_")
        if doc.long_description:
            parts.append(f"{prefix}_{doc.long_description}_")
        if doc.params:
            parts.append(f"{prefix}_Parameters:_")
            for param in doc.params:
                type_str = f": {param.type_name}" if param.type_name else ""
                parts.append(f"{prefix}  - `{param.arg_name}{type_str}` - {param.description}")
        if doc.returns:
            parts.append(f"{prefix}_Returns:_ {doc.returns.description}")
        if doc.raises:
            parts.append(f"{prefix}_Raises:_")
            for exc in doc.raises:
                parts.append(f"{prefix}  - `{exc.type_name}` - {exc.description}")

        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Error formatting docstring: {e}")
        return str(doc_dict)


@cached(lambda cls: make_hashable(cls))
def convert_type_to_markdown(cls: Any) -> str:
    """Get a clean type name for display, handling generics correctly."""
    if cls is None or cls is type(None):
        return "None"
    if hasattr(cls, "__name__") and cls.__name__ != "<lambda>":
        return cls.__name__

    if hasattr(cls, "annotation"):
        annotation = cls.annotation
        if annotation is not None:
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                args = get_args(annotation)
                if len(args) == 2 and args[1] is type(None):
                    inner_type = args[0]
                    inner_type_name = convert_type_to_markdown(inner_type)
                    return f"Optional[{inner_type_name}]"
            return convert_type_to_markdown(annotation)

    origin = ti.get_origin(cls)
    args = ti.get_args(cls)

    if origin is not None:
        if ti.is_optional_type(cls):
            inner_type = args[0]
            inner_type_name = convert_type_to_markdown(inner_type)
            return f"Optional[{inner_type_name}]"

        if ti.is_union_type(cls):
            args_str = " | ".join(convert_type_to_markdown(arg) for arg in args)
            return f"Union[{args_str}]"

        origin_name = getattr(origin, "__name__", str(origin).split(".")[-1])
        if origin_name.startswith("_"):
            origin_name = origin_name[1:]

        if args:
            args_str = ", ".join(convert_type_to_markdown(arg) for arg in args)
            return f"{origin_name}[{args_str}]"
        else:
            return origin_name

    if ti.is_typevar(cls):
        return str(cls)
    if ti.is_forward_ref(cls):
        return str(cls)
    if ti.is_literal_type(cls):
        return f"Literal[{', '.join(str(arg) for arg in args)}]"
    if ti.is_typeddict(cls):
        return f"TypedDict[{', '.join(convert_type_to_markdown(arg) for arg in args)}]"
    if ti.is_protocol(cls):
        return f"Protocol[{', '.join(convert_type_to_markdown(arg) for arg in args)}]"
    if ti.is_classvar(cls):
        return f"ClassVar[{convert_type_to_markdown(args[0])}]" if args else "ClassVar"
    if ti.is_final_type(cls):
        return f"Final[{convert_type_to_markdown(args[0])}]" if args else "Final"
    if ti.is_new_type(cls):
        return str(cls)

    if str(cls).startswith("typing.Optional"):
        inner_type_str = str(cls).replace("typing.Optional[", "").rstrip("]")
        return f"Optional[{inner_type_str}]"

    return str(cls).replace("typing.", "").replace("__main__.", "")


@cached(lambda obj, use_getdoc=True: make_hashable((obj, use_getdoc)))
def _parse_docstring(obj: Any, use_getdoc: bool = True) -> Optional[dict]:
    """Extract and parse docstring from an object.

    Args:
        obj: The object to extract docstring from.
        use_getdoc: If True, use inspect.getdoc (follows MRO). If False, use obj.__doc__ directly.

    Returns:
        Dictionary containing parsed docstring components
    """
    try:
        import docstring_parser
    except ImportError:
        docstring_parser = None

    doc = getdoc(obj) if use_getdoc else getattr(obj, "__doc__", None)

    if not doc:
        return None

    if not docstring_parser:
        return {"short": doc.strip()}

    try:
        parsed = docstring_parser.parse(doc)
        result = {
            "short": parsed.short_description,
            "long": parsed.long_description,
            "params": [(p.arg_name, p.type_name, p.description) for p in parsed.params],
            "returns": parsed.returns.description if parsed.returns else None,
            "raises": [(e.type_name, e.description) for e in parsed.raises],
        }
        return {
            k: v
            for k, v in result.items()
            if v and (not isinstance(v, list) or len(v) > 0)
        }
    except Exception as e:
        logger.warning(f"Failed to parse docstring for {obj}: {e}")
        return {"short": doc.strip()}


# ------------------------------------------------------------------------
# Scoped Instance Converters
# ------------------------------------------------------------------------


@cached(lambda target, settings=None: make_hashable((target, settings)))
def convert_function_to_markdown(
    target : Callable,
    settings : MarkdownSettings | None = None
) -> str:
    """
    Converts a function into a markdown string.
    """
    if settings is None:
        settings = {}
    
    func_name = target.__name__
    prefix = "  " * settings.get("indent", 0)
    
    # Create title based on title_level
    title_level = settings.get("title_level", "h1")
    title = settings.get("override_title") or func_name
    
    if title_level == "h1":
        title_md = f"# {title}"
    elif title_level == "h2":
        title_md = f"## {title}"
    elif title_level == "h3":
        title_md = f"### {title}"
    elif title_level == "bold":
        title_md = f"**{title}**"
    else:
        title_md = f"# {title}"
    
    parts = []
    
    if settings.get("show_title", True):
        parts.append(f"{prefix}{title_md}")
    
    if settings.get("show_docs", True):
        doc_dict = _parse_docstring(target, use_getdoc=True)
        if doc_dict:
            doc_md = _format_docstring(doc_dict, prefix + "  ")
            if doc_md:
                parts.append(doc_md)
    
    result = "\n".join(parts)
    
    if settings.get("as_code_block", False):
        lang = settings.get("code_block_language", "python")
        return f"```{lang}\n{result}\n```"
    
    return result


@cached(lambda target, settings=None: make_hashable((target, settings)))
def convert_dataclass_to_markdown(
    target : Any,
    settings : MarkdownSettings | None = None
) -> str:
    """
    Converts a dataclass into a markdown string.
    """
    if settings is None:
        settings = {}
    
    if not is_dataclass(target):
        raise ValueError("Target must be a dataclass")
    
    is_class = isinstance(target, type)
    class_name = target.__name__ if is_class else target.__class__.__name__
    
    prefix = "  " * settings.get("indent", 0)
    bullet = f"{settings.get('bullet_style', '-')} " if settings.get("show_bullets", True) else ""
    
    # Create title
    title_level = settings.get("title_level", "h1")
    title = settings.get("override_title") or class_name
    
    if title_level == "h1":
        title_md = f"# {title}"
    elif title_level == "h2":
        title_md = f"## {title}"
    elif title_level == "h3":
        title_md = f"### {title}"
    elif title_level == "bold":
        title_md = f"**{title}**"
    else:
        title_md = f"# {title}"
    
    parts = []
    
    if settings.get("show_title", True):
        parts.append(f"{prefix}{title_md}")
    
    # Add docstring if enabled
    if settings.get("show_docs", True) and settings.get("show_header", True):
        doc_obj = target if is_class else target.__class__
        doc_dict = _parse_docstring(doc_obj, use_getdoc=False)
        if doc_dict:
            doc_dict_filtered = doc_dict.copy()
            doc_dict_filtered.pop("params", None)
            doc_md = _format_docstring(doc_dict_filtered, prefix + "  ")
            if doc_md:
                parts.append(doc_md)
    
    # Handle fields
    fields_list = dataclass_fields(target)
    exclude = settings.get("exclude", []) or []
    
    if settings.get("as_natural_language", False):
        # Natural language format
        if not is_class:
            parts.append(f"{prefix}{class_name} is currently set with the following values:")
            for field in fields_list:
                if field.name in exclude:
                    continue
                value = getattr(target, field.name)
                type_name = convert_type_to_markdown(field.type).replace("_", " ").lower()
                parts.append(f"{prefix}{bullet}{field.name.title()} (A {type_name}) is defined as {repr(value)}")
    else:
        # Standard format
        for field in fields_list:
            if field.name in exclude:
                continue
            
            if settings.get("split", False):
                # Create subheading for each field
                field_title_level = settings.get("title_level", "h1")
                if field_title_level == "h1":
                    field_title = f"## {field.name}"
                elif field_title_level == "h2":
                    field_title = f"### {field.name}"
                elif field_title_level == "h3":
                    field_title = f"#### {field.name}"
                else:
                    field_title = f"### {field.name}"
                
                parts.append(f"{prefix}{field_title}")
                
                # Add type information
                if settings.get("show_types", True):
                    type_name = convert_type_to_markdown(field.type)
                    parts.append(f"{prefix}**Type:** `{type_name}`")
                
                # Add field description if available and enabled
                if settings.get("show_field_descriptions", True):
                    field_desc = _get_dataclass_field_description(field)
                    if field_desc:
                        parts.append(f"{prefix}**Description:** {field_desc}")
                
                # Add value
                if settings.get("show_values", True) and not is_class:
                    value = getattr(target, field.name)
                    parts.append(f"{prefix}**Value:** `{repr(value)}`")
                elif settings.get("show_defaults", True) and hasattr(field, "default") and field.default is not dataclass_fields:
                    if field.default_factory is not dataclass_fields:
                        parts.append(f"{prefix}**Default:** `{field.default_factory()}`")
                    else:
                        parts.append(f"{prefix}**Default:** `{repr(field.default)}`")
                
                parts.append("")  # Add spacing between fields
            else:
                # Standard bullet format
                field_parts = [f"{prefix}{bullet}{field.name}"]
                
                if settings.get("show_types", True):
                    type_name = convert_type_to_markdown(field.type)
                    field_parts.append(f" : {type_name}")
                
                # Show values for instances (not classes) when show_values is True
                if settings.get("show_values", True) and not is_class:
                    value = getattr(target, field.name)
                    field_parts.append(f" = {repr(value)}")
                elif settings.get("show_defaults", True) and hasattr(field, "default") and field.default is not dataclass_fields:
                    if field.default_factory is not dataclass_fields:
                        field_parts.append(f" = {field.default_factory()}")
                    else:
                        field_parts.append(f" = {repr(field.default)}")
                
                # Add field description as comment if available and enabled
                if settings.get("show_field_descriptions", True):
                    field_desc = _get_dataclass_field_description(field)
                    if field_desc:
                        field_parts.append(f"  # {field_desc}")
                
                parts.append("".join(field_parts))
    
    result = "\n".join(parts)
    
    if settings.get("as_code_block", False):
        lang = settings.get("code_block_language", "python")
        return f"```{lang}\n{result}\n```"
    
    return result


@cached(lambda target, settings=None: make_hashable((target, settings)))
def convert_pydantic_model_to_markdown(
    target : BaseModel,
    settings : MarkdownSettings | None = None
) -> str:
    """
    Converts a Pydantic model into a markdown string.
    """
    if settings is None:
        settings = {}
    
    if not (isinstance(target, BaseModel) or (isinstance(target, type) and issubclass(target, BaseModel))):
        raise ValueError("Target must be a Pydantic model")
    
    is_class = isinstance(target, type)
    model_name = target.__name__ if is_class else target.__class__.__name__
    
    prefix = "  " * settings.get("indent", 0)
    bullet = f"{settings.get('bullet_style', '-')} " if settings.get("show_bullets", True) else ""
    
    # Create title
    title_level = settings.get("title_level", "h1")
    title = settings.get("override_title") or model_name
    
    if title_level == "h1":
        title_md = f"# {title}"
    elif title_level == "h2":
        title_md = f"## {title}"
    elif title_level == "h3":
        title_md = f"### {title}"
    elif title_level == "bold":
        title_md = f"**{title}**"
    else:
        title_md = f"# {title}"
    
    parts = []
    
    if settings.get("show_title", True):
        parts.append(f"{prefix}{title_md}")
    
    # Add docstring if enabled
    if settings.get("show_docs", True) and settings.get("show_header", True):
        doc_obj = target if is_class else target.__class__
        doc_dict = _parse_docstring(doc_obj, use_getdoc=False)
        if doc_dict:
            doc_dict_filtered = doc_dict.copy()
            doc_dict_filtered.pop("params", None)
            doc_md = _format_docstring(doc_dict_filtered, prefix + "  ")
            if doc_md:
                parts.append(doc_md)
    
    # Handle fields
    model_fields = target.__class__.model_fields if is_class else target.__class__.model_fields
    exclude = settings.get("exclude", []) or []
    
    if settings.get("as_natural_language", False):
        # Natural language format
        if not is_class:
            parts.append(f"{prefix}{model_name} is currently set with the following values:")
            for field_name, field_info in model_fields.items():
                if field_name in exclude:
                    continue
                value = getattr(target, field_name)
                type_name = convert_type_to_markdown(field_info.annotation).replace("_", " ").lower()
                parts.append(f"{prefix}{bullet}{field_name.title()} (A {type_name}) is defined as {repr(value)}")
    else:
        # Standard format
        for field_name, field_info in model_fields.items():
            if field_name in exclude:
                continue
            
            if settings.get("split", False):
                # Create subheading for each field
                field_title_level = settings.get("title_level", "h1")
                if field_title_level == "h1":
                    field_title = f"## {field_name}"
                elif field_title_level == "h2":
                    field_title = f"### {field_name}"
                elif field_title_level == "h3":
                    field_title = f"#### {field_name}"
                else:
                    field_title = f"### {field_name}"
                
                parts.append(f"{prefix}{field_title}")
                
                # Add type information
                if settings.get("show_types", True):
                    type_name = convert_type_to_markdown(field_info.annotation)
                    parts.append(f"{prefix}**Type:** `{type_name}`")
                
                # Add field description if available and enabled
                if settings.get("show_field_descriptions", True):
                    field_desc = _get_field_description(field_info)
                    if field_desc:
                        parts.append(f"{prefix}**Description:** {field_desc}")
                
                # Add value
                if settings.get("show_values", True) and not is_class:
                    value = getattr(target, field_name)
                    parts.append(f"{prefix}**Value:** `{repr(value)}`")
                elif settings.get("show_defaults", True) and field_info.default is not None:
                    parts.append(f"{prefix}**Default:** `{repr(field_info.default)}`")
                elif settings.get("show_defaults", True) and hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                    parts.append(f"{prefix}**Default:** `{field_info.default_factory()}`")
                
                parts.append("")  # Add spacing between fields
            else:
                # Standard bullet format
                field_parts = [f"{prefix}{bullet}{field_name}"]
                
                if settings.get("show_types", True):
                    type_name = convert_type_to_markdown(field_info.annotation)
                    field_parts.append(f" : {type_name}")
                
                # Show values for instances (not classes) when show_values is True
                if settings.get("show_values", True) and not is_class:
                    value = getattr(target, field_name)
                    field_parts.append(f" = {repr(value)}")
                elif settings.get("show_defaults", True) and field_info.default is not None:
                    field_parts.append(f" = {repr(field_info.default)}")
                elif settings.get("show_defaults", True) and hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                    field_parts.append(f" = {field_info.default_factory()}")
                
                # Add field description as comment if available and enabled
                if settings.get("show_field_descriptions", True):
                    field_desc = _get_field_description(field_info)
                    if field_desc:
                        field_parts.append(f"  # {field_desc}")
                
                parts.append("".join(field_parts))
    
    result = "\n".join(parts)
    
    if settings.get("as_code_block", False):
        lang = settings.get("code_block_language", "python")
        return f"```{lang}\n{result}\n```"
    
    return result


def convert_object_to_markdown(
    target : Any,
    settings : MarkdownSettings | None = None
) -> str:
    """
    Converts an object into a markdown string.
    """
    if settings is None:
        settings = {}
    
    # Dispatch to appropriate converter based on object type
    if isinstance(target, Callable):
        return convert_function_to_markdown(target, settings)
    elif isinstance(target, BaseModel) or (isinstance(target, type) and issubclass(target, BaseModel)):
        return convert_pydantic_model_to_markdown(target, settings)
    elif is_dataclass(target):
        return convert_dataclass_to_markdown(target, settings)
    else:
        # Fallback for other objects
        prefix = "  " * settings.get("indent", 0)
        bullet = f"{settings.get('bullet_style', '-')} " if settings.get("show_bullets", True) else ""
        
        obj_name = target.__class__.__name__ if hasattr(target, '__class__') else str(type(target).__name__)
        
        # Create title
        title_level = settings.get("title_level", "h1")
        title = settings.get("override_title") or obj_name
        
        if title_level == "h1":
            title_md = f"# {title}"
        elif title_level == "h2":
            title_md = f"## {title}"
        elif title_level == "h3":
            title_md = f"### {title}"
        elif title_level == "bold":
            title_md = f"**{title}**"
        else:
            title_md = f"# {title}"
        
        parts = []
        
        if settings.get("show_title", True):
            parts.append(f"{prefix}{title_md}")
        
        # Handle different object types
        if isinstance(target, (list, tuple, set)):
            if target:
                for i, item in enumerate(target):
                    parts.append(f"{prefix}{bullet}Item {i}: {repr(item)}")
            else:
                parts.append(f"{prefix}{bullet}Empty {obj_name.lower()}")
        elif isinstance(target, dict):
            if target:
                for key, value in target.items():
                    parts.append(f"{prefix}{bullet}{key}: {repr(value)}")
            else:
                parts.append(f"{prefix}{bullet}Empty dictionary")
        else:
            # Generic object - try to get attributes
            if hasattr(target, '__dict__'):
                for attr_name, attr_value in target.__dict__.items():
                    if not attr_name.startswith('_'):  # Skip private attributes
                        parts.append(f"{prefix}{bullet}{attr_name}: {repr(attr_value)}")
            else:
                parts.append(f"{prefix}{bullet}Value: {repr(target)}")
        
        result = "\n".join(parts)
        
        if settings.get("as_code_block", False):
            lang = settings.get("code_block_language", "text")
            return f"```{lang}\n{result}\n```"
        
        return result


def convert_to_markdown(
    target : Any,
    indent : int = 0,
    split : bool = False,
    exclude : Optional[List[str]] = None,
    as_code_block : bool = False,
    as_natural_language : bool = False,
    show_schema : bool = False,
    show_field_descriptions : bool = True,
    show_types : bool = True,
    show_values : bool = True,
    show_defaults : bool = True,
    show_title : bool = True,
    show_bullets : bool = True,
    show_docs : bool = True,
    bullet_style : str = "-",
    title_level : Literal["h1", "h2", "h3", "bold"] = "h1",
    code_block_language : str | None = None,
    show_header : bool = True,
    override_title : str | None = None,
    override_description : str | None = None,
) -> str:
    """
    Converts a target object into a markdown string.

    Args:
        target: Any - The target object to convert to markdown.
        split: bool - Whether to split the markdown into multiple parts. (Each field will be a subheading)
        indent: int - The number of spaces to indent the markdown.
        exclude: Optional[List[str]] - A list of field names to exclude from the markdown.
        as_code_block: bool - Whether to render the markdown as a code block.
        as_natural_language: bool - Whether to render the markdown as natural language.
        show_schema: bool - Whether to show the schema of the target object.
        show_field_descriptions: bool - Whether to show the field descriptions of the target object.
        show_types: bool - Whether to show the types of the target object.
        show_values: bool - Whether to show the values of the target object.
        show_defaults: bool - Whether to show the default values of the target object.
        show_title: bool - Whether to show the title of the target object.
        show_bullets: bool - Whether to show bullets in the markdown.
        show_docs: bool - Whether to show the docs of the target object.
        bullet_style: str - The style of the bullets in the markdown.
        title_level: Literal["h1", "h2", "h3", "bold"] - The level of the title in the markdown.
        code_block_language: Optional[str] - The language of the code block in the markdown.
        show_header: bool - Whether to show the header of the target object.
        override_title: Optional[str] - The title to use for the target object.
        override_description: Optional[str] - The description to use for the target object.
    """
    # Create settings dict from parameters
    settings: MarkdownSettings = {
        "indent": indent,
        "split": split,
        "exclude": exclude,
        "as_code_block": as_code_block,
        "as_natural_language": as_natural_language,
        "show_schema": show_schema,
        "show_field_descriptions": show_field_descriptions,
        "show_types": show_types,
        "show_values": show_values,
        "show_defaults": show_defaults,
        "show_title": show_title,
        "show_bullets": show_bullets,
        "show_docs": show_docs,
        "bullet_style": bullet_style,
        "title_level": title_level,
        "code_block_language": code_block_language,
        "show_header": show_header,
        "override_title": override_title,
        "override_description": override_description,
    }
    
    return convert_object_to_markdown(target, settings)

