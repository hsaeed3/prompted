"""
💬 prompted.utils.formatting

Contains utilities used for text formatting & rendering.
"""

import json
import logging
from dataclasses import is_dataclass, fields as dataclass_fields
from inspect import getdoc
from pydantic import BaseModel
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    get_args,
)
import typing_inspect as ti

from ..common.cache import cached, make_hashable
from ..types.chat_completions import Message


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# MARKDOWN : HELPERS
# ------------------------------------------------------------------------


def _get_field_description(field_info: Any) -> Optional[str]:
    """Extract field description from Pydantic field info.

    Args:
        field_info: The Pydantic field info object to extract description from

    Returns:
        The field description if available, None otherwise
    """
    try:
        # Attempt to import docstring_parser locally within the function
        import docstring_parser
    except ImportError:
        # logging.warning("docstring_parser not found. Field descriptions from docstrings will not be available.")
        docstring_parser = None  # Set to None if import fails

    try:
        if (
            docstring_parser
            and hasattr(field_info, "__doc__")
            and field_info.__doc__
        ):
            doc = docstring_parser.parse(field_info.__doc__)
            if doc.short_description:
                return doc.short_description

        if hasattr(field_info, "description"):
            return field_info.description

        return None
    except Exception:
        # Catch any errors during parsing or access
        return None


def format_docstring(
    doc_dict: dict, prefix: str = "", compact: bool = False
) -> str:
    """Format parsed docstring into markdown.

    Args:
        doc_dict: Dictionary containing parsed docstring sections or a raw docstring string.
        prefix: String to prepend to each line for indentation
        compact: If True, produces more compact output (currently not fully utilized for docstrings)

    Returns:
        Formatted markdown string
    """
    try:
        # Attempt to import docstring_parser locally within the function
        import docstring_parser
    except ImportError:
        # logging.warning("docstring_parser not found. Docstring formatting will be basic.")
        docstring_parser = None  # Set to None if import fails

    try:
        if not doc_dict:
            return ""

        # If docstring_parser is not available, just return the short description if it exists
        if not docstring_parser:
            if isinstance(doc_dict, dict) and doc_dict.get("short"):
                return f"{prefix}_{doc_dict['short']}_"
            elif isinstance(doc_dict, str):
                return f"{prefix}_{doc_dict.strip()}_"
            return ""

        # Handle raw docstring string input as well
        if isinstance(doc_dict, str):
            doc = docstring_parser.parse(doc_dict)
        elif isinstance(doc_dict, dict):
            # Format directly from the dictionary structure created by _parse_docstring
            parts = []
            if doc_dict.get("short"):
                parts.append(f"{prefix}_{doc_dict['short']}_")
            if doc_dict.get("long"):
                parts.append(f"{prefix}_{doc_dict['long']}_")
            # Only include parameters if they are present in the dictionary
            if doc_dict.get("params"):
                parts.append(f"{prefix}_Parameters:_")
                for name, type_name, desc in doc_dict["params"]:
                    type_str = f": {type_name}" if type_name else ""
                    parts.append(
                        f"{prefix}  - `{name}{type_str}` - {desc}"
                    )
            if doc_dict.get("returns"):
                parts.append(f"{prefix}_Returns:_ {doc_dict['returns']}")
            if doc_dict.get("raises"):
                parts.append(f"{prefix}_Raises:_")
                for type_name, desc in doc_dict["raises"]:
                    parts.append(
                        f"{prefix}  - `{type_name}` - {exc.description}"
                    )  # Corrected exc.description
            return "\n".join(parts)
        else:
            # Fallback for unexpected doc_dict types
            return str(doc_dict)

        # If we parsed a string docstring using docstring_parser
        parts = []

        if doc.short_description:
            parts.append(f"{prefix}_{doc.short_description}_")

        if doc.long_description:
            parts.append(f"{prefix}_{doc.long_description}_")

        if doc.params:
            parts.append(f"{prefix}_Parameters:_")
            for param in doc.params:
                type_str = (
                    f": {param.type_name}" if param.type_name else ""
                )
                parts.append(
                    f"{prefix}  - `{param.arg_name}{type_str}` - {param.description}"
                )

        if doc.returns:
            parts.append(f"{prefix}_Returns:_ {doc.returns.description}")

        if doc.raises:
            parts.append(f"{prefix}_Raises:_")
            for exc in doc.raises:
                parts.append(
                    f"{prefix}  - `{exc.type_name}` - {exc.description}"
                )

        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Error formatting docstring: {e}")
        # Return original as string in case of error
        return str(doc_dict)


@cached(lambda cls: make_hashable(cls) if cls else "")
def get_type_name(cls: Any) -> str:
    """Get a clean type name for display, handling generics correctly."""
    # Handle None type
    if cls is None or cls is type(None):
        return "None"
    # Handle basic types with __name__ attribute
    if hasattr(cls, "__name__") and cls.__name__ != "<lambda>":
        return cls.__name__

    # Try to get type directly from Pydantic field info if available
    if hasattr(cls, "annotation"):
        # Get the annotation directly from the field
        annotation = cls.annotation
        if annotation is not None:
            # Handle Optional types from Pydantic field
            if (
                hasattr(annotation, "__origin__")
                and annotation.__origin__ is Union
            ):
                args = get_args(annotation)
                if len(args) == 2 and args[1] is type(None):
                    inner_type = args[0]
                    inner_type_name = get_type_name(inner_type)
                    return f"Optional[{inner_type_name}]"
            return get_type_name(annotation)

    # Get origin and args using typing_inspect for better type handling
    origin = ti.get_origin(cls)
    args = ti.get_args(cls)

    if origin is not None:
        # Handle Optional (Union[T, None])
        if ti.is_optional_type(cls):
            # Recursively get the name of the inner type (the one not None)
            inner_type = args[0]
            inner_type_name = get_type_name(inner_type)
            return f"Optional[{inner_type_name}]"

        # Handle other Union types
        if ti.is_union_type(cls):
            # Recursively get names of all arguments in the Union
            args_str = " | ".join(get_type_name(arg) for arg in args)
            return f"Union[{args_str}]"

        # Handle other generic types (List, Dict, Tuple, Set, etc.)
        # Use origin.__name__ for built-in generics like list, dict, tuple, set
        origin_name = getattr(
            origin, "__name__", str(origin).split(".")[-1]
        )
        if origin_name.startswith(
            "_"
        ):  # Handle internal typing names like _List
            origin_name = origin_name[1:]

        if args:  # If there are type arguments
            # Recursively get names of type arguments
            args_str = ", ".join(get_type_name(arg) for arg in args)
            return f"{origin_name}[{args_str}]"
        else:  # Generic without arguments (e.g., typing.List)
            return origin_name

    # Handle special cases with typing_inspect
    if ti.is_typevar(cls):
        return str(cls)
    if ti.is_forward_ref(cls):
        return str(cls)
    if ti.is_literal_type(cls):
        return f"Literal[{', '.join(str(arg) for arg in args)}]"
    if ti.is_typeddict(cls):
        return (
            f"TypedDict[{', '.join(get_type_name(arg) for arg in args)}]"
        )
    if ti.is_protocol(cls):
        return f"Protocol[{', '.join(get_type_name(arg) for arg in args)}]"
    if ti.is_classvar(cls):
        return (
            f"ClassVar[{get_type_name(args[0])}]" if args else "ClassVar"
        )
    if ti.is_final_type(cls):
        return f"Final[{get_type_name(args[0])}]" if args else "Final"
    if ti.is_new_type(cls):
        return str(cls)

    # Special handling for Optional type
    if str(cls).startswith("typing.Optional"):
        # Extract the inner type from the string representation
        inner_type_str = (
            str(cls).replace("typing.Optional[", "").rstrip("]")
        )
        return f"Optional[{inner_type_str}]"

    # Fallback for any other types
    # Clean up 'typing.' prefix and handle other common representations
    return str(cls).replace("typing.", "").replace("__main__.", "")


def _parse_docstring(obj: Any, use_getdoc: bool = True) -> Optional[dict]:
    """
    Extract and parse docstring from an object.

    Args:
        obj: The object to extract docstring from.
        use_getdoc: If True, use inspect.getdoc (follows MRO). If False, use obj.__doc__ directly.

    Returns:
        Dictionary containing parsed docstring components:
        - short: Brief description
        - long: Detailed description
        - params: List of parameters (name, type_name, description)
        - returns: Return value description
        - raises: List of exceptions (type_name, description)
    """
    try:
        # Attempt to import docstring_parser locally within the function
        import docstring_parser
    except ImportError:
        # logging.warning("docstring_parser not found. Docstring parsing will be basic.")
        docstring_parser = None  # Set to None if import fails

    # *** MODIFICATION HERE: Use obj.__doc__ if use_getdoc is False ***
    # Safely access __doc__ attribute
    doc = getdoc(obj) if use_getdoc else getattr(obj, "__doc__", None)

    if not doc:
        return None

    if not docstring_parser:
        # If parser not available, just return the raw docstring as short description
        return {"short": doc.strip()}

    try:
        parsed = docstring_parser.parse(doc)
        result = {
            "short": parsed.short_description,
            "long": parsed.long_description,
            "params": [
                (p.arg_name, p.type_name, p.description)
                for p in parsed.params
            ],
            "returns": parsed.returns.description
            if parsed.returns
            else None,
            "raises": [
                (e.type_name, e.description) for e in parsed.raises
            ],
        }
        # Filter out empty lists or None values for cleaner dictionary
        return {
            k: v
            for k, v in result.items()
            if v and (not isinstance(v, list) or len(v) > 0)
        }
    except Exception as e:
        logger.warning(f"Failed to parse docstring for {obj}: {e}")
        # Fallback to simple docstring if parsing fails
        return {"short": doc.strip()}


# -----------------------------------------------------------------------------
# Public API: format_to_markdown
# -----------------------------------------------------------------------------


def format_to_markdown(
    target: Any,
    indent: int = 0,
    code_block: bool = False,
    compact: bool = False,
    show_types: bool = True,
    show_title: bool = True,
    show_bullets: bool = True,
    show_docs: bool = True,
    bullet_style: str = "-",
    language: str | None = None,
    show_header: bool = True,
    schema: bool = False,
    _visited: set[int] | None = None,
) -> str:
    """
    Formats a target object into markdown optimized for LLM prompts.

    This function takes a target object and converts it into a markdown string
    that is optimized for use in language model prompts. It supports various
    options to customize the output, including indentation, code blocks,
    compact formatting, type annotations, and more.

    Args:
        target (Any): The object to format into markdown.
        indent (int, optional): The number of indentation levels to apply. Defaults to 0.
        code_block (bool, optional): Whether to format the output as a code block. Defaults to False.
        compact (bool, optional): Whether to use compact formatting. Defaults to False.
        show_types (bool, optional): Whether to include type annotations. Defaults to True.
        show_title (bool, optional): Whether to include the title of the object. Defaults to True.
        show_bullets (bool, optional): Whether to include bullet points. Defaults to True.
        show_docs (bool, optional): Whether to include documentation strings. Defaults to True.
        bullet_style (str, optional): The style of bullet points to use. Defaults to "-".
        language (str | None, optional): The language for code block formatting. Defaults to None.
        show_header (bool, optional): Whether to include the header of the object. Defaults to True.
        schema (bool, optional): If True, only show schema. If False, show values for initialized objects. Defaults to False.
        _visited (set[int] | None, optional): A set of visited object IDs to avoid circular references. Defaults to None.

    Returns:
        str: The formatted markdown string.
    """

    # Key function for caching. Uses make_hashable on a tuple of relevant arguments.
    # Use id(target) for the target object itself as make_hashable might be expensive
    # for complex objects, and the cache key should reflect the specific object instance
    # and the formatting options.
    # Pass id(_visited) as well to differentiate calls with different visited sets
    # (though recursive calls pass copies, the initial call's set matters).
    @cached(
        lambda target,
        indent,
        code_block,
        compact,
        show_types,
        show_title,
        show_bullets,
        show_docs,
        bullet_style,
        language,
        show_header,
        schema,
        _visited: (
            id(target),  # Use object ID for the target itself
            indent,
            code_block,
            compact,
            show_types,
            show_title,
            show_bullets,
            show_docs,
            bullet_style,
            language,
            show_header,
            schema,
            id(_visited)
            if _visited is not None
            else None,  # Use ID for the visited set
        )
    )
    def _format_to_markdown(
        target: Any,
        indent: int = 0,
        code_block: bool = False,
        compact: bool = False,
        show_types: bool = True,
        show_title: bool = True,
        show_bullets: bool = True,
        show_docs: bool = True,
        bullet_style: str = "-",
        language: str | None = None,
        show_header: bool = True,
        schema: bool = False,
        _visited: set[int] | None = None,
    ) -> str:
        # Initialize visited set if not provided (for the initial call)
        visited = _visited if _visited is not None else set()
        obj_id = id(target)

        # Check for circular references
        if obj_id in visited:
            return "<circular>"

        # Add current object to visited set for recursive calls
        # Pass a copy of the set to recursive calls to isolate their visited state
        visited_copy = visited.copy()
        visited_copy.add(obj_id)

        prefix = "  " * indent
        bullet = f"{bullet_style} " if show_bullets else ""

        # Handle primitive types and bytes
        if target is None or isinstance(target, (str, int, float, bool)):
            return str(target)
        if isinstance(target, bytes):
            return f"b'{target.hex()}'"

        # Handle Pydantic models
        if isinstance(target, BaseModel) or (
            isinstance(target, type) and issubclass(target, BaseModel)
        ):
            is_class = isinstance(target, type)
            model_name = (
                target.__name__ if is_class else target.__class__.__name__
            )

            if code_block:
                if (
                    schema or is_class
                ):  # If schema is True or it's the class itself, show schema
                    data = {}
                    for field, field_info in target.model_fields.items():
                        # Get the full type name including Optional if present
                        type_name = get_type_name(field_info.annotation)
                        # If the field has a default value and it's not None, add it to the type
                        if not is_class and field_info.default is not None:
                            type_name = (
                                f"{type_name} = {field_info.default}"
                            )
                        data[field] = type_name
                    # Format schema dictionary as JSON-like structure
                    json_lines = [
                        f'{prefix}  "{k}": "{v}"' for k, v in data.items()
                    ]
                    json_str = (
                        "{\n" + ",\n".join(json_lines) + f"\n{prefix}}}"
                    )
                else:  # Show actual values for an instance if schema is False
                    # Use model_dump to get a JSON-serializable dictionary representation
                    data = target.model_dump(mode="json")
                    # Use json.dumps for proper JSON formatting with indentation
                    json_str = json.dumps(data, indent=2)

                lang_tag = f"{language or 'json'}"
                return f"```{lang_tag}\n{json_str}\n```"

            # Non-code-block formatting for Pydantic models
            header_parts = (
                [f"{prefix}{bullet}**{model_name}**:"]
                if show_title
                else []
            )
            if show_docs and show_header:
                try:
                    # Parse docstring from the class or instance's class
                    doc_obj = target if is_class else target.__class__
                    # *** MODIFICATION HERE: Use use_getdoc=False to get direct docstring ***
                    doc_dict = _parse_docstring(doc_obj, use_getdoc=False)
                    if doc_dict:
                        # Filter out 'params' as they are listed as fields
                        doc_dict_filtered = doc_dict.copy()
                        doc_dict_filtered.pop("params", None)
                        doc_md = format_docstring(
                            doc_dict_filtered, prefix + "  ", compact
                        )
                        if doc_md:
                            header_parts.append(doc_md)
                except Exception as e:
                    logger.warning(
                        f"Error parsing docstring for {model_name}: {e}"
                    )

            header = "\n".join(header_parts) if header_parts else ""

            # Iterate over model_fields which correctly represent user-defined fields
            fields = target.model_fields.items()
            field_lines = []
            # Determine base indentation for fields based on compact mode
            field_indent_base = indent + (1 if not compact else 0)

            for key, field_info in fields:
                field_line_parts = []
                # Get the full type name including Optional if present
                type_name = get_type_name(field_info.annotation)
                type_info = f": {type_name}" if show_types else ""

                # *** MODIFICATION HERE: Explicitly check schema and is_class to show value ***
                should_show_value = not schema and not is_class

                if should_show_value:  # Show value only if it's an instance AND schema is False
                    # Get the actual value from the instance
                    value = getattr(
                        target, key, "<missing>"
                    )  # Use getattr with default for robustness

                    # Recursively format the value
                    formatted_value = _format_to_markdown(
                        value,
                        field_indent_base
                        + (
                            1 if not compact else 0
                        ),  # Increase indent for nested values
                        code_block=False,  # Don't force code block for nested values unless specified
                        compact=compact,
                        show_types=show_types,
                        show_title=False,  # Don't show title for nested fields
                        show_bullets=False,  # Don't show bullets for nested values
                        show_docs=False,
                        bullet_style=bullet_style,
                        language=language,
                        show_header=False,
                        schema=schema,  # Pass schema flag down
                        _visited=visited_copy,  # Pass the copy of visited set
                    ).lstrip()  # Remove any leading indentation from the recursive call

                    # Format the field line: "key: Type = Value" or "key: Type:\n  Value"
                    if "\n" in formatted_value:
                        # If the formatted value is multi-line, put it on the next line
                        field_line_parts.append(
                            f"{prefix}{bullet}{key}{type_info}:"
                        )
                        field_line_parts.append(f"{formatted_value}")
                    else:
                        # Single line value
                        field_line_parts.append(
                            f"{prefix}{bullet}{key}{type_info} = {formatted_value}"
                        )
                else:  # Schema mode or is_class: show only key and type
                    field_line_parts.append(
                        f"{prefix}{bullet}{key}{type_info}"
                    )

                # Add the formatted field line(s) to the list
                field_lines.extend(field_line_parts)

            if compact and field_lines:
                # For compact, join fields on a single line
                # Re-generate fields_str to correctly apply schema/instance logic
                fields_str_parts = []
                for key, field_info in fields:
                    field_part = f"{key}{f': {get_type_name(field_info.annotation)}' if show_types else ''}"
                    # *** MODIFICATION HERE: Use should_show_value flag ***
                    if should_show_value:
                        field_part += f"={getattr(target, key)}"  # Add value for instance if not schema
                    fields_str_parts.append(field_part)
                fields_str = ", ".join(fields_str_parts)

                if show_title:
                    return (
                        f"{header.strip()} {fields_str}"
                        if header
                        else fields_str
                    )
                else:
                    return fields_str
            else:
                # Non-compact: header followed by indented fields
                content_lines = []
                if header:
                    content_lines.append(header)
                content_lines.extend(field_lines)
                return "\n".join(content_lines)

        # Handle collections (list, tuple, set)
        if isinstance(target, (list, tuple, set)):
            if not target:
                return (
                    "[]"
                    if isinstance(target, list)
                    else "()"
                    if isinstance(target, tuple)
                    else "{}"
                )

            # If code_block is true and items are complex (dicts, BaseModels),
            # try to dump as JSON.
            if code_block and all(
                isinstance(item, (dict, BaseModel)) for item in target
            ):
                try:
                    # Convert BaseModel instances to dictionaries for JSON serialization
                    # Apply schema logic here as well
                    data_to_dump = []
                    for item in target:
                        if isinstance(item, BaseModel):
                            if schema:  # If schema, represent as schema
                                data_to_dump.append(
                                    {
                                        field: get_type_name(
                                            field_info.annotation
                                        )
                                        for field, field_info in item.model_fields.items()
                                    }
                                )
                            else:  # If not schema, dump values
                                data_to_dump.append(
                                    item.model_dump(mode="json")
                                )
                        elif isinstance(item, dict):  # Handle nested dicts
                            if schema:
                                # Simplified schema for dict items: key and type of value
                                data_to_dump.append(
                                    {
                                        k: get_type_name(type(v))
                                        for k, v in item.items()
                                    }
                                )
                            else:
                                data_to_dump.append(item)
                        else:  # Handle other types in collection
                            data_to_dump.append(item)

                    json_str = json.dumps(data_to_dump, indent=2)
                    return f"```{language or 'json'}\n{json_str}\n```"
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize collection to JSON code block: {e}"
                    )
                    # Fallback to non-code-block formatting if JSON serialization fails

            # Non-code-block formatting for collections
            type_name = target.__class__.__name__ if show_types else ""
            header = ""
            if show_title:
                header = (
                    f"{prefix}{bullet}**{type_name}**:"
                    if show_types
                    else f"{prefix}{bullet}Collection:"
                )

            indent_step = 1 if compact else 2
            item_indent = indent + indent_step

            items = []
            for item in target:
                # Recursively format each item in the collection
                formatted_item = _format_to_markdown(
                    item,
                    item_indent,  # Increase indent for nested items
                    code_block,
                    compact,
                    show_types,
                    show_title,  # Pass show_title to nested items
                    show_bullets,  # Pass show_bullets to nested items
                    show_docs,
                    bullet_style,
                    language,
                    show_header,
                    schema,  # Pass schema flag down
                    visited_copy,  # Pass the copy of visited set
                )
                # Add bullet and indentation to the formatted item
                items.append(
                    f"{'  ' * item_indent}{bullet_style} {formatted_item.lstrip()}"
                )

            # Remove empty strings from lines before joining
            final_lines = [
                line for line in [header] + items if line.strip()
            ]
            return "\n".join(final_lines)

        # Handle dictionaries
        if isinstance(target, dict):
            if not target:
                return "{}"

            if code_block:
                # For dictionaries in code block, just dump as JSON
                # Apply schema logic for nested models/dataclasses within the dict
                if schema:
                    # Create a schema representation of the dictionary
                    schema_data = {}
                    for key, value in target.items():
                        # *** MODIFICATION HERE: Check if value is a model or dataclass type OR instance ***
                        if (
                            isinstance(value, (BaseModel))
                            or (
                                isinstance(value, type)
                                and issubclass(value, BaseModel)
                            )
                            or is_dataclass(value)
                            or (
                                isinstance(value, type)
                                and is_dataclass(value)
                            )
                        ):
                            # If value is a model/dataclass (type or instance), show its schema
                            schema_data[key] = _format_to_markdown(
                                value,
                                schema=True,
                                code_block=False,
                                show_title=False,
                                show_bullets=False,
                                show_docs=False,
                                _visited=visited_copy,
                            ).strip()
                        else:
                            # Otherwise, show the type name of the value
                            schema_data[key] = get_type_name(type(value))
                    json_str = json.dumps(schema_data, indent=2)
                else:
                    # Dump actual values
                    json_str = json.dumps(target, indent=2)

                return f"```{language or 'json'}\n{json_str}\n```"

            # Non-code-block formatting for dictionaries
            type_name = target.__class__.__name__ if show_types else ""
            header = ""
            if show_title:
                header = (
                    f"{prefix}{bullet}**{type_name}**:"
                    if show_types
                    else f"{prefix}{bullet}Dictionary:"
                )

            indent_step = 1 if compact else 2
            item_indent = indent + indent_step

            items = []
            for key, value in target.items():
                # Recursively format each value in the dictionary
                formatted_value = _format_to_markdown(
                    value=value,
                    item_indent=item_indent,  # Increase indent for nested values
                    code_block=code_block,
                    compact=compact,
                    show_types=show_types,
                    show_title=show_title,  # Pass show_title to nested values
                    show_bullets=False,  # Usually don't bullet the value itself within a dict item line
                    show_docs=show_docs,
                    bullet_style=bullet_style,
                    language=language,
                    show_header=show_header,
                    schema=schema,  # Pass schema flag down
                    visited_copy=visited_copy,  # Pass the copy of visited set
                )
                # Format the dictionary item: "key: Value"
                # Ensure the key itself gets bulleted if show_bullets is true for the parent dict
                if show_bullets:
                    items.append(
                        f"{prefix}{'  ' * item_indent}{bullet_style} {key}: {formatted_value.lstrip()}"
                    )
                else:
                    items.append(
                        f"{prefix}{'  ' * item_indent}{key}: {formatted_value.lstrip()}"
                    )

            final_lines = [
                line for line in [header] + items if line.strip()
            ]
            return "\n".join(final_lines)

        # Handle dataclasses
        if is_dataclass(target):
            is_class = isinstance(
                target, type
            )  # Check if it's a dataclass type or instance
            type_name = (
                target.__name__ if is_class else target.__class__.__name__
            )

            if code_block:  # Dataclasses don't have model_dump, so we manually build dict
                if (
                    schema or is_class
                ):  # If schema is True or it's the class itself, show schema
                    data = {
                        f.name: get_type_name(f.type)
                        for f in dataclass_fields(target)
                    }
                    # Format schema dictionary as JSON-like structure
                    json_lines = [
                        f'{prefix}  "{k}": "{v}"' for k, v in data.items()
                    ]
                    json_str = (
                        "{\n" + ",\n".join(json_lines) + f"\n{prefix}}}"
                    )
                else:  # Show actual values for an instance if schema is False
                    # Manually build dictionary from dataclass fields and values
                    data = {
                        f.name: getattr(target, f.name)
                        for f in dataclass_fields(target)
                    }
                    # Use json.dumps for proper JSON formatting with indentation
                    json_str = json.dumps(data, indent=2)

                lang_tag = f"{language or 'json'}"
                return f"```{lang_tag}\n{json_str}\n```"

            # Non-code-block formatting for dataclasses
            header_parts = (
                [f"{prefix}{bullet}**{type_name}**:"] if show_title else []
            )
            if show_docs and show_header:
                try:
                    # Parse docstring from the class or instance's class
                    doc_obj = target if is_class else target.__class__
                    # *** MODIFICATION HERE: Use use_getdoc=False to get direct docstring ***
                    doc_dict = _parse_docstring(doc_obj, use_getdoc=False)
                    if doc_dict:
                        # Filter out 'params' as they are listed as fields
                        doc_dict_filtered = doc_dict.copy()
                        doc_dict_filtered.pop("params", None)
                        doc_md = format_docstring(
                            doc_dict_filtered, prefix + "  ", compact
                        )
                        if doc_md:
                            header_parts.append(doc_md)
                except Exception as e:
                    logger.warning(
                        f"Error parsing docstring for {type_name}: {e}"
                    )

            header = "\n".join(header_parts) if header_parts else ""

            fields_list = dataclass_fields(target)
            field_lines = []
            # Determine base indentation for fields based on compact mode
            field_indent_base = indent + (1 if not compact else 0)

            for f in fields_list:
                field_line_parts = []
                type_info = (
                    f": {get_type_name(f.type)}" if show_types else ""
                )

                # *** MODIFICATION HERE: Explicitly check schema and is_class to show value ***
                should_show_value = not schema and not is_class

                if should_show_value:  # Show value only if it's an instance AND schema is False
                    # Get the actual value from the instance
                    value = getattr(target, f.name)

                    # Recursively format the value
                    formatted_value = _format_to_markdown(
                        value,
                        field_indent_base
                        + (
                            1 if not compact else 0
                        ),  # Increase indent for nested values
                        code_block=False,  # Don't force code block for nested values unless specified
                        compact=compact,
                        show_types=show_types,
                        show_title=False,  # Don't show title for nested fields
                        show_bullets=False,  # Don't show bullets for nested values
                        show_docs=False,
                        bullet_style=bullet_style,
                        language=language,
                        show_header=False,
                        schema=schema,  # Pass schema flag down
                        _visited=visited_copy,  # Pass the copy of visited set
                    ).lstrip()  # Remove any leading indentation from the recursive call

                    # Format the field line: "key: Type = Value" or "key: Type:\n  Value"
                    if "\n" in formatted_value:
                        # If the formatted value is multi-line, put it on the next line
                        field_line_parts.append(
                            f"{prefix}{bullet}{f.name}{type_info}:"
                        )
                        field_line_parts.append(f"{formatted_value}")
                    else:
                        # Single line value
                        field_line_parts.append(
                            f"{prefix}{bullet}{f.name}{type_info} = {formatted_value}"
                        )
                else:  # Schema mode or is_class: show only key and type
                    field_line_parts.append(
                        f"{prefix}{bullet}{f.name}{type_info}"
                    )

                # Add the formatted field line(s) to the list
                field_lines.extend(field_line_parts)

            if compact and field_lines:
                # For compact, join fields on a single line
                # Re-generate fields_str to correctly apply schema/instance logic
                fields_str_parts = []
                for f in fields_list:
                    field_part = f"{f.name}{f': {get_type_name(f.type)}' if show_types else ''}"
                    # *** MODIFICATION HERE: Use should_show_value flag ***
                    if should_show_value:
                        field_part += f"={getattr(target, f.name)}"  # Add value for instance if not schema
                    fields_str_parts.append(field_part)
                fields_str = ", ".join(fields_str_parts)

                if show_title:
                    return (
                        f"{header.strip()} {fields_str}"
                        if header
                        else fields_str
                    )
                else:
                    return fields_str
            else:
                # Non-compact: header followed by indented fields
                content_lines = []
                if header:
                    content_lines.append(header)
                content_lines.extend(field_lines)
                return "\n".join(content_lines)

        # Fallback for any other types, just return string representation
        return str(target)

    # Initial call to the inner cached function
    # The key_fn for the outer @cached decorator will be executed here
    return _format_to_markdown(
        target,
        indent,
        code_block,
        compact,
        show_types,
        show_title,
        show_bullets,
        show_docs,
        bullet_style,
        language,
        show_header,
        schema,
        _visited,
    )


def format_messages(messages: Any) -> List[Message]:
    """Formats the input into a list of chat completion messages."""

    @cached(lambda messages: make_hashable(messages) if messages else "")
    def _format_messages(messages: Any) -> List[Message]:
        try:
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]
            if not isinstance(messages, list):
                messages = [messages]

            normalized = []
            for message in messages:
                if isinstance(message, dict):
                    # Create a new dict to avoid modifying the original
                    normalized.append({**message})
                elif hasattr(message, "model_dump"):
                    normalized.append(message.model_dump())
                else:
                    raise ValueError(f"Invalid message format: {message}")
            return normalized
        except Exception as e:
            logger.debug(f"Error normalizing messages: {e}")
            raise

    return _format_messages(messages)


def format_system_prompt(
    messages: List[Message],
    system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
    blank: bool = False,
) -> List[Message]:
    """
    Normalizes a message thread by gathering all system messages at the start.

    Args:
        messages: List of messages to normalize.
        system_prompt: Optional system prompt to prepend.
        blank: If True, ensures at least one system message exists (even empty).

    Returns:
        A normalized list of messages.
    """

    @cached(
        lambda messages, system_prompt=None, blank=False: make_hashable(
            (messages, system_prompt, blank)
        )
    )
    def _format_system_prompt(
        messages: Any,
        system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
        blank: bool = False,
    ) -> List[Message]:
        try:
            system_messages = [
                msg for msg in messages if msg.get("role") == "system"
            ]
            other_messages = [
                msg for msg in messages if msg.get("role") != "system"
            ]

            if system_prompt:
                if isinstance(system_prompt, str):
                    new_system = {
                        "role": "system",
                        "content": system_prompt,
                    }
                elif isinstance(system_prompt, dict):
                    new_system = {**system_prompt, "role": "system"}
                    if "content" not in new_system:
                        raise ValueError(
                            "System prompt dict must contain 'content' field"
                        )
                else:
                    raise ValueError(
                        "System prompt must be string or dict"
                    )
                system_messages.insert(0, new_system)

            if not system_messages and blank:
                system_messages = [{"role": "system", "content": ""}]
            elif not system_messages:
                return messages

            if len(system_messages) > 1:
                combined_content = "\n".join(
                    msg["content"] for msg in system_messages
                )
                system_messages = [
                    {"role": "system", "content": combined_content}
                ]

            return system_messages + other_messages
        except Exception as e:
            logger.debug(f"Error normalizing system prompt: {e}")
            raise

    return _format_system_prompt(messages, system_prompt, blank)


__all__ = [
    "format_docstring",
    "format_to_markdown",
    "get_type_name",
    "format_messages",
    "format_system_prompt",
]
