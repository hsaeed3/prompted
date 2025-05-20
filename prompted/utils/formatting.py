"""
💬 prompted.utils.formatting

Contains utilities used for text formatting & rendering.
"""

import json
import logging
from dataclasses import is_dataclass, fields as dataclass_fields
from inspect import getdoc
from pydantic import BaseModel
from typing import Any, Optional

from ..common.cache import (
    cached,
    make_hashable
)


logger = logging.getLogger(__name__)

__all__ = [
    "format_docstring",
    "format_to_markdown",
    "get_type_name",
]


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
    import docstring_parser

    try:
        if hasattr(field_info, "__doc__") and field_info.__doc__:
            doc = docstring_parser.parse(field_info.__doc__)
            if doc.short_description:
                return doc.short_description

        if hasattr(field_info, "description"):
            return field_info.description

        return None
    except Exception:
        return None


def format_docstring(
    doc_dict: dict, prefix: str = "", compact: bool = False
) -> str:
    """Format parsed docstring into markdown.

    Args:
        doc_dict: Dictionary containing parsed docstring sections
        prefix: String to prepend to each line for indentation
        compact: If True, produces more compact output

    Returns:
        Formatted markdown string
    """
    import docstring_parser

    try:
        if not doc_dict:
            return ""

        if isinstance(doc_dict, str):
            doc = docstring_parser.parse(doc_dict)
        else:
            doc = docstring_parser.parse(str(doc_dict))

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
    except Exception:
        return str(doc_dict)


@cached(lambda cls: make_hashable(cls) if cls else "")
def get_type_name(cls: Any) -> str:
    """Get a clean type name for display"""
    # Handle None type
    if cls is None:
        return "None"
    # Handle basic types with __name__ attribute
    if hasattr(cls, "__name__"):
        return cls.__name__
    # Handle typing types like Optional, List etc
    if hasattr(cls, "__origin__"):
        # Get the base type (List, Optional etc)
        origin = cls.__origin__.__name__
        # Handle special case of Optional which is really Union[T, None]
        if (
            origin == "Union"
            and len(cls.__args__) == 2
            and cls.__args__[1] is type(None)
        ):
            return f"Optional[{get_type_name(cls.__args__[0])}]"
        # For other generic types, recursively get names of type arguments
        args = ", ".join(get_type_name(arg) for arg in cls.__args__)
        return f"{origin}[{args}]"

    # Fallback for any other types
    return str(cls)


def _parse_docstring(obj: Any) -> Optional[dict]:
    """
    Extract and parse docstring from an object using docstring-parser.

    Returns:
        Dictionary containing parsed docstring components:
        - short_description: Brief description
        - long_description: Detailed description
        - params: List of parameters
        - returns: Return value description
        - raises: List of exceptions
    """
    import docstring_parser

    doc = getdoc(obj)
    if not doc:
        return None

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
        return {k: v for k, v in result.items() if v}
    except:
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

    @cached(
        lambda target,
        indent=0,
        code_block=False,
        compact=False,
        show_types=True,
        show_title=True,
        show_bullets=True,
        show_docs=True,
        bullet_style="-",
        language=None,
        show_header=True,
        schema=False,
        _visited=None: make_hashable(
            (
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
        visited = _visited or set()
        obj_id = id(target)
        if obj_id in visited:
            return "<circular>"
        visited.add(obj_id)

        prefix = "  " * indent
        bullet = f"{bullet_style} " if show_bullets else ""

        if target is None or isinstance(target, (str, int, float, bool)):
            return str(target)
        if isinstance(target, bytes):
            return f"b'{target.hex()}'"

        # Handle Pydantic models
        try:
            if isinstance(target, BaseModel) or (
                isinstance(target, type) and issubclass(target, BaseModel)
            ):
                is_class = isinstance(target, type)
                model_name = (
                    target.__name__
                    if is_class
                    else target.__class__.__name__
                )

                if code_block:
                    data = (
                        target.model_dump()
                        if not is_class and not schema
                        else {
                            field: f"{get_type_name(field_info.annotation)}"
                            if show_types
                            else "..."
                            for field, field_info in target.model_fields.items()
                        }
                    )
                    # Format JSON with proper indentation
                    json_str = (
                        json.dumps(data, indent=2)
                        if not is_class and not schema
                        else "{\n"
                        + "\n".join(
                            f'  "{k}": "{v}"' for k, v in data.items()
                        )
                        + "\n}"
                    )
                    lang_tag = f"{language or ''}"
                    return f"```{lang_tag}\n{json_str}\n```"

                header_parts = (
                    [f"{prefix}{bullet}**{model_name}**:"]
                    if show_title
                    else []
                )
                if show_docs and show_header:
                    try:
                        doc_dict = _parse_docstring(target)
                        if doc_dict:
                            doc_md = format_docstring(
                                doc_dict, prefix + "  ", compact
                            )
                            if doc_md:
                                header_parts.append(doc_md)
                    except Exception as e:
                        logger.warning(
                            f"Error parsing docstring for {model_name}: {e}"
                        )

                header = "\n".join(header_parts) if header_parts else ""

                fields = target.model_fields.items()
                field_lines = []
                field_prefix = prefix + ("  " if not compact else "")

                for key, field_info in fields:
                    if compact:
                        field_parts = [
                            f"{key}: {get_type_name(field_info.annotation)}"
                            if show_types
                            else key
                        ]
                        if not schema and not is_class:
                            field_parts.append(f"= {getattr(target, key)}")
                        field_lines.append(", ".join(field_parts))
                    else:
                        field_parts = [
                            f"{field_prefix}{bullet}{key}"
                            + (
                                f": {get_type_name(field_info.annotation)}"
                                if show_types
                                else ""
                            )
                        ]
                        if not schema and not is_class:
                            field_parts.append(f" = {getattr(target, key)}")
                        field_lines.extend(field_parts)

                if compact and field_lines:
                    return (
                        f"{header} {', '.join(field_lines)}"
                        if show_title
                        else ", ".join(field_lines)
                    )
                else:
                    if show_bullets:
                        if show_title:
                            return "\n".join(
                                filter(None, [header] + field_lines)
                            )
                        else:
                            # When show_title is False, don't indent the field lines
                            field_lines = [
                                f"{prefix}{bullet}{key}"
                                + (
                                    f": {get_type_name(field_info.annotation)}"
                                    if show_types
                                    else ""
                                )
                                + (
                                    f" = {getattr(target, key)}"
                                    if not schema and not is_class
                                    else ""
                                )
                                for key, field_info in fields
                            ]
                            return "\n".join(field_lines)
                    else:
                        # Remove indentation when show_bullets is False
                        field_lines = [
                            line.lstrip() for line in field_lines
                        ]
                        return "\n".join(
                            filter(None, [header] + field_lines)
                        )
        except Exception as e:
            logger.error(
                f"Error formatting pydantic model target {target} to markdown: {e}"
            )
            raise e

        # Handle collections
        if isinstance(target, (list, tuple, set)):
            if not target:
                return (
                    "[]"
                    if isinstance(target, list)
                    else "()"
                    if isinstance(target, tuple)
                    else "{}"
                )

            if code_block and isinstance(target[0], (dict, BaseModel)):
                json_str = json.dumps(list(target), indent=2)
                return f"```{language or 'json'}\n{json_str}\n```"

            type_name = target.__class__.__name__ if show_types else ""
            header = (
                f"{prefix}{bullet}**{type_name}**:"
                if show_types and show_title
                else f"{prefix}{bullet}"
            )
            indent_step = 1 if compact else 2
            item_prefix = prefix + ("  " if not compact else "")

            items = [
                f"{item_prefix}{bullet}{format_to_markdown(item, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, schema, visited.copy())}"
                for item in target
            ]
            return (
                "\n".join([header] + items)
                if show_types and show_title
                else "\n".join(items)
            )

        # Handle dictionaries
        if isinstance(target, dict):
            if not target:
                return "{}"

            if code_block:
                json_str = json.dumps(target, indent=2)
                return f"```{language or 'json'}\n{json_str}\n```"

            type_name = target.__class__.__name__ if show_types else ""
            header = (
                f"{prefix}{bullet}**{type_name}**:"
                if show_types and show_title
                else f"{prefix}{bullet}"
            )
            indent_step = 1 if compact else 2
            item_prefix = prefix + ("  " if not compact else "")

            items = [
                f"{item_prefix}{bullet}{key}: {format_to_markdown(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, schema, visited.copy())}"
                for key, value in target.items()
            ]
            return (
                "\n".join([header] + items)
                if show_types and show_title
                else "\n".join(items)
            )

        # Handle dataclasses
        if is_dataclass(target):
            type_name = target.__class__.__name__ if show_types else ""
            header = (
                f"{prefix}{bullet}**{type_name}**:"
                if show_types and show_title
                else f"{prefix}{bullet}"
            )
            indent_step = 1 if compact else 2
            item_prefix = prefix + ("  " if not compact else "")

            fields_list = [
                (f.name, getattr(target, f.name))
                for f in dataclass_fields(target)
            ]
            items = [
                f"{item_prefix}{bullet}{name}: {format_to_markdown(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, schema, visited.copy())}"
                for name, value in fields_list
            ]
            return (
                "\n".join([header] + items)
                if show_types and show_title
                else "\n".join(items)
            )

        return str(target)

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