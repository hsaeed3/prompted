"""
## 💭 chatspec.markdown

Contains the `markdownify` function, a utility for cleanly formatting any object
into a markdown text string or code block. Additionally, contains the
`MarkdownObject` class, a container for objects that can hold their original values
as well as their markdown or code block representations.
"""

import json
from cachetools import cached
from dataclasses import is_dataclass, fields as dataclass_fields
from inspect import getdoc
from pydantic import BaseModel
from typing import Any, Generic, Optional, TypeVar
from typing_extensions import TypedDict

from .utils import _make_hashable, logger, _cached

__all__ = [
    "MarkdownObject",
    "MarkdownConfig",
    "markdownify",
]


# -----------------------------------------------------------------------------
# [MarkdownObject & MarkdownParams]
# -----------------------------------------------------------------------------


_MarkdownObjectT = TypeVar("_MarkdownObjectT")
"""
A type variable representing the type of the value stored within a `MarkdownObject`
instance.
"""


class MarkdownConfig(TypedDict, total=False):
    markdown: bool
    """
    If `True`, the value will be rendered as markdown.
    
    *See `chatspec.utils.markdownify()` for the method that powers this feature.*
    """
    code_block: bool
    """
    If `True`, the value will be rendered as a code block.
    """
    # [Markdown Specific Config -- Only Used When `markdown = True`]
    indent: int
    """
    The number of spaces markdown formatting will be indented by. This applies to
    both code blocks and items rendered as tables or lists.
    """
    compact: bool
    """
    If `True`, the markdown will be rendered in a more compact format, with less
    vertical space between items.
    """
    show_header: bool
    """
    If `True`, the header of the value will be displayed within a markdown text block.
    """
    show_types: bool
    """
    If `True`, the type of the value will be displayed within a markdown text block.
    """
    show_title: bool
    """
    If `True`, the title of the value will be displayed within a markdown text block.
    """
    show_bullets: bool
    """
    If `True`, the value will be rendered with bullet points.
    """
    show_docs: bool
    """
    If `True`, the documentation string of the value will be displayed within a
    markdown text block.
    """
    bullet_style: str
    """
    The style of bullet points to use.
    """
    language: str | None
    """
    The language to use for code block formatting.
    """
    
    
class MarkdownObject(Generic[_MarkdownObjectT]):
    """
    A container for easily managing & rendering markdown representations of
    various objects, while retaining their original values & functionality.
    """
    config: Optional[MarkdownConfig] = None
    """
    The configuration for markdown rendering.
    """
    value: _MarkdownObjectT
    """
    The original `value` or instance of the object.
    """
    markdown: Optional[str] = None
    """
    The markdown representation of the object.
    """
    
    def __init__(
        self,
        value: _MarkdownObjectT,
        indent: int = 0,
        code_block: bool = False,
        compact: bool = False,
        show_types: bool = True,
        show_title: bool = True,
        show_bullets: bool = True,
        show_docs: bool = True,
        bullet_style: str = "-",
        language: Optional[str] = None,
        show_header: bool = True,
        config: Optional[MarkdownConfig] = None
    ) -> None:
        """
        Initializes the MarkdownObject with a given value and optional configuration.
        Immediately updates the markdown representation.
        """
        self.value = value
        self.config = config or {
            "indent": indent,
            "code_block": code_block,
            "compact": compact,
            "show_types": show_types,
            "show_title": show_title,
            "show_bullets": show_bullets,
            "show_docs": show_docs,
            "bullet_style": bullet_style,
            "language": language,
            "show_header": show_header,
        }
        self.markdown = None
        self.update()

    def update(
        self,
        indent: Optional[int] = None,
        code_block: Optional[bool] = None,
        compact: Optional[bool] = None,
        show_types: Optional[bool] = None,
        show_title: Optional[bool] = None,
        show_bullets: Optional[bool] = None,
        show_docs: Optional[bool] = None,
        bullet_style: Optional[str] = None,
        language: Optional[str] = None,
        show_header: Optional[bool] = None,
        config: Optional[MarkdownConfig] = None
    ) -> None:
        """
        Updates the markdown representation using the `markdownify` function.
        If a configuration is provided, it applies those parameters; otherwise, defaults are used.
        """
        config = config or self.config
        self.markdown = markdownify(
            self.value,
            indent=indent if indent is not None else config.get("indent", 0),
            code_block=code_block if code_block is not None else config.get("code_block", False),
            compact=compact if compact is not None else config.get("compact", False),
            show_types=show_types if show_types is not None else config.get("show_types", True),
            show_title=show_title if show_title is not None else config.get("show_title", True),
            show_bullets=show_bullets if show_bullets is not None else config.get("show_bullets", True),
            show_docs=show_docs if show_docs is not None else config.get("show_docs", True),
            bullet_style=bullet_style if bullet_style is not None else config.get("bullet_style", "-"),
            language=language if language is not None else config.get("language", None),
            show_header=show_header if show_header is not None else config.get("show_header", True),
        )

    def __str__(self) -> str:
        """
        Returns the markdown representation when the object is converted to a string.
        """
        return self.markdown or str(self.value)

    def __repr__(self) -> str:
        """
        Returns a developer-friendly representation of the MarkdownObject.
        """
        return (
            f"MarkdownObject(value={self.value!r}, "
            f"markdown={self.markdown!r})"
        )


# -----------------------------------------------------------------------------
# [.markdownify()]
# -----------------------------------------------------------------------------


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


def _format_docstring(
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


@_cached(
    lambda cls: _make_hashable(cls) if cls else ""
)
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
# Public API: markdownify
# -----------------------------------------------------------------------------


@_cached(
    lambda target, indent=0, code_block=False, compact=False, show_types=True, show_title=True, show_bullets=True, show_docs=True, bullet_style="-", language=None, show_header=True, _visited=None: _make_hashable(
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
            _visited
        )
    )
)  
def markdownify(
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
        _visited (set[int] | None, optional): A set of visited object IDs to avoid circular references. Defaults to None.

    Returns:
        str: The formatted markdown string.

    Example:
        ```python
        from pydantic import BaseModel

        class ExampleModel(BaseModel):
            name: str
            value: int

        example_instance = ExampleModel(name="example", value=42)

        # Example with indent
        markdown_indent = markdownify(example_instance, indent=1)
        print(markdown_indent)
        # Output:
        #   - **ExampleModel**:
        #     - name: str
        #     - value: int

        # Example with code_block
        markdown_code_block = markdownify(example_instance, code_block=True)
        print(markdown_code_block)
        # Output:
        # ```
        # {
        #   "name": "example",
        #   "value": 42
        # }
        # ```

        # Example with compact
        markdown_compact = markdownify(example_instance, compact=True)
        print(markdown_compact)
        # Output:
        # - **ExampleModel**: name: str, value: int

        # Example with show_types
        markdown_show_types = markdownify(example_instance, show_types=False)
        print(markdown_show_types)
        # Output:
        # - **ExampleModel**:
        #   - name
        #   - value

        # Example with show_title
        markdown_show_title = markdownify(example_instance, show_title=False)
        print(markdown_show_title)
        # Output:
        # - name: str
        # - value: int

        # Example with show_bullets
        markdown_show_bullets = markdownify(example_instance, show_bullets=False)
        print(markdown_show_bullets)
        # Output:
        # **ExampleModel**:
        # name: str
        # value: int

        # Example with show_docs
        markdown_show_docs = markdownify(example_instance, show_docs=False)
        print(markdown_show_docs)
        # Output:
        # - **ExampleModel**:
        #   - name: str
        #   - value: int

        # Example with bullet_style
        markdown_bullet_style = markdownify(example_instance, bullet_style="*")
        print(markdown_bullet_style)
        # Output:
        # * **ExampleModel**:
        #   * name: str
        #   * value: int

        # Example with language
        markdown_language = markdownify(example_instance, code_block=True, language="python")
        print(markdown_language)
        # Output:
        # ```python
        # {
        #   "name": "example",
        #   "value": 42
        # }
        # ```

        # Example with show_header
        markdown_show_header = markdownify(example_instance, show_header=False)
        print(markdown_show_header)
        # Output:
        # - name: str
        # - value: int
        ```
    """
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
                target.__name__ if is_class else target.__class__.__name__
            )

            if code_block:
                data = (
                    target.model_dump()
                    if not is_class
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
                    if not is_class
                    else "{\n" + "\n".join(
                        f'  "{k}": "{v}"' 
                        for k, v in data.items()
                    ) + "\n}"
                )
                lang_tag = f"{language or ''}"
                return f"```{lang_tag}\n{json_str}\n```"

            header_parts = [f"{prefix}{bullet}**{model_name}**:"] if show_title else []
            if show_docs and show_header:
                try:
                    doc_dict = _parse_docstring(target)
                    if doc_dict:
                        doc_md = _format_docstring(
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
                    field_lines.append(", ".join(field_parts))
                else:
                    field_parts = [
                        f"{field_prefix}{bullet}{key}"
                        + (f": {get_type_name(field_info.annotation)}" if show_types else "")
                    ]
                    field_lines.extend(field_parts)

            if compact and field_lines:
                return f"{header} {', '.join(field_lines)}" if show_title else ', '.join(field_lines)
            else:
                if show_bullets:
                    if show_title:
                        return "\n".join(filter(None, [header] + field_lines))
                    else:
                        # When show_title is False, don't indent the field lines
                        field_lines = [f"{prefix}{bullet}{key}" + (f": {get_type_name(field_info.annotation)}" if show_types else "")
                                     for key, field_info in fields]
                        return "\n".join(field_lines)
                else:
                    # Remove indentation when show_bullets is False
                    field_lines = [line.lstrip() for line in field_lines]
                    return "\n".join(filter(None, [header] + field_lines))
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
            json_str = json.dumps(
                list(target), indent=2
            )
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
            f"{item_prefix}{bullet}{markdownify(item, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, visited.copy())}"
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
            json_str = json.dumps(
                target, indent=2
            )
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
            f"{item_prefix}{bullet}{key}: {markdownify(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, visited.copy())}"
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
            f"{item_prefix}{bullet}{name}: {markdownify(value, indent + indent_step, code_block, compact, show_types, show_title, show_bullets, show_docs, bullet_style, language, show_header, visited.copy())}"
            for name, value in fields_list
        ]
        return (
            "\n".join([header] + items)
            if show_types and show_title
            else "\n".join(items)
        )

    return str(target)


