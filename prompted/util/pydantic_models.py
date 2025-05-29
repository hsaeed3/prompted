"""
prompted.resources.pydantic_models
"""

from docstring_parser import parse
from dataclasses import is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    get_type_hints,
    Union,
    Sequence,
    Literal,
)
from pydantic import (
    BaseModel,
    create_model,
    Field,
)

from .._cache import cached, make_hashable, TYPE_MAPPING
from ..logger import _get_logger

logger = _get_logger(__name__)

__all__ = [
    "convert_to_pydantic_field",
    "convert_to_pydantic_model",
    "convert_to_selection_pydantic_model",
    "convert_to_boolean_pydantic_model",
]


# ------------------------------------------------------------------------------
# CONVERTERS
# ------------------------------------------------------------------------------


def convert_to_pydantic_field(
    type_hint: Type,
    index: Optional[int] = None,
    description: Optional[str] = None,
    default: Any = ...,
) -> Dict[str, Any]:
    """
    Creates a Pydantic field mapping from a type hint.

    Args:
        type_hint: The Python type to convert
        index: Optional index to append to field name for uniqueness
        description: Optional field description
        default: Optional default value

    Returns:
        Dictionary mapping field name to (type, Field) tuple
    """

    @cached(
        lambda type_hint, index=None, description=None, default=...: make_hashable(
            (type_hint, index, description, default)
        )
    )
    def _create_field_mapping(
        type_hint: Type,
        index: Optional[int] = None,
        description: Optional[str] = None,
        default: Any = ...,
    ) -> Dict[str, Any]:
        try:
            base_name, _ = TYPE_MAPPING.get(type_hint, ("value", type_hint))
            field_name = f"{base_name}_{index}" if index is not None else base_name
            return {
                field_name: (
                    type_hint,
                    Field(default=default, description=description),
                )
            }
        except Exception as e:
            logger.debug(f"Error creating field mapping: {e}")
            raise

    return _create_field_mapping(type_hint, index, description, default)


def convert_to_pydantic_model(
    target: Union[Type, Sequence[Type], Dict[str, Any], BaseModel, Callable],
    init: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
    field_name: Optional[str] = None,
    default: Any = ...,
) -> Union[Type[BaseModel], BaseModel]:
    """
    Converts various input types into a pydantic model class or instance.

    Args:
        target: The target to convert (type, sequence, dict, model, or function)
        init: Whether to initialize the model with values (for dataclasses/dicts)
        name: Optional name for the generated model
        description: Optional description for the model/field
        field_name: Optional field name for the generated model (If the target is a single type)
        default: Optional default value for single-type models

    Returns:
        A pydantic model class or instance if init=True
    """

    @cached(
        lambda target,
        init=False,
        name=None,
        description=None,
        field_name=None,
        default=...: make_hashable((target, init, name, description, field_name, default))
    )
    def _convert_to_pydantic_model(
        target: Union[Type, Sequence[Type], Dict[str, Any], BaseModel, Callable],
        init: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
        field_name: Optional[str] = None,
        default: Any = ...,
    ) -> Union[Type[BaseModel], BaseModel]:
        model_name = name or "GeneratedModel"

        # Handle existing Pydantic models
        if isinstance(target, type) and issubclass(target, BaseModel):
            return target

        # Handle dataclasses
        if is_dataclass(target):
            hints = get_type_hints(target)
            fields = {}

            # Parse docstring if available
            docstring = target.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            for field_name, hint in hints.items():
                description = ""
                if doc_info and doc_info.params:
                    description = next(
                        (
                            p.description
                            for p in doc_info.params
                            if p.arg_name == field_name
                        ),
                        "",
                    )

                fields[field_name] = (
                    hint,
                    Field(
                        default=getattr(target, field_name) if init else ...,
                        description=description,
                    ),
                )

            model_class = create_model(
                model_name,
                __doc__=description or (doc_info.short_description if doc_info else None),
                **fields,
            )

            if init and isinstance(target, type):
                return model_class
            elif init:
                return model_class(
                    **{field_name: getattr(target, field_name) for field_name in hints}
                )
            return model_class

        # Handle callable (functions)
        if callable(target) and not isinstance(target, type):
            fields = extract_function_fields(target)

            # Extract just the short description from the docstring
            doc_info = parse(target.__doc__ or "")
            clean_description = doc_info.short_description if doc_info else None

            return create_model(
                name or target.__name__,
                __doc__=description or clean_description,
                **fields,
            )

        # Handle single types
        if isinstance(target, type):
            field_mapping = convert_to_pydantic_field(
                target, description=description, default=default
            )
            # If field_name is provided, override the default field name
            if field_name:
                # Get the first (and only) key-value pair from field_mapping
                _, field_value = next(iter(field_mapping.items()))
                field_mapping = {field_name: field_value}
            return create_model(model_name, __doc__=description, **field_mapping)

        # Handle sequences of types
        if isinstance(target, (list, tuple)):
            field_mapping = {}
            for i, type_hint in enumerate(target):
                if not isinstance(type_hint, type):
                    raise ValueError("Sequence elements must be types")
                # If field_name is provided and this is the first type, use it
                if field_name and i == 0:
                    field_mapping.update(
                        {
                            field_name: convert_to_pydantic_field(
                                type_hint,
                                description=description,
                                default=default,
                            )[next(iter(convert_to_pydantic_field(type_hint).keys()))]
                        }
                    )
                else:
                    field_mapping.update(convert_to_pydantic_field(type_hint, index=i))
            return create_model(model_name, __doc__=description, **field_mapping)

        # Handle dictionaries
        if isinstance(target, dict):
            if init:
                model_class = create_model(
                    model_name,
                    __doc__=description,
                    **{k: (type(v), Field(default=v)) for k, v in target.items()},
                )
                return model_class(**target)
            return create_model(model_name, __doc__=description, **target)

        # Handle model instances
        if isinstance(target, BaseModel):
            # Parse docstring from the model's class
            docstring = target.__class__.__doc__
            doc_info = None
            if docstring:
                doc_info = parse(docstring)

            if init:
                fields = {}
                for k, v in target.model_dump().items():
                    description = ""
                    if doc_info and doc_info.params:
                        description = next(
                            (p.description for p in doc_info.params if p.arg_name == k),
                            "",
                        )
                    fields[k] = (
                        type(v),
                        Field(default=v, description=description),
                    )

                model_class = create_model(
                    model_name,
                    __doc__=description
                    or (doc_info.short_description if doc_info else None),
                    **fields,
                )
                return model_class(**target.model_dump())
            return target.__class__

        raise ValueError(
            f"Unsupported target type: {type(target)}. Must be a type, "
            "sequence of types, dict, dataclass, function, or Pydantic model."
        )

    return _convert_to_pydantic_model(
        target, init, name, description, field_name, default
    )


def convert_to_selection_pydantic_model(
    fields: List[str] = [],
    name: str = "Selection",
    description: str | None = None,
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for making a selection from a list of string options.

    The model will have a single field named `selection`. The type of this field
    will be `Literal[*fields]`, meaning its value must be one of the strings
    provided in the `fields` list.

    Args:
        name: The name for the created Pydantic model. Defaults to "Selection".
        description: An optional description for the model (becomes its docstring).
        fields: A list of strings representing the allowed choices for the selection.
                This list cannot be empty.

    Returns:
        A new Pydantic BaseModel class with a 'selection' field.

    Raises:
        ValueError: If the `fields` list is empty, as Literal requires at least one option.
    """
    if not fields:
        raise ValueError(
            "`fields` list cannot be empty for `create_selection_model` "
            "as it defines the possible selections for the Literal type."
        )

    # Create the Literal type from the list of field strings.
    # We can't use unpacking syntax directly with Literal, so we need to handle it differently
    if len(fields) == 1:
        selection_type = Literal[fields[0]]
    else:
        # For multiple fields, we need to use eval to create the Literal type
        # This is because Literal needs to be constructed with the actual string values
        # as separate arguments, not as a list
        literal_str = f"Literal[{', '.join(repr(f) for f in fields)}]"
        selection_type = eval(literal_str)

    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        "selection": (
            selection_type,
            Field(
                ...,
                description="The selected value from the available options.",
            ),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        if fields:
            model_docstring = (
                f"A model for selecting one option from: {', '.join(fields)}."
            )
        else:  # Should not be reached due to the check above, but for completeness
            model_docstring = "A selection model."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel


def convert_to_boolean_pydantic_model(
    name: str = "Confirmation",
    description: str | None = None,
    field_name: str = "choice",
) -> Type[BaseModel]:
    """
    Creates a Pydantic model for boolean confirmation/response.

    The model will have a single field named `confirmed`. The type of this field
    will be `bool`, meaning its value must be either True or False.

    Args:
        name: The name for the created Pydantic model. Defaults to "Confirmation".
        description: An optional description for the model (becomes its docstring).

    Returns:
        A new Pydantic BaseModel class with a 'confirmed' field.
    """
    # Define the field for the model. It's required (...).
    model_fields_definitions = {
        field_name: (
            bool,
            Field(..., description="The boolean confirmation value."),
        )
    }

    # Determine the docstring for the created model
    model_docstring = description
    if model_docstring is None:
        model_docstring = "A model for boolean confirmation."

    NewModel: Type[BaseModel] = create_model(
        name,
        __base__=BaseModel,
        __doc__=model_docstring,
        **model_fields_definitions,
    )
    return NewModel
