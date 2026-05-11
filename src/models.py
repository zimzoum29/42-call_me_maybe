"""Pydantic models representing prompts, functions and calls.

This module centralizes the lightweight schemas used throughout the
project and includes small validation helpers that raise `JsonError`
on invalid input.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from src.utils import JsonError

JsonType = Literal["string", "number", "integer", "boolean"]


class Prompt(BaseModel):

    """Model describing a single input prompt item."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class ReturnType(BaseModel):

    """Model describing a function return type."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class Parameter(BaseModel):

    """Model describing a single parameter's type."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):

    """Schema for an available function including parameters and return."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure the function name is not empty."""
        if not value:
            raise ValueError("Function name cannot be empty.")
        return value

    def ordered_parameter_items(self) -> list[tuple[str, Parameter]]:
        """Return ordered (name, Parameter) pairs for iteration."""
        return list(self.parameters.items())


class FunctionCall(BaseModel):

    """Schema representing a generated function call.

    Attributes:
        name: Name of the function to call.
        parameters: Mapping of parameter names to generated values.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any]


def validate_function_definitions(data: Any) -> list[FunctionDefinition]:
    """Validate a raw data structure as function definitions.

    Args:
        data: Parsed JSON data expected to be a list of function
            definitions.

    Returns:
        A list of `FunctionDefinition` instances.

    Raises:
        JsonError: If the input is malformed or validation fails.
    """
    if not isinstance(data, list):
        raise JsonError("Function definitions root must be a list.")
    try:
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except Exception as error:
        raise JsonError(f"Invalid function definition: {error}") from error
    if not functions:
        raise JsonError("At least one function definition is required.")
    return functions


def validate_prompt_items(data: Any) -> list[Prompt]:
    """Validate a list of prompt items parsed from JSON.

    Args:
        data: Parsed JSON expected to be a list of prompt objects.

    Returns:
        A list of validated `Prompt` instances.

    Raises:
        JsonError: If input format or validation fails.
    """
    if not isinstance(data, list):
        raise JsonError("Prompts root must be a list.")
    try:
        return [Prompt.model_validate(item) for item in data]
    except Exception as error:
        raise JsonError(f"Invalid prompt item: {error}") from error
