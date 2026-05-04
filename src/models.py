from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from src.utils import JsonError

JsonType = Literal["string", "number", "integer", "boolean"]


class Prompt(BaseModel):

    model_config = ConfigDict(extra="forbid")

    prompt: str


class ReturnType(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class Parameter(BaseModel):

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("Function name cannot be empty.")
        return value

    def ordered_parameter_items(self) -> list[tuple[str, Parameter]]:
        return list(self.parameters.items())


class FunctionCall(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any]


def validate_function_definitions(data: Any) -> list[FunctionDefinition]:
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
    if not isinstance(data, list):
        raise JsonError("Prompts root must be a list.")
    try:
        return [Prompt.model_validate(item) for item in data]
    except Exception as error:
        raise JsonError(f"Invalid prompt item: {error}") from error
