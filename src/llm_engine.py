from typing import Any

from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition
from src.prompting import build_generation_prompt
from src.utils import JsonError


class FunctionCallingEngine:

    def __init__(
        self,
        function_definitions: list[FunctionDefinition],
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        if not function_definitions:
            raise JsonError("At least one function definition is required.")
        self._function_definitions = function_definitions
        self._functions_by_name = {
            function.name: function for function in function_definitions
        }
        self._model = Small_LLM_Model(model_name=model_name)

    def build_prompt(self, user_prompt: str) -> str:
        return build_generation_prompt(self._function_definitions, user_prompt)

    def encode(self, text: str) -> Any:
        return self._model.encode(text)

    def decode(self, token_ids: Any) -> str:
        return self._model.decode(token_ids)

    def get_next_token_logits(self, input_ids: list[int]) -> list[float]:
        return self._model.get_logits_from_input_ids(input_ids)

    def get_function_definition(
        self,
        function_name: str,
    ) -> FunctionDefinition:
        try:
            return self._functions_by_name[function_name]
        except KeyError as error:
            raise JsonError(
                f"Unknown function name: {function_name}"
            ) from error
