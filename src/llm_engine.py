"""LLM engine wrapper for function-calling behavior.

This module provides a thin adapter around the installed LLM SDK to
support constrained function-calling workflows used by the rest of
the application.

The `FunctionCallingEngine` exposes helper methods for prompt
generation, token encoding/decoding, logits retrieval and lookup of
function definitions by name.
"""

from typing import Any

from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition
from src.prompting import build_generation_prompt
from src.utils import JsonError


class FunctionCallingEngine:
    """Adapter around a small LLM model for function selection.

    Args:
        function_definitions: List of validated `FunctionDefinition`
            objects describing available functions.
        model_name: Optional model identifier passed to `llm_sdk`.

    Raises:
        JsonError: If `function_definitions` is empty.
    """

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
        """Build the text prompt passed to the LLM.

        Args:
            user_prompt: The user's natural-language request.

        Returns:
            A complete prompt string containing function descriptions
            and the user's request suitable for the model.
        """
        return build_generation_prompt(self._function_definitions, user_prompt)

    def encode(self, text: str) -> Any:
        """Encode `text` into the token representation used by the model.

        This is a thin proxy to the underlying `llm_sdk` encode method.

        Args:
            text: Text to encode.

        Returns:
            The SDK-specific encoded token object.
        """
        return self._model.encode(text)

    def decode(self, token_ids: Any) -> str:
        """Decode token ids (or token-like object) into a string.

        Args:
            token_ids: Token ids or SDK token object to decode.

        Returns:
            The decoded string fragment for the provided tokens.
        """
        return self._model.decode(token_ids)

    def get_next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the provided input ids.

        Args:
            input_ids: Sequence of token ids representing the prompt +
                generated tokens so far.

        Returns:
            A list of logits for the model's vocabulary.
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def get_function_definition(
        self,
        function_name: str,
    ) -> FunctionDefinition:
        """Retrieve a validated `FunctionDefinition` by name.

        Args:
            function_name: Name of the function to look up.

        Returns:
            The corresponding `FunctionDefinition` instance.

        Raises:
            JsonError: If `function_name` is not known.
        """
        try:
            return self._functions_by_name[function_name]
        except KeyError as error:
            raise JsonError(
                f"Unknown function name: {function_name}"
            ) from error
