"""Lightweight JSON-like grammar validation utilities.

This module provides a simple, efficient matcher used by the
constrained decoder to validate partial and complete JSON that
represents function calls with typed parameters.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from src.models import FunctionDefinition, JsonType
from src.utils import JsonError


class MatchResult(BaseModel):
    """Result of analyzing a text fragment against a branch.

    Attributes:
        possible: Whether the fragment can still be extended to a
            valid branch (prefix is allowed).
        complete: Whether the fragment is a complete and valid branch.
    """

    model_config = ConfigDict(frozen=True)

    possible: bool
    complete: bool


class FunctionBranchMatcher:

    """Checks whether a text matches the JSON layout of one function.

    The matcher performs incremental checks used by the constrained
    decoder to decide if a prefix remains valid or represents a
    complete function call.
    """

    def __init__(self, definition: FunctionDefinition) -> None:
        """Create a matcher for `definition`.

        Args:
            definition: The `FunctionDefinition` describing expected
                parameter names and types.
        """
        self._definition = definition

    def analyze(self, text: str) -> MatchResult:
        """Analyze `text` and return whether it's a valid prefix/complete.

        Args:
            text: Fragment to analyze.

        Returns:
            `MatchResult` indicating if the fragment is a valid prefix
            and whether it's a complete match.
        """
        index = 0
        ok, complete, index = self._check_literal(text, index, '{"name":"')
        if not ok or not complete:
            return MatchResult(possible=ok, complete=False)

        ok, complete, index = self._check_literal(
            text,
            index,
            self._definition.name,
        )
        if not ok or not complete:
            return MatchResult(possible=ok, complete=False)

        ok, complete, index = self._check_literal(
            text,
            index,
            '","parameters":{',
        )
        if not ok or not complete:
            return MatchResult(possible=ok, complete=False)

        parameter_items = self._definition.ordered_parameter_items()
        for position, (name, parameter) in enumerate(parameter_items):
            ok, complete, index = self._check_literal(
                text,
                index,
                f'"{name}":',
            )
            if not ok or not complete:
                return MatchResult(possible=ok, complete=False)

            ok, complete, index = self._check_parameter_value(
                text,
                index,
                parameter.type,
            )
            if not ok or not complete:
                return MatchResult(possible=ok, complete=False)

            if position < len(parameter_items) - 1:
                ok, complete, index = self._check_literal(text, index, ",")
                if not ok or not complete:
                    return MatchResult(possible=ok, complete=False)

        ok, complete, index = self._check_literal(text, index, "}}")
        if not ok or not complete:
            return MatchResult(possible=ok, complete=False)
        return MatchResult(
            possible=index == len(text),
            complete=index == len(text),
        )

    def _check_literal(
        self,
        text: str,
        index: int,
        literal: str,
    ) -> tuple[bool, bool, int]:
        """Check that `literal` matches at `index`.

        Returns a tuple (possible, complete, new_index) where "possible"
        indicates the characters so far could match and "complete"
        indicates the full literal was present.
        """
        remaining = text[index:]
        max_len = min(len(remaining), len(literal))
        if remaining[:max_len] != literal[:max_len]:
            return False, False, index
        if len(remaining) < len(literal):
            return True, False, len(text)
        return True, True, index + len(literal)

    def _check_parameter_value(
        self,
        text: str,
        index: int,
        param_type: JsonType,
    ) -> tuple[bool, bool, int]:
        """Dispatch to the correct typed checker for a parameter value.

        Args:
            text: Input text.
            index: Cursor position to start checking.
            param_type: Expected JSON-like type.

        Returns:
            Tuple matching the same (possible, complete, new_index)
            contract used across check helpers.
        """
        if param_type == "string":
            return self._check_json_string(text, index)
        if param_type == "number":
            return self._check_json_number(text, index, allow_float=True)
        if param_type == "integer":
            return self._check_json_number(text, index, allow_float=False)
        if param_type == "boolean":
            return self._check_json_boolean(text, index)
        return False, False, index

    def _check_json_boolean(
        self,
        text: str,
        index: int,
    ) -> tuple[bool, bool, int]:
        """Check a JSON boolean value at `index`.

        The checker supports incremental matches (prefixes) used while
        streaming tokens from the model.
        """
        remaining = text[index:]
        if remaining == "":
            return True, False, index
        for value in ("true", "false"):
            if value.startswith(remaining):
                return True, False, len(text)
            if remaining.startswith(value):
                return True, True, index + len(value)
        return False, False, index

    def _check_json_string(
        self,
        text: str,
        index: int,
    ) -> tuple[bool, bool, int]:
        """Check a JSON string starting at `index`.

        This validates quoting, escapes and basic unicode escapes, and
        supports prefix matches when a closing quote has not yet been
        produced by the generator.
        """
        if index >= len(text):
            return True, False, index
        if text[index] != '"':
            return False, False, index

        cursor = index + 1
        while cursor < len(text):
            char = text[cursor]
            if char == '"':
                return True, True, cursor + 1
            if ord(char) < 32:
                return False, False, index
            if char != "\\":
                cursor += 1
                continue

            cursor += 1
            if cursor >= len(text):
                return True, False, len(text)
            escaped = text[cursor]
            if escaped in '"\\/bfnrt':
                cursor += 1
                continue
            if escaped == "u":
                cursor += 1
                unicode_end = cursor + 4
                if unicode_end > len(text):
                    for digit in text[cursor:]:
                        if digit not in "0123456789abcdefABCDEF":
                            return False, False, index
                    return True, False, len(text)
                for digit in text[cursor:unicode_end]:
                    if digit not in "0123456789abcdefABCDEF":
                        return False, False, index
                cursor = unicode_end
                continue
            return False, False, index
        return True, False, len(text)

    def _check_json_number(
        self,
        text: str,
        index: int,
        allow_float: bool,
    ) -> tuple[bool, bool, int]:
        """Check an integer or floating-point JSON number.

        Returns the same (possible, complete, new_index) tuple used by
        other helpers.
        """
        cursor = index
        if cursor >= len(text):
            return True, False, cursor
        if text[cursor] == "-":
            cursor += 1
            if cursor >= len(text):
                return True, False, cursor

        digit_start = cursor
        if cursor < len(text) and text[cursor] == "0":
            cursor += 1
        else:
            while cursor < len(text) and text[cursor].isdigit():
                cursor += 1
        if cursor == digit_start:
            return False, False, index

        if cursor < len(text) and text[cursor] == ".":
            if not allow_float:
                return False, False, index
            cursor += 1
            if cursor >= len(text):
                return True, False, cursor
            fraction_start = cursor
            while cursor < len(text) and text[cursor].isdigit():
                cursor += 1
            if cursor == fraction_start:
                return False, False, index

        return True, True, cursor


class Grammar:

    """Composite grammar built from multiple function branches.

    The `Grammar` exposes two convenience methods used by the decoder:
    `is_valid_prefix` and `is_complete` which respectively determine
    whether a partial generated string can still become valid and if a
    string is a finished, valid representation.
    """

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self._branches = [
            FunctionBranchMatcher(function)
            for function in functions
        ]

    def is_valid_prefix(self, text: str) -> bool:
        """Return True if `text` is a valid prefix for any branch."""
        return any(branch.analyze(text).possible for branch in self._branches)

    def is_complete(self, text: str) -> bool:
        """Return True if `text` is a complete valid branch."""
        return any(branch.analyze(text).complete for branch in self._branches)


def normalize_types(
    parameters: dict[str, Any],
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Normalize parameter values according to the function schema.

    This converts numeric values to `float`/`int` as required and
    validates type constraints, raising `JsonError` on mismatch.

    Args:
        parameters: Mapping of parameter names to generated values.
        function: The function schema to normalize against.

    Returns:
        A new dict with normalized parameter values.

    Raises:
        JsonError: If required names are missing or types mismatch.
    """
    normalized: dict[str, Any] = {}
    expected_names = [name for name, _ in function.ordered_parameter_items()]
    if set(parameters) != set(expected_names):
        raise JsonError(
            f"Generated parameters do not match {function.name} schema."
        )

    for name, parameter in function.ordered_parameter_items():
        value = parameters[name]
        if parameter.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise JsonError(f"Parameter '{name}' must be a number.")
            normalized[name] = float(value)
        elif parameter.type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise JsonError(f"Parameter '{name}' must be an integer.")
            normalized[name] = int(value)
        elif parameter.type == "boolean":
            if not isinstance(value, bool):
                raise JsonError(f"Parameter '{name}' must be a boolean.")
            normalized[name] = value
        elif parameter.type == "string":
            if not isinstance(value, str):
                raise JsonError(f"Parameter '{name}' must be a string.")
            normalized[name] = value
    return normalized
