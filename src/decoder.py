"""Constrained decoding helper that enforces a JSON schema grammar.

This module implements a deterministic decoder which consults a
lightweight `Grammar` to ensure that only tokens leading to schema-
valid JSON are accepted during generation.
"""

from math import inf
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.grammar import Grammar
from src.llm_engine import FunctionCallingEngine
from src.utils import JsonError


class GeneratedObject(BaseModel):

    """Small container for generated output.

    Attributes:
        text: The generated text string.
        token_ids: List of token ids produced by the model.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: list[int]


class ConstrainedDecoder:

    """Decoder that selects only tokens valid under a `Grammar`.

    The decoder uses `FunctionCallingEngine` for token-level access and
    `Grammar` to validate prefixes and completion.
    """

    def __init__(
        self,
        engine: FunctionCallingEngine,
        grammar: Grammar,
    ) -> None:
        """Initialize the decoder.

        Args:
            engine: Engine wrapper providing encode/decode and logits.
            grammar: Grammar instance used for prefix/completion checks.
        """
        self._engine = engine
        self._grammar = grammar
        self._token_cache: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token id, caching results.

        Args:
            token_id: The token id to decode.

        Returns:
            The decoded string for the token id.
        """
        if token_id not in self._token_cache:
            self._token_cache[token_id] = self._engine.decode([token_id])
        return self._token_cache[token_id]

    def _encode_to_list(self, text: str) -> list[int]:
        """Encode `text` to a list of integer token ids.

        The underlying SDK may return different shapes (numpy arrays,
        lists, nested lists). This helper normalizes those into a flat
        list of ints.

        Args:
            text: Text to encode.

        Returns:
            List of integer token ids.

        Raises:
            JsonError: If the SDK returns an unsupported format.
        """
        encoded = self._engine.encode(text)
        if hasattr(encoded, "tolist"):
            data = encoded.tolist()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [int(item) for item in data[0]]
            if isinstance(data, list):
                return [int(item) for item in data]
        if isinstance(encoded, list):
            return [int(item) for item in encoded]
        raise JsonError(
            "Unsupported encoded token format returned by llm_sdk."
        )

    def _is_valid_token(self, generated_text: str, token_id: int) -> bool:
        """Return whether appending a token keeps the prefix valid.

        Args:
            generated_text: Text generated so far.
            token_id: Candidate token id to append.

        Returns:
            True if the resulting prefix is valid under the grammar.
        """
        token_text = self._decode_token(token_id)
        if not token_text:
            return False
        return self._grammar.is_valid_prefix(generated_text + token_text)

    def _masked_best_token(
        self,
        generated_text: str,
        logits: list[float],
        first_pass_size: int,
    ) -> int:
        """Choose the best token that keeps generation valid.

        The method first inspects the top `first_pass_size` logits and
        falls back to scanning the full vocabulary looking for the
        highest-scoring valid token.
        """
        ranked_ids = sorted(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
            reverse=True,
        )
        valid_best: int | None = None
        valid_best_logit = -inf

        for token_id in ranked_ids[:first_pass_size]:
            if self._is_valid_token(generated_text, token_id):
                valid_best = token_id
                valid_best_logit = logits[token_id]
                break

        if valid_best is not None:
            return valid_best

        for token_id in ranked_ids[first_pass_size:]:
            if self._is_valid_token(generated_text, token_id):
                if logits[token_id] > valid_best_logit:
                    valid_best = token_id
                    valid_best_logit = logits[token_id]

        if valid_best is None:
            raise JsonError(
                "Constrained decoding failed: no valid token candidate found."
            )
        return valid_best

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        first_pass_size: int = 2048,
    ) -> GeneratedObject:
        """Generate text while enforcing grammar constraints.

        Args:
            prompt_text: Full prompt to seed the model.
            max_new_tokens: Maximum tokens to generate before failing.
            first_pass_size: Number of top logits to inspect first.

        Returns:
            `GeneratedObject` containing the produced text and token ids.

        Raises:
            JsonError: If generation cannot produce a valid completion.
        """
        prompt_ids = self._encode_to_list(prompt_text)
        generated_ids: list[int] = []
        generated_text = ""

        for _ in range(max_new_tokens):
            input_ids = prompt_ids + generated_ids
            logits_any: Any = self._engine.get_next_token_logits(input_ids)
            logits = [float(value) for value in logits_any]
            token_id = self._masked_best_token(
                generated_text=generated_text,
                logits=logits,
                first_pass_size=first_pass_size,
            )
            token_text = self._decode_token(token_id)
            generated_ids.append(token_id)
            generated_text += token_text
            if self._grammar.is_complete(generated_text):
                return GeneratedObject(
                    text=generated_text,
                    token_ids=generated_ids,
                )

        raise JsonError(
            "Constrained decoding failed: maximum number of tokens reached."
        )
