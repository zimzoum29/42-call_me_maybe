from math import inf
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.grammar import Grammar
from src.llm_engine import FunctionCallingEngine
from src.utils import JsonError


class GeneratedObject(BaseModel):

    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: list[int]


class ConstrainedDecoder:

    def __init__(
        self,
        engine: FunctionCallingEngine,
        grammar: Grammar,
    ) -> None:
        self._engine = engine
        self._grammar = grammar
        self._token_cache: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_cache:
            self._token_cache[token_id] = self._engine.decode([token_id])
        return self._token_cache[token_id]

    def _encode_to_list(self, text: str) -> list[int]:
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

    def _debug_mask_logits(
        self,
        generated_text: str,
        logits: list[float],
    ) -> list[float]:
        masked = [-inf for _ in logits]
        for token_id, logit in enumerate(logits):
            if self._is_valid_token(generated_text, token_id):
                masked[token_id] = logit
        return masked

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 256,
        first_pass_size: int = 2048,
    ) -> GeneratedObject:
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
