"""Safe structured-output wrapper around an existing watsonx.ai model."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .prompts import json_repair_messages

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when a model response cannot be converted into the requested schema."""


class WatsonxStructuredClient:
    """Adds validated JSON generation without creating another ModelInference instance."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate_structured(
        self, messages: list[dict[str, str]], schema_type: type[ModelT]
    ) -> ModelT:
        schema = schema_type.model_json_schema()
        raw_output = self._chat(messages)

        try:
            return self._parse_and_validate(raw_output, schema_type)
        except (json.JSONDecodeError, ValidationError, StructuredOutputError) as first_error:
            repaired_output = self._chat(
                json_repair_messages(raw_output, schema, str(first_error))
            )
            try:
                return self._parse_and_validate(repaired_output, schema_type)
            except (json.JSONDecodeError, ValidationError, StructuredOutputError) as repair_error:
                raise StructuredOutputError(
                    f"Structured output failed after one repair attempt: {repair_error}"
                ) from repair_error

    def _chat(self, messages: list[dict[str, str]]) -> str:
        response = self._model.chat(messages=messages)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise StructuredOutputError("watsonx.ai returned an unexpected chat response.") from error
        if not isinstance(content, str):
            raise StructuredOutputError("watsonx.ai returned non-text chat content.")
        return content

    @staticmethod
    def _parse_and_validate(raw_output: str, schema_type: type[ModelT]) -> ModelT:
        cleaned = WatsonxStructuredClient._strip_markdown_code_fences(raw_output)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Some models add one introductory sentence despite the instruction.
            object_start, object_end = cleaned.find("{"), cleaned.rfind("}")
            if object_start < 0 or object_end <= object_start:
                raise
            payload = json.loads(cleaned[object_start : object_end + 1])
        return schema_type.model_validate(payload)

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else text.strip()


_structured_client: WatsonxStructuredClient | None = None


def configure_structured_client(model: Any) -> WatsonxStructuredClient:
    """Configure the wrapper with the application's already-created watsonx model."""

    global _structured_client
    _structured_client = WatsonxStructuredClient(model)
    return _structured_client


def get_structured_client() -> WatsonxStructuredClient:
    if _structured_client is None:
        raise RuntimeError(
            "No watsonx.ai model is configured. Call configure_structured_client(llm_base) "
            "during application setup."
        )
    return _structured_client
