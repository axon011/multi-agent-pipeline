"""Schema-validated LLM output across both LLM backends.

Two backends, two strategies:

* OpenAI-compatible chat models (GLM, OpenAI, OpenRouter, ...) support tool
  calling, so we use ``llm.with_structured_output(schema,
  method="function_calling")`` and get a validated Pydantic object back.
* The Claude CLI backend (``ChatClaudeCode``) executes tools inside the CLI
  and never surfaces tool calls to LangChain, so native structured output
  isn't available. There we append the JSON Schema to the prompt, extract
  the JSON from the reply, validate it with Pydantic, and retry once with
  the validation error fed back to the model.

Either way the caller gets an instance of ``schema`` or a
``StructuredOutputError`` — never an unvalidated dict or a regex guess.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


class StructuredOutputError(ValueError):
    """The model's reply could not be validated against the requested schema."""


def supports_native_structured_output(llm) -> bool:
    """True for chat models that surface tool calls to LangChain."""
    from langchain_openai import ChatOpenAI

    return isinstance(llm, ChatOpenAI)


def extract_json(text: str) -> str:
    """Return the JSON object/array embedded in a model reply.

    Handles code fences and leading/trailing prose by slicing from the first
    opening bracket to its last matching closing bracket.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        return cleaned
    start = min(starts)
    closer = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(closer)
    return cleaned[start : end + 1] if end > start else cleaned[start:]


def parse_model(text: str, schema: type[T]) -> T:
    """Validate a raw model reply against ``schema``.

    Raises ``StructuredOutputError`` (wrapping the JSON or validation error)
    when the reply doesn't conform.
    """
    try:
        return schema.model_validate_json(extract_json(text))
    except ValidationError as exc:
        raise StructuredOutputError(str(exc)) from exc


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Respond with ONLY a JSON object that validates against this JSON "
        "Schema. No prose, no markdown fences.\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


def invoke_structured(
    llm,
    prompt: ChatPromptTemplate,
    payload: dict,
    schema: type[T],
    max_retries: int = 1,
) -> T:
    """Run ``prompt | llm`` and return a validated ``schema`` instance."""
    if supports_native_structured_output(llm):
        chain = prompt | llm.with_structured_output(schema, method="function_calling")
        result = chain.invoke(payload)
        if result is None:
            raise StructuredOutputError(f"model returned no {schema.__name__} tool call")
        return result

    messages: list[BaseMessage] = prompt.format_messages(**payload)
    messages.append(HumanMessage(content=_schema_instructions(schema)))

    last_error: StructuredOutputError | None = None
    for attempt in range(max_retries + 1):
        response = llm.invoke(messages)
        try:
            return parse_model(response.content, schema)
        except StructuredOutputError as exc:
            last_error = exc
            logger.warning(
                "%s validation failed (attempt %d): %s",
                schema.__name__, attempt + 1, str(exc)[:300],
            )
            messages = messages + [
                AIMessage(content=response.content),
                HumanMessage(
                    content=(
                        "That reply did not validate against the schema:\n"
                        f"{str(exc)[:1000]}\n\nReturn the corrected JSON object only."
                    )
                ),
            ]
    raise last_error  # type: ignore[misc]
