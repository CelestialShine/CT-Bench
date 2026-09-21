"""Shared, label-free contract for CT-Bench multiple-choice predictions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHOICE_OUTPUT_SCHEMA_VERSION = "care-ct-choice-v2-2026-08-24"
POSTHOC_REVISION_ID = (
    "test-informed-component-fusion-v8-2026-09-08"
)


class MultipleChoiceOutput(BaseModel):
    """Structured inference output containing one forced A-D choice."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rationale: str = Field(min_length=1)
    selected_option: Literal["A", "B", "C", "D"]


def extract_multiple_choice(value: Any) -> str | None:
    """Extract an explicit A-D prediction without consulting a label.

    The runtime selector, output validator, and offline scorer all import this
    function so an answer cannot be considered usable during inference but
    unparseable during evaluation.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        choices = []
        for key in ("selected_option", "answer", "choice"):
            if key in value:
                choice = extract_multiple_choice(value[key])
                if choice is None:
                    return None
                choices.append(choice)
        unique_choices = set(choices)
        return choices[0] if len(unique_choices) == 1 else None
    if not isinstance(value, str):
        return None

    text = value.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        parsed = json.loads(fenced)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if parsed is not None and parsed != value:
        # A syntactically valid JSON value is authoritative.  In particular,
        # do not fall back to regex parsing when a JSON object contains
        # conflicting answer fields.
        return extract_multiple_choice(parsed)

    # A bare letter is unambiguous, including when it came from a mapping
    # value such as {"choice": "b"}.  For prose, keep the option token
    # case-sensitive so an ordinary article ("answer a question") cannot be
    # mistaken for option A.
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()

    strong_patterns = (
        r'(?i:"selected_option")\s*:\s*"?([A-D])\b',
        r"\b(?i:selected(?:\s+option)?)\s*(?:(?i:is)|:)\s*"
        r"\*{0,2}([A-D])\b",
        r"\b(?i:answer)\s*(?:(?i:is)|:)\s*\*{0,2}([A-D])\b",
        r"\b(?i:best\s+option|correct\s+(?:choice|option))\s*"
        r"(?:(?i:is)|:)\s*\*{0,2}([A-D])\b",
    )
    strong_choices = {
        match.group(1).upper()
        for pattern in strong_patterns
        for match in re.finditer(pattern, text, flags=re.M)
    }
    if strong_choices:
        return strong_choices.pop() if len(strong_choices) == 1 else None

    weak_patterns = (
        r"\b(?i:option)\s+\*{0,2}([A-D])\b",
        r"^\s*\*{0,2}([A-D])\*{0,2}\s*(?:[.)]|$)",
    )
    weak_choices = {
        match.group(1).upper()
        for pattern in weak_patterns
        for match in re.finditer(pattern, text, flags=re.M)
    }
    return weak_choices.pop() if len(weak_choices) == 1 else None


def validate_structured_multiple_choice_output(value: Any) -> dict[str, str]:
    """Validate the persisted structured answer representation.

    Runtime adapters may normalize legacy explicit text before persistence, but
    resume and scoring accept only this exact structured object.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected a structured object with rationale and selected_option."
        )
    try:
        return MultipleChoiceOutput.model_validate(dict(value)).model_dump()
    except ValidationError as error:
        raise ValueError("Invalid structured multiple-choice output.") from error


def normalize_multiple_choice_output(value: Any) -> dict[str, str]:
    """Return a JSON-serializable structured answer or fail closed.

    Azure/OpenAI engines normally return ``MultipleChoiceOutput`` directly.
    The explicit-text fallback keeps other engines compatible only when they
    provide an unambiguous A-D prediction; free-form guessing is forbidden.
    """

    if isinstance(value, MultipleChoiceOutput):
        return value.model_dump()
    if isinstance(value, Mapping):
        return validate_structured_multiple_choice_output(value)
    choice = extract_multiple_choice(value)
    if choice is None:
        raise ValueError("Model output lacks an explicit A-D prediction.")
    return MultipleChoiceOutput(
        rationale=str(value).strip(),
        selected_option=choice,
    ).model_dump()
