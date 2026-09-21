import json
import re
from collections.abc import Mapping
from typing import Any, List

from pydantic import BaseModel, Field, ValidationError

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool


class ReflectionAudit(BaseModel):
    """Inference-time, label-free audit of a completed CARE-CT trajectory."""

    reasoning_score: int = Field(ge=1, le=10)
    key_observations: List[str]
    identified_issues: List[str]
    error_type: str
    should_rerun: bool
    recommended_tool: str
    recovery_sub_goal: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be used safely by the agent."""


def parse_reflection_audit(response: Any) -> ReflectionAudit:
    """Normalize a structured or JSON reflection audit.

    Unsupported free-form text is rejected instead of being interpreted as a
    request to rerun. Formal outer inference treats that audit failure as an
    incomplete diagnostic that must be retried.
    """

    if isinstance(response, ReflectionAudit):
        return response
    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="replace")
    if isinstance(response, Mapping):
        try:
            return ReflectionAudit.model_validate(dict(response))
        except ValidationError as error:
            raise LLMResponseError("LLM returned an invalid reflection audit.") from error
    if not isinstance(response, str) or not response.strip():
        raise LLMResponseError("LLM returned no usable reflection audit.")

    text = response.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        decoded = json.loads(fenced)
    except (json.JSONDecodeError, TypeError) as error:
        raise LLMResponseError(
            "LLM reflection audit was not valid structured JSON."
        ) from error
    if not isinstance(decoded, Mapping):
        raise LLMResponseError("LLM reflection audit JSON must be an object.")
    return parse_reflection_audit(decoded)


def parse_memory_verification(response: Any) -> tuple[str, str]:
    """Normalize structured or textual memory-verification responses.

    Provider error envelopes and missing responses are raised instead of being
    interpreted as CONTINUE, which would silently alter experiment behavior.
    """

    if isinstance(response, MemoryVerification):
        return response.analysis, "STOP" if response.stop_signal else "CONTINUE"

    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="replace")

    if isinstance(response, Mapping):
        error = response.get("error")
        if error:
            if isinstance(error, Mapping):
                error_name = error.get("type") or error.get("code") or "provider_error"
                error_message = error.get("message") or response.get("message") or str(error)
            else:
                error_name = str(error)
                error_message = response.get("message") or str(error)
            raise LLMResponseError(
                f"LLM memory verification failed [{error_name}]: {error_message}"
            )

        try:
            parsed = MemoryVerification.model_validate(dict(response))
        except ValidationError:
            lowered = {str(key).lower(): value for key, value in response.items()}
            analysis = lowered.get("analysis") or lowered.get("explanation")
            conclusion = lowered.get("conclusion")
            if isinstance(conclusion, str):
                normalized = conclusion.strip().upper()
                if normalized in {"STOP", "CONTINUE"}:
                    return str(analysis or response), normalized
            keys = ", ".join(sorted(str(key) for key in response))
            raise LLMResponseError(
                "LLM returned an invalid memory-verification object"
                f" (keys: {keys or '<none>'})."
            )
        return parsed.analysis, "STOP" if parsed.stop_signal else "CONTINUE"

    if response is None:
        raise LLMResponseError("LLM returned no memory-verification response.")
    if not isinstance(response, str):
        raise LLMResponseError(
            "LLM returned an unsupported memory-verification response type: "
            f"{type(response).__name__}."
        )

    text = response.strip()
    if not text:
        raise LLMResponseError("LLM returned an empty memory-verification response.")

    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, Mapping):
        return parse_memory_verification(decoded)

    matches = list(
        re.finditer(
            r"conclusion\**:?\s*\**\s*(STOP|CONTINUE)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    if matches:
        return text, matches[-1].group(1).upper()

    keywords = re.findall(r"\b(STOP|CONTINUE)\b", text, re.IGNORECASE)
    if keywords:
        return text, keywords[-1].upper()

    print("No valid conclusion (STOP or CONTINUE) found. Continuing...")
    return text, "CONTINUE"

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str
