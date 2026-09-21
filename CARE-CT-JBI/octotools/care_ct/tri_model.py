"""Contracts for independent GPT, Gemini, and BiomedCLIP candidates.

The GPT and Gemini candidates are elicited before either model can observe the
other candidate or any trajectory/tool result.  BiomedCLIP remains an image
classifier and is represented through its existing frozen A--D option
contract.  This module contains no benchmark labels and performs no fusion.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .option_contract import CHOICE_IDS, canonical_choice_options


TRI_MODEL_CANDIDATE_SCHEMA_VERSION = (
    "care-ct-tri-model-candidate-v1-2026-09-17"
)
TRI_MODEL_OPTION_CONTRACT_VERSION = (
    "care-ct-tri-model-option-contract-v1-2026-09-17"
)
TRI_MODEL_MODEL_FAMILIES = ("gpt", "gemini", "biomedclip")
GPT_CANDIDATE_TOOL_NAME = "GPT_Independent_Candidate"
GEMINI_CANDIDATE_TOOL_NAME = "Gemini_Independent_Candidate"


class ChoiceProbabilities(BaseModel):
    """Self-reported A--D probabilities from one hosted model call."""

    model_config = ConfigDict(extra="forbid")

    A: float = Field(ge=0.0, le=1.0)
    B: float = Field(ge=0.0, le=1.0)
    C: float = Field(ge=0.0, le=1.0)
    D: float = Field(ge=0.0, le=1.0)


class TriModelChoiceOutput(BaseModel):
    """Provider-facing structured response for an independent candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rationale: str = Field(min_length=1)
    selected_option: Literal["A", "B", "C", "D"]
    probabilities: ChoiceProbabilities


def independent_candidate_prompt(
    question: str,
    choices: Mapping[str, str],
) -> str:
    """Return the identical, label-free prompt used for GPT and Gemini."""

    options = canonical_choice_options(choices)
    rendered = "\n".join(
        f"{choice_id}. {option}"
        for choice_id, option in zip(CHOICE_IDS, options)
    )
    return f"""
Task: Independently answer this medical-image forced-choice item using only the
attached source image, the question, and the four options below. You have not
seen and must not infer another model's answer or any later tool result.

Question:
{question}

Options (preserve this exact A--D mapping):
{rendered}

Return exactly one selected option, a concise image-grounded rationale, and a
probability for every option A, B, C, and D. Probabilities must be finite,
non-negative, and express your uncertainty; do not omit an option and do not
abstain. The selected option must have the highest probability.
""".strip()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def _normalized_probabilities(value: Mapping[str, Any]) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for choice_id in CHOICE_IDS:
        raw = value.get(choice_id)
        if isinstance(raw, bool):
            raise ValueError("Choice probabilities must be numeric, not boolean.")
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("Every A-D probability must be numeric.") from error
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError("Every A-D probability must be finite and in [0, 1].")
        probabilities[choice_id] = parsed
    total = sum(probabilities.values())
    if total <= 0.0:
        raise ValueError("The A-D probability vector has zero mass.")
    return {
        choice_id: round(probabilities[choice_id] / total, 12)
        for choice_id in CHOICE_IDS
    }


def normalize_tri_model_choice(value: Any) -> dict[str, Any]:
    """Validate and normalize one provider response without consulting gold."""

    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise ValueError("Independent candidate output must be a mapping.")
    try:
        parsed = TriModelChoiceOutput.model_validate(dict(value)).model_dump()
    except ValidationError as error:
        raise ValueError("Invalid independent tri-model candidate output.") from error
    probabilities = _normalized_probabilities(parsed["probabilities"])
    selected = parsed["selected_option"]
    maximum = max(probabilities.values())
    if not math.isclose(
        probabilities[selected], maximum, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Independent candidate selected_option is not an argmax probability."
        )
    ranked = sorted(
        CHOICE_IDS,
        key=lambda choice_id: (-probabilities[choice_id], choice_id),
    )
    return {
        "rationale": str(parsed["rationale"]).strip(),
        "selected_option": selected,
        "probabilities": probabilities,
        "confidence": round(probabilities[selected], 12),
        "margin": round(
            probabilities[selected]
            - max(
                probabilities[choice_id]
                for choice_id in CHOICE_IDS
                if choice_id != selected
            ),
            12,
        ),
        "ranked_options": ranked,
    }


def build_candidate_evidence_result(
    *,
    output: Any,
    model_family: str,
    model_name: str,
    choices: Mapping[str, str],
    image_sha256: str,
    image_source: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """Wrap one valid independent response in an auditable evidence contract."""

    family = str(model_family or "").strip().casefold()
    if family not in {"gpt", "gemini"}:
        raise ValueError("Hosted independent candidate family must be GPT or Gemini.")
    normalized = normalize_tri_model_choice(output)
    options = canonical_choice_options(choices)
    image_hash = str(image_sha256 or "").strip().casefold()
    if len(image_hash) != 64 or any(
        character not in "0123456789abcdef" for character in image_hash
    ):
        raise ValueError("Independent candidate requires a valid image SHA-256.")
    prompt_digest = str(prompt_hash or "").strip().casefold()
    if len(prompt_digest) != 64 or any(
        character not in "0123456789abcdef" for character in prompt_digest
    ):
        raise ValueError("Independent candidate requires a valid prompt SHA-256.")
    if image_source not in {"original_image", "provided_bbox"}:
        raise ValueError("Independent candidate has an unsupported image source.")
    return {
        "status": "success",
        "candidate_schema_version": TRI_MODEL_CANDIDATE_SCHEMA_VERSION,
        "model_family": family,
        "model_name": str(model_name or "unknown"),
        "rationale": normalized["rationale"],
        "selected_option": normalized["selected_option"],
        "choice_scores": normalized["probabilities"],
        "confidence": normalized["confidence"],
        "margin": normalized["margin"],
        "care_ct_call_contract": {
            "version": TRI_MODEL_OPTION_CONTRACT_VERSION,
            "evidence_role": "answer_option_classification",
            "choice_ids": list(CHOICE_IDS),
            "effective_options": list(options),
            "model_family": family,
            "independent": True,
            "stage": "pre_tool",
            "prompt_sha256": prompt_digest,
            "image_routing": {
                "image_source": image_source,
                "image_sha256": image_hash,
            },
        },
    }


def candidate_error_result(
    *, model_family: str, model_name: str, error: str
) -> dict[str, Any]:
    """Persist a bounded, explicit failed-candidate record for ITT accounting."""

    family = str(model_family or "").strip().casefold()
    if family not in {"gpt", "gemini"}:
        raise ValueError("Candidate error family must be GPT or Gemini.")
    return {
        "status": "error",
        "error": str(error or "Independent candidate failed")[:2000],
        "candidate_schema_version": TRI_MODEL_CANDIDATE_SCHEMA_VERSION,
        "model_family": family,
        "model_name": str(model_name or "unknown"),
    }


def parse_candidate_evidence_result(
    value: Any,
    choices: Mapping[str, str],
) -> dict[str, Any] | None:
    """Fail closed unless ``value`` satisfies the full candidate contract."""

    if not isinstance(value, Mapping):
        return None
    if value.get("candidate_schema_version") != TRI_MODEL_CANDIDATE_SCHEMA_VERSION:
        return None
    if str(value.get("status") or "").strip().casefold() != "success":
        return None
    family = str(value.get("model_family") or "").strip().casefold()
    if family not in {"gpt", "gemini"}:
        return None
    contract = value.get("care_ct_call_contract")
    if not isinstance(contract, Mapping):
        return None
    options = canonical_choice_options(choices)
    if not bool(
        contract.get("version") == TRI_MODEL_OPTION_CONTRACT_VERSION
        and contract.get("evidence_role") == "answer_option_classification"
        and contract.get("choice_ids") == list(CHOICE_IDS)
        and contract.get("effective_options") == list(options)
        and contract.get("model_family") == family
        and contract.get("independent") is True
        and contract.get("stage") == "pre_tool"
    ):
        return None
    routing = contract.get("image_routing")
    if not isinstance(routing, Mapping):
        return None
    image_hash = str(routing.get("image_sha256") or "").strip().casefold()
    if (
        routing.get("image_source") not in {"original_image", "provided_bbox"}
        or len(image_hash) != 64
        or any(character not in "0123456789abcdef" for character in image_hash)
    ):
        return None
    prompt_hash = str(contract.get("prompt_sha256") or "").strip().casefold()
    if len(prompt_hash) != 64 or any(
        character not in "0123456789abcdef" for character in prompt_hash
    ):
        return None
    selected = str(value.get("selected_option") or "").strip().upper()
    scores = value.get("choice_scores")
    if selected not in CHOICE_IDS or not isinstance(scores, Mapping):
        return None
    try:
        normalized_scores = _normalized_probabilities(scores)
    except ValueError:
        return None
    maximum = max(normalized_scores.values())
    if not math.isclose(
        normalized_scores[selected], maximum, rel_tol=0.0, abs_tol=1e-12
    ):
        return None
    other = max(
        normalized_scores[choice_id]
        for choice_id in CHOICE_IDS
        if choice_id != selected
    )
    return {
        "model_family": family,
        "model_name": str(value.get("model_name") or "unknown"),
        "prediction": selected,
        "choice_scores": normalized_scores,
        "confidence": normalized_scores[selected],
        "margin": normalized_scores[selected] - other,
        "image_sha256": image_hash,
        "image_source": routing.get("image_source"),
        "prompt_sha256": prompt_hash,
    }


__all__ = [
    "ChoiceProbabilities",
    "GEMINI_CANDIDATE_TOOL_NAME",
    "GPT_CANDIDATE_TOOL_NAME",
    "TRI_MODEL_CANDIDATE_SCHEMA_VERSION",
    "TRI_MODEL_MODEL_FAMILIES",
    "TRI_MODEL_OPTION_CONTRACT_VERSION",
    "TriModelChoiceOutput",
    "build_candidate_evidence_result",
    "candidate_error_result",
    "independent_candidate_prompt",
    "normalize_tri_model_choice",
    "parse_candidate_evidence_result",
    "prompt_sha256",
]
