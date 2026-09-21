"""Canonical, label-free prompt contract for the untuned CARE-CT ARM audit."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict


ARM_AUDIT_PROMPT_VERSION = "care_ct_untuned_arm_audit_v1"

ARM_AUDIT_SYSTEM_PROMPT = """You are the CARE-CT Adaptive Reflection Module (ARM).
Audit only the supplied inference-time trajectory evidence. The benchmark
ground-truth answer is unavailable: do not request it, infer it from hidden
metadata, or claim to have compared against it. Identify only material,
fixable problems that are supported by the supplied actions, tool results,
evidence graph, and consistency report. Return exactly one JSON object matching
the requested ReflectionAudit schema, with no Markdown or extra text."""

ARM_ERROR_TYPES = (
    "none",
    "tool_error",
    "localization_error",
    "semantic_ambiguity",
    "organ_conflict",
    "missing_measurement",
    "insufficient_evidence",
    "cross_modal_inconsistency",
)

_IDENTITY_OR_LABEL_KEYS = {
    "answer",
    "answers",
    "benchmark_index",
    "correct_answer",
    "correct_choice",
    "gold",
    "gold_answer",
    "ground_truth",
    "ground_truth_answer",
    "ground_truth_available_to_reflection",
    "lesion_idx",
    "patient",
    "patient_id",
    "patient_index",
    "pid",
    "qa_id",
    "reference_answer",
    "source_index",
    "target_answer",
}
_PATH_OR_BINARY_KEYS = {
    "base64",
    "bytes",
    "cache_dir",
    "data_url",
    "file",
    "file_name",
    "filepath",
    "image",
    "image_bytes",
    "image_path",
    "output_dir",
    "output_path",
    "path",
    "saved_path",
}
_NON_SEMANTIC_KEYS = {
    "created_at",
    "dependencies",
    "depends_on",
    "evidence_id",
}
_MAX_STRING_CHARS = 12_000
_MAX_SEQUENCE_ITEMS = 128
_MAX_MAPPING_ITEMS = 256
_MAX_DEPTH = 16

_DATA_IMAGE_RE = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\r\n]+", re.IGNORECASE
)
_FILE_URI_RE = re.compile(r"file://[^\s\"'`<>]+", re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\s\"'`<>]+")
_POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])/(?:[^\s\"'`<>:/]+/)*[^\s\"'`<>:/]+"
)


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_forbidden_key(value: Any) -> bool:
    key = _normalized_key(value)
    return (
        key in _IDENTITY_OR_LABEL_KEYS
        or key in _PATH_OR_BINARY_KEYS
        or key in _NON_SEMANTIC_KEYS
        or key == "id"
        or key.endswith("_id")
        or key.endswith("_ids")
    )


def _sanitize_string(value: str) -> str:
    text = _DATA_IMAGE_RE.sub("[IMAGE_BYTES_OMITTED]", value)
    text = _FILE_URI_RE.sub("[PATH_OMITTED]", text)
    text = _WINDOWS_PATH_RE.sub("[PATH_OMITTED]", text)
    text = _POSIX_PATH_RE.sub("[PATH_OMITTED]", text)
    if len(text) > _MAX_STRING_CHARS:
        suffix = "...[TRUNCATED]"
        text = text[: _MAX_STRING_CHARS - len(suffix)] + suffix
    return text


def sanitize_audit_value(value: Any, *, _depth: int = 0) -> Any:
    """Return deterministic JSON-safe evidence without labels, IDs, or paths."""

    if _depth >= _MAX_DEPTH:
        return "[MAX_DEPTH_OMITTED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[IMAGE_BYTES_OMITTED]"
    if isinstance(value, os.PathLike):
        return "[PATH_OMITTED]"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize_audit_value(dataclasses.asdict(value), _depth=_depth + 1)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return sanitize_audit_value(model_dump(mode="json"), _depth=_depth + 1)
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if _is_forbidden_key(key):
                continue
            safe_key = _sanitize_string(str(key))
            safe_item = sanitize_audit_value(item, _depth=_depth + 1)
            items.append((safe_key, safe_item))
        items.sort(key=lambda pair: pair[0])
        result = dict(items[:_MAX_MAPPING_ITEMS])
        if len(items) > _MAX_MAPPING_ITEMS:
            result["_truncated_mapping_items"] = len(items) - _MAX_MAPPING_ITEMS
        return result
    if isinstance(value, Sequence):
        items = [
            sanitize_audit_value(item, _depth=_depth + 1)
            for item in list(value)[:_MAX_SEQUENCE_ITEMS]
        ]
        if len(value) > _MAX_SEQUENCE_ITEMS:
            items.append(f"[TRUNCATED_{len(value) - _MAX_SEQUENCE_ITEMS}_ITEMS]")
        return items
    return _sanitize_string(str(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        sanitize_audit_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_image_summary(image_info: Any) -> Dict[str, Any]:
    if not isinstance(image_info, Mapping):
        return {}
    return {
        key: sanitize_audit_value(image_info[key])
        for key in ("width", "height", "mode")
        if key in image_info
    }


def build_arm_audit_prompt(
    question: str,
    image_info: Mapping[str, Any] | None,
    query_analysis: Any,
    available_tools: Sequence[str],
    original_output: Any,
    actions: Any,
    evidence_graph: Any,
    consistency_report: Any,
) -> str:
    """Build an untuned ARM prompt with no benchmark-answer input parameter."""

    tools = sorted(
        {
            str(tool).strip()
            for tool in available_tools
            if isinstance(tool, str) and str(tool).strip()
        }
    )
    blocks = {
        "query": sanitize_audit_value(question),
        "image_summary": _safe_image_summary(image_info),
        "initial_query_analysis": sanitize_audit_value(query_analysis),
        "available_tools": tools,
        "original_prediction": sanitize_audit_value(original_output),
        "original_actions": sanitize_audit_value(actions),
        "evidence_graph": sanitize_audit_value(evidence_graph),
        "consistency_report": sanitize_audit_value(consistency_report),
    }
    audit_input = json.dumps(
        blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"""Task: Audit one completed CARE-CT trajectory and decide whether one
fresh, independent trajectory is justified.

Ground-truth policy:
- The benchmark answer is unavailable.
- Judge only evidence completeness, tool failures, contradictions,
  localization, semantic support, anatomical agreement, numerical support,
  and cross-modal consistency.
- A high CTS is supporting context, not proof that the prediction is correct.

Audit input (canonical JSON):
{audit_input}

Return one ReflectionAudit JSON object with exactly these fields:
- reasoning_score: integer from 1 to 10.
- key_observations: list of concise evidence-grounded strings.
- identified_issues: list of material, fixable issues; use [] if none.
- error_type: one of {_canonical_json(list(ARM_ERROR_TYPES))}.
- should_rerun: boolean; true only for a material, fixable issue likely to
  change the answer or its evidentiary support.
- recommended_tool: exactly one listed available tool when should_rerun=true;
  otherwise "NONE".
- recovery_sub_goal: one concrete recovery goal when should_rerun=true;
  otherwise "NONE".
- confidence: number from 0 to 1.
- rationale: concise evidence-based explanation without hidden-answer claims.

The fresh trajectory starts from empty Memory. Do not copy unsupported
conclusions, expose internal identifiers, or include Markdown."""
