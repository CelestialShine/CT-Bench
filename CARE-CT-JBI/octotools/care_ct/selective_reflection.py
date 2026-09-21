"""Versioned, answer-label-free admission and stopping rules for outer ARM."""
from __future__ import annotations

from .arm_strategies import trajectory_quality
from .config import CareCTConfig

REFLECTION_POLICY_VERSION = "care-ct-selective-reflection-v1-2026-09-08"
ORIGINAL_STEP_LIMIT = 6
REFLECTED_STEP_LIMIT = 4


def evidence_state(trajectory, task_context):
    quality = trajectory_quality(trajectory, task_context=task_context)
    profile = quality["task_evidence_profile"]
    flags = (trajectory.get("care_ct", {}).get("consistency") or {}).get("flags") or {}
    config = CareCTConfig()
    records = profile["option_records"]
    latest = records[-1] if records else {}
    confidence, margin = latest.get("confidence"), latest.get("margin")
    return quality, profile, flags, {
        "low_confidence": confidence is not None and confidence < config.classification_threshold,
        "low_margin": margin is not None and margin < config.classification_margin_threshold,
        "strong_primary": (confidence is not None and margin is not None
                           and confidence >= config.classification_threshold
                           and margin >= config.classification_margin_threshold),
    }


def evidence_stop_reason(trajectory, task_context):
    """Stop only on complete task evidence; never stop on a neutral CTS alone."""
    quality, profile, flags, scores = evidence_state(trajectory, task_context)
    if (quality["unresolved_tool_errors"] or not profile["localization_ready"]
            or not profile["primary_evidence_complete"] or flags.get("weak_detection")):
        return None
    if profile["size_task"]:
        return "calibrated_measurement_complete"
    if (scores["strong_primary"] and not profile["classifier_disagreement"]
            and not flags.get("organ_conflict")):
        return "complete_unambiguous_option_evidence"
    return None


def reflection_admission(trajectory, *, task_context, audit, available_tools,
                         max_recovery_attempts=1):
    """Audit suggestions require observable evidence and an actionable target.

    Existing thresholds are reused; none are tuned against scored case labels.
    Missing output/evidence remains recoverable. Low CTS or a generic semantic
    concern alone cannot launch another complete trajectory.
    """
    quality, profile, flags, scores = evidence_state(trajectory, task_context)
    available = sorted(set(available_tools))
    evidence = trajectory.get("care_ct", {}).get("evidence") or []
    attempted = {str(item.get("tool_name", "")).casefold() for item in evidence}
    reasons, target = [], None
    if not quality["valid_structured_choice"]:
        reasons.append("invalid_structured_output")
    if quality["unresolved_tool_errors"]:
        reasons.append("unresolved_tool_error")
        target = quality["unresolved_tool_errors"][0]
    if not profile["localization_ready"]:
        reasons.append("missing_localization")
        target = "MaskRCNN_Object_Detector_Tool"
    elif not profile["primary_evidence_complete"]:
        reasons.append("missing_primary_evidence")
        target = ("CT_Lesion_Measurement_Tool" if profile["size_task"]
                  else "Biomedclip_Tunedbox_Tool")
    elif profile["primary_prediction_aligned"] is False:
        reasons.append("answer_evidence_mismatch")
    # A new verifier must not simply replay a successful call with the same inputs.
    if not reasons and not profile["size_task"]:
        corroborated = (profile["classifier_disagreement"] or flags.get("organ_conflict")
                        or (scores["low_margin"] and (scores["low_confidence"]
                                                     or flags.get("weak_detection"))))
        candidates = [audit.get("recommended_tool"), "BiomedCLIP_Tool",
                      "Biomedclip_Tunednobox_Tool"]
        if audit.get("should_rerun") is True and corroborated:
            target = next((tool for tool in candidates if tool in available
                           and tool.casefold() not in attempted
                           and "biomedclip" in tool.casefold()), None)
            if target:
                reasons.append("corroborated_uncertainty_with_unused_verifier")
    actionable = target is None or target in available
    admitted = bool(reasons and actionable and max_recovery_attempts > 0)
    return {
        "policy_version": REFLECTION_POLICY_VERSION,
        "should_rerun": admitted,
        "reasons": reasons,
        "target_tool": target if admitted else None,
        "available_tools": available,
        "max_recovery_attempts": max_recovery_attempts,
        "reflected_step_limit": REFLECTED_STEP_LIMIT,
        "audit_requested": audit.get("should_rerun") is True,
        "decision": ("admitted" if admitted else
                     "recovery_disabled" if max_recovery_attempts <= 0 else
                     "target_unavailable" if not actionable else
                     "no_actionable_evidence_defect"),
    }
