"""Deterministic post-processing for evidence and user-history rules."""

from __future__ import annotations

from typing import Any

from schema import merge_risk_flags, normalize_prediction

GENERIC_PARTS = {"body", "unknown"}


def _flag_set(flags: str) -> set[str]:
    if not flags or flags == "none":
        return set()
    return {part.strip() for part in flags.split(";") if part.strip()}


def _claimed_parts(parsed_claim: dict[str, Any] | None) -> set[str]:
    if not parsed_claim:
        return set()
    parts = set(parsed_claim.get("claimed_parts") or [])
    primary = str(parsed_claim.get("primary_part", "")).strip()
    if primary:
        parts.add(primary)
    return {part for part in parts if part and part != "unknown"}


def _apply_object_part_hints(
    updated: dict[str, str],
    parsed_claim: dict[str, Any] | None,
    claim_object: str,
) -> dict[str, str]:
    if not parsed_claim or claim_object not in {"car", "laptop", "package"}:
        return updated

    claimed = _claimed_parts(parsed_claim)
    primary = str(parsed_claim.get("primary_part", "")).strip()
    current = updated.get("object_part", "unknown")
    status = updated.get("claim_status", "not_enough_information")

    if primary and primary in claimed:
        if current in GENERIC_PARTS and status in {"supported", "contradicted"}:
            updated["object_part"] = primary
        elif (
            status == "supported"
            and current not in claimed
            and primary != "unknown"
        ):
            updated["object_part"] = primary

    if status == "contradicted" and "claim_mismatch" in _flag_set(
        updated.get("risk_flags", "none")
    ):
        if current in GENERIC_PARTS and primary:
            updated["object_part"] = primary

    return updated


def history_risk_flags(history: dict[str, str] | None) -> str:
    if not history:
        return "none"

    flags: list[str] = []
    history_flags = history.get("history_flags", "none").strip()
    if history_flags and history_flags.lower() != "none":
        for flag in history_flags.split(";"):
            flag = flag.strip()
            if flag:
                flags.append(flag)

    rejected = int(history.get("rejected_claim", "0") or 0)
    manual = int(history.get("manual_review_claim", "0") or 0)
    past = int(history.get("past_claim_count", "0") or 0)

    if rejected >= 2 or (past >= 5 and rejected >= 1):
        if "user_history_risk" not in flags:
            flags.append("user_history_risk")

    if manual >= 2 or "manual_review_required" in history_flags:
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")

    return ";".join(flags) if flags else "none"


def apply_consistency_rules(
    prediction: dict[str, str],
    parsed_claim: dict[str, Any] | None,
    claim_object: str = "",
) -> dict[str, str]:
    updated = dict(prediction)
    flags = updated.get("risk_flags", "none")
    status = updated.get("claim_status", "not_enough_information")
    supporting = updated.get("supporting_image_ids", "none")
    valid_image = updated.get("valid_image", "false")
    evidence_met = updated.get("evidence_standard_met", "false")
    issue_type = updated.get("issue_type", "unknown")
    flag_set = _flag_set(flags)

    if status == "supported" and issue_type == "none":
        updated["claim_status"] = "contradicted"
        flags = merge_risk_flags(flags, "damage_not_visible")
        status = updated["claim_status"]

    if (
        status == "contradicted"
        and "wrong_object" in flag_set
        and evidence_met == "false"
        and "claim_mismatch" not in flag_set
    ):
        updated["claim_status"] = "not_enough_information"
        status = updated["claim_status"]

    if (
        status == "not_enough_information"
        and "blurry_image" in flag_set
        and supporting != "none"
        and evidence_met == "true"
    ):
        updated["claim_status"] = "supported"
        status = updated["claim_status"]
        trimmed = [flag for flag in flag_set if flag != "wrong_object"]
        flags = ";".join(trimmed) if trimmed else "none"

    if status == "supported" and supporting == "none":
        flags = merge_risk_flags(flags, "manual_review_required")
    if valid_image == "false" and evidence_met == "true":
        flags = merge_risk_flags(flags, "manual_review_required")
    if status == "contradicted" and issue_type == "none":
        flags = merge_risk_flags(flags, "claim_mismatch")

    updated["risk_flags"] = flags
    updated = _apply_object_part_hints(updated, parsed_claim, claim_object)
    return updated


def apply_history_rules(prediction: dict, history: dict[str, str] | None) -> dict:
    updated = dict(prediction)
    updated["risk_flags"] = merge_risk_flags(
        prediction.get("risk_flags"),
        history_risk_flags(history),
    )

    if history and history.get("history_flags", "none") != "none":
        justification = updated.get("claim_status_justification", "")
        summary = history.get("history_summary", "").strip()
        if summary and summary.lower() not in justification.lower():
            if "user_history_risk" in updated["risk_flags"] and "history" not in justification.lower():
                updated["claim_status_justification"] = (
                    f"{justification} User history: {summary}".strip()
                )

    return updated


def finalize_prediction(
    raw_prediction: dict,
    claim_object: str,
    history: dict[str, str] | None,
    parsed_claim: dict[str, Any] | None = None,
) -> dict:
    normalized = normalize_prediction(raw_prediction, claim_object)
    normalized = apply_consistency_rules(
        normalized, parsed_claim, claim_object=claim_object
    )
    return apply_history_rules(normalized, history)
