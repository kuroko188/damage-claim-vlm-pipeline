"""Allowed output values and normalization helpers."""

from __future__ import annotations

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

CAR_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
}

LAPTOP_PARTS = {
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
}

PACKAGE_PARTS = {
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
}

RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

SEVERITIES = {"none", "low", "medium", "high", "unknown"}

OBJECT_PARTS_BY_CLAIM = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

PART_ALIASES = {
    "front_bumper": {"front_bumper", "front bumper", "front_bumper_area", "bumper"},
    "rear_bumper": {"rear_bumper", "rear bumper", "back_bumper"},
    "side_mirror": {"side_mirror", "mirror", "side mirror"},
    "windshield": {"windshield", "front_glass", "front glass", "windscreen"},
    "headlight": {"headlight", "front_light", "front light"},
    "taillight": {"taillight", "rear_light", "back_light", "tail_light"},
    "door": {"door", "door_panel", "door panel"},
    "hood": {"hood", "top_panel", "top panel", "bonnet"},
    "package_corner": {"package_corner", "corner", "box_corner"},
    "package_side": {"package_side", "side", "box_side"},
    "seal": {"seal", "tape", "flap"},
}


def normalize_bool(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return "false"


def normalize_enum_strict(value: object, allowed: set[str], default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in allowed else default


def normalize_enum_fuzzy(value: object, allowed: set[str], default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if text in allowed:
        return text
    for canonical, aliases in PART_ALIASES.items():
        if canonical in allowed and text in aliases:
            return canonical
    for candidate in sorted(allowed, key=len, reverse=True):
        if candidate in text or text in candidate:
            return candidate
    return default


def normalize_object_part(value: object, claim_object: str) -> str:
    allowed = OBJECT_PARTS_BY_CLAIM.get(claim_object, {"unknown"})
    return normalize_enum_fuzzy(value, allowed, "unknown")


def normalize_risk_flags(value: object) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "none"
    parts = []
    for raw in text.replace(",", ";").split(";"):
        flag = normalize_enum_strict(raw, RISK_FLAGS, "")
        if flag and flag != "none" and flag not in parts:
            parts.append(flag)
    return ";".join(parts) if parts else "none"


def merge_risk_flags(*values: object) -> str:
    merged: list[str] = []
    for value in values:
        normalized = normalize_risk_flags(value)
        if normalized == "none":
            continue
        for flag in normalized.split(";"):
            if flag not in merged:
                merged.append(flag)
    return ";".join(merged) if merged else "none"


def normalize_supporting_image_ids(value: object) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "none"
    parts = []
    for raw in text.replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if not item.startswith("img_"):
            item = image_id_from_path(item)
        if item not in parts:
            parts.append(item)
    return ";".join(parts) if parts else "none"


def image_id_from_path(path: str) -> str:
    return path.rsplit(".", 1)[0].split("/")[-1].split("\\")[-1]


def normalize_prediction(raw: dict, claim_object: str) -> dict:
    return {
        "evidence_standard_met": normalize_bool(raw.get("evidence_standard_met", False)),
        "evidence_standard_met_reason": str(
            raw.get("evidence_standard_met_reason", "")
        ).strip(),
        "risk_flags": normalize_risk_flags(raw.get("risk_flags")),
        "issue_type": normalize_enum_strict(
            raw.get("issue_type"), ISSUE_TYPES, "unknown"
        ),
        "object_part": normalize_object_part(raw.get("object_part"), claim_object),
        "claim_status": normalize_enum_strict(
            raw.get("claim_status"), CLAIM_STATUSES, "not_enough_information"
        ),
        "claim_status_justification": str(
            raw.get("claim_status_justification", "")
        ).strip(),
        "supporting_image_ids": normalize_supporting_image_ids(
            raw.get("supporting_image_ids")
        ),
        "valid_image": normalize_bool(raw.get("valid_image", False)),
        "severity": normalize_enum_strict(raw.get("severity"), SEVERITIES, "unknown"),
    }
