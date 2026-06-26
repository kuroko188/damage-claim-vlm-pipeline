"""Load CSV inputs and auxiliary datasets."""

from __future__ import annotations

import csv
from pathlib import Path

from config import DATASET_DIR, INPUT_COLUMNS, REPO_ROOT

ALWAYS_GENERAL_RULE_IDS = {"REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"}
MULTI_IMAGE_RULE_ID = "REQ_GENERAL_MULTI_IMAGE"

CAR_BODY_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "fender",
    "quarter_panel",
    "body",
}
CAR_GLASS_LIGHT_PARTS = {
    "windshield",
    "headlight",
    "taillight",
    "side_mirror",
}
LAPTOP_INPUT_PARTS = {"screen", "keyboard", "trackpad"}
LAPTOP_BODY_PARTS = {"hinge", "lid", "corner", "port", "base", "body"}
PACKAGE_EXTERIOR_ISSUES = {"crushed_packaging", "torn_packaging"}
PACKAGE_STAIN_ISSUES = {"water_damage", "stain"}
PACKAGE_CONTENTS_PARTS = {"contents", "item"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def split_image_paths(image_paths: str) -> list[str]:
    return [part.strip() for part in image_paths.split(";") if part.strip()]


def resolve_image_path(relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").strip()
    candidates = [
        REPO_ROOT / normalized,
        DATASET_DIR / normalized,
    ]
    if normalized.startswith("images/"):
        candidates.append(DATASET_DIR / normalized)
        candidates.append(REPO_ROOT / "dataset" / normalized)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_user_history() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(DATASET_DIR / "user_history.csv")
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements() -> list[dict[str, str]]:
    return read_csv_rows(DATASET_DIR / "evidence_requirements.csv")


def _normalize_tokens(values: list[str] | None) -> set[str]:
    tokens: set[str] = set()
    for value in values or []:
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if text:
            tokens.add(text)
    return tokens


def _family_matches_applies(families: set[str], applies: str) -> bool:
    if applies in families:
        return True
    return any(family in applies or applies in family for family in families)


def _rule_matches(
    rule: dict[str, str],
    *,
    claim_object: str,
    families: set[str],
    parts: set[str],
    issues: set[str],
    image_count: int,
) -> bool:
    requirement_id = rule["requirement_id"]
    obj = rule["claim_object"]
    applies = rule["applies_to"].lower()

    if requirement_id in ALWAYS_GENERAL_RULE_IDS:
        return True
    if requirement_id == MULTI_IMAGE_RULE_ID:
        return image_count > 1
    if obj not in {claim_object, "all"}:
        return False

    if _family_matches_applies(families, applies):
        return True

    if claim_object == "car":
        if requirement_id == "REQ_CAR_BODY_PANEL":
            return bool(
                parts & CAR_BODY_PARTS
                or issues & {"dent", "scratch"}
                or families & {"dent or scratch"}
            )
        if requirement_id == "REQ_CAR_GLASS_LIGHT_MIRROR":
            return bool(
                parts & CAR_GLASS_LIGHT_PARTS
                or issues & {"crack", "glass_shatter", "broken_part", "missing_part"}
            )
        if requirement_id == "REQ_CAR_IDENTITY_OR_SIDE":
            return image_count > 1 or "vehicle identity or orientation" in families

    if claim_object == "laptop":
        if requirement_id == "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD":
            return bool(parts & LAPTOP_INPUT_PARTS or "screen, keyboard, or trackpad" in applies)
        if requirement_id == "REQ_LAPTOP_BODY_HINGE_PORT":
            return bool(parts & LAPTOP_BODY_PARTS)

    if claim_object == "package":
        if requirement_id == "REQ_PACKAGE_EXTERIOR":
            return bool(
                parts & {"box", "package_corner", "package_side", "seal"}
                or issues & PACKAGE_EXTERIOR_ISSUES
            )
        if requirement_id == "REQ_PACKAGE_LABEL_OR_STAIN":
            return bool(
                parts & {"label", "package_side", "box"}
                or issues & PACKAGE_STAIN_ISSUES
            )
        if requirement_id == "REQ_PACKAGE_CONTENTS":
            return bool(parts & PACKAGE_CONTENTS_PARTS or issues & {"missing_part"})

    return False


def relevant_evidence_rules(
    requirements: list[dict[str, str]],
    claim_object: str,
    issue_families: list[str],
    *,
    claimed_parts: list[str] | None = None,
    claimed_issue_types: list[str] | None = None,
    image_count: int = 1,
) -> list[dict[str, str]]:
    families = _normalize_tokens(issue_families)
    parts = _normalize_tokens(claimed_parts)
    issues = _normalize_tokens(claimed_issue_types)

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for rule in requirements:
        if _rule_matches(
            rule,
            claim_object=claim_object,
            families=families,
            parts=parts,
            issues=issues,
            image_count=image_count,
        ):
            requirement_id = rule["requirement_id"]
            if requirement_id not in seen:
                selected.append(rule)
                seen.add(requirement_id)

    if not selected:
        fallback = next(
            (
                rule
                for rule in requirements
                if rule["requirement_id"] == "REQ_GENERAL_OBJECT_PART"
            ),
            None,
        )
        if fallback:
            selected = [fallback]
    return selected


def claim_input_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [{key: row[key] for key in INPUT_COLUMNS} for row in rows]
