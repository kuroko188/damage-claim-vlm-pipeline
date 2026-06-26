"""Validate output.csv schema and row count against claims.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import DATASET_DIR, DEFAULT_INPUT, DEFAULT_OUTPUT, OUTPUT_COLUMNS
from data_loader import claim_input_rows
from schema import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    RISK_FLAGS,
    SEVERITIES,
    normalize_bool,
    normalize_enum_strict,
    normalize_object_part,
    normalize_risk_flags,
    normalize_supporting_image_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated output.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(
                f"{path}: expected columns {OUTPUT_COLUMNS}, got {reader.fieldnames}"
            )
        return list(reader)


def validate_row(row: dict[str, str], index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"row {index + 1} ({row.get('user_id', '?')})"

    for column in OUTPUT_COLUMNS:
        if column not in row or row[column] is None:
            errors.append(f"{prefix}: missing column {column}")

    if normalize_bool(row.get("evidence_standard_met", "")) not in {"true", "false"}:
        errors.append(f"{prefix}: invalid evidence_standard_met")
    if normalize_bool(row.get("valid_image", "")) not in {"true", "false"}:
        errors.append(f"{prefix}: invalid valid_image")

    status = normalize_enum_strict(row.get("claim_status", ""), CLAIM_STATUSES, "")
    if status not in CLAIM_STATUSES:
        errors.append(f"{prefix}: invalid claim_status")

    issue = normalize_enum_strict(row.get("issue_type", ""), ISSUE_TYPES, "")
    if issue not in ISSUE_TYPES:
        errors.append(f"{prefix}: invalid issue_type")

    part = normalize_object_part(row.get("object_part", ""), row.get("claim_object", ""))
    if part == "unknown" and row.get("object_part", "").strip().lower() not in {
        "unknown",
        "",
    }:
        errors.append(f"{prefix}: invalid object_part for {row.get('claim_object')}")

    severity = normalize_enum_strict(row.get("severity", ""), SEVERITIES, "")
    if severity not in SEVERITIES:
        errors.append(f"{prefix}: invalid severity")

    flags = normalize_risk_flags(row.get("risk_flags", ""))
    for flag in flags.split(";"):
        if flag and flag not in RISK_FLAGS:
            errors.append(f"{prefix}: invalid risk flag {flag}")

    supporting = normalize_supporting_image_ids(row.get("supporting_image_ids", ""))
    if supporting != "none":
        for image_id in supporting.split(";"):
            if not image_id.startswith("img_"):
                errors.append(f"{prefix}: invalid supporting_image_id {image_id}")

    for text_col in (
        "evidence_standard_met_reason",
        "claim_status_justification",
    ):
        if not str(row.get(text_col, "")).strip():
            errors.append(f"{prefix}: empty {text_col}")

    return errors


def main() -> None:
    args = parse_args()
    if not args.output.exists():
        raise SystemExit(f"Missing output file: {args.output}")

    input_rows = claim_input_rows(args.input) if args.input.exists() else []
    output_rows = read_rows(args.output)

    errors: list[str] = []
    if input_rows and len(output_rows) != len(input_rows):
        errors.append(
            f"Row count mismatch: input={len(input_rows)} output={len(output_rows)}"
        )

    for index, row in enumerate(output_rows):
        errors.extend(validate_row(row, index))

    if errors:
        print(f"Validation failed with {len(errors)} issue(s):")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        raise SystemExit(1)

    print(
        f"OK: {len(output_rows)} rows, columns match schema, "
        f"input={args.input.name if args.input.exists() else 'n/a'}"
    )


if __name__ == "__main__":
    main()
