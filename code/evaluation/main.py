"""Evaluate two-stage vs single-shot strategies on sample_claims.csv."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import DATASET_DIR, OUTPUT_COLUMNS
from data_loader import claim_input_rows, read_csv_rows, write_csv_rows
from gemini_client import UsageStats
from pipeline import ClaimPipeline, process_single_shot_row

EXACT_SCORE_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "severity",
    "valid_image",
    "risk_flags",
    "supporting_image_ids",
]

TEXT_SCORE_FIELDS = [
    "evidence_standard_met_reason",
    "claim_status_justification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate claim verification strategies.")
    parser.add_argument(
        "--sample",
        type=Path,
        default=DATASET_DIR / "sample_claims.csv",
        help="Labeled sample CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N rows (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def _normalize_semicolon_field(value: str) -> str:
    text = value.strip().lower()
    if not text or text == "none":
        return "none"
    parts = sorted(part.strip() for part in text.split(";") if part.strip())
    return ";".join(parts)


def field_matches(field: str, expected: str, predicted: str) -> bool:
    if field in {"risk_flags", "supporting_image_ids"}:
        return _normalize_semicolon_field(expected) == _normalize_semicolon_field(
            predicted
        )
    return expected.strip().lower() == predicted.strip().lower()


def score_predictions(
    expected_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]
) -> dict[str, float | int | dict[str, dict[str, int]]]:
    total = len(expected_rows)
    exact = 0
    field_matches_count = {field: 0 for field in EXACT_SCORE_FIELDS}
    text_nonempty_matches = {field: 0 for field in TEXT_SCORE_FIELDS}

    by_object: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "claim_status_correct": 0}
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for expected, predicted in zip(expected_rows, predicted_rows, strict=True):
        row_exact = True
        for field in EXACT_SCORE_FIELDS:
            if field_matches(field, expected[field], predicted[field]):
                field_matches_count[field] += 1
            else:
                row_exact = False

        for field in TEXT_SCORE_FIELDS:
            if (
                expected[field].strip()
                and predicted[field].strip()
                and expected[field].strip().lower() == predicted[field].strip().lower()
            ):
                text_nonempty_matches[field] += 1

        claim_object = expected["claim_object"]
        by_object[claim_object]["rows"] += 1
        if field_matches("claim_status", expected["claim_status"], predicted["claim_status"]):
            by_object[claim_object]["claim_status_correct"] += 1

        confusion[expected["claim_status"]][predicted["claim_status"]] += 1

        if row_exact:
            exact += 1

    metrics: dict[str, float | int | dict[str, dict[str, int]]] = {
        "rows": total,
        "exact_row_matches": exact,
    }
    for field, count in field_matches_count.items():
        metrics[f"{field}_accuracy"] = round(count / total, 4) if total else 0.0
    for field, count in text_nonempty_matches.items():
        metrics[f"{field}_exact_match_rate"] = round(count / total, 4) if total else 0.0
    metrics["exact_row_accuracy"] = round(exact / total, 4) if total else 0.0
    metrics["by_claim_object"] = {
        claim_object: {
            **counts,
            "claim_status_accuracy": round(
                counts["claim_status_correct"] / counts["rows"], 4
            )
            if counts["rows"]
            else 0.0,
        }
        for claim_object, counts in sorted(by_object.items())
    }
    metrics["claim_status_confusion"] = {
        expected: dict(predicted_counts)
        for expected, predicted_counts in sorted(confusion.items())
    }
    return metrics


def export_mismatches(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    path: Path,
    *,
    strategy: str,
) -> int:
    mismatch_rows: list[dict[str, str]] = []
    for index, (expected, predicted) in enumerate(
        zip(expected_rows, predicted_rows, strict=True), start=1
    ):
        diff_fields = [
            field
            for field in EXACT_SCORE_FIELDS
            if not field_matches(field, expected[field], predicted[field])
        ]
        if not diff_fields:
            continue
        mismatch_rows.append(
            {
                "row_index": str(index),
                "strategy": strategy,
                "user_id": expected["user_id"],
                "claim_object": expected["claim_object"],
                "diff_fields": ";".join(diff_fields),
                "expected_claim_status": expected["claim_status"],
                "predicted_claim_status": predicted["claim_status"],
                "expected_risk_flags": expected["risk_flags"],
                "predicted_risk_flags": predicted["risk_flags"],
                "expected_supporting_image_ids": expected["supporting_image_ids"],
                "predicted_supporting_image_ids": predicted["supporting_image_ids"],
            }
        )

    write_csv_rows(
        path,
        mismatch_rows,
        [
            "row_index",
            "strategy",
            "user_id",
            "claim_object",
            "diff_fields",
            "expected_claim_status",
            "predicted_claim_status",
            "expected_risk_flags",
            "predicted_risk_flags",
            "expected_supporting_image_ids",
            "predicted_supporting_image_ids",
        ],
    )
    return len(mismatch_rows)


def estimate_cost_usd(usage: UsageStats) -> dict[str, float]:
    """Approximate paid-tier cost using official Gemini list prices."""
    text_calls = usage.by_model.get("gemini-2.5-flash-lite", 0)
    vision_calls = usage.by_model.get("gemini-2.5-flash", 0)
    input_cost = usage.input_tokens / 1_000_000 * 0.18
    output_cost = usage.output_tokens / 1_000_000 * 1.10
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "model_calls": usage.model_calls,
        "images_processed": usage.images_processed,
        "text_calls": text_calls,
        "vision_calls": vision_calls,
        "estimated_usd": round(input_cost + output_cost, 4),
    }


def main() -> None:
    args = parse_args()
    labeled_rows = read_csv_rows(args.sample)
    if args.limit > 0:
        labeled_rows = labeled_rows[: args.limit]

    input_rows = [{key: row[key] for key in OUTPUT_COLUMNS[:4]} for row in labeled_rows]

    two_stage = ClaimPipeline()
    two_stage_predictions = two_stage.process_rows(input_rows)

    single_usage = UsageStats()
    single_predictions = [
        process_single_shot_row(row, usage=single_usage) for row in input_rows
    ]

    two_stage_metrics = score_predictions(labeled_rows, two_stage_predictions)
    single_metrics = score_predictions(labeled_rows, single_predictions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    two_stage_mismatch_count = export_mismatches(
        labeled_rows,
        two_stage_predictions,
        output_dir / "sample_mismatches_two_stage.csv",
        strategy="two_stage",
    )
    single_mismatch_count = export_mismatches(
        labeled_rows,
        single_predictions,
        output_dir / "sample_mismatches_single_shot.csv",
        strategy="single_shot",
    )

    results = {
        "sample_rows": len(labeled_rows),
        "strategies": {
            "two_stage": {
                "description": "gemini-2.5-flash-lite text parse + gemini-2.5-flash vision",
                "metrics": two_stage_metrics,
                "usage": estimate_cost_usd(two_stage.usage),
                "mismatch_rows": two_stage_mismatch_count,
            },
            "single_shot": {
                "description": "gemini-2.5-flash only, one call per claim",
                "metrics": single_metrics,
                "usage": estimate_cost_usd(single_usage),
                "mismatch_rows": single_mismatch_count,
            },
        },
        "final_strategy": "two_stage",
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv_rows(
        output_dir / "sample_predictions_two_stage.csv",
        two_stage_predictions,
        OUTPUT_COLUMNS,
    )
    write_csv_rows(
        output_dir / "sample_predictions_single_shot.csv",
        single_predictions,
        OUTPUT_COLUMNS,
    )

    print(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
