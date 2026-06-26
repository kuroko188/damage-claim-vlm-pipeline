"""CLI entry point for generating output.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_INPUT, DEFAULT_OUTPUT, OUTPUT_COLUMNS
from data_loader import claim_input_rows, write_csv_rows
from pipeline import ClaimPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify damage claims and write structured predictions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with claim rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV for predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 = all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output file.",
    )
    return parser.parse_args()


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["user_id"],
        row["image_paths"],
        row["user_claim"],
        row["claim_object"],
    )


def _load_existing_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    from data_loader import read_csv_rows

    existing: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_csv_rows(path):
        if all(column in row for column in OUTPUT_COLUMNS):
            existing[_row_key(row)] = {column: row[column] for column in OUTPUT_COLUMNS}
    return existing


def main() -> None:
    args = parse_args()
    rows = claim_input_rows(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    existing = _load_existing_rows(args.output) if args.resume else {}
    pipeline = ClaimPipeline()
    predictions: list[dict[str, str]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        key = _row_key(row)
        if key in existing:
            predictions.append(existing[key])
            print(f"[{index}/{total}] skipped (cached output): {row['user_id']}")
            continue
        print(f"[{index}/{total}] processing: {row['user_id']}")
        predictions.append(pipeline.process_row(row))

    write_csv_rows(args.output, predictions, OUTPUT_COLUMNS)

    print(f"Wrote {len(predictions)} rows to {args.output}")
    print(
        "Usage: "
        f"calls={pipeline.usage.model_calls}, "
        f"input_tokens={pipeline.usage.input_tokens}, "
        f"output_tokens={pipeline.usage.output_tokens}, "
        f"images={pipeline.usage.images_processed}"
    )


if __name__ == "__main__":
    main()
