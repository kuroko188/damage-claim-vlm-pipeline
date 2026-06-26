"""End-to-end claim processing pipeline."""

from __future__ import annotations

from typing import Any

from claim_parser import parse_claim
from config import OUTPUT_COLUMNS
from data_loader import (
    load_evidence_requirements,
    load_user_history,
    relevant_evidence_rules,
    resolve_image_path,
    split_image_paths,
)
from gemini_client import UsageStats, build_client, generate_json
from rules import finalize_prediction
from schema import image_id_from_path
from vision_analyzer import VISION_SYSTEM, analyze_images, build_vision_context


class ClaimPipeline:
    def __init__(self) -> None:
        self.client = build_client()
        self.usage = UsageStats()
        self.evidence_requirements = load_evidence_requirements()
        self.user_history = load_user_history()

    def process_row(self, row: dict[str, str]) -> dict[str, str]:
        history = self.user_history.get(row["user_id"])
        parsed_claim = parse_claim(
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
            client=self.client,
            usage=self.usage,
        )
        image_paths = split_image_paths(row["image_paths"])
        evidence_rules = relevant_evidence_rules(
            self.evidence_requirements,
            row["claim_object"],
            parsed_claim.get("issue_families", []),
            claimed_parts=parsed_claim.get("claimed_parts", []),
            claimed_issue_types=parsed_claim.get("claimed_issue_types", []),
            image_count=len(image_paths),
        )
        vision_result = analyze_images(
            image_paths=row["image_paths"],
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
            parsed_claim=parsed_claim,
            evidence_rules=evidence_rules,
            user_history=history,
            client=self.client,
            usage=self.usage,
        )

        if parsed_claim.get("ignore_instructions_in_chat"):
            from schema import merge_risk_flags

            vision_result["risk_flags"] = merge_risk_flags(
                vision_result.get("risk_flags"),
                "text_instruction_present",
            )

        finalized = finalize_prediction(
            vision_result,
            row["claim_object"],
            history,
            parsed_claim=parsed_claim,
        )
        output = {key: row[key] for key in row}
        output.update(finalized)
        return {column: output[column] for column in OUTPUT_COLUMNS}

    def process_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return [self.process_row(row) for row in rows]


def process_single_shot_row(
    row: dict[str, str],
    *,
    client=None,
    usage: UsageStats | None = None,
    user_history: dict[str, dict[str, str]] | None = None,
    evidence_requirements: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    """Single-call baseline for evaluation comparisons."""
    from config import VISION_MODEL
    from google.genai import types

    usage = usage or UsageStats()
    client = client or build_client()
    user_history = user_history or load_user_history()
    evidence_requirements = evidence_requirements or load_evidence_requirements()
    history = user_history.get(row["user_id"])
    image_paths = split_image_paths(row["image_paths"])
    rules = relevant_evidence_rules(
        evidence_requirements,
        row["claim_object"],
        ["general claim review"],
        image_count=len(image_paths),
    )

    parts: list[Any] = [
        VISION_SYSTEM,
        "\nSingle-shot mode: extract the claim from the conversation and review all images.\n",
        build_vision_context(
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
            parsed_claim={"mode": "single_shot"},
            evidence_rules=rules,
            user_history=history,
            single_shot=True,
        ),
    ]
    for relative in image_paths:
        absolute = resolve_image_path(relative)
        image_id = image_id_from_path(relative)
        parts.append(f"\nImage ID: {image_id}\n")
        data = absolute.read_bytes()
        suffix = absolute.suffix.lower()
        mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    vision_result = generate_json(
        client,
        model=VISION_MODEL,
        contents=parts,
        usage=usage,
        images=len(image_paths),
        thinking_budget=0,
    )
    finalized = finalize_prediction(
        vision_result,
        row["claim_object"],
        history,
        parsed_claim=None,
    )
    output = {key: row[key] for key in row}
    output.update(finalized)
    return {column: output[column] for column in OUTPUT_COLUMNS}
