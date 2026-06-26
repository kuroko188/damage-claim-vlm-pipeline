"""Runtime configuration for the claim verification pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_INPUT = DATASET_DIR / "claims.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output.csv"

load_dotenv(CODE_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env")

TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite")
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

TEMPERATURE = 0.0
MAX_RETRIES = 5
RETRY_BASE_DELAY_SEC = 2.0

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY before running the pipeline."
        )
    return key
