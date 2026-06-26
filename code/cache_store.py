"""Disk cache for model JSON responses (deterministic reruns, resume support)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config import CODE_ROOT

CACHE_DIR = CODE_ROOT / ".cache" / "responses"


def _cache_key(stage: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{stage}_{digest}"


def load_cached(stage: str, payload: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{_cache_key(stage, payload)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cached(stage: str, payload: str, result: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(stage, payload)}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
