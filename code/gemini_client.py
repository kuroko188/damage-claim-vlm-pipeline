"""Shared Gemini client helpers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from config import MAX_RETRIES, RETRY_BASE_DELAY_SEC, TEMPERATURE, get_api_key


@dataclass
class UsageStats:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images_processed: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, usage: Any | None, images: int = 0) -> None:
        self.model_calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1
        self.images_processed += images
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)


def build_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_json(
    client: genai.Client,
    *,
    model: str,
    contents: list[Any],
    usage: UsageStats,
    images: int = 0,
    thinking_budget: int | None = 0,
    cache_stage: str | None = None,
    cache_payload: str | None = None,
) -> dict[str, Any]:
    if cache_stage and cache_payload:
        from cache_store import load_cached, save_cached

        cached = load_cached(cache_stage, cache_payload)
        if cached is not None:
            return cached

    config_kwargs: dict[str, Any] = {
        "temperature": TEMPERATURE,
        "response_mime_type": "application/json",
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            usage.record(model, response.usage_metadata, images=images)
            text = response.text or ""
            if not text.strip():
                raise RuntimeError(f"Empty response from {model}")
            result = extract_json(text)
            if cache_stage and cache_payload:
                from cache_store import save_cached

                save_cached(cache_stage, cache_payload, result)
            return result
        except Exception as exc:  # noqa: BLE001 - retry on transient API failures
            last_error = exc
            if attempt + 1 >= MAX_RETRIES:
                break
            time.sleep(RETRY_BASE_DELAY_SEC * (2**attempt))
    raise RuntimeError(f"Gemini call failed for {model}") from last_error
