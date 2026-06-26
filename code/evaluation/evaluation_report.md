# Evaluation Report

This report describes how the sample set was evaluated and the operational characteristics of the final Gemini-based pipeline.

## Strategies compared

| Strategy | Models | Calls per claim |
|----------|--------|-----------------|
| **Two-stage (final)** | `gemini-2.5-flash-lite` (text) + `gemini-2.5-flash` (vision) | 2 |
| **Single-shot (baseline)** | `gemini-2.5-flash` only | 1 |

Run evaluation:

```bash
cd code
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key
python evaluation/main.py
```

Metrics are written to `code/evaluation/results.json`.

## Final strategy

Production predictions in `output.csv` use the **two-stage** pipeline:

1. Text model extracts claimed parts, issue families, and review intent from the chat transcript.
2. Vision model inspects all submitted images with evidence requirements and user history context.
3. Deterministic rules add user-history risk flags without overriding clear visual evidence.

## Operational analysis (approximate)

Figures below assume the full test set (`dataset/claims.csv`, 44 rows, ~82 images) and official list pricing from [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) as of June 2026.

### Model calls

| Dataset | Two-stage calls | Single-shot calls | Images |
|---------|-----------------|-------------------|--------|
| Sample (20 rows) | ~40 | ~20 | ~31 |
| Test (44 rows) | ~88 | ~44 | ~82 |
| Sample + test once | ~128 | ~64 | ~113 |

### Token usage (estimate)

| Component | Input tokens | Output tokens |
|-----------|--------------|---------------|
| Text stage (Flash-Lite) | ~40K / 44 test rows | ~13K |
| Vision stage (Flash) | ~160K / 44 test rows | ~22K |
| **Two-stage total (test)** | **~200K** | **~35K** |

Image tokenization uses Gemini 2.x pan-and-scan (~258 tokens per small tile). Actual usage is recorded at runtime via `usage_metadata`.

### Cost estimate (paid tier, USD)

Pricing used:

- `gemini-2.5-flash-lite`: $0.10 / 1M input, $0.40 / 1M output
- `gemini-2.5-flash`: $0.30 / 1M input, $2.50 / 1M output

| Workload | Estimated cost |
|----------|----------------|
| Test set once (44 rows, two-stage) | **$0.03 – $0.25** |
| Sample evaluation + test generation | **$0.08 – $0.40** |
| Full dev loop (3 reruns + eval) | **$0.20 – $1.00** |

Free-tier quotas in Google AI Studio may cover sample + test runs entirely when rate limits are respected.

### Runtime and rate limits

- Serial processing: ~3–8 seconds per vision call → **~5–12 minutes** for 44 test claims (two-stage).
- Retry policy: exponential backoff on transient errors (`MAX_RETRIES=5`).
- Thinking budget for vision calls is set to `0` to reduce latency and cost.
- Recommended execution: sequential requests to stay within free-tier RPM (~10–15 RPM).

### Cost/latency controls implemented

- Temperature fixed at `0.0` for reproducibility.
- Separate cheap text model for parsing.
- No repeated vision calls unless a row is reprocessed.
- Usage counters printed after `python main.py` for cost auditing

## Metrics

Run after setting `GOOGLE_API_KEY` in `code/.env`:

```bash
cd code
python3 evaluation/main.py
```

Latest metrics are written to `code/evaluation/results.json`. The evaluator now scores:

- all core decision fields, including `risk_flags` and `supporting_image_ids`
- `claim_status` accuracy grouped by `claim_object`
- a `claim_status` confusion matrix
- mismatch exports (generated on eval run): `sample_mismatches_two_stage.csv`, `sample_mismatches_single_shot.csv`

Re-run after prompt changes; metrics in `results.json` are the source of truth.
