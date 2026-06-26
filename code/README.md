# Implementation

Python source for the multimodal claim verification pipeline. See the [root README](../README.md) for overview, results, and setup.

## Run

From repository root:

```bash
python3 code/main.py
python3 code/validate_output.py
python3 code/evaluation/main.py
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | — | Required |
| `GEMINI_TEXT_MODEL` | `gemini-2.5-flash-lite` | Stage 1 parsing |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Stage 2 vision |

Copy `.env.example` to `.env` and set your API key locally.

## Modules

| File | Role |
|------|------|
| `main.py` | CLI: batch CSV in → CSV out |
| `pipeline.py` | `ClaimPipeline.process_row()` orchestration |
| `claim_parser.py` | Extract claimed parts/issues from chat |
| `vision_analyzer.py` | Multimodal image review + JSON context |
| `rules.py` | Deterministic post-processing |
| `schema.py` | Enum normalization |
| `data_loader.py` | CSV I/O, evidence rule matching, image paths |
| `gemini_client.py` | API client, retries, usage tracking, cache |
| `cache_store.py` | Disk cache for model responses |
| `validate_output.py` | Output schema validation |

## Evaluation

`evaluation/main.py` compares **two-stage** vs **single-shot** on `dataset/sample_claims.csv`.

Outputs:

- `evaluation/results.json` — accuracy metrics, confusion matrix, token usage
- `evaluation/evaluation_report.md` — operational notes

Mismatch CSVs are generated when you run evaluation (not required in git).
