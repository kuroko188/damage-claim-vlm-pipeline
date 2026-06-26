# Multimodal Damage Claim Verification

A two-stage Gemini pipeline that verifies insurance-style damage claims from **chat transcripts**, **photos**, **user history**, and **evidence checklists**. Built for the HackerRank Orchestrate hackathon (June 2026).

For each claim (car, laptop, or package), the system decides whether images **support**, **contradict**, or provide **not enough information** — with structured fields for issue type, object part, severity, risk flags, and grounded justifications.

## Highlights

- **Two-stage architecture**: cheap text parsing (`gemini-2.5-flash-lite`) + multimodal review (`gemini-2.5-flash`)
- **Vision-first policy**: images are primary truth; user history only adds risk flags
- **Deterministic rule layer**: catches logical contradictions after the LLM (e.g. supported + no damage → contradicted)
- **Case-specific evidence rules**: injects matched rows from `evidence_requirements.csv` per claim
- **Eval-driven design**: two-stage vs single-shot comparison with metrics and confusion matrix
- **Production-minded ops**: disk cache, `--resume`, schema validation, retry with backoff

## Results (sample set, n=20)

| Metric | Two-stage (final) | Single-shot baseline |
|--------|-------------------|----------------------|
| **claim_status accuracy** | **90%** | 80% |
| **car claim_status** | **100%** (8/8) | 75% |
| evidence_standard_met | 85% | 90% |
| supporting_image_ids | 85% | 80% |

Full test set: **44 claims** processed in ~6 minutes (~88 API calls).

See [`code/evaluation/results.json`](code/evaluation/results.json) and [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md).

## Architecture

```text
claims.csv row
  → claim_parser.py      (Flash-Lite: parts, issues, injection flag)
  → data_loader.py       (matched evidence rules + image paths)
  → vision_analyzer.py   (Flash: multimodal review → JSON)
  → rules.py + schema.py (normalize enums, consistency checks, history flags)
  → output.csv
```

## Quick start

```bash
git clone <your-repo-url>
cd multimodal-claim-verification   # or your repo name

cd code
pip install -r requirements.txt
cp .env.example .env
# Set GOOGLE_API_KEY in .env
```

Run on the test set (from repo root):

```bash
python3 code/main.py
python3 code/validate_output.py
```

Evaluate on labeled sample data:

```bash
python3 code/evaluation/main.py
```

Options: `--limit N`, `--resume` (skip rows already in output), custom `--input` / `--output`.

## Repository layout

```text
.
├── README.md
├── output.csv                 # Example predictions (44 test claims)
├── code/
│   ├── main.py                # CLI entry point
│   ├── pipeline.py            # Orchestration
│   ├── claim_parser.py        # Stage 1: text
│   ├── vision_analyzer.py     # Stage 2: vision
│   ├── rules.py               # Post-LLM safety rules
│   ├── evaluation/            # Strategy comparison + metrics
│   └── ...
└── dataset/
    ├── claims.csv             # Test inputs
    ├── sample_claims.csv      # Labeled dev set
    ├── user_history.csv
    ├── evidence_requirements.csv
    └── images/
```

## Tech stack

- **Python 3.10+**
- **Google Gemini API** (`google-genai`) — multimodal JSON generation
- **python-dotenv** — local API key management (never committed)

No FastAPI, LangChain, or database — this is an offline batch pipeline optimized for reproducible evaluation.

## License

MIT (add a `LICENSE` file if you publish publicly).
