# news-insights

Analyze news articles from the web to determine sentiment, tone/emotions, and intent. Results are stored for continual learning and future fine-tuning.

## Quickstart (Windows + Poetry)

1. Install Python 3.12 and Poetry.
2. Install dependencies:
   - `cd D:\\code\\news-insights`
   - `poetry install`
   - `poetry run pre-commit install`
3. Initialize the database and ingest some articles:
   - `poetry run news-insights init-db`
   - `poetry run news-insights ingest --limit 30`
   - Add a per-source time budget if needed (in seconds): `poetry run news-insights ingest --limit 30 --time-budget-s 120`
4. Run baseline predictions (sentiment, emotion, intent):
   - `poetry run news-insights predict`
   - With zero-shot presets: `poetry run news-insights predict --preset genre`
   - With custom labels: `poetry run news-insights predict --labels "news report,analysis,opinion"`
   - Handle long articles with chunking: `poetry run news-insights predict --chunking --max-tokens 512 --overlap 64 --agg mean`
5. Export a labeled dataset (once you add labels):
   - `poetry run news-insights export-dataset`

## Commands

- `init-db` — Create tables and seed default RSS sources
- `ingest` — Fetch latest articles from RSS, extract full text, store (supports per-source time budget)
- `predict` — Run baseline Transformers pipelines and store predictions (supports zero-shot label presets and long-document chunking)
- `export-dataset` — Export labeled samples to Parquet for training
- `train` — Placeholder for fine-tuning (to be enabled when labels are available)
- `info` — Show environment/config info

## Configuration

Environment variables (prefix `NEWS_INSIGHTS_`) configure the app. You can copy `configs/.env.example` to `.env` and adjust.

Key settings:
- `DATABASE_URL` (default: `sqlite:///./data/news_insights.db`)
- `FEEDS` — Comma-separated list of RSS feed URLs
- `ZERO_SHOT_LABELS` — Comma-separated candidate labels for intent classification
- `ZERO_SHOT_PRESET` — Optional preset name to use instead of `ZERO_SHOT_LABELS` (e.g., `genre`, `purpose`, `verification`, `provenance`, `coverage`)
- `REQUEST_TIMEOUT_S` — Per-request timeout (seconds) for fetching article pages (default 20)
- `PER_SOURCE_TIME_BUDGET_S` — Max seconds to spend per source during ingestion (default 120; set to empty/None to disable)
- Chunking controls (long articles):
  - `ENABLE_CHUNKING` — Enable token-based chunking (default true)
  - `MAX_TOKENS_PER_CHUNK` — Override tokenizer max length (optional)
  - `CHUNK_OVERLAP` — Token overlap between chunks (default 64)
  - `LONG_DOC_AGG` — Aggregation method for chunk scores: mean or max (default mean)
  - `MAX_CHUNKS_PER_DOC` — Optional cap on chunks per document (optional)

## Notes
- Baseline models are CPU-friendly but still sizable (Transformers + Torch). First run will download model weights.
- Use Windows Task Scheduler to periodically run `scripts/ingest.ps1` and `scripts/predict.ps1` for continuous updates.

