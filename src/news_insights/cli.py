from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from .config import get_settings, get_zero_shot_presets, resolve_zero_shot_labels
from .db import get_session, init_db
from .db.models import Source
from .ingestion.feeds import ingest_from_feeds
from .logging import setup_logging
from .ml.inference import predict_pending
from .data.dataset import export_labeled_dataset

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def info() -> None:
    """Show environment and config info."""
    s = get_settings()
    logger = setup_logging(s.log_level)
    logger.info("Data directory: %s", s.data_dir.resolve())
    logger.info("Database URL: %s", s.database_url)
    logger.info("Feeds: %s", ", ".join(s.feeds))


@app.command("init-db")
def init_db_cmd() -> None:
    """Initialize the database (create tables)."""
    s = get_settings()
    setup_logging(s.log_level)
    init_db()
    typer.echo("Database initialized.")

@app.command()

def ingest(
    limit: Optional[int] = typer.Option(None, help="Max items per feed"),
    time_budget_s: Optional[int] = typer.Option(
        None,
        "--time-budget-s",
        help="Max seconds to spend per source before moving on (overrides default)",
    ),
) -> None:
    s = get_settings()
    logger = setup_logging(s.log_level)
    init_db()
    with get_session() as session:
        added = ingest_from_feeds(
            session,
            limit_per_feed=limit,
            per_source_time_budget_s=time_budget_s,
        )
    logger.info("Ingested %d new article(s)", added)
@app.command()

def predict(
    limit: Optional[int] = typer.Option(None, help="Max articles to predict"),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Zero-shot label preset to use (e.g., "
        + ", ".join(sorted(get_zero_shot_presets().keys()))
        + ")",
    ),
    labels: Optional[str] = typer.Option(
        None,
        "--labels",
        help="Comma-separated custom zero-shot labels (overrides --preset)",
    ),
    # Chunking controls
    chunking: Optional[bool] = typer.Option(
        None,
        "--chunking/--no-chunking",
        help="Enable token-based chunking and aggregation for long articles",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Max tokens per chunk (default: tokenizer max)",
    ),
    overlap: Optional[int] = typer.Option(
        None,
        "--overlap",
        help="Token overlap between chunks (default from config)",
    ),
    agg: Optional[str] = typer.Option(
        None,
        "--agg",
        help="Aggregation for chunked scores: mean or max",
    ),
    max_chunks: Optional[int] = typer.Option(
        None,
        "--max-chunks",
        help="Optional cap on number of chunks per document",
    ),
    use_gpu: Optional[bool] = typer.Option(
        None,
        "--use-gpu/--no-use-gpu",
        help="Enable or disable GPU usage for inference",
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        help="Number of articles to process before saving predictions to the database",
    ),
) -> None:
    s = get_settings()
    logger = setup_logging(s.log_level)
    init_db()

    # Resolve candidate labels
    custom_labels = None
    if labels:
        custom_labels = [x.strip() for x in labels.split(",") if x.strip()]
    if preset and preset not in get_zero_shot_presets():
        raise typer.BadParameter(
            f"Unknown preset '{preset}'. Available: {', '.join(sorted(get_zero_shot_presets().keys()))}"
        )
    if agg and agg.lower() not in {"mean", "max"}:
        raise typer.BadParameter("--agg must be 'mean' or 'max'")

    with get_session() as session:
        n = predict_pending(
            session,
            limit=limit,
            candidate_labels=custom_labels,
            preset=preset,
            chunking=chunking,
            max_tokens_per_chunk=max_tokens,
            chunk_overlap=overlap,
            agg=(agg.lower() if agg else None),
            max_chunks=max_chunks,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    logger.info("Stored predictions for %d article(s)", n)


@app.command("export-dataset")
def export_dataset(out: Path = typer.Option(Path("data/processed/labeled.parquet"), help="Output Parquet path")) -> None:
    s = get_settings()
    logger = setup_logging(s.log_level)
    init_db()
    with get_session() as session:
        n = export_labeled_dataset(session, out)
    if n == 0:
        logger.warning("No labeled samples found. Add labels before exporting.")
    else:
        logger.info("Exported %d labeled records to %s", n, out)


@app.command()
def train() -> None:
    """Placeholder for future fine-tuning pipeline."""
    s = get_settings()
    logger = setup_logging(s.log_level)
    logger.warning("Training pipeline not yet implemented. Collect labels first, then fine-tune.")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

