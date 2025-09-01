from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://ir.thomsonreuters.com/rss/news-releases.xml",
]

# Predefined zero-shot label presets
ZERO_SHOT_PRESETS: dict[str, list[str]] = {
    # Recommended genre/intent preset
    "genre": [
        "news report",
        "analysis",
        "opinion",
        "editorial",
        "explainer",
        "investigative report",
        "interview",
        "fact-check",
        "live update",
        "press release",
        "sponsored content",
        "satire",
    ],
    # Alternative purpose preset
    "purpose": [
        "inform",
        "persuade",
        "criticize",
        "advocate",
        "promote",
        "warn",
        "mobilize",
        "entertain",
        "reassure",
        "call to action",
    ],
    # Verification/status preset
    "verification": [
        "unverified",
        "rumor",
        "speculative",
        "confirmation",
        "correction",
    ],
    # Provenance/business preset
    "provenance": [
        "corporate statement",
        "government statement",
        "NGO statement",
        "academic report",
        "earnings release",
        "market commentary",
    ],
    # Coverage/stage preset
    "coverage": [
        "breaking news",
        "developing story",
        "recap",
        "follow-up",
        "feature",
    ],
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEWS_INSIGHTS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path("data"))

    # Database
    database_url: str = "sqlite:///./data/news_insights.db"

    # Ingestion
    feeds: List[str] = Field(default_factory=lambda: DEFAULT_FEEDS.copy())
    request_timeout_s: int = 20
    # Max seconds to spend per source during ingestion before moving to the next (None to disable)
    per_source_time_budget_s: int | None = 120

    # ML
    batch_size: int = 8
    hf_model_sentiment: str = "distilbert-base-uncased-finetuned-sst-2-english"
    hf_model_emotion: str = "j-hartmann/emotion-english-distilroberta-base"
    hf_model_zeroshot: str = "facebook/bart-large-mnli"
    zero_shot_labels: List[str] = Field(
        default_factory=lambda: [
            "informational",
            "opinion",
            "analysis",
            "advertisement",
            "press release",
        ]
    )
    # Optional preset name to override zero-shot labels (e.g., "genre", "purpose")
    zero_shot_preset: str | None = None

    # Long-document handling (token-based chunking)
    enable_chunking: bool = True
    max_tokens_per_chunk: int | None = None  # None => use tokenizer max - safety margin
    chunk_overlap: int = 64
    long_doc_agg: str = "mean"  # "mean" or "max"
    max_chunks_per_doc: int | None = None  # optional cap per document

    # Logging
    log_level: str = "INFO"


def get_zero_shot_presets() -> dict[str, list[str]]:
    return ZERO_SHOT_PRESETS


def resolve_zero_shot_labels(preset: str | None = None, labels: list[str] | None = None) -> list[str]:
    """Resolve the zero-shot labels to use, based on an explicit list, a preset name,
    or the default labels from settings.
    Precedence: explicit labels > preset > settings.zero_shot_labels
    """
    if labels:
        return labels
    if preset and preset in ZERO_SHOT_PRESETS:
        return ZERO_SHOT_PRESETS[preset]
    # Fall back to configured labels
    return get_settings().zero_shot_labels


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    # Ensure data_dir exists at runtime
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s

