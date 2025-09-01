from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import dateparser
import feedparser
import calendar
import time
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Article, Source
from .extract import canonicalize_url, extract_content, url_hash

logger = logging.getLogger("news_insights")


def ensure_sources(session: Session, feed_urls: Iterable[str]) -> List[Source]:
    existing = {s.url for s in session.scalars(select(Source))}
    created: list[Source] = []
    for url in feed_urls:
        if url in existing:
            continue
        src = Source(name=url, kind="rss", url=url, active=True)
        session.add(src)
        created.append(src)
    if created:
        session.flush()
    return list(session.scalars(select(Source).where(Source.active == True)))  # noqa: E712


def to_text(v: object | None) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        try:
            s = v.decode("utf-8", "replace")
        except Exception:
            s = v.decode(errors="replace")
    else:
        s = str(v)
    s = s.strip()
    return s or None


def parse_datetime(val: object | None) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        dt = val
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(val, time.struct_time):
        try:
            ts = calendar.timegm(val)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None
    # Fallback: parse as string
    s = to_text(val)
    if not s:
        return None
    dt = dateparser.parse(s)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ingest_from_feeds(
    session: Session,
    limit_per_feed: Optional[int] = None,
    per_source_time_budget_s: Optional[int] = None,
) -> int:
    settings = get_settings()
    sources = ensure_sources(session, settings.feeds)
    budget = per_source_time_budget_s if per_source_time_budget_s is not None else getattr(settings, "per_source_time_budget_s", None)

    # Track URL hashes we've already seen to avoid inserting duplicates within the same run
    # Initialize with hashes already present in the DB to short-circuit early
    try:
        seen_hashes = set(session.scalars(select(Article.url_hash)))
    except Exception:
        seen_hashes = set()

    added = 0
    for source in sources:
        start_ts = time.monotonic()
        parsed = feedparser.parse(source.url)
        entries = parsed.entries[: limit_per_feed or len(parsed.entries)]
        logger.info("Fetched %d entries from %s", len(entries), source.url)
        processed_for_source = 0
        for entry in entries:
            if budget is not None and (time.monotonic() - start_ts) >= budget:
                logger.warning(
                    "Time budget of %ss reached for %s; processed %d/%d entries",
                    budget,
                    source.url,
                    processed_for_source,
                    len(entries),
                )
                break
            # Link may live in different places; normalize to a clean string
            raw_url = entry.get("link")
            if not raw_url and isinstance(entry.get("links"), list) and entry.get("links"):
                links = entry.get("links")
                first = links[0] if links else None
                if isinstance(first, dict):
                    raw_url = first.get("href")
            if not raw_url:
                raw_url = entry.get("id") or entry.get("guid")

            url = to_text(raw_url)
            title = to_text(entry.get("title"))

            published = parse_datetime(
                entry.get("published")
                or entry.get("updated")
                or entry.get("pubDate")
                or entry.get("published_parsed")
                or entry.get("updated_parsed")
            )
            if not url:
                continue

            canon = canonicalize_url(url)
            h = url_hash(canon)

            # Skip if already seen in this run or already present in DB
            if h in seen_hashes or session.scalar(select(Article.id).where(Article.url_hash == h)):
                continue

            logger.debug("Extracting content from %s", url)
            content = extract_content(url, timeout=settings.request_timeout_s)
            if not content or len(content.strip()) < 200:
                # Skip very short or failed extractions
                continue

            art = Article(
                source_id=source.id,
                url=url,
                canonical_url=canon,
                url_hash=h,
                title=title,
                content=content,
                published_at=published,
            )

            # Insert with a savepoint so a uniqueness violation doesn't abort the whole batch
            try:
                with session.begin_nested():
                    session.add(art)
                    session.flush()  # trigger INSERT and constraints
                seen_hashes.add(h)
                added += 1
            except IntegrityError:
                # Another process (or earlier iteration) inserted it; skip gracefully
                try:
                    session.expunge(art)
                except Exception:
                    pass
                logger.debug("Duplicate article url_hash=%s for %s; skipping", h, canon)
                continue
            processed_for_source += 1
        # Flush batched changes for this source
        session.flush()
    return added
