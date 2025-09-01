from __future__ import annotations

import hashlib
import logging
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
from requests.exceptions import RequestException
import trafilatura

logger = logging.getLogger("news_insights")


_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}


def canonicalize_url(url: str) -> str:
    """Normalize URL for deduplication.

    - lowercases scheme/host
    - removes common tracking params and fragments
    - removes trailing slash
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in _TRACKING_PARAMS]
    query = urlencode(query_pairs)

    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    return normalized


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


DEFAULT_HEADERS = {
    "User-Agent": "news-insights/0.1 (+https://example.com) Python-requests",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def extract_content(url: str, timeout: Optional[int] = None) -> Optional[str]:
    t = timeout or 20
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=(t, t), allow_redirects=True)
        if not resp.ok:
            logger.warning("HTTP %s fetching %s", resp.status_code, url)
            return None
        html = resp.text
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if not text:
            logger.warning("Extraction returned empty text for %s", url)
        return text
    except RequestException as e:
        logger.warning("Request error for %s: %s", url, e)
        return None
    except Exception as e:
        logger.exception("Extraction error for %s: %s", url, e)
        return None

