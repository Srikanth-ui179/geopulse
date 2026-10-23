"""Feedparser-based parsing of RSS and Atom entries into normalized article records."""

import logging
from datetime import UTC, datetime
from typing import Any

import feedparser

from geopulse.ingestion.models import ArticleRecord
from geopulse.ingestion.normalization import clean_text, compute_content_hash, normalize_url
from geopulse.ingestion.relevance import classify_relevance
from geopulse.ingestion.timestamps import parse_entry_timestamp

logger = logging.getLogger(__name__)


def extract_entry_url(entry: dict[str, Any]) -> str | None:
    """Extract a valid article URL from link, links, or id fields."""
    # Check link field directly
    link = entry.get("link")
    if isinstance(link, str) and link.strip():
        return link.strip()

    # Check links list (common in Atom feeds)
    links = entry.get("links")
    if isinstance(links, list):
        for candidate in links:
            if isinstance(candidate, dict) and candidate.get("rel") in {"alternate", None}:
                href = candidate.get("href")
                if isinstance(href, str) and href.strip():
                    return href.strip()

    # Check id field as fallback if it looks like an HTTP URL
    entry_id = entry.get("id")
    if isinstance(entry_id, str) and entry_id.strip().startswith(("http://", "https://")):
        return entry_id.strip()

    return None


def extract_entry_author(entry: dict[str, Any]) -> str | None:
    """Extract author string from entry."""
    author = entry.get("author")
    if isinstance(author, str) and author.strip():
        return clean_text(author)

    author_detail = entry.get("author_detail")
    if isinstance(author_detail, dict):
        name = author_detail.get("name")
        if isinstance(name, str) and name.strip():
            return clean_text(name)

    authors = entry.get("authors")
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict) and first.get("name"):
            return clean_text(str(first["name"]))

    return None


def parse_feed_content(
    raw_content: str,
    source_name: str,
    fetched_at: datetime | None = None,
) -> list[ArticleRecord]:
    """Parse RSS or Atom XML string into a deduplicated list of normalized ArticleRecords.

    Silently ignores malformed or unidentifiable entries and deduplicates within the batch.
    """
    timestamp = fetched_at or datetime.now(UTC)
    parsed = feedparser.parse(raw_content)

    articles: list[ArticleRecord] = []
    seen_canonical_urls: set[str] = set()

    feed_lang = clean_text(parsed.feed.get("language")) if hasattr(parsed, "feed") else None

    for entry in parsed.entries:
        raw_title = entry.get("title")
        title = clean_text(raw_title)
        if not title:
            # Skip entries without a valid title
            continue

        raw_url = extract_entry_url(entry)
        if not raw_url:
            # Skip entries without a link
            continue

        try:
            canonical_url = normalize_url(raw_url)
        except ValueError:
            # Skip malformed URLs
            continue

        # In-batch deduplication: avoid processing the exact same canonical URL twice
        if canonical_url in seen_canonical_urls:
            continue
        seen_canonical_urls.add(canonical_url)

        snippet = clean_text(entry.get("summary") or entry.get("description"))
        published_at = parse_entry_timestamp(entry)
        author = extract_entry_author(entry)
        content_hash = compute_content_hash(canonical_url, title, snippet)
        relevance = classify_relevance(title=title, snippet=snippet, source_name=source_name)

        articles.append(
            ArticleRecord(
                source_name=source_name,
                title=title,
                url=raw_url,
                canonical_url=canonical_url,
                published_at=published_at,
                fetched_at=timestamp,
                snippet=snippet,
                author=author,
                content_hash=content_hash,
                language=feed_lang or "en",
                relevance_status=relevance,
            )
        )

    return articles
