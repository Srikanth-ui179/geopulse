"""Feedparser-based parsing of RSS and Atom entries into normalized article records."""

import logging
import re
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


_FEED_BOILERPLATE = re.compile(
    r"\b(?:subscribe to (?:our )?newsletter|sign up for (?:our )?newsletter|"
    r"you may also like|related stories|read more:|all rights reserved|"
    r"cookie policy|terms of use|advertisement)\b.*$",
    re.I,
)
_FEED_STOPWORDS = frozenset(
    {"and", "the", "for", "with", "from", "into", "over", "under", "about", "after", "during"}
)


def _feed_terms(value: str) -> set[str]:
    """Return simple content words for conservative cross-field relevance checks."""
    return {
        word for word in re.findall(r"[a-z0-9]{3,}", value.lower()) if word not in _FEED_STOPWORDS
    }


def extract_entry_text(entry: dict[str, Any], title: str = "") -> str | None:
    """Extract the most informative text already supplied by the feed entry.

    Some feeds provide only a short RSS summary while placing the article body in
    Atom ``content`` or RSS ``content:encoded``. Never fetch article URLs here.
    """
    candidates: list[str] = []
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            value = item.get("value") if isinstance(item, dict) else None
            cleaned = _clean_feed_candidate(value) if isinstance(value, str) else None
            if cleaned:
                candidates.append(cleaned)

    for key in ("summary", "description"):
        value = entry.get(key)
        cleaned = _clean_feed_candidate(value) if isinstance(value, str) else None
        if cleaned:
            candidates.append(cleaned)

    # Remove exact duplicates so a repeated body cannot win merely by appearing in
    # multiple content fields. If a long candidate is unrelated to the headline but
    # a shorter excerpt is on-topic, discard the unrelated candidate.
    unique_candidates = list(dict.fromkeys(candidates))
    title_terms = _feed_terms(title)
    if title_terms and len(unique_candidates) > 1:
        candidate_terms = {value: _feed_terms(value) for value in unique_candidates}

        def is_related(left: set[str], right: set[str]) -> bool:
            overlap = len(left & right)
            denominator = min(len(left), len(right))
            return denominator > 0 and overlap >= 2 and overlap / denominator >= 0.2

        on_topic = []
        for value in unique_candidates:
            shares_title_terms = is_related(title_terms, candidate_terms[value])
            shares_excerpt_terms = any(
                is_related(candidate_terms[value], other_terms)
                for other, other_terms in candidate_terms.items()
                if other != value
            )
            if shares_title_terms or shares_excerpt_terms:
                on_topic.append(value)
        if on_topic:
            unique_candidates = on_topic

    # Feed bodies can contain navigation and subscription material. Bound prompt/storage
    # growth while retaining the longest substantive feed-supplied excerpt.
    return max(unique_candidates, key=len, default=None)[:12000] if unique_candidates else None


_SPONSORED_TEASER = re.compile(
    r"(?:^|\s+)\[sponsored(?:\s+content)?\][^.\n]*(?:\.|$)",
    re.I,
)

_NEWSLETTER_INTRO_BOILERPLATE = re.compile(
    r"^\s*(?:welcome to (?:the adversarial|our newsletter)[^.\n]*(?:\.|\n)|"
    r"read more below\.\s*(?:\*{3,}|---|___)?\s*)",
    re.I,
)


def _clean_feed_candidate(value: str) -> str | None:
    """Clean HTML and remove trailing feed chrome, sponsored teasers, and intro boilerplate."""
    cleaned = clean_text(value)
    if not cleaned:
        return None
    cleaned = _FEED_BOILERPLATE.sub("", cleaned).strip()
    cleaned = _SPONSORED_TEASER.sub(" ", cleaned).strip()
    cleaned = _NEWSLETTER_INTRO_BOILERPLATE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


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

        snippet = extract_entry_text(entry, title=title)
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
