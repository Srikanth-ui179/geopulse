"""Geopulse RSS ingestion package."""

from geopulse.ingestion.extractor import ArticleExtractor, ExtractedContent
from geopulse.ingestion.feeds import DEFAULT_FEEDS, find_feed, get_default_feeds
from geopulse.ingestion.fetcher import FeedFetchError, fetch_feed_data
from geopulse.ingestion.models import (
    ArticleRecord,
    FeedIngestionResult,
    FeedSource,
    IngestionSummary,
)
from geopulse.ingestion.normalization import clean_text, compute_content_hash, normalize_url
from geopulse.ingestion.parser import parse_feed_content
from geopulse.ingestion.relevance import classify_relevance
from geopulse.ingestion.repository import (
    ArticleRepository,
    InMemoryArticleRepository,
    SupabaseArticleRepository,
)
from geopulse.ingestion.service import ingest_feed, run_ingestion
from geopulse.ingestion.timestamps import parse_entry_timestamp

__all__ = [
    "DEFAULT_FEEDS",
    "ArticleExtractor",
    "ExtractedContent",
    "ArticleRecord",
    "ArticleRepository",
    "FeedFetchError",
    "FeedIngestionResult",
    "FeedSource",
    "InMemoryArticleRepository",
    "IngestionSummary",
    "SupabaseArticleRepository",
    "classify_relevance",
    "clean_text",
    "compute_content_hash",
    "fetch_feed_data",
    "find_feed",
    "get_default_feeds",
    "ingest_feed",
    "normalize_url",
    "parse_entry_timestamp",
    "parse_feed_content",
    "run_ingestion",
]
