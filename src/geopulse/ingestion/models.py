"""Data models for RSS feed configuration, article ingestion, and immutable revisions."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RevisionClassification = Literal[
    "initial_publication",
    "content_update",
    "metadata_update",
    "explicit_correction",
    "explicit_retraction",
    "unavailability",
]


class FeedSource(BaseModel):
    """Configuration for an RSS or Atom news feed."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Human-readable name of the feed source")
    feed_url: str = Field(..., min_length=1, description="Full HTTP(S) URL of the RSS/Atom feed")
    enabled: bool = Field(default=True, description="Whether this feed is active for ingestion")


class ArticleRevisionRecord(BaseModel):
    """Immutable snapshot of an article's content and metadata at a specific point in time."""

    model_config = ConfigDict(frozen=True)

    revision_id: str
    article_id: str
    canonical_url: str
    content_hash: str
    title: str
    publisher: str
    published_at: datetime | None = None
    first_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_modified: str | None = None
    etag: str | None = None
    body_text: str | None = None
    extraction_status: str = "success"
    extraction_method: str = "trafilatura"
    revision_sequence: int = 1
    previous_revision_id: str | None = None
    revision_classification: RevisionClassification = "initial_publication"
    relevant_metadata: dict[str, Any] = Field(default_factory=dict)


class ArticleRecord(BaseModel):
    """Normalized article metadata parsed from an RSS/Atom entry."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    title: str
    url: str
    canonical_url: str
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    snippet: str | None = None
    author: str | None = None
    content_hash: str
    language: str | None = "en"
    relevance_status: Literal["pending", "relevant", "irrelevant", "review"] = "pending"
    topic: str | None = None
    body_text: str | None = None
    extraction_status: str | None = None
    extracted_at: datetime | None = None
    is_full_text: bool = False
    article_id: str | None = None
    current_revision_id: str | None = None
    revision_sequence: int = 1
    revisions: tuple[ArticleRevisionRecord, ...] = Field(default_factory=tuple)


class FeedIngestionResult(BaseModel):
    """Execution result for a single feed."""

    feed_name: str
    feed_url: str
    success: bool
    articles_fetched: int = 0
    articles_new: int = 0
    articles: list[ArticleRecord] = Field(default_factory=list)
    error_message: str | None = None
    elapsed_seconds: float = 0.0


class IngestionSummary(BaseModel):
    """Aggregate summary across all processed feeds."""

    total_feeds: int = 0
    successful_feeds: int = 0
    failed_feeds: int = 0
    total_articles_fetched: int = 0
    total_articles_new: int = 0
    feed_results: list[FeedIngestionResult] = Field(default_factory=list)
