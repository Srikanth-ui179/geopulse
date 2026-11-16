"""Data models for geopolitical event clustering."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from geopulse.classification.models import TopicCategory
from geopulse.ingestion.models import ArticleRecord

ConfidenceLabel = Literal[
    "corroborated",
    "single_source",
    "disputed",
    "uncertain",
    "independently_corroborated",
    "multiple_reports_shared_origin",
    "conflicting_evidence",
    "insufficient_information",
]


class EventCluster(BaseModel):
    """A cluster of articles reporting on the same real-world geopolitical event."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Representative title for the event")
    category: TopicCategory = Field(..., description="Primary topic category")
    region: str | None = Field(default=None, description="Inferred geographic region")
    first_seen_at: datetime = Field(..., description="Earliest publication/retrieval timestamp")
    last_seen_at: datetime = Field(..., description="Latest publication/retrieval timestamp")
    published_at: datetime | None = Field(
        default=None,
        description="Earliest verified article publication timestamp (or None if unavailable)",
    )
    is_analysis: bool = Field(
        default=False,
        description="Whether this event represents strategic analysis rather than breaking news",
    )
    is_update: bool = Field(
        default=False,
        description="Whether this event represents an update to a previously reported event",
    )
    persistent_event_id: str | None = Field(
        default=None,
        description="Persistent canonical event ID across daily briefings",
    )
    confidence_label: ConfidenceLabel = Field(
        default="uncertain",
        description="corroborated, single_source, disputed, or uncertain",
    )
    corroboration_status: str = Field(
        default="unassessed",
        description="Phase 3 corroboration status",
    )
    independent_chains_count: int = Field(
        default=0,
        ge=0,
        description="Demonstrably independent evidence chains",
    )

    @field_validator("corroboration_status", mode="before")
    @classmethod
    def _normalize_corroboration_status(cls, v: Any) -> str:
        if v is None or str(v).strip().lower() in ("", "none", "null"):
            return "unassessed"
        return str(v).strip()

    @field_validator("independent_chains_count", mode="before")
    @classmethod
    def _normalize_independent_chains(cls, v: Any) -> int:
        if v is None:
            return 0
        return int(v)

    articles: tuple[ArticleRecord, ...] = Field(
        default_factory=tuple,
        description="Source articles reporting on this event",
    )

    @property
    def source_count(self) -> int:
        """Count distinct publishers linked to distinct articles in this event."""
        publishers = {
            a.source_name.strip().casefold() for a in self.articles if a.source_name.strip()
        }
        article_urls = {a.canonical_url for a in self.articles if a.canonical_url}
        return min(len(publishers), len(article_urls))

    @property
    def article_count(self) -> int:
        """Total number of article reports attached to this event."""
        return len(self.articles)


class ClusteringSummary(BaseModel):
    """Aggregate summary of event clustering output."""

    total_articles: int
    total_clusters: int
    corroborated_clusters: int
    single_source_clusters: int
    disputed_clusters: int
    uncertain_clusters: int
    clusters: tuple[EventCluster, ...] = Field(default_factory=tuple)
    total_articles_evaluated: int = 0
    articles_filtered_out: int = 0
