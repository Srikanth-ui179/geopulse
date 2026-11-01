"""Data models for geopolitical event clustering."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from geopulse.classification.models import TopicCategory
from geopulse.ingestion.models import ArticleRecord

ConfidenceLabel = Literal["corroborated", "single_source", "disputed", "uncertain"]


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
    confidence_label: ConfidenceLabel = Field(
        default="uncertain",
        description="corroborated, single_source, disputed, or uncertain",
    )
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
