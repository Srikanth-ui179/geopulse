"""Geopulse event clustering package."""

from geopulse.clustering.clusterer import EventClusterer
from geopulse.clustering.models import (
    ClusteringSummary,
    ConfidenceLabel,
    EventCluster,
)
from geopulse.clustering.repository import (
    EventRepository,
    InMemoryEventRepository,
    SupabaseEventRepository,
    reconstruct_event_summary,
)
from geopulse.ingestion.models import ArticleRecord

_DEFAULT_CLUSTERER = EventClusterer()


def cluster_articles(
    articles: list[ArticleRecord] | tuple[ArticleRecord, ...],
) -> ClusteringSummary:
    """Convenience helper to cluster articles using default settings."""
    return _DEFAULT_CLUSTERER.cluster_articles(articles)


__all__ = [
    "ClusteringSummary",
    "ConfidenceLabel",
    "EventCluster",
    "EventClusterer",
    "EventRepository",
    "InMemoryEventRepository",
    "SupabaseEventRepository",
    "cluster_articles",
    "reconstruct_event_summary",
]
