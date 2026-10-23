"""Geopulse local Ollama event summarization package."""

from geopulse.clustering.models import EventCluster
from geopulse.summarization.client import (
    BaseOllamaClient,
    MockOllamaClient,
    OllamaClient,
)
from geopulse.summarization.exceptions import (
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    SummaryValidationError,
)
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryStatus,
    reconstruct_event_summary,
)
from geopulse.summarization.service import EventSummarizer


def summarize_event(
    event: EventCluster,
    client: BaseOllamaClient | None = None,
) -> SummarizationResult:
    """Convenience helper to summarize a single EventCluster."""
    summarizer = EventSummarizer(client=client)
    return summarizer.summarize_event(event)


def summarize_events(
    events: list[EventCluster] | tuple[EventCluster, ...],
    client: BaseOllamaClient | None = None,
) -> tuple[SummarizationResult, ...]:
    """Convenience helper to summarize multiple EventClusters."""
    summarizer = EventSummarizer(client=client)
    return summarizer.summarize_events(events)


__all__ = [
    "BaseOllamaClient",
    "EventKeyDetails",
    "EventSourceAttribution",
    "EventSummarizer",
    "EventSummary",
    "MockOllamaClient",
    "OllamaClient",
    "OllamaError",
    "OllamaModelNotFoundError",
    "OllamaResponseError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
    "SummarizationResult",
    "SummaryStatus",
    "SummaryValidationError",
    "reconstruct_event_summary",
    "summarize_event",
    "summarize_events",
]
