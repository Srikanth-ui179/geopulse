"""Geopulse event summarization package supporting local Ollama and Google Gemini."""

from typing import TYPE_CHECKING

from geopulse.clustering.models import EventCluster
from geopulse.summarization.client import (
    STAGE_BUDGET_CLAIMS,
    STAGE_BUDGET_CORRECTION,
    STAGE_BUDGET_QA,
    STAGE_BUDGET_VERIFIER,
    BaseLLMClient,
    BaseOllamaClient,
    GeminiClient,
    MockGeminiClient,
    MockOllamaClient,
    OllamaClient,
    resolve_summarization_clients,
)
from geopulse.summarization.evidence_matcher import (
    check_evidence_span_match,
    normalize_evidence_text,
)
from geopulse.summarization.exceptions import (
    GEMINI_FALLBACK_ELIGIBLE_EXCEPTIONS,
    GeminiAuthError,
    GeminiConfigError,
    GeminiError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    LLMError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    SummaryValidationError,
)
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
    SummaryStatus,
    reconstruct_event_summary,
)
from geopulse.summarization.publication_gate import (
    PublicationGateResult,
    validate_publication_gate,
)
from geopulse.summarization.service import EventSummarizer
from geopulse.summarization.validator import (
    GroundingValidationResult,
    are_sources_genuinely_independent,
    count_genuinely_independent_sources,
    is_evidence_sufficient,
    validate_summary_grounding,
)

if TYPE_CHECKING:
    from geopulse.config import Settings


def summarize_event(
    event: EventCluster,
    client: BaseLLMClient | None = None,
    fallback_client: BaseLLMClient | None = None,
    settings: "Settings | None" = None,
) -> SummarizationResult:
    """Convenience helper to summarize a single EventCluster."""
    summarizer = EventSummarizer(
        client=client,
        fallback_client=fallback_client,
        settings=settings,
    )
    return summarizer.summarize_event(event)


def summarize_events(
    events: list[EventCluster] | tuple[EventCluster, ...],
    client: BaseLLMClient | None = None,
    fallback_client: BaseLLMClient | None = None,
    settings: "Settings | None" = None,
    max_events: int | None = None,
    min_relevance_rank: float | None = None,
    include_low_priority: bool = False,
) -> tuple[SummarizationResult, ...]:
    """Convenience helper to summarize multiple EventClusters."""
    summarizer = EventSummarizer(
        client=client,
        fallback_client=fallback_client,
        settings=settings,
    )
    return summarizer.summarize_events(
        events,
        max_events=max_events,
        min_relevance_rank=min_relevance_rank,
        include_low_priority=include_low_priority,
    )


__all__ = [
    "GEMINI_FALLBACK_ELIGIBLE_EXCEPTIONS",
    "BaseLLMClient",
    "BaseOllamaClient",
    "EventKeyDetails",
    "ClaimEvidenceReference",
    "EventSourceAttribution",
    "EventSummarizer",
    "EventSummary",
    "SummaryClaim",
    "GeminiAuthError",
    "GeminiClient",
    "GeminiConfigError",
    "GeminiError",
    "GeminiModelNotFoundError",
    "GeminiQuotaError",
    "GeminiResponseError",
    "GeminiTimeoutError",
    "GeminiUnavailableError",
    "GroundingValidationResult",
    "LLMError",
    "MockGeminiClient",
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
    "is_evidence_sufficient",
    "reconstruct_event_summary",
    "STAGE_BUDGET_CLAIMS",
    "STAGE_BUDGET_CORRECTION",
    "STAGE_BUDGET_QA",
    "STAGE_BUDGET_VERIFIER",
    "resolve_summarization_clients",
    "summarize_event",
    "summarize_events",
    "validate_summary_grounding",
    "are_sources_genuinely_independent",
    "count_genuinely_independent_sources",
    "PublicationGateResult",
    "validate_publication_gate",
    "check_evidence_span_match",
    "normalize_evidence_text",
]
