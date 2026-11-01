"""Data models for Ollama-generated geopolitical event summaries."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from geopulse.clustering.models import EventCluster

SummaryStatus = Literal[
    "success",
    "failed",
    "unavailable",
    "quarantined",
    "insufficient_evidence",
]


class EventSourceAttribution(BaseModel):
    """Source article attribution preserved for evidence grounding."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    title: str
    url: str
    published_at: datetime | None = None


class EventKeyDetails(BaseModel):
    """Factual components extracted from source articles."""

    model_config = ConfigDict(frozen=True)

    actors: tuple[str, ...] = Field(default_factory=tuple)
    location: str | None = None
    timing: str | None = None
    action: str | None = None


ClaimClassification = Literal[
    "reported_fact", "attributed_statement", "source_analysis", "inference", "disputed_claim"
]
ClaimVerificationStatus = Literal["pending", "accepted", "rejected", "uncertain"]


class ClaimEvidenceReference(BaseModel):
    """An exact text span from one attached article supporting a claim."""

    model_config = ConfigDict(frozen=True)

    source_url: str
    evidence_span: str
    verification_status: ClaimVerificationStatus = "pending"
    verification_reason: str | None = None


class SummaryClaim(BaseModel):
    """A material statement tied to an exact, event-specific source-text span."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    text: str
    evidence_refs: tuple[ClaimEvidenceReference, ...] = Field(min_length=1)
    classification: ClaimClassification
    attribution: str | None = None
    central: bool = False
    verification_status: ClaimVerificationStatus = "pending"
    verification_reason: str | None = None


class EventSummary(BaseModel):
    """Structured, source-grounded intelligence summary for an event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    headline: str = Field(..., min_length=5, description="Concise, factual headline")
    what_happened: str = Field(..., min_length=10, description="Short summary of the event")
    key_details: EventKeyDetails = Field(default_factory=EventKeyDetails)
    significance: str | None = Field(
        default=None,
        description="Strategic significance only when supported by source articles",
    )
    confidence_qualification: str = Field(
        ...,
        description="Clear flag for single-source, disputed, or corroborated reporting",
    )
    sources: tuple[EventSourceAttribution, ...] = Field(
        default_factory=tuple,
        description="Preserved source article references",
    )
    claims: tuple[SummaryClaim, ...] = Field(
        default_factory=tuple,
        description="Verified claim-level evidence used to render the narrative",
    )
    model_used: str = Field(..., description="LLM model identifier used for generation")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SummarizationResult(BaseModel):
    """Outcome of summarizing an EventCluster."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    status: SummaryStatus
    summary: EventSummary | None = None
    error_message: str | None = None
    quarantine_reason: str | None = None
    event: EventCluster


def reconstruct_event_summary(row: dict[str, Any]) -> EventSummary | None:
    """Reconstruct an EventSummary domain object from a database row dictionary.

    Supports row formats from both public.events and public.briefing_events.
    Returns None if essential event identification or summary text is missing.
    """
    ev_id = str(row.get("event_id") or row.get("id") or "").strip()
    raw_summary = row.get("summary") or row.get("what_happened")
    if not ev_id or not raw_summary:
        return None

    what_happened_str = str(raw_summary).strip()
    if len(what_happened_str) < 10:
        what_happened_str = f"Summary: {what_happened_str}".ljust(10)

    headline = str(row.get("headline") or row.get("title") or "Event Summary").strip()
    if len(headline) < 5:
        headline = f"Event: {headline}" if headline else "Event Summary"

    kd_raw = row.get("key_details")
    evidence_metadata = kd_raw.get("_evidence") if isinstance(kd_raw, dict) else None
    if isinstance(kd_raw, dict):
        actors = tuple(str(a) for a in kd_raw.get("actors", ()))
        key_details = EventKeyDetails(
            actors=actors,
            location=kd_raw.get("location"),
            timing=kd_raw.get("timing"),
            action=kd_raw.get("action"),
        )
    elif isinstance(kd_raw, EventKeyDetails):
        key_details = kd_raw
    else:
        key_details = EventKeyDetails()

    claims: tuple[SummaryClaim, ...] = ()
    sources: tuple[EventSourceAttribution, ...] = ()
    if isinstance(evidence_metadata, dict):
        raw_claims = evidence_metadata.get("claims", ())
        raw_sources = evidence_metadata.get("sources", ())
        if isinstance(raw_claims, (list, tuple)):
            claims = tuple(SummaryClaim.model_validate(item) for item in raw_claims)
        if isinstance(raw_sources, (list, tuple)):
            sources = tuple(EventSourceAttribution.model_validate(item) for item in raw_sources)

    significance = row.get("significance") or row.get("why_it_matters")
    if claims:
        source_by_url = {source.url: source for source in sources}
        central_claims = [claim for claim in claims if claim.central]
        claim_support = []
        for claim in central_claims:
            accepted_refs = [
                ref for ref in claim.evidence_refs if ref.verification_status == "accepted"
            ]
            publishers = {
                source_by_url[ref.source_url].source_name.strip().casefold()
                for ref in accepted_refs
                if ref.source_url in source_by_url
            }
            urls = {ref.source_url for ref in accepted_refs if ref.source_url in source_by_url}
            claim_support.append(len(publishers) >= 2 and len(urls) >= 2)
        confidence_qual = (
            "Corroborated across distinct sources for each central claim."
            if claim_support and all(claim_support)
            else "Single-source reporting; not independently verified."
        )
    else:
        confidence_qual = "Source evidence unavailable; confidence unverified."
    model_used = str(row.get("model_used") or "persisted")

    gen_at: datetime | None = None
    for field in ("updated_at", "created_at", "first_seen_at"):
        val = row.get(field)
        if val:
            if isinstance(val, datetime):
                gen_at = val if val.tzinfo else val.replace(tzinfo=UTC)
                break
            try:
                dt = datetime.fromisoformat(str(val))
                gen_at = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                break
            except (ValueError, TypeError):
                pass
    if gen_at is None:
        gen_at = datetime.now(UTC)

    return EventSummary(
        event_id=ev_id,
        headline=headline,
        what_happened=what_happened_str,
        key_details=key_details,
        significance=str(significance) if significance else None,
        confidence_qualification=confidence_qual,
        sources=sources,
        claims=claims,
        model_used=model_used,
        generated_at=gen_at,
    )
