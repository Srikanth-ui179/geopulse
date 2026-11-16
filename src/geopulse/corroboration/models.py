"""Domain models for source lineage and independent source corroboration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LineageClassification = Literal[
    "original_reporting",
    "syndicated",
    "wire_service_citation",
    "official_statement",
    "derivative",
    "unknown",
]

SourceOriginType = Literal[
    "official_statement",
    "eyewitness_account",
    "independent_reporting",
    "wire_syndication",
    "unknown",
]

EvidenceLabel = Literal[
    "Single-source",
    "Partially corroborated",
    "Independently corroborated",
]

CorroborationStatus = Literal[
    "independently_corroborated",
    "single_source",
    "multiple_reports_shared_origin",
    "conflicting_evidence",
    "insufficient_information",
]


class ArticleLineage(BaseModel):
    """Lineage assessment for a single news article report."""

    model_config = ConfigDict(frozen=True)

    article_url: str = Field(..., description="Canonical or fetched URL of the article")
    publisher: str = Field(..., description="Publishing outlet or platform name")
    original_source: str = Field(
        ...,
        description="Identified originating reporting outlet, wire, or official source",
    )
    lineage_classification: LineageClassification = Field(
        ...,
        description="Lineage: original, syndicated, wire citation, official, derivative, unknown",
    )
    origin_type: SourceOriginType = Field(
        default="unknown",
        description="Source origin: official statement, eyewitness, independent, or wire",
    )
    citation_source: str | None = Field(
        default=None,
        description="Explicit cited entity (wire, government, or other publisher) if applicable",
    )
    evidence_text: str | None = Field(
        default=None,
        description="Snippet or dateline excerpt indicating the source lineage",
    )
    canonical_origin_url: str | None = Field(
        default=None,
        description="Upstream canonical URL if syndicated or republished",
    )


class ClaimCorroboration(BaseModel):
    """Corroboration audit result for an individual factual claim."""

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., description="Verbatim text of the claim")
    status: CorroborationStatus = Field(..., description="Corroboration determination status")
    evidence_label: EvidenceLabel = Field(
        default="Single-source",
        description="Editorial label: Single-source, Partially, or Independently corroborated",
    )
    independent_evidence_chains: int = Field(
        ...,
        ge=0,
        description="Count of demonstrably independent evidence chains",
    )
    original_sources: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Distinct original reporting sources behind supporting evidence",
    )
    supporting_articles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Article URLs providing accepted supporting evidence",
    )
    derivative_articles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Article URLs identified as derivative, syndicated, or repeating shared wire",
    )
    contradictory_evidence: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence excerpts or claims contradicting this statement",
    )
    unresolved_relationships: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Sources or articles whose provenance or independence cannot be established",
    )
    explanation: str = Field(
        ...,
        description="Concise, factual explanation of how the corroboration was derived",
    )


class EventCorroboration(BaseModel):
    """Aggregate corroboration assessment for an entire geopolitical event cluster."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., description="UUID or identifier of the event cluster")
    status: CorroborationStatus = Field(..., description="Overall event corroboration status")
    evidence_label: EvidenceLabel = Field(
        default="Single-source",
        description="Editorial label: Single-source, Partially, or Independently corroborated",
    )
    independent_chains_count: int = Field(
        ...,
        ge=0,
        description="Demonstrably independent evidence chains supporting the event",
    )
    primary_origin: str | None = Field(
        default=None,
        description="Shared wire, ministry, or originator name if single origin",
    )
    claims_corroboration: tuple[ClaimCorroboration, ...] = Field(
        default_factory=tuple,
        description="Detailed claim-level corroboration records",
    )
    lineages: tuple[ArticleLineage, ...] = Field(
        default_factory=tuple,
        description="Article-by-article lineage records",
    )
    summary_explanation: str = Field(
        ...,
        description="Concise summary explanation suitable for editorial review or briefing",
    )
