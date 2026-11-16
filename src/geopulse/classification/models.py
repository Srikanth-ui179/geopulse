"""Data models for geopolitical topic and relevance classification."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RelevanceStatus = Literal["pending", "relevant", "irrelevant", "review"]


class TopicCategory(StrEnum):
    """Structured topic categories for defence and geopolitical affairs."""

    ARMED_CONFLICT = "armed_conflict"
    MILITARY_OPERATIONS = "military_operations"
    DEFENCE_POLICY = "defence_policy"
    WEAPONS_PROCUREMENT = "weapons_procurement"
    MILITARY_TECHNOLOGY = "military_technology"
    INTELLIGENCE_ESPIONAGE = "intelligence_espionage"
    DIPLOMACY = "diplomacy"
    SANCTIONS = "sanctions"
    TERRORISM_INSURGENCY = "terrorism_insurgency"
    CYBER_HYBRID_WARFARE = "cyber_hybrid_warfare"
    SECURITY_INCIDENT = "security_incident"
    OTHER_GEOPOLITICAL = "other_geopolitical"
    UNCERTAIN = "uncertain"


class ClassificationResult(BaseModel):
    """Outcome of classifying an article's relevance and topic."""

    model_config = ConfigDict(frozen=True)

    relevance_status: RelevanceStatus
    primary_topic: TopicCategory
    secondary_topics: tuple[TopicCategory, ...] = Field(default_factory=tuple)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    explanation: str | None = Field(
        default=None,
        description="Brief explanation of the classification rationale",
    )
