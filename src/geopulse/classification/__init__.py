"""Geopulse classification package."""

from geopulse.classification.base import BaseClassifier
from geopulse.classification.models import (
    ClassificationResult,
    RelevanceStatus,
    TopicCategory,
)
from geopulse.classification.rules import RuleBasedClassifier

_DEFAULT_CLASSIFIER = RuleBasedClassifier()


def classify_text(
    title: str,
    snippet: str | None = None,
    source_name: str | None = None,
    url: str | None = None,
) -> ClassificationResult:
    """Convenience helper to classify text using the default rule-based classifier."""
    return _DEFAULT_CLASSIFIER.classify_text(
        title=title,
        snippet=snippet,
        source_name=source_name,
        url=url,
    )


__all__ = [
    "BaseClassifier",
    "ClassificationResult",
    "RelevanceStatus",
    "RuleBasedClassifier",
    "TopicCategory",
    "classify_text",
]
