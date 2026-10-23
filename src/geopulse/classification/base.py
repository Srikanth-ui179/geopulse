"""Abstract interface for article classifiers."""

from abc import ABC, abstractmethod

from geopulse.classification.models import ClassificationResult
from geopulse.ingestion.models import ArticleRecord


class BaseClassifier(ABC):
    """Abstract base class for article classifiers."""

    @abstractmethod
    def classify_text(
        self,
        title: str,
        snippet: str | None = None,
        source_name: str | None = None,
    ) -> ClassificationResult:
        """Classify article text into relevance status and structured topic category."""
        ...

    def classify_article(self, article: ArticleRecord) -> ClassificationResult:
        """Classify an ArticleRecord instance."""
        return self.classify_text(
            title=article.title,
            snippet=article.snippet,
            source_name=article.source_name,
        )
