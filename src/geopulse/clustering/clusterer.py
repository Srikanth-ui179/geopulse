"""Deterministic event clustering engine for geopolitical news."""

import re
import uuid
from datetime import UTC, datetime, timedelta

from geopulse.classification.base import BaseClassifier
from geopulse.classification.models import TopicCategory
from geopulse.classification.rules import RuleBasedClassifier
from geopulse.clustering.models import ClusteringSummary, ConfidenceLabel, EventCluster
from geopulse.clustering.similarity import (
    compute_article_similarity,
    extract_locations,
    has_dispute_indicators,
    has_single_origin_dependency,
)
from geopulse.ingestion.models import ArticleRecord

ANALYSIS_SOURCES: frozenset[str] = frozenset({"war on the rocks"})
ANALYSIS_TITLE_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(analysis|analyst\s+argues|analysts?\b|commentary|op-ed|opinion|interview|"
    r"exclusive interview|strategic review|perspective|cogs of war|book review|essay|"
    r"roundtable|discussion|podcast|debrief|newsletter|reading list|"
    r"lessons?\s+(?:from|for)|should inform|"
    r"(?:is|are)\s+(?:destabilizing|dangerous|flawed|a\s+mistake|misguided|unnecessary|unworkable|doomed)|"
    r"why\s+(?:the\s+)?.*?\s+(?:must|should|is|won['’]t|cannot|fails?)|"
    r"the\s+case\s+(?:for|against)|"
    r"time\s+to\s+(?:rethink|reconsider|abandon|escalate))\b",
    re.I,
)
ANALYSIS_URL_PATTERNS: re.Pattern[str] = re.compile(
    r"/(?:opinion|commentary|analysis|voices|ideas)/",
    re.I,
)


class _ClusterBuilder:
    """Internal mutable container used during cluster construction."""

    def __init__(self, article: ArticleRecord, category: TopicCategory) -> None:
        article_time = article.published_at or article.fetched_at
        # Generate deterministic UUIDv5 based on first canonical URL
        self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"geopulse:event:{article.canonical_url}"))
        self.title = article.title
        self.category = category
        self.first_seen_at = article_time
        self.last_seen_at = article_time
        self.articles: list[ArticleRecord] = [article]

        combined = f"{article.title} {article.snippet or ''}"
        self.has_dispute = has_dispute_indicators(combined)
        self.locations = extract_locations(combined)

    def add_article(self, article: ArticleRecord) -> None:
        article_time = article.published_at or article.fetched_at
        self.articles.append(article)
        if article_time < self.first_seen_at:
            self.first_seen_at = article_time
        if article_time > self.last_seen_at:
            self.last_seen_at = article_time

        combined = f"{article.title} {article.snippet or ''}"
        if has_dispute_indicators(combined):
            self.has_dispute = True
        self.locations.update(extract_locations(combined))

    def to_event_cluster(self) -> EventCluster:
        distinct_sources = {a.source_name.strip().lower() for a in self.articles}
        distinct_article_urls = {a.canonical_url for a in self.articles if a.canonical_url}
        confidence: ConfidenceLabel

        if self.has_dispute:
            confidence = "disputed"
        elif len(distinct_sources) < 2 or len(distinct_article_urls) < 2:
            confidence = "single_source"
        else:
            is_derived, _ = has_single_origin_dependency(self.articles)
            if is_derived:
                confidence = "single_source"
            else:
                confidence = "corroborated"

        # Determine inferred region if available
        region = next(iter(sorted(self.locations)), None)

        # Collect verified published_at timestamps
        published_dates = [a.published_at for a in self.articles if a.published_at is not None]
        earliest_pub = min(published_dates) if published_dates else None

        is_analysis = any(
            a.source_name.strip().lower() in ANALYSIS_SOURCES
            or bool(ANALYSIS_TITLE_PATTERNS.search(f"{a.title} {a.snippet or ''}"))
            or bool(ANALYSIS_URL_PATTERNS.search(a.canonical_url or a.url or ""))
            for a in self.articles
        )

        return EventCluster(
            id=self.id,
            title=self.title,
            category=self.category,
            region=region,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            published_at=earliest_pub,
            is_analysis=is_analysis,
            confidence_label=confidence,
            articles=tuple(self.articles),
        )


class EventClusterer:
    """Deterministic, entity- and temporal-aware geopolitical event clustering."""

    def __init__(
        self,
        similarity_threshold: float = 0.50,
        max_time_delta: timedelta = timedelta(hours=48),
        classifier: BaseClassifier | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_time_delta = max_time_delta
        self.classifier = classifier or RuleBasedClassifier()

    def cluster_articles(
        self,
        articles: list[ArticleRecord] | tuple[ArticleRecord, ...],
    ) -> ClusteringSummary:
        """Cluster a collection of articles into discrete real-world events."""
        if not articles:
            return ClusteringSummary(
                total_articles=0,
                total_clusters=0,
                corroborated_clusters=0,
                single_source_clusters=0,
                disputed_clusters=0,
                uncertain_clusters=0,
                clusters=(),
                total_articles_evaluated=0,
                articles_filtered_out=0,
            )

        # 1. Deterministic sort: order by timestamp (published or fetched) then canonical_url
        sorted_articles = sorted(
            articles,
            key=lambda a: (
                a.published_at or a.fetched_at or datetime.min.replace(tzinfo=UTC),
                a.canonical_url,
            ),
        )

        # 2. Pre-classify articles for topic awareness
        classified: list[tuple[ArticleRecord, TopicCategory]] = []
        for art in sorted_articles:
            res = self.classifier.classify_article(art)
            if res.relevance_status == "relevant" or (
                res.relevance_status != "irrelevant"
                and res.primary_topic != TopicCategory.UNCERTAIN
            ):
                classified.append((art, res.primary_topic))

        # 3. Incremental clustering with temporal & entity gating
        builders: list[_ClusterBuilder] = []

        for article, topic in classified:
            article_time = article.published_at or article.fetched_at
            best_match_idx: int | None = None
            highest_score: float = 0.0

            for idx, builder in enumerate(builders):
                # Check temporal window against cluster boundary
                if abs(article_time - builder.last_seen_at) > self.max_time_delta:
                    continue

                # Compute similarity across articles in cluster builder
                sim = max(
                    compute_article_similarity(
                        article_a=article,
                        article_b=existing_article,
                        max_time_delta=self.max_time_delta,
                        topic_a=topic,
                        topic_b=builder.category,
                    )
                    for existing_article in builder.articles
                )

                if sim >= self.similarity_threshold and sim > highest_score:
                    highest_score = sim
                    best_match_idx = idx

            if best_match_idx is not None:
                builders[best_match_idx].add_article(article)
            else:
                builders.append(_ClusterBuilder(article=article, category=topic))

        # 4. Finalize immutable event clusters (breaking news first, analysis de-prioritized)
        raw_clusters = [b.to_event_cluster() for b in builders]
        final_clusters = tuple(sorted(raw_clusters, key=lambda c: 1 if c.is_analysis else 0))

        corroborated = sum(1 for c in final_clusters if c.confidence_label == "corroborated")
        single_source = sum(1 for c in final_clusters if c.confidence_label == "single_source")
        disputed = sum(1 for c in final_clusters if c.confidence_label == "disputed")
        uncertain = sum(1 for c in final_clusters if c.confidence_label == "uncertain")

        total_evaluated = len(articles)
        classified_count = len(classified)
        filtered_out = total_evaluated - classified_count

        return ClusteringSummary(
            total_articles=classified_count,
            total_clusters=len(final_clusters),
            corroborated_clusters=corroborated,
            single_source_clusters=single_source,
            disputed_clusters=disputed,
            uncertain_clusters=uncertain,
            clusters=final_clusters,
            total_articles_evaluated=total_evaluated,
            articles_filtered_out=filtered_out,
        )
