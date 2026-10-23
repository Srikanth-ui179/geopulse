"""End-to-end intelligence pipeline orchestrator for Geopulse.

Coordinates the complete, audit-grade OSINT workflow across:
1. Ingestion & normalization of RSS articles
2. Article persistence and unique deduplication
3. Temporal & entity event clustering
4. Event and event-article relationship persistence
5. Source-grounded summarization via local Ollama
6. Summary and significance persistence to event records
7. Telegram HTML digest chunk formatting
8. Briefing delivery with resumption and audit logging
9. Briefing events and source evidence relational linking
"""

import logging
from dataclasses import dataclass
from datetime import date

from geopulse.clustering import cluster_articles
from geopulse.clustering.models import EventCluster
from geopulse.clustering.repository import EventRepository, SupabaseEventRepository
from geopulse.config import Settings, get_settings
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.delivery.models import DeliveryResult
from geopulse.delivery.repository import (
    DeliveryRepository,
    JsonFileDeliveryRepository,
    SupabaseDeliveryRepository,
)
from geopulse.delivery.service import TelegramDeliveryService
from geopulse.ingestion.models import ArticleRecord, FeedSource
from geopulse.ingestion.repository import ArticleRepository, SupabaseArticleRepository
from geopulse.ingestion.service import run_ingestion
from geopulse.preflight import run_supabase_preflight
from geopulse.summarization.models import SummarizationResult
from geopulse.summarization.service import EventSummarizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outcome of an end-to-end briefing pipeline execution."""

    briefing_date: date
    is_preview: bool
    is_persistent: bool
    articles_count: int
    new_articles_count: int
    events_count: int
    events: tuple[EventCluster, ...]
    summaries: tuple[SummarizationResult, ...]
    digest_chunks: tuple[str, ...]
    briefing_id: str | None = None
    delivery_result: DeliveryResult | None = None
    persisted_briefing_events: int = 0
    persisted_evidence_records: int = 0
    error_message: str | None = None


class BriefingPipeline:
    """Orchestrates the Geopulse intelligence lifecycle with dependency injection."""

    def __init__(
        self,
        settings: Settings | None = None,
        article_repo: ArticleRepository | None = None,
        event_repo: EventRepository | None = None,
        delivery_repo: DeliveryRepository | None = None,
        summarizer: EventSummarizer | None = None,
        formatter: TelegramDigestFormatter | None = None,
        delivery_service: TelegramDeliveryService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.article_repo = article_repo
        self.event_repo = event_repo
        self.delivery_repo = delivery_repo
        self.summarizer = summarizer or EventSummarizer(settings=self.settings)
        self.formatter = formatter or TelegramDigestFormatter()
        self.delivery_service = delivery_service

    def _resolve_repositories(self, persistent: bool) -> tuple[
        ArticleRepository | None,
        EventRepository | None,
        DeliveryRepository,
    ]:
        """Resolve repository implementations based on persistence mode."""
        if persistent:
            # If all persistent repositories are injected (e.g. in-memory fakes for unit tests),
            # use them directly without requiring Supabase credentials.
            if (
                self.article_repo is not None
                and self.event_repo is not None
                and self.delivery_repo is not None
            ):
                return self.article_repo, self.event_repo, self.delivery_repo

            client = getattr(self.article_repo, "client", None)
            if not self.settings.is_supabase_configured and client is None:
                raise ValueError(
                    "Persistent pipeline requires SUPABASE_URL and SUPABASE_KEY to be configured."
                )
            art_repo = self.article_repo or SupabaseArticleRepository(self.settings, client=client)
            ev_repo = self.event_repo or SupabaseEventRepository(self.settings, client=client)
            deliv_repo = self.delivery_repo or SupabaseDeliveryRepository(
                self.settings, client=client
            )
            return art_repo, ev_repo, deliv_repo

        # In-memory / local-only preview mode
        deliv_repo = self.delivery_repo or JsonFileDeliveryRepository(
            self.settings.delivery_state_file
        )
        return self.article_repo, self.event_repo, deliv_repo

    def run(
        self,
        briefing_date: date | None = None,
        feeds: list[FeedSource] | None = None,
        limit_per_feed: int = 3,
        articles: list[ArticleRecord] | None = None,
        persistent: bool = False,
        send: bool = False,
        force: bool = False,
        source_id: str = "default-source",
    ) -> PipelineResult:
        """Execute the end-to-end pipeline in strict, verifiable sequence."""
        target_date = briefing_date or date.today()

        # Resolve repositories
        try:
            art_repo, ev_repo, deliv_repo = self._resolve_repositories(persistent)
        except Exception as exc:
            logger.error("Failed to resolve pipeline repositories: %s", exc)
            return PipelineResult(
                briefing_date=target_date,
                is_preview=not send,
                is_persistent=persistent,
                articles_count=0,
                new_articles_count=0,
                events_count=0,
                events=(),
                summaries=(),
                digest_chunks=(),
                error_message=str(exc),
            )

        # Preflight schema & connectivity check for Supabase persistence
        if persistent and isinstance(art_repo, SupabaseArticleRepository):
            preflight = run_supabase_preflight(
                client=art_repo.client,
                settings=self.settings,
                scope="full",
            )
            if not preflight.success:
                logger.error("Supabase preflight check failed: %s", preflight.error_message)
                return PipelineResult(
                    briefing_date=target_date,
                    is_preview=not send,
                    is_persistent=persistent,
                    articles_count=0,
                    new_articles_count=0,
                    events_count=0,
                    events=(),
                    summaries=(),
                    digest_chunks=(),
                    error_message=f"Preflight schema check failed: {preflight.error_message}",
                )

        # Stage 1 & 2: Ingestion and Article Persistence
        new_articles_count = 0
        if articles is not None:
            raw_articles = list(articles)
            if persistent and art_repo is not None:
                try:
                    _, new_articles_count = art_repo.persist_articles(source_id, raw_articles)
                except Exception as exc:
                    logger.error("Failed to persist articles: %s", exc)
                    return PipelineResult(
                        briefing_date=target_date,
                        is_preview=not send,
                        is_persistent=persistent,
                        articles_count=len(raw_articles),
                        new_articles_count=0,
                        events_count=0,
                        events=(),
                        summaries=(),
                        digest_chunks=(),
                        error_message=f"Failed to persist articles: {exc}",
                    )
        else:
            try:
                summary = run_ingestion(
                    feeds=feeds,
                    limit_per_feed=limit_per_feed,
                    dry_run=not persistent,
                    settings=self.settings,
                    repository=art_repo,
                )
                raw_articles = [a for res in summary.feed_results for a in res.articles]
                new_articles_count = summary.total_articles_new
            except Exception as exc:
                logger.error("Failed to run ingestion: %s", exc)
                return PipelineResult(
                    briefing_date=target_date,
                    is_preview=not send,
                    is_persistent=persistent,
                    articles_count=0,
                    new_articles_count=0,
                    events_count=0,
                    events=(),
                    summaries=(),
                    digest_chunks=(),
                    error_message=f"Failed to run ingestion: {exc}",
                )

        articles_count = len(raw_articles)
        if not raw_articles:
            return PipelineResult(
                briefing_date=target_date,
                is_preview=not send,
                is_persistent=persistent,
                articles_count=0,
                new_articles_count=0,
                events_count=0,
                events=(),
                summaries=(),
                digest_chunks=(),
            )

        # Stage 3: Retrieve article UUID mapping for relational foreign keys
        url_to_id: dict[str, str] = {}
        if persistent and art_repo is not None:
            try:
                canonical_urls = [a.canonical_url for a in raw_articles]
                url_to_id = art_repo.get_article_ids_by_urls(canonical_urls)
            except Exception as exc:
                logger.error("Failed to retrieve article UUID mapping: %s", exc)
                return PipelineResult(
                    briefing_date=target_date,
                    is_preview=not send,
                    is_persistent=persistent,
                    articles_count=articles_count,
                    new_articles_count=new_articles_count,
                    events_count=0,
                    events=(),
                    summaries=(),
                    digest_chunks=(),
                    error_message=f"Failed to retrieve article UUID mapping: {exc}",
                )

        # Stage 4 & 5: Event Clustering and Persistence
        try:
            clustering_summary = cluster_articles(raw_articles)
            events = clustering_summary.clusters
        except Exception as exc:
            logger.error("Failed to cluster articles: %s", exc)
            return PipelineResult(
                briefing_date=target_date,
                is_preview=not send,
                is_persistent=persistent,
                articles_count=articles_count,
                new_articles_count=new_articles_count,
                events_count=0,
                events=(),
                summaries=(),
                digest_chunks=(),
                error_message=f"Failed to cluster articles: {exc}",
            )

        if persistent and ev_repo is not None and events:
            try:
                ev_repo.persist_events(events, article_url_to_id=url_to_id)
            except Exception as exc:
                logger.error("Failed to persist events: %s", exc)
                return PipelineResult(
                    briefing_date=target_date,
                    is_preview=not send,
                    is_persistent=persistent,
                    articles_count=articles_count,
                    new_articles_count=new_articles_count,
                    events_count=len(events),
                    events=events,
                    summaries=(),
                    digest_chunks=(),
                    error_message=f"Failed to persist events: {exc}",
                )

        # Stage 6 & 7: Summarization and Summary Persistence
        summarization_results: tuple[SummarizationResult, ...] = ()
        if events:
            try:
                summarization_results = self.summarizer.summarize_events(events)
            except Exception as exc:
                logger.error("Failed to summarize events: %s", exc)
                return PipelineResult(
                    briefing_date=target_date,
                    is_preview=not send,
                    is_persistent=persistent,
                    articles_count=articles_count,
                    new_articles_count=new_articles_count,
                    events_count=len(events),
                    events=events,
                    summaries=(),
                    digest_chunks=(),
                    error_message=f"Failed to summarize events: {exc}",
                )

            if persistent and ev_repo is not None:
                try:
                    ev_repo.update_event_summaries(summarization_results)
                except Exception as exc:
                    logger.error("Failed to persist event summaries: %s", exc)
                    return PipelineResult(
                        briefing_date=target_date,
                        is_preview=not send,
                        is_persistent=persistent,
                        articles_count=articles_count,
                        new_articles_count=new_articles_count,
                        events_count=len(events),
                        events=events,
                        summaries=summarization_results,
                        digest_chunks=(),
                        error_message=f"Failed to persist event summaries: {exc}",
                    )

        # Stage 8: Digest Formatting
        chunks = self.formatter.format_digest(
            results=summarization_results,
            briefing_date=target_date,
        )

        # Stage 9: Preview Mode Check (Free of Telegram and database delivery writes)
        if not send:
            return PipelineResult(
                briefing_date=target_date,
                is_preview=True,
                is_persistent=persistent,
                articles_count=articles_count,
                new_articles_count=new_articles_count,
                events_count=len(events),
                events=events,
                summaries=summarization_results,
                digest_chunks=chunks,
            )

        # Stage 10: Telegram Delivery & Provenance Persistence
        if self.delivery_service is None:
            return PipelineResult(
                briefing_date=target_date,
                is_preview=False,
                is_persistent=persistent,
                articles_count=articles_count,
                new_articles_count=new_articles_count,
                events_count=len(events),
                events=events,
                summaries=summarization_results,
                digest_chunks=chunks,
                error_message="DeliveryService must be configured to execute delivery (send=True).",
            )

        try:
            deliv_res = self.delivery_service.deliver_digest(
                digest_chunks=chunks,
                briefing_date=target_date,
                force=force,
            )
        except Exception as exc:
            logger.error("Unexpected error during delivery execution: %s", exc)
            return PipelineResult(
                briefing_date=target_date,
                is_preview=False,
                is_persistent=persistent,
                articles_count=articles_count,
                new_articles_count=new_articles_count,
                events_count=len(events),
                events=events,
                summaries=summarization_results,
                digest_chunks=chunks,
                error_message=f"Failed during delivery execution: {exc}",
            )

        persisted_events = 0
        persisted_evidence = 0

        # Persist briefing events and article evidence if delivery record exists
        if persistent and deliv_res.briefing_id and deliv_res.briefing_id != "dry-run":
            try:
                persisted_events, persisted_evidence = deliv_repo.record_briefing_events(
                    briefing_id=deliv_res.briefing_id,
                    results=list(summarization_results),
                    article_url_to_id=url_to_id,
                )
            except Exception as exc:
                logger.error("Failed to persist briefing events and evidence: %s", exc)
                post_err = f"Failed to persist briefing events: {exc}"

        final_error = deliv_res.error_message
        if "post_err" in locals() and post_err:
            final_error = f"{final_error}; {post_err}" if final_error else post_err

        return PipelineResult(
            briefing_date=target_date,
            is_preview=False,
            is_persistent=persistent,
            articles_count=articles_count,
            new_articles_count=new_articles_count,
            events_count=len(events),
            events=events,
            summaries=summarization_results,
            digest_chunks=chunks,
            briefing_id=deliv_res.briefing_id,
            delivery_result=deliv_res,
            persisted_briefing_events=persisted_events,
            persisted_evidence_records=persisted_evidence,
            error_message=final_error,
        )
