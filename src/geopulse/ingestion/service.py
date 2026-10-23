"""RSS Ingestion orchestration service."""

import logging

from geopulse.config import Settings, get_settings
from geopulse.ingestion.feeds import get_default_feeds
from geopulse.ingestion.fetcher import FeedFetchError, fetch_feed_data
from geopulse.ingestion.models import (
    ArticleRecord,
    FeedIngestionResult,
    FeedSource,
    IngestionSummary,
)
from geopulse.ingestion.parser import parse_feed_content
from geopulse.ingestion.repository import ArticleRepository, SupabaseArticleRepository

logger = logging.getLogger(__name__)


def ingest_feed(
    source: FeedSource,
    limit: int | None = None,
    repository: ArticleRepository | None = None,
    dry_run: bool = True,
) -> FeedIngestionResult:
    """Ingest articles from a single feed.

    Never raises on network or parsing failure; always captures error in result.
    """
    try:
        raw_xml = fetch_feed_data(source.feed_url)
        articles: list[ArticleRecord] = parse_feed_content(raw_xml, source_name=source.name)

        if limit is not None and limit > 0:
            articles = articles[:limit]

        articles_new = len(articles)

        if not dry_run and repository is not None:
            source_id = repository.get_or_create_source(source)
            _, articles_new = repository.persist_articles(source_id, articles)

        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=True,
            articles_fetched=len(articles),
            articles_new=articles_new,
            articles=articles,
        )
    except FeedFetchError as exc:
        logger.warning("Feed fetch error for %s (%s): %s", source.name, source.feed_url, exc)
        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=False,
            error_message=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error ingesting feed %s: %s", source.name, exc)
        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=False,
            error_message=f"Ingestion failure: {exc}",
        )


def run_ingestion(
    feeds: list[FeedSource] | None = None,
    limit_per_feed: int | None = None,
    dry_run: bool = True,
    settings: Settings | None = None,
    repository: ArticleRepository | None = None,
) -> IngestionSummary:
    """Run ingestion over a collection of feeds with graceful error isolation."""
    target_feeds = feeds if feeds is not None else get_default_feeds()
    cfg = settings or get_settings()

    active_repo: ArticleRepository | None = repository
    effective_dry_run = dry_run

    if active_repo is None and not dry_run:
        if cfg.is_supabase_configured:
            try:
                active_repo = SupabaseArticleRepository(cfg)
            except Exception as exc:
                logger.warning(
                    "Could not initialize Supabase repository; falling back to dry-run: %s",
                    exc,
                )
                effective_dry_run = True
        else:
            effective_dry_run = True

    results: list[FeedIngestionResult] = []
    total_fetched = 0
    total_new = 0

    for feed in target_feeds:
        if not feed.enabled:
            continue
        result = ingest_feed(
            source=feed,
            limit=limit_per_feed,
            repository=active_repo,
            dry_run=effective_dry_run,
        )
        results.append(result)
        total_fetched += result.articles_fetched
        total_new += result.articles_new

    successful_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)

    return IngestionSummary(
        total_feeds=len(results),
        successful_feeds=successful_count,
        failed_feeds=failed_count,
        total_articles_fetched=total_fetched,
        total_articles_new=total_new,
        feed_results=results,
    )
