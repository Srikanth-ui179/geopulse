"""RSS Ingestion orchestration service."""

import concurrent.futures
import logging
import threading
import time

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
    repo_lock: threading.Lock | None = None,
) -> FeedIngestionResult:
    """Ingest articles from a single feed.

    Never raises on network or parsing failure; always captures error in result.
    """
    start_time = time.monotonic()
    try:
        raw_xml = fetch_feed_data(source.feed_url)
        articles: list[ArticleRecord] = parse_feed_content(raw_xml, source_name=source.name)

        if limit is not None and limit > 0:
            articles = articles[:limit]

        articles_new = len(articles)

        if not dry_run and repository is not None:
            if repo_lock is not None:
                with repo_lock:
                    source_id = repository.get_or_create_source(source)
                    _, articles_new = repository.persist_articles(source_id, articles)
            else:
                source_id = repository.get_or_create_source(source)
                _, articles_new = repository.persist_articles(source_id, articles)

        elapsed = round(time.monotonic() - start_time, 2)
        logger.info(
            "Feed %s fetched %d articles in %.2fs",
            source.name,
            len(articles),
            elapsed,
        )
        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=True,
            articles_fetched=len(articles),
            articles_new=articles_new,
            articles=articles,
            elapsed_seconds=elapsed,
        )
    except FeedFetchError as exc:
        elapsed = round(time.monotonic() - start_time, 2)
        logger.warning(
            "Feed fetch error for %s (%s) after %.2fs: %s",
            source.name,
            source.feed_url,
            elapsed,
            exc,
        )
        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=False,
            error_message=str(exc),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - start_time, 2)
        logger.exception(
            "Unexpected error ingesting feed %s after %.2fs: %s",
            source.name,
            elapsed,
            exc,
        )
        return FeedIngestionResult(
            feed_name=source.name,
            feed_url=source.feed_url,
            success=False,
            error_message=f"Ingestion failure: {exc}",
            elapsed_seconds=elapsed,
        )


def run_ingestion(
    feeds: list[FeedSource] | None = None,
    limit_per_feed: int | None = None,
    dry_run: bool = True,
    settings: Settings | None = None,
    repository: ArticleRepository | None = None,
) -> IngestionSummary:
    """Run ingestion over a collection of feeds with graceful error isolation and concurrency."""
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

    enabled_feeds = [f for f in target_feeds if f.enabled]
    if not enabled_feeds:
        return IngestionSummary()

    effective_limit = (
        limit_per_feed if limit_per_feed is not None else getattr(cfg, "feed_limit", 10)
    )

    repo_lock = threading.Lock() if (active_repo is not None and not effective_dry_run) else None

    max_workers = min(len(enabled_feeds), 6)
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    ingest_feed,
                    source=feed,
                    limit=effective_limit,
                    repository=active_repo,
                    dry_run=effective_dry_run,
                    repo_lock=repo_lock,
                )
                for feed in enabled_feeds
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            ingest_feed(
                source=enabled_feeds[0],
                limit=effective_limit,
                repository=active_repo,
                dry_run=effective_dry_run,
                repo_lock=repo_lock,
            )
        ]

    total_fetched = sum(r.articles_fetched for r in results)
    total_new = sum(r.articles_new for r in results)
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
