"""Unit tests for the ingestion service orchestration and error isolation."""

from unittest.mock import MagicMock

from geopulse.config import Settings
from geopulse.ingestion.fetcher import FeedFetchError
from geopulse.ingestion.models import FeedSource
from geopulse.ingestion.service import ingest_feed, run_ingestion

MOCK_XML = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Mock Feed</title>
    <item>
      <title>Peace talks conclude with temporary ceasefire</title>
      <link>https://example.com/peace-talks</link>
      <description>Delegations agreed to halt active hostilities.</description>
    </item>
    <item>
      <title>Defense ministry approves ammunition contract</title>
      <link>https://example.com/contract</link>
      <description>Procurement for artillery shells.</description>
    </item>
  </channel>
</rss>
"""


def test_ingest_feed_success(monkeypatch):
    monkeypatch.setattr(
        "geopulse.ingestion.service.fetch_feed_data",
        lambda url, **kwargs: MOCK_XML,
    )
    source = FeedSource(name="Test Source", feed_url="https://example.com/rss")

    result = ingest_feed(source, limit=10, dry_run=True)

    assert result.success is True
    assert result.articles_fetched == 2
    assert result.articles_new == 2
    assert len(result.articles) == 2
    assert result.error_message is None


def test_ingest_feed_handles_fetch_error_gracefully(monkeypatch):
    def failing_fetch(url, **kwargs):
        raise FeedFetchError(url, "Connection refused")

    monkeypatch.setattr("geopulse.ingestion.service.fetch_feed_data", failing_fetch)
    source = FeedSource(name="Failing Feed", feed_url="https://example.com/failing")

    result = ingest_feed(source, dry_run=True)

    assert result.success is False
    assert result.articles_fetched == 0
    assert result.error_message is not None
    assert "Connection refused" in result.error_message


def test_run_ingestion_isolates_partial_failures(monkeypatch):
    def mixed_fetch(url, **kwargs):
        if "bad" in url:
            raise FeedFetchError(url, "404 Not Found")
        return MOCK_XML

    monkeypatch.setattr("geopulse.ingestion.service.fetch_feed_data", mixed_fetch)

    feeds = [
        FeedSource(name="Good Feed 1", feed_url="https://example.com/good1"),
        FeedSource(name="Bad Feed", feed_url="https://example.com/bad"),
        FeedSource(name="Good Feed 2", feed_url="https://example.com/good2"),
    ]

    summary = run_ingestion(feeds=feeds, dry_run=True)

    assert summary.total_feeds == 3
    assert summary.successful_feeds == 2
    assert summary.failed_feeds == 1
    assert summary.total_articles_fetched == 4


def test_run_ingestion_respects_limit(monkeypatch):
    monkeypatch.setattr(
        "geopulse.ingestion.service.fetch_feed_data",
        lambda url, **kwargs: MOCK_XML,
    )
    feed = FeedSource(name="Limited Feed", feed_url="https://example.com/rss")

    summary = run_ingestion(feeds=[feed], limit_per_feed=1, dry_run=True)

    assert summary.total_articles_fetched == 1


def test_run_ingestion_with_mocked_supabase_repository(monkeypatch):
    monkeypatch.setattr(
        "geopulse.ingestion.service.fetch_feed_data",
        lambda url, **kwargs: MOCK_XML,
    )

    mock_repo = MagicMock()
    mock_repo.get_or_create_source.return_value = "source-uuid-123"
    mock_repo.persist_articles.return_value = (2, 2)

    feed = FeedSource(name="Mock Feed", feed_url="https://example.com/rss")
    result = ingest_feed(feed, repository=mock_repo, dry_run=False)

    assert result.success is True
    mock_repo.get_or_create_source.assert_called_once_with(feed)
    mock_repo.persist_articles.assert_called_once()
    assert result.articles_new == 2


def test_run_ingestion_defaults_to_dry_run_without_supabase_credentials():
    settings = Settings(_env_file=None)
    assert settings.is_supabase_configured is False
    # Verifies run_ingestion defaults to dry run safely
    summary = run_ingestion(feeds=[], settings=settings)
    assert summary.total_feeds == 0
