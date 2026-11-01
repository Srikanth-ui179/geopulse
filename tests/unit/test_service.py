"""Unit tests for the ingestion service orchestration and error isolation."""

from unittest.mock import MagicMock

import pytest

from geopulse.config import Settings
from geopulse.ingestion.fetcher import FeedFetchError, fetch_feed_data
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


def test_consequential_diplomacy_retained_when_displaced_by_unrelated_news_entries(monkeypatch):
    """Verify that important diplomacy news is not lost when preceding items are unrelated."""
    from geopulse.classification.models import TopicCategory
    from geopulse.clustering import cluster_articles

    xml_with_unrelated_front = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>General News Outlet</title>
        <item>
          <title>Municipal election results certified across districts</title>
          <link>https://example.com/local-election</link>
          <description>Electoral board releases final tallies for city council seats.</description>
        </item>
        <item>
          <title>Tech conglomerate launches new smartphone series</title>
          <link>https://example.com/tech-launch</link>
          <description>Consumer electronics showcase features upgraded camera sensors.</description>
        </item>
        <item>
          <title>Regional soccer tournament concludes with penalty shootout</title>
          <link>https://example.com/sports-final</link>
          <description>Championship trophy awarded following extra time.</description>
        </item>
        <item>
          <title>Severe weekend storm causes localized traffic delays</title>
          <link>https://example.com/weather-traffic</link>
          <description>Highway crews clear fallen branches along route.</description>
        </item>
        <item>
          <title>Diplomatic envoys convene for high-stakes ceasefire and peace negotiations</title>
          <link>https://example.com/peace-talks-accord</link>
          <description>Belligerents agree to restart tripartite peace talks.</description>
        </item>
      </channel>
    </rss>
    """
    monkeypatch.setattr(
        "geopulse.ingestion.service.fetch_feed_data",
        lambda url, **kwargs: xml_with_unrelated_front,
    )
    feed = FeedSource(name="General News Outlet", feed_url="https://example.com/rss")

    # 1. Under limit=3, only the first 3 unrelated stories are fetched; diplomacy is lost
    narrow_summary = run_ingestion(feeds=[feed], limit_per_feed=3, dry_run=True)
    assert narrow_summary.total_articles_fetched == 3
    narrow_articles = [a for res in narrow_summary.feed_results for a in res.articles]
    narrow_clusters = cluster_articles(narrow_articles).clusters
    # No relevant events formed because all 3 entries are unrelated local news
    assert len(narrow_clusters) == 0

    # 2. Under expanded default limit (e.g. 10), all 5 entries are fetched
    wide_summary = run_ingestion(feeds=[feed], limit_per_feed=10, dry_run=True)
    assert wide_summary.total_articles_fetched == 5
    wide_articles = [a for res in wide_summary.feed_results for a in res.articles]

    # Verify non-diplomacy general items are not deemed relevant
    assert wide_articles[0].relevance_status != "relevant"
    assert wide_articles[1].relevance_status == "irrelevant"
    assert wide_articles[2].relevance_status == "irrelevant"
    assert wide_articles[3].relevance_status != "relevant"

    # Verify the diplomacy story is recognized as relevant and clustered
    assert wide_articles[4].relevance_status == "relevant"
    wide_clusters = cluster_articles(wide_articles).clusters
    assert len(wide_clusters) == 1
    assert wide_clusters[0].category == TopicCategory.DIPLOMACY
    assert (
        "ceasefire" in wide_clusters[0].title.lower() or "peace" in wide_clusters[0].title.lower()
    )


def test_fetch_feed_data_retry_and_timeout(monkeypatch):
    import httpx

    calls = 0

    def mock_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("Connection timed out")

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    with pytest.raises(FeedFetchError) as exc_info:
        fetch_feed_data("https://example.com/feed", max_retries=1, retry_backoff=0.01)

    assert calls == 2  # 1 initial + 1 retry
    assert "Timed out" in str(exc_info.value)


def test_run_ingestion_concurrent_and_elapsed(monkeypatch):
    import time

    def slow_fetch(url, **kwargs):
        if "slow" in url:
            time.sleep(0.05)
        return MOCK_XML

    monkeypatch.setattr("geopulse.ingestion.service.fetch_feed_data", slow_fetch)

    feeds = [
        FeedSource(name="Fast 1", feed_url="https://example.com/fast1"),
        FeedSource(name="Slow", feed_url="https://example.com/slow"),
        FeedSource(name="Fast 2", feed_url="https://example.com/fast2"),
    ]

    summary = run_ingestion(feeds=feeds, dry_run=True)

    assert summary.total_feeds == 3
    assert summary.successful_feeds == 3
    # Preserves original order
    assert [r.feed_name for r in summary.feed_results] == ["Fast 1", "Slow", "Fast 2"]
    # Elapsed seconds recorded
    for r in summary.feed_results:
        assert r.elapsed_seconds >= 0.0
