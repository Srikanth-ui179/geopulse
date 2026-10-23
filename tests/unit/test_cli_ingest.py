"""Unit tests for the CLI ingest subcommand."""

from geopulse.cli import main
from geopulse.ingestion.models import (
    ArticleRecord,
    FeedIngestionResult,
    IngestionSummary,
)


def test_cli_ingest_dry_run_command_succeeds(monkeypatch, capsys):
    sample_article = ArticleRecord(
        source_name="Test Source",
        title="Armed forces initiate peacekeeping mandate",
        url="https://example.com/peacekeeping",
        canonical_url="https://example.com/peacekeeping",
        content_hash="mock-hash-12345",
        relevance_status="relevant",
    )
    mock_summary = IngestionSummary(
        total_feeds=1,
        successful_feeds=1,
        failed_feeds=0,
        total_articles_fetched=1,
        total_articles_new=1,
        feed_results=[
            FeedIngestionResult(
                feed_name="Test Source",
                feed_url="https://example.com/feed",
                success=True,
                articles_fetched=1,
                articles_new=1,
                articles=[sample_article],
            )
        ],
    )

    monkeypatch.setattr(
        "geopulse.cli.run_ingestion",
        lambda **kwargs: mock_summary,
    )

    result = main(["ingest", "--dry-run", "--limit", "5"])

    captured = capsys.readouterr().out
    assert result == 0
    assert "Starting Geopulse RSS Ingestion [Dry-run" in captured
    assert "Test Source" in captured
    assert "Ingestion Summary:" in captured
    assert "Articles Fetched: 1" in captured


def test_cli_ingest_unknown_feed_filter_fails(capsys):
    result = main(["ingest", "--feed", "NonexistentFeed99"])
    captured = capsys.readouterr().out
    assert result == 1
    assert "No feed matched 'NonexistentFeed99'" in captured


def test_cli_ingest_persist_without_credentials_fails(monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    result = main(["ingest", "--persist"])
    captured = capsys.readouterr().out
    assert result == 1
    assert "--persist requires SUPABASE_URL and SUPABASE_KEY" in captured


def test_cli_ingest_show_articles_prints_details(monkeypatch, capsys):
    sample_article = ArticleRecord(
        source_name="Test Source",
        title="Armed forces initiate peacekeeping mandate",
        url="https://example.com/peacekeeping?utm_source=twitter",
        canonical_url="https://example.com/peacekeeping",
        content_hash="mock-hash-12345",
        relevance_status="relevant",
    )
    mock_summary = IngestionSummary(
        total_feeds=1,
        successful_feeds=1,
        failed_feeds=0,
        total_articles_fetched=1,
        total_articles_new=1,
        feed_results=[
            FeedIngestionResult(
                feed_name="Test Source",
                feed_url="https://example.com/feed",
                success=True,
                articles_fetched=1,
                articles_new=1,
                articles=[sample_article],
            )
        ],
    )

    monkeypatch.setattr(
        "geopulse.cli.run_ingestion",
        lambda **kwargs: mock_summary,
    )

    result = main(["ingest", "--dry-run", "--show-articles"])

    captured = capsys.readouterr().out
    assert result == 0
    assert "Armed forces initiate peacekeeping mandate" in captured
    assert "https://example.com/peacekeeping" in captured


def test_cli_ingest_with_cluster_flag_displays_clustering_summary(monkeypatch, capsys):
    sample_article = ArticleRecord(
        source_name="BBC World News",
        title="US and UK launch joint airstrikes on Houthi military positions in Yemen",
        url="https://bbc.com/houthi-strikes",
        canonical_url="https://bbc.com/houthi-strikes",
        content_hash="mock-hash-111",
        relevance_status="relevant",
    )
    mock_summary = IngestionSummary(
        total_feeds=1,
        successful_feeds=1,
        failed_feeds=0,
        total_articles_fetched=1,
        total_articles_new=1,
        feed_results=[
            FeedIngestionResult(
                feed_name="BBC World News",
                feed_url="https://example.com/feed",
                success=True,
                articles_fetched=1,
                articles_new=1,
                articles=[sample_article],
            )
        ],
    )

    monkeypatch.setattr(
        "geopulse.cli.run_ingestion",
        lambda **kwargs: mock_summary,
    )

    result = main(["ingest", "--dry-run", "--cluster", "--show-articles"])
    captured = capsys.readouterr().out
    assert result == 0
    assert "Event Clustering Summary:" in captured
    assert "Events Formed:      1" in captured
    assert "Identified Events:" in captured
    assert "US and UK launch joint airstrikes on Houthi military positions in Yemen" in captured


def test_cli_cluster_subcommand_executes_cleanly(monkeypatch, capsys):
    sample_article = ArticleRecord(
        source_name="BBC World News",
        title="Ukrainian drone strikes set ablaze oil depot in Russia's Rostov region",
        url="https://bbc.com/rostov-depot",
        canonical_url="https://bbc.com/rostov-depot",
        content_hash="mock-hash-222",
        relevance_status="relevant",
    )
    mock_summary = IngestionSummary(
        total_feeds=1,
        successful_feeds=1,
        failed_feeds=0,
        total_articles_fetched=1,
        total_articles_new=1,
        feed_results=[
            FeedIngestionResult(
                feed_name="BBC World News",
                feed_url="https://example.com/feed",
                success=True,
                articles_fetched=1,
                articles_new=1,
                articles=[sample_article],
            )
        ],
    )

    monkeypatch.setattr(
        "geopulse.cli.run_ingestion",
        lambda **kwargs: mock_summary,
    )

    result = main(["cluster", "--limit", "2"])
    captured = capsys.readouterr().out
    assert result == 0
    assert "Fetching feeds to cluster events" in captured
    assert "Event Clustering Summary:" in captured
    assert "Events Formed:      1" in captured

