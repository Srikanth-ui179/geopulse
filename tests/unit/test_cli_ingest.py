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
    from geopulse import cli as cli_mod
    from geopulse.config import Settings

    monkeypatch.setattr(
        cli_mod,
        "Settings",
        lambda: Settings(_env_file=None, supabase_url=None, supabase_key=None),
    )

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


def test_cli_cluster_persist_flow_ingests_and_links_articles(monkeypatch, capsys):
    from geopulse.config import Settings
    from geopulse.preflight import PreflightResult

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

    ingest_kwargs_captured = {}

    def mock_run_ingest(**kwargs):
        ingest_kwargs_captured.update(kwargs)
        return mock_summary

    persisted_events_args = {}

    class FakeEvRepo:
        def __init__(self, *args, **kwargs):
            pass

        def persist_events(self, events, article_url_to_id=None):
            persisted_events_args["events"] = events
            persisted_events_args["article_url_to_id"] = article_url_to_id
            return len(events), len(events)

    class FakeArtRepo:
        client = None

        def __init__(self, *args, **kwargs):
            pass

        def get_article_ids_by_urls(self, urls):
            return {u: f"uuid-{u}" for u in urls}

    monkeypatch.setattr(
        "geopulse.cli.Settings",
        lambda: Settings(
            _env_file=None,
            supabase_url="https://test.supabase.co",
            supabase_key="sb_secret_fake_key_12345",
        ),
    )
    monkeypatch.setattr(
        "geopulse.cli.run_supabase_preflight",
        lambda **kwargs: PreflightResult(success=True),
    )
    monkeypatch.setattr("geopulse.cli.run_ingestion", mock_run_ingest)
    monkeypatch.setattr("geopulse.cli.SupabaseArticleRepository", FakeArtRepo)
    monkeypatch.setattr("geopulse.cli.SupabaseEventRepository", FakeEvRepo)

    result = main(["cluster", "--persist"])
    captured = capsys.readouterr().out
    assert result == 0
    # Ingestion was run with dry_run=False and repository supplied
    assert ingest_kwargs_captured.get("dry_run") is False
    assert isinstance(ingest_kwargs_captured.get("repository"), FakeArtRepo)
    # Event repo received resolved article_url_to_id
    assert "article_url_to_id" in persisted_events_args
    assert (
        persisted_events_args["article_url_to_id"].get("https://bbc.com/rostov-depot")
        == "uuid-https://bbc.com/rostov-depot"
    )
    assert "Persisted 1 event cluster(s) to Supabase." in captured
