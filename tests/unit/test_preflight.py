"""Offline unit tests for Supabase preflight connectivity and schema validation."""

from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock

from geopulse.cli import main
from geopulse.config import Settings
from geopulse.ingestion.models import ArticleRecord
from geopulse.ingestion.repository import SupabaseArticleRepository
from geopulse.pipeline import BriefingPipeline
from geopulse.preflight import (
    CLUSTER_REQUIRED_SCHEMA,
    FULL_REQUIRED_SCHEMA,
    INGEST_REQUIRED_SCHEMA,
    run_supabase_preflight,
)
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
)
from geopulse.summarization.service import EventSummarizer


class MockPostgrestQueryBuilder:
    """Mock query builder simulating PostgREST table queries and schema errors."""

    def __init__(
        self,
        table_name: str,
        missing_tables: set[str],
        missing_columns: dict[str, set[str]],
        write_tracker: list[tuple[str, str, Any]],
    ) -> None:
        self.table_name = table_name
        self.missing_tables = missing_tables
        self.missing_columns = missing_columns
        self.write_tracker = write_tracker
        self._selected_cols: list[str] = []

    def select(self, cols: str = "*", *args, **kwargs) -> "MockPostgrestQueryBuilder":
        if self.table_name in self.missing_tables:
            raise Exception(f"relation 'public.{self.table_name}' does not exist (404)")

        # Check if any requested column is simulated missing
        if cols != "*":
            requested = [c.strip() for c in cols.split(",")]
            missing_for_tbl = self.missing_columns.get(self.table_name, set())
            for col in requested:
                if col in missing_for_tbl:
                    raise Exception(
                        f"Could not find the column '{col}' of '{self.table_name}' "
                        "in the schema cache (400)"
                    )
        return self

    def limit(self, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        return self

    def insert(self, payload: Any, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        self.write_tracker.append((self.table_name, "insert", payload))
        return self

    def upsert(self, payload: Any, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        self.write_tracker.append((self.table_name, "upsert", payload))
        return self

    def update(self, payload: Any, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        self.write_tracker.append((self.table_name, "update", payload))
        return self

    def eq(self, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        return self

    def in_(self, *args, **kwargs) -> "MockPostgrestQueryBuilder":
        return self

    def execute(self) -> MagicMock:
        mock_res = MagicMock()
        mock_res.data = []
        return mock_res


class MockPreflightSupabaseClient:
    """Mock Supabase client providing controlled schema presence and write tracking."""

    def __init__(
        self,
        missing_tables: set[str] | None = None,
        missing_columns: dict[str, set[str]] | None = None,
    ) -> None:
        self.missing_tables = missing_tables or set()
        self.missing_columns = missing_columns or {}
        self.writes: list[tuple[str, str, Any]] = []

    def table(self, table_name: str) -> MockPostgrestQueryBuilder:
        return MockPostgrestQueryBuilder(
            table_name=table_name,
            missing_tables=self.missing_tables,
            missing_columns=self.missing_columns,
            write_tracker=self.writes,
        )


def _create_sample_article() -> ArticleRecord:
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)
    return ArticleRecord(
        source_name="Defense News",
        title="Hypersonic Missile Test",
        url="https://defensenews.com/hypersonic-test",
        canonical_url="https://defensenews.com/hypersonic-test",
        published_at=now,
        fetched_at=now,
        snippet="Air Force conducted hypersonic flight test.",
        content_hash="h-hypersonic",
        relevance_status="relevant",
        topic="weapons_procurement",
    )


def test_preflight_missing_credentials():
    """Verify preflight fails closed when Supabase credentials are missing."""
    settings = Settings(supabase_url=None, supabase_key=None)
    res = run_supabase_preflight(client=None, settings=settings)

    assert res.success is False
    assert res.error_message is not None
    assert "Supabase credentials are not configured" in res.error_message


def test_preflight_connection_failure(monkeypatch):
    """Verify preflight handles network/connection failure gracefully without leaking keys."""
    settings = Settings(
        supabase_url="https://nonexistent-project-xyz.supabase.co",
        supabase_key="dummy-service-key-secret-12345",
    )

    def mock_create_client(url, key):
        raise ConnectionError("Host unreachable: name resolution failed")

    import supabase

    monkeypatch.setattr(supabase, "create_client", mock_create_client)

    res = run_supabase_preflight(client=None, settings=settings)
    assert res.success is False
    assert res.error_message is not None
    assert "Failed to connect to Supabase" in res.error_message
    # Confirm secret key is never leaked in error message
    assert "dummy-service-key-secret-12345" not in res.error_message


def test_preflight_missing_required_table():
    """Verify preflight pinpoints missing table when migration 001 is incomplete."""
    mock_client = MockPreflightSupabaseClient(missing_tables={"briefing_events"})
    res = run_supabase_preflight(client=mock_client, scope="full")

    assert res.success is False
    assert "briefing_events" in res.missing_tables
    assert res.error_message is not None
    assert "Missing required table(s): briefing_events" in res.error_message


def test_preflight_missing_required_column():
    """Verify preflight detects missing column from unapplied migration 005."""
    mock_client = MockPreflightSupabaseClient(
        missing_columns={"events": {"headline", "key_details"}}
    )
    res = run_supabase_preflight(client=mock_client, scope="full")

    assert res.success is False
    assert "events" in res.missing_columns
    assert "headline" in res.missing_columns["events"]
    assert "key_details" in res.missing_columns["events"]
    assert res.error_message is not None
    assert "Missing required column(s): events (headline, key_details)" in res.error_message


def test_schema_preflight_fails_before_any_write():
    """Verify that if preflight fails, BriefingPipeline aborts with zero database writes."""
    mock_client = MockPreflightSupabaseClient(missing_tables={"briefings"})
    art = _create_sample_article()

    art_repo = SupabaseArticleRepository(client=mock_client)
    pipeline = BriefingPipeline(
        article_repo=art_repo,
        settings=Settings(),
    )

    result = pipeline.run(
        briefing_date=date(2026, 10, 24),
        articles=[art],
        persistent=True,
        send=False,
    )

    # Pipeline failed on preflight
    assert result.error_message is not None
    assert "Preflight schema check failed" in result.error_message
    assert "Missing required table(s): briefings" in result.error_message

    # Strict invariant: ZERO writes occurred to any table
    assert len(mock_client.writes) == 0


def test_successful_preflight_followed_by_pipeline():
    """Verify that when preflight succeeds, the persistent pipeline proceeds normally."""
    mock_client = MockPreflightSupabaseClient()
    art = _create_sample_article()

    art_repo = SupabaseArticleRepository(client=mock_client)

    mock_summarizer = MagicMock(spec=EventSummarizer)
    summary_obj = EventSummary(
        event_id="dummy-id",
        headline="Hypersonic Test Completed",
        what_happened="Test completed successfully.",
        key_details=EventKeyDetails(actors=("Air Force",), action="Completed flight test"),
        significance="Advanced hypersonic capabilities.",
        confidence_qualification="Single-source defence reporting",
        sources=(
            EventSourceAttribution(
                source_name=art.source_name, title=art.title, url=art.url
            ),
        ),
        model_used="mock-model",
    )

    mock_summarizer.summarize_events.side_effect = lambda evs: tuple(
        SummarizationResult(
            event_id=e.id,
            status="success",
            summary=summary_obj,
            event=e,
        )
        for e in evs
    )

    pipeline = BriefingPipeline(
        article_repo=art_repo,
        summarizer=mock_summarizer,
        settings=Settings(),
    )

    result = pipeline.run(
        briefing_date=date(2026, 10, 24),
        articles=[art],
        persistent=True,
        send=False,
    )

    assert result.error_message is None
    assert result.is_preview is True
    assert result.is_persistent is True
    assert result.articles_count == 1
    # Articles were persisted because preflight passed
    written_tables = {w[0] for w in mock_client.writes}
    assert "articles" in written_tables


def test_preview_mode_never_sends_messages(monkeypatch):
    """Verify that --preview mode runs in-memory and never calls Telegram client."""
    mock_telegram = MagicMock()
    from geopulse.delivery import client as tg_client_mod

    monkeypatch.setattr(tg_client_mod.TelegramClient, "send_message", mock_telegram)

    ret = main(["brief", "--preview", "--limit", "1", "--fixture", "--mock-model"])
    assert ret == 0
    # Telegram was never called
    assert mock_telegram.call_count == 0


def test_persist_without_send_never_sends_messages(monkeypatch):
    """Verify that --persist without --send never calls Telegram client."""
    mock_telegram = MagicMock()
    from geopulse.delivery import client as tg_client_mod

    monkeypatch.setattr(tg_client_mod.TelegramClient, "send_message", mock_telegram)

    # Calling brief with --send omitted defaults to dry-run delivery
    ret = main(["brief", "--limit", "1", "--fixture", "--mock-model"])
    assert ret == 0
    assert mock_telegram.call_count == 0


def test_send_blocked_when_preflight_fails(monkeypatch, capsys):
    """Verify that --send is blocked and no messages are sent when preflight fails."""
    mock_telegram = MagicMock()
    from geopulse.delivery import client as tg_client_mod

    monkeypatch.setattr(tg_client_mod.TelegramClient, "send_message", mock_telegram)

    # Mock preflight to fail with missing migration 005 column
    import geopulse.cli as cli_mod

    def mock_failing_preflight(*args, **kwargs):
        from geopulse.preflight import PreflightResult

        return PreflightResult(
            success=False,
            missing_columns={"events": ["headline", "key_details"]},
            error_message="Missing required column(s): events (headline, key_details)",
            details="Pending database migrations must be applied.",
        )

    monkeypatch.setattr(cli_mod, "run_supabase_preflight", mock_failing_preflight)

    # Provide dummy environment settings so is_supabase_configured is True
    test_settings = Settings(
        supabase_url="https://mock-test.supabase.co",
        supabase_key="mock-key-12345",
        telegram_bot_token="mock-token",
        telegram_chat_id="123456",
    )
    monkeypatch.setattr(cli_mod, "Settings", lambda: test_settings)

    ret = main(["brief", "--persist", "--send", "--limit", "1", "--fixture", "--mock-model"])
    captured = capsys.readouterr().out

    # Failed closed with exit code 1
    assert ret == 1
    assert "Supabase schema preflight failed" in captured
    assert "events (headline, key_details)" in captured
    # Invariant: Zero Telegram messages sent
    assert mock_telegram.call_count == 0


def test_preflight_scopes():
    """Verify ingest, cluster, and full scopes inspect appropriate table sets."""
    assert "sources" in INGEST_REQUIRED_SCHEMA
    assert "articles" in INGEST_REQUIRED_SCHEMA
    assert "events" not in INGEST_REQUIRED_SCHEMA

    assert "events" in CLUSTER_REQUIRED_SCHEMA
    assert "event_articles" in CLUSTER_REQUIRED_SCHEMA
    assert "briefings" not in CLUSTER_REQUIRED_SCHEMA

    assert "briefings" in FULL_REQUIRED_SCHEMA
    assert "briefing_events" in FULL_REQUIRED_SCHEMA
    assert "briefing_event_evidence" in FULL_REQUIRED_SCHEMA


def test_ingest_preflight_failure_blocks_ingestion(monkeypatch, capsys):
    """Verify that 'geopulse ingest --persist' halts before fetching feeds if preflight fails."""
    import geopulse.cli as cli_mod
    from geopulse.preflight import PreflightResult

    def mock_failing_preflight(*args, **kwargs):
        return PreflightResult(
            success=False,
            missing_tables=("articles",),
            error_message="Missing required table(s): articles",
        )

    monkeypatch.setattr(cli_mod, "run_supabase_preflight", mock_failing_preflight)
    test_settings = Settings(
        supabase_url="https://mock-test.supabase.co",
        supabase_key="mock-key-12345",
    )
    monkeypatch.setattr(cli_mod, "Settings", lambda: test_settings)

    mock_run_ingest = MagicMock()
    monkeypatch.setattr(cli_mod, "run_ingestion", mock_run_ingest)

    ret = main(["ingest", "--persist", "--limit", "1"])
    captured = capsys.readouterr().out

    assert ret == 1
    assert "Supabase schema preflight failed" in captured
    assert "Missing required table(s): articles" in captured
    # run_ingestion was NEVER called
    assert mock_run_ingest.call_count == 0


def test_cluster_preflight_failure_blocks_clustering(monkeypatch, capsys):
    """Verify that 'geopulse cluster --persist' halts if preflight fails."""
    import geopulse.cli as cli_mod
    from geopulse.preflight import PreflightResult

    def mock_failing_preflight(*args, **kwargs):
        return PreflightResult(
            success=False,
            missing_tables=("events",),
            error_message="Missing required table(s): events",
        )

    monkeypatch.setattr(cli_mod, "run_supabase_preflight", mock_failing_preflight)
    test_settings = Settings(
        supabase_url="https://mock-test.supabase.co",
        supabase_key="mock-key-12345",
    )
    monkeypatch.setattr(cli_mod, "Settings", lambda: test_settings)

    ret = main(["cluster", "--persist", "--limit", "1"])
    captured = capsys.readouterr().out

    assert ret == 1
    assert "Supabase schema preflight failed" in captured
    assert "Missing required table(s): events" in captured

