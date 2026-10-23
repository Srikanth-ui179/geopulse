"""Offline unit tests for Phase 6 Supabase integration and repository contracts."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest

from geopulse.classification.models import TopicCategory
from geopulse.cli import main
from geopulse.clustering.models import EventCluster
from geopulse.clustering.repository import InMemoryEventRepository, SupabaseEventRepository
from geopulse.config import Settings
from geopulse.delivery.models import TelegramMessageResponse
from geopulse.delivery.repository import JsonFileDeliveryRepository, SupabaseDeliveryRepository
from geopulse.delivery.service import TelegramDeliveryService, compute_digest_hash
from geopulse.ingestion.models import ArticleRecord
from geopulse.ingestion.repository import SupabaseArticleRepository
from geopulse.summarization.models import EventSummary, SummarizationResult


class MockPostgrestBuilder:
    """Mock for chained PostgREST / Supabase table query builder."""

    def __init__(self, data: list[dict] | None = None) -> None:
        self._data = data if data is not None else []
        self._payload: dict | list | None = None
        self._upsert_params: dict | None = None
        self._table_name: str = ""

    def select(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def insert(self, payload: dict | list, *args, **kwargs) -> "MockPostgrestBuilder":
        self._payload = payload
        return self

    def upsert(self, payload: dict | list, *args, **kwargs) -> "MockPostgrestBuilder":
        self._payload = payload
        self._upsert_params = kwargs
        return self

    def update(self, payload: dict, *args, **kwargs) -> "MockPostgrestBuilder":
        self._payload = payload
        return self

    def eq(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def in_(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def gte(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def lte(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def limit(self, *args, **kwargs) -> "MockPostgrestBuilder":
        return self

    def execute(self) -> MagicMock:
        mock_res = MagicMock()
        mock_res.data = self._data
        return mock_res


class MockSupabaseClient:
    """In-memory mock Supabase client providing controlled query responses."""

    def __init__(self) -> None:
        self.tables: dict[str, MockPostgrestBuilder] = {}
        self.executed_queries: list[tuple[str, str, dict | list | None]] = []

    def table(self, table_name: str) -> MockPostgrestBuilder:
        builder = self.tables.get(table_name)
        if builder is None:
            builder = MockPostgrestBuilder()
            self.tables[table_name] = builder
        return builder


def test_article_insert_and_upsert_behavior():
    """Verify SupabaseArticleRepository constructs compliant payloads and executes upsert."""
    mock_client = MockSupabaseClient()
    mock_client.tables["articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseArticleRepository(client=mock_client)
    now = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)

    article = ArticleRecord(
        source_name="Defense News",
        title="Missile Contract Awarded",
        url="https://defensenews.com/missile",
        canonical_url="https://defensenews.com/missile",
        published_at=now,
        fetched_at=now,
        snippet="Missile contract details",
        content_hash="hash-123",
        relevance_status="relevant",
        topic="weapons_procurement",
    )

    total, new = repo.persist_articles(source_id="src-uuid-1", articles=[article])
    assert total == 1
    assert new == 1

    builder = mock_client.tables["articles"]
    assert builder._payload is not None
    assert len(builder._payload) == 1
    row = builder._payload[0]
    assert row["source_id"] == "src-uuid-1"
    assert row["canonical_url"] == "https://defensenews.com/missile"
    assert row["topic"] == "weapons_procurement"
    assert row["relevance_status"] == "relevant"
    assert builder._upsert_params == {"on_conflict": "canonical_url", "ignore_duplicates": True}


def test_duplicate_ingestion_is_idempotent():
    """Verify that previously ingested articles with existing canonical URLs are skipped."""
    mock_client = MockSupabaseClient()
    mock_client.tables["articles"] = MockPostgrestBuilder(
        data=[{"canonical_url": "https://defensenews.com/existing"}]
    )

    repo = SupabaseArticleRepository(client=mock_client)
    now = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)

    article = ArticleRecord(
        source_name="Defense News",
        title="Existing Article",
        url="https://defensenews.com/existing",
        canonical_url="https://defensenews.com/existing",
        published_at=now,
        fetched_at=now,
        snippet="Already seen",
        content_hash="hash-seen",
    )

    total, new = repo.persist_articles(source_id="src-uuid-1", articles=[article])
    assert total == 1
    assert new == 0


def test_get_article_ids_by_urls():
    """Verify batch lookup of article UUIDs by canonical URLs."""
    mock_client = MockSupabaseClient()
    mock_client.tables["articles"] = MockPostgrestBuilder(
        data=[
            {"id": "uuid-art-1", "canonical_url": "https://example.com/1"},
            {"id": "uuid-art-2", "canonical_url": "https://example.com/2"},
        ]
    )

    repo = SupabaseArticleRepository(client=mock_client)
    mapping = repo.get_article_ids_by_urls(["https://example.com/1", "https://example.com/2"])
    assert mapping == {
        "https://example.com/1": "uuid-art-1",
        "https://example.com/2": "uuid-art-2",
    }


def test_event_to_article_source_associations():
    """Verify SupabaseEventRepository writes events and public.event_articles relationships."""
    mock_client = MockSupabaseClient()
    mock_client.tables["events"] = MockPostgrestBuilder(data=[])
    mock_client.tables["event_articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    art1 = ArticleRecord(
        source_name="Defense News",
        title="Report 1",
        url="https://defensenews.com/1",
        canonical_url="https://defensenews.com/1",
        published_at=now,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="Report 2",
        url="https://breakingdefense.com/2",
        canonical_url="https://breakingdefense.com/2",
        published_at=now,
        content_hash="h2",
    )

    event = EventCluster(
        id="ev-uuid-101",
        title="Denmark Strike Missile Procurement",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        published_at=now,
        is_analysis=False,
        confidence_label="corroborated",
        articles=(art1, art2),
    )

    article_url_to_id = {
        "https://defensenews.com/1": "art-uuid-1",
        "https://breakingdefense.com/2": "art-uuid-2",
    }

    total, count = repo.persist_events([event], article_url_to_id=article_url_to_id)
    assert total == 1
    assert count == 1

    events_builder = mock_client.tables["events"]
    assert events_builder._payload[0]["id"] == "ev-uuid-101"
    assert events_builder._payload[0]["category"] == "weapons_procurement"
    assert events_builder._payload[0]["is_analysis"] is False

    assoc_builder = mock_client.tables["event_articles"]
    assert len(assoc_builder._payload) == 2
    assert assoc_builder._payload[0]["event_id"] == "ev-uuid-101"
    assert assoc_builder._payload[0]["article_id"] == "art-uuid-1"
    assert assoc_builder._payload[1]["article_id"] == "art-uuid-2"


def test_in_memory_event_repository_contract():
    """Verify InMemoryEventRepository preserves events and article associations."""
    repo = InMemoryEventRepository()
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="BBC World News",
        title="Incident",
        url="https://bbc.com/inc",
        canonical_url="https://bbc.com/inc",
        published_at=now,
        content_hash="h-inc",
    )
    event = EventCluster(
        id="ev-inc-1",
        title="Security Alert",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    repo.persist_events([event], article_url_to_id={"https://bbc.com/inc": "art-1"})
    by_date = repo.get_events_by_date(date(2026, 10, 23))
    assert len(by_date) == 1
    assert by_date[0].id == "ev-inc-1"
    assert repo.get_event_article_ids("ev-inc-1") == ["art-1"]


def test_briefing_persistence_and_delivery_state_recovery():
    """Verify SupabaseDeliveryRepository round-trips chunk progress and digest hash."""
    now = datetime(2026, 10, 23, 14, 0, 0, tzinfo=UTC)
    mock_client = MockSupabaseClient()
    mock_client.tables["briefings"] = MockPostgrestBuilder(
        data=[
            {
                "id": "briefing-uuid-1",
                "briefing_date": "2026-10-23",
                "status": "partial",
                "started_at": now.isoformat(),
                "completed_at": None,
                "telegram_message_count": 2,
                "chunks_delivered": 1,
                "telegram_message_ids": [101],
                "digest_hash": "hash-digest-abc",
                "error_message": "Network timeout on chunk 2",
            }
        ]
    )

    repo = SupabaseDeliveryRepository(client=mock_client)
    record = repo.get_briefing_by_date(date(2026, 10, 23))

    assert record is not None
    assert record.id == "briefing-uuid-1"
    assert record.status == "partial"
    assert record.chunks_delivered == 1
    assert record.telegram_message_ids == (101,)
    assert record.digest_hash == "hash-digest-abc"
    assert record.error_message == "Network timeout on chunk 2"


def test_briefing_events_and_evidence_persistence():
    """Verify record_briefing_events writes to briefing_events and briefing_event_evidence."""
    mock_client = MockSupabaseClient()
    mock_client.tables["briefing_events"] = MockPostgrestBuilder(data=[])
    mock_client.tables["briefing_event_evidence"] = MockPostgrestBuilder(data=[])

    repo = SupabaseDeliveryRepository(client=mock_client)
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Breaking Defense",
        title="DARC Base",
        url="https://breakingdefense.com/darc",
        canonical_url="https://breakingdefense.com/darc",
        published_at=now,
        content_hash="hd",
    )
    event = EventCluster(
        id="ev-darc-1",
        title="DARC Programme",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art,),
    )
    summary_obj = EventSummary(
        event_id="ev-darc-1",
        headline="UK to host DARC radar",
        what_happened="Cawdor Barracks selected for deep-space tracking radar.",
        significance="Enhances AUKUS space domain awareness.",
        confidence_qualification="Single-source defence reporting",
        model_used="mistral:latest",
    )
    result = SummarizationResult(
        event_id="ev-darc-1",
        status="success",
        summary=summary_obj,
        event=event,
    )

    repo.record_briefing_events(
        briefing_id="b-uuid-1",
        results=[result],
        article_url_to_id={"https://breakingdefense.com/darc": "art-uuid-darc"},
    )

    ev_builder = mock_client.tables["briefing_events"]
    assert len(ev_builder._payload) == 1
    assert ev_builder._payload[0]["briefing_id"] == "b-uuid-1"
    assert ev_builder._payload[0]["event_id"] == "ev-darc-1"
    assert ev_builder._payload[0]["rank"] == 1
    assert (
        ev_builder._payload[0]["summary"]
        == "Cawdor Barracks selected for deep-space tracking radar."
    )

    evi_builder = mock_client.tables["briefing_event_evidence"]
    assert len(evi_builder._payload) == 1
    assert evi_builder._payload[0]["briefing_id"] == "b-uuid-1"
    assert evi_builder._payload[0]["article_id"] == "art-uuid-darc"


def test_partial_delivery_resume_with_supabase_repository():
    """Verify delivery service safely resumes partial delivery using Supabase repo."""
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
    mock_client = MockSupabaseClient()

    chunks = ("Chunk 1 content here", "Chunk 2 content here")
    expected_hash = compute_digest_hash(chunks)

    # Pre-existing briefing with chunk 1 already confirmed
    initial_row = {
        "id": "briefing-uuid-resume",
        "briefing_date": "2026-10-23",
        "status": "partial",
        "started_at": now.isoformat(),
        "completed_at": None,
        "telegram_message_count": 2,
        "chunks_delivered": 1,
        "telegram_message_ids": [777],
        "digest_hash": expected_hash,
        "error_message": "Network timeout on chunk 2",
    }
    mock_client.tables["briefings"] = MockPostgrestBuilder(data=[initial_row])

    repo = SupabaseDeliveryRepository(client=mock_client)
    mock_tg = MagicMock()
    # Mock TG should only be called once (for chunk 2)
    mock_tg.send_message.return_value = TelegramMessageResponse(
        message_id=778,
        chat_id="test-chat",
        date=1729684800,
    )

    service = TelegramDeliveryService(
        client=mock_tg,
        repository=repo,
        settings=Settings(),
    )

    chunks = ("Chunk 1 content here", "Chunk 2 content here")
    res = service.deliver_digest(digest_chunks=chunks, briefing_date=date(2026, 10, 23))

    assert res.status == "sent"
    assert res.resumed_from_chunk == 1
    assert res.chunks_delivered == 2
    # Only chunk 2 was sent to Telegram (chunk 1 was NOT resent)
    assert mock_tg.send_message.call_count == 1
    assert mock_tg.send_message.call_args[0][0] == "Chunk 2 content here"
    assert res.message_ids == (777, 778)


def test_unconfigured_supabase_fails_gracefully():
    """Verify that unconfigured Supabase raises ValueError with clear guidance."""
    settings = Settings(supabase_url=None, supabase_key=None)
    with pytest.raises(ValueError, match="Supabase is not configured"):
        SupabaseArticleRepository(settings=settings)

    with pytest.raises(ValueError, match="Supabase is not configured"):
        SupabaseEventRepository(settings=settings)

    with pytest.raises(ValueError, match="Supabase is not configured"):
        SupabaseDeliveryRepository(settings=settings)


def test_repeated_article_ingestion_idempotency():
    """Verify repeated ingestion of same canonical URLs reports 0 new articles."""
    mock_client = MockSupabaseClient()
    now = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)
    existing_url = "https://defensenews.com/existing"

    # Simulate article already in database
    mock_client.tables["articles"] = MockPostgrestBuilder(data=[{"canonical_url": existing_url}])

    repo = SupabaseArticleRepository(client=mock_client)
    article = ArticleRecord(
        source_name="Defense News",
        title="Existing Article",
        url=existing_url,
        canonical_url=existing_url,
        published_at=now,
        fetched_at=now,
        content_hash="h1",
    )

    total, new = repo.persist_articles(source_id="src-1", articles=[article])
    assert total == 1
    assert new == 0
    # No insert/upsert should be executed if no articles are new
    assert mock_client.tables["articles"]._payload is None


def test_event_evidence_preservation_across_runs():
    """Verify updating an event with new articles appends associations without losing existing."""
    mock_client = MockSupabaseClient()
    now = datetime(2026, 10, 23, 10, 0, 0, tzinfo=UTC)

    art1 = ArticleRecord(
        source_name="Source 1",
        title="Article 1",
        url="https://s1.com/1",
        canonical_url="https://s1.com/1",
        published_at=now,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Source 2",
        title="Article 2",
        url="https://s2.com/2",
        canonical_url="https://s2.com/2",
        published_at=now,
        content_hash="h2",
    )

    ev_repo = SupabaseEventRepository(client=mock_client)
    event_initial = EventCluster(
        id="ev-100",
        title="Hypersonic Test",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1,),
    )

    ev_repo.persist_events([event_initial], article_url_to_id={"https://s1.com/1": "art-uuid-1"})
    assert len(mock_client.tables["event_articles"]._payload) == 1
    assert mock_client.tables["event_articles"]._payload[0]["article_id"] == "art-uuid-1"

    # Run 2: Event corroborated by second source
    event_updated = EventCluster(
        id="ev-100",
        title="Hypersonic Test",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    ev_repo.persist_events(
        [event_updated],
        article_url_to_id={"https://s1.com/1": "art-uuid-1", "https://s2.com/2": "art-uuid-2"},
    )
    assoc_payloads = mock_client.tables["event_articles"]._payload
    assert len(assoc_payloads) == 2
    article_ids = {row["article_id"] for row in assoc_payloads}
    assert article_ids == {"art-uuid-1", "art-uuid-2"}
    # Asserts ignore_duplicates=True so existing rows are not overwritten or dropped
    assert mock_client.tables["event_articles"]._upsert_params.get("ignore_duplicates") is True


def test_delivery_lock_mutual_exclusion(tmp_path):
    """Verify that lock acquisition prevents concurrent execution and releases cleanly."""
    lock_file = tmp_path / "test_delivery.lock"
    repo1 = SupabaseDeliveryRepository(client=MockSupabaseClient(), lock_path=lock_file)
    repo2 = SupabaseDeliveryRepository(client=MockSupabaseClient(), lock_path=lock_file)

    # 1. Process 1 acquires lock
    assert repo1.acquire_lock() is True

    # 2. Process 2 attempts lock on same path and is rejected
    assert repo2.acquire_lock() is False

    # 3. Process 1 releases lock
    repo1.release_lock()

    # 4. Process 2 can now acquire lock
    assert repo2.acquire_lock() is True
    repo2.release_lock()


def test_cross_repository_locking_compatibility(tmp_path):
    """Verify SupabaseDeliveryRepository and JsonFileDeliveryRepository share kernel lock."""
    lock_file = tmp_path / "shared_state.lock"
    state_file = tmp_path / "shared_state.json"

    json_repo = JsonFileDeliveryRepository(file_path=state_file)
    json_repo.lock_path = lock_file

    supa_repo = SupabaseDeliveryRepository(client=MockSupabaseClient(), lock_path=lock_file)

    assert json_repo.acquire_lock() is True
    assert supa_repo.acquire_lock() is False

    json_repo.release_lock()
    assert supa_repo.acquire_lock() is True
    supa_repo.release_lock()


def test_cli_flags_conflicting_persist_and_dry_run(capsys):
    """Verify that passing both --persist and --dry-run fails closed."""
    ret = main(["ingest", "--persist", "--dry-run"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Cannot specify both --persist and --dry-run" in captured


def test_cli_flags_conflicting_delivery_state_options(capsys):
    """Verify that passing contradictory state storage flags fails closed."""
    ret = main(["brief", "--in-memory-state", "--supabase-state"])
    captured = capsys.readouterr().out
    assert ret == 1
    assert "Cannot specify both --in-memory-state and --supabase-state" in captured
