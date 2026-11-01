"""Offline unit tests for Phase 6 Supabase integration and repository contracts."""

from datetime import UTC, date, datetime
from typing import Any
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


class MockRpcSupabaseClient(MockSupabaseClient):
    """In-memory mock Supabase client with RPC procedure support for testing atomic operations."""

    def __init__(self) -> None:
        super().__init__()
        self.rpc_calls: list[tuple[str, dict]] = []
        self.rpc_side_effect: Exception | None = None
        self.rpc_data: dict | list | None = {"events_persisted": 1, "articles_linked": 1}

    def rpc(self, func_name: str, params: dict | None = None) -> Any:
        self.rpc_calls.append((func_name, params or {}))
        builder = MagicMock()
        if self.rpc_side_effect is not None:
            builder.execute.side_effect = self.rpc_side_effect
        else:
            mock_res = MagicMock()
            mock_res.data = self.rpc_data
            builder.execute.return_value = mock_res
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
    """Verify SupabaseEventRepository writes events and event_articles via RPC."""
    mock_client = MockRpcSupabaseClient()
    mock_client.rpc_data = {"events_persisted": 1, "articles_linked": 2}

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

    assert len(mock_client.rpc_calls) == 1
    func_name, params = mock_client.rpc_calls[0]
    assert func_name == "upsert_events_with_articles"
    assert params["p_events"][0]["id"] == "ev-uuid-101"
    assert params["p_events"][0]["category"] == "weapons_procurement"
    assert params["p_events"][0]["is_analysis"] is False

    assocs = params["p_event_articles"]
    assert len(assocs) == 2
    assert assocs[0]["event_id"] == "ev-uuid-101"
    assert assocs[0]["article_id"] == "art-uuid-1"
    assert assocs[1]["article_id"] == "art-uuid-2"


def test_persist_events_skips_unmatched_article_ids_before_writing_events():
    """Verify events with no resolvable article IDs are skipped before writing to events table."""
    mock_client = MockRpcSupabaseClient()
    mock_client.tables["articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    unmatched_art = ArticleRecord(
        source_name="Unknown Feed",
        title="Orphan Candidate",
        url="https://unknown.com/orphan",
        canonical_url="https://unknown.com/orphan",
        published_at=now,
        content_hash="h-orphan",
    )
    orphan_event = EventCluster(
        id="ev-orphan-1",
        title="Unmatched Article Event",
        category=TopicCategory.OTHER_GEOPOLITICAL,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(unmatched_art,),
    )

    # Empty article_url_to_id and empty articles table mock
    total, persisted = repo.persist_events([orphan_event], article_url_to_id={})
    assert total == 1
    assert persisted == 0

    # Ensure RPC was NOT called and tables were NOT written
    assert len(mock_client.rpc_calls) == 0
    assert "events" not in mock_client.tables or mock_client.tables["events"]._payload is None
    assert (
        "event_articles" not in mock_client.tables
        or mock_client.tables["event_articles"]._payload is None
    )


def test_persist_events_without_rpc_support_raises_clear_error_and_zero_writes():
    """Verify client lacking RPC raises actionable error and performs zero table writes."""
    mock_client = MockSupabaseClient()  # has table() but not rpc()
    mock_client.tables["events"] = MockPostgrestBuilder(data=[])
    mock_client.tables["event_articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Valid Article",
        url="https://defensenews.com/valid",
        canonical_url="https://defensenews.com/valid",
        published_at=now,
        content_hash="h-valid",
    )
    event = EventCluster(
        id="ev-valid-1",
        title="Valid Event",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    with pytest.raises(
        RuntimeError,
        match="Atomic event persistence is unavailable: Supabase client does not support RPC",
    ):
        repo.persist_events(
            [event],
            article_url_to_id={"https://defensenews.com/valid": "art-uuid-valid"},
        )

    # Ensure zero writes were made to events or event_articles tables
    assert mock_client.tables["events"]._payload is None
    assert mock_client.tables["event_articles"]._payload is None


def test_persist_events_duplicate_links_idempotent():
    """Verify event-article links deduplicate in payloads and remain idempotent."""
    mock_client = MockRpcSupabaseClient()
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Valid Article",
        url="https://defensenews.com/valid",
        canonical_url="https://defensenews.com/valid",
        published_at=now,
        content_hash="h-valid",
    )
    event = EventCluster(
        id="ev-valid-1",
        title="Valid Event",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    mapping = {"https://defensenews.com/valid": "art-uuid-valid"}
    repo.persist_events([event], article_url_to_id=mapping)
    assert len(mock_client.rpc_calls) == 1
    params = mock_client.rpc_calls[0][1]
    assert len(params["p_event_articles"]) == 1


def test_persist_events_atomic_rpc_success_all_articles_resolved():
    """Verify SupabaseEventRepository delegates to atomic RPC when available and succeeds."""
    mock_client = MockRpcSupabaseClient()
    mock_client.rpc_data = {"events_persisted": 1, "articles_linked": 2}
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art1 = ArticleRecord(
        source_name="Defense News",
        title="Procurement Notice",
        url="https://defensenews.com/proc",
        canonical_url="https://defensenews.com/proc",
        published_at=now,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="Procurement Analysis",
        url="https://breakingdefense.com/proc",
        canonical_url="https://breakingdefense.com/proc",
        published_at=now,
        content_hash="h2",
    )
    event = EventCluster(
        id="ev-atomic-1",
        title="Joint Missile Procurement",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        published_at=now,
        is_analysis=False,
        confidence_label="corroborated",
        articles=(art1, art2),
    )

    mapping = {
        "https://defensenews.com/proc": "art-uuid-1",
        "https://breakingdefense.com/proc": "art-uuid-2",
    }

    total, persisted = repo.persist_events([event], article_url_to_id=mapping)
    assert total == 1
    assert persisted == 1

    # Verify RPC called with correct function and payloads
    assert len(mock_client.rpc_calls) == 1
    func_name, params = mock_client.rpc_calls[0]
    assert func_name == "upsert_events_with_articles"

    p_events = params["p_events"]
    assert len(p_events) == 1
    assert p_events[0]["id"] == "ev-atomic-1"
    assert p_events[0]["category"] == "weapons_procurement"
    assert p_events[0]["is_analysis"] is False

    p_assocs = params["p_event_articles"]
    assert len(p_assocs) == 2
    assert p_assocs[0]["event_id"] == "ev-atomic-1"
    assert p_assocs[0]["article_id"] == "art-uuid-1"
    assert p_assocs[1]["article_id"] == "art-uuid-2"

    # Verify fallback tables were NOT used
    assert "events" not in mock_client.tables or mock_client.tables["events"]._payload is None


def test_persist_events_atomic_rpc_skips_unmatched_articles_before_rpc():
    """Verify events without resolvable article IDs are skipped before invoking RPC."""
    mock_client = MockRpcSupabaseClient()
    mock_client.tables["articles"] = MockPostgrestBuilder(data=[])
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Feed",
        title="Unmatched",
        url="https://feed.com/unmatched",
        canonical_url="https://feed.com/unmatched",
        published_at=now,
        content_hash="h-unmatched",
    )
    event = EventCluster(
        id="ev-orphan-rpc",
        title="Orphan Candidate",
        category=TopicCategory.OTHER_GEOPOLITICAL,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    total, persisted = repo.persist_events([event], article_url_to_id={})
    assert total == 1
    assert persisted == 0
    # RPC must NOT be called for events with zero resolvable articles
    assert len(mock_client.rpc_calls) == 0


def test_persist_events_atomic_rpc_association_failure_surfaced_safely():
    """Verify RPC failure (e.g. FK constraint or DB error) is surfaced without fallback write."""
    mock_client = MockRpcSupabaseClient()
    mock_client.rpc_side_effect = Exception("foreign key violation: article_id does not exist")
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Feed",
        title="Valid Article",
        url="https://feed.com/valid",
        canonical_url="https://feed.com/valid",
        published_at=now,
        content_hash="h-valid",
    )
    event = EventCluster(
        id="ev-fail-1",
        title="Failing Assoc Event",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    with pytest.raises(RuntimeError, match="Failed to persist events atomically via RPC"):
        repo.persist_events(
            [event],
            article_url_to_id={"https://feed.com/valid": "art-uuid-valid"},
        )

    # Ensure no fallback writes were attempted to events table
    assert "events" not in mock_client.tables or mock_client.tables["events"]._payload is None


def test_persist_events_atomic_rpc_repeated_execution_idempotent():
    """Verify repeated persistence calls with same event and articles remain idempotent."""
    mock_client = MockRpcSupabaseClient()
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Idempotent Article",
        url="https://defensenews.com/idemp",
        canonical_url="https://defensenews.com/idemp",
        published_at=now,
        content_hash="h-idemp",
    )
    event = EventCluster(
        id="ev-idemp-1",
        title="Idempotent Event",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    mapping = {"https://defensenews.com/idemp": "art-uuid-idemp"}

    total1, p1 = repo.persist_events([event], article_url_to_id=mapping)
    total2, p2 = repo.persist_events([event], article_url_to_id=mapping)

    assert total1 == 1 and p1 == 1
    assert total2 == 1 and p2 == 1
    assert len(mock_client.rpc_calls) == 2
    # Verify both calls passed identical parameters
    assert mock_client.rpc_calls[0][1] == mock_client.rpc_calls[1][1]


def test_persist_events_atomic_rpc_multi_event_batch_atomic_rollback():
    """Verify a multi-event batch failure rolls back completely without committing partial data."""
    mock_client = MockRpcSupabaseClient()
    # Simulate a transaction rollback halfway through batch
    mock_client.rpc_side_effect = Exception("transaction aborted: constraint failure on batch")
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

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
    ev1 = EventCluster(
        id="ev-batch-1",
        title="Batch Event 1",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art1,),
    )
    ev2 = EventCluster(
        id="ev-batch-2",
        title="Batch Event 2",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art2,),
    )

    mapping = {"https://s1.com/1": "art-1", "https://s2.com/2": "art-2"}

    with pytest.raises(RuntimeError, match="Failed to persist events atomically via RPC"):
        repo.persist_events([ev1, ev2], article_url_to_id=mapping)

    # Ensure no partial sequential fallback was executed
    assert "events" not in mock_client.tables or mock_client.tables["events"]._payload is None


def test_persist_events_missing_rpc_procedure_raises_actionable_error_zero_writes(caplog):
    """Verify that when the stored function is missing, error is raised with 0 writes."""
    import logging

    mock_client = MockRpcSupabaseClient()
    # Simulate PostgREST 404 / PGRST202 schema cache error
    mock_client.rpc_side_effect = Exception(
        '{"code": "PGRST202", "message": '
        '"Could not find the function public.upsert_events_with_articles"}'
    )
    mock_client.tables["events"] = MockPostgrestBuilder(data=[])
    mock_client.tables["event_articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Missing RPC Article",
        url="https://defensenews.com/missing",
        canonical_url="https://defensenews.com/missing",
        published_at=now,
        content_hash="h-missing",
    )
    event = EventCluster(
        id="ev-missing-1",
        title="Missing RPC Event",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(
            RuntimeError,
            match="Atomic event persistence is unavailable: stored procedure "
            "'upsert_events_with_articles' not found in database",
        ),
    ):
        repo.persist_events(
            [event],
            article_url_to_id={"https://defensenews.com/missing": "art-uuid-missing"},
        )

    assert "006_atomic_event_persistence.sql" in caplog.text
    assert "Sequential fallback writes are disabled" in caplog.text

    # Verify zero writes were attempted to events or event_articles
    assert mock_client.tables["events"]._payload is None
    assert mock_client.tables["event_articles"]._payload is None


def test_persist_events_ambiguous_rpc_timeout_not_mistaken_for_missing_procedure_zero_writes():
    """Verify RPC timeout/ambiguous error is not treated as missing RPC and performs zero writes."""
    mock_client = MockRpcSupabaseClient()
    mock_client.rpc_side_effect = Exception(
        "504 Gateway Timeout: connection timed out while awaiting database response"
    )
    mock_client.tables["events"] = MockPostgrestBuilder(data=[])
    mock_client.tables["event_articles"] = MockPostgrestBuilder(data=[])

    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Timeout Article",
        url="https://defensenews.com/timeout",
        canonical_url="https://defensenews.com/timeout",
        published_at=now,
        content_hash="h-timeout",
    )
    event = EventCluster(
        id="ev-timeout-1",
        title="Timeout Event",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )

    with pytest.raises(
        RuntimeError,
        match="Atomic event persistence RPC timed out or connection was lost: 504 Gateway Timeout",
    ) as exc_info:
        repo.persist_events(
            [event],
            article_url_to_id={"https://defensenews.com/timeout": "art-uuid-timeout"},
        )

    # Confirm it does NOT claim zero records were written and explicitly notes unknown status
    err_msg = str(exc_info.value)
    assert "database commit status is unknown" in err_msg
    assert "Do NOT perform manual partial writes" in err_msg
    assert "upsert_events_with_articles uses deterministic event IDs" in err_msg

    # Ensure zero fallback writes were attempted
    assert mock_client.tables["events"]._payload is None
    assert mock_client.tables["event_articles"]._payload is None


def test_persist_events_atomic_rpc_deduplicates_duplicate_events_in_batch():
    """Verify duplicate EventCluster instances within the same batch are deduplicated."""
    mock_client = MockRpcSupabaseClient()
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art = ArticleRecord(
        source_name="Defense News",
        title="Dedup Article",
        url="https://defensenews.com/dedup",
        canonical_url="https://defensenews.com/dedup",
        published_at=now,
        content_hash="h-dedup",
    )
    event = EventCluster(
        id="ev-dup-1",
        title="Duplicate Event",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    mapping = {"https://defensenews.com/dedup": "art-uuid-dedup"}

    total, persisted = repo.persist_events([event, event], article_url_to_id=mapping)
    assert total == 2
    assert persisted == 1

    assert len(mock_client.rpc_calls) == 1
    params = mock_client.rpc_calls[0][1]
    assert len(params["p_events"]) == 1
    assert params["p_events"][0]["id"] == "ev-dup-1"


def test_persist_events_conflicting_duplicate_event_ids_rejected():
    """Verify duplicate event IDs with conflicting titles/categories raise ValueError."""
    mock_client = MockRpcSupabaseClient()
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art1 = ArticleRecord(
        source_name="Defense News",
        title="Article 1",
        url="https://defensenews.com/1",
        canonical_url="https://defensenews.com/1",
        published_at=now,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="Article 2",
        url="https://breakingdefense.com/2",
        canonical_url="https://breakingdefense.com/2",
        published_at=now,
        content_hash="h2",
    )
    ev1 = EventCluster(
        id="ev-conflict-id",
        title="First Title",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1,),
    )
    ev2 = EventCluster(
        id="ev-conflict-id",
        title="Conflicting Title",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art2,),
    )
    mapping = {"https://defensenews.com/1": "art-1", "https://breakingdefense.com/2": "art-2"}

    with pytest.raises(ValueError, match="Conflicting duplicate event ID 'ev-conflict-id'"):
        repo.persist_events([ev1, ev2], article_url_to_id=mapping)

    # RPC must NOT be called on conflicting input
    assert len(mock_client.rpc_calls) == 0


def test_persist_events_identical_duplicate_event_merges_articles():
    """Verify identical duplicate events with different articles merge associations."""
    mock_client = MockRpcSupabaseClient()
    repo = SupabaseEventRepository(client=mock_client)
    now = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)

    art1 = ArticleRecord(
        source_name="Defense News",
        title="Article 1",
        url="https://defensenews.com/1",
        canonical_url="https://defensenews.com/1",
        published_at=now,
        content_hash="h1",
    )
    art2 = ArticleRecord(
        source_name="Breaking Defense",
        title="Article 2",
        url="https://breakingdefense.com/2",
        canonical_url="https://breakingdefense.com/2",
        published_at=now,
        content_hash="h2",
    )
    ev1 = EventCluster(
        id="ev-merge-id",
        title="Common Title",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1,),
    )
    ev2 = EventCluster(
        id="ev-merge-id",
        title="Common Title",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art2,),
    )
    mapping = {"https://defensenews.com/1": "art-1", "https://breakingdefense.com/2": "art-2"}

    total, persisted = repo.persist_events([ev1, ev2], article_url_to_id=mapping)
    assert total == 2
    assert persisted == 1

    assert len(mock_client.rpc_calls) == 1
    params = mock_client.rpc_calls[0][1]
    assert len(params["p_events"]) == 1
    # Both associations merged into the single event
    assert len(params["p_event_articles"]) == 2
    art_ids = {a["article_id"] for a in params["p_event_articles"]}
    assert art_ids == {"art-1", "art-2"}


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
    """Verify updating an event with new articles appends associations via atomic RPC."""
    mock_client = MockRpcSupabaseClient()
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
    assert len(mock_client.rpc_calls) == 1
    call1_params = mock_client.rpc_calls[0][1]
    assert len(call1_params["p_event_articles"]) == 1
    assert call1_params["p_event_articles"][0]["article_id"] == "art-uuid-1"

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
    assert len(mock_client.rpc_calls) == 2
    call2_params = mock_client.rpc_calls[1][1]
    assoc_payloads = call2_params["p_event_articles"]
    assert len(assoc_payloads) == 2
    article_ids = {row["article_id"] for row in assoc_payloads}
    assert article_ids == {"art-uuid-1", "art-uuid-2"}
    # Asserts that the atomic RPC handles idempotent conflicts via migration 006


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
