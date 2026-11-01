"""Unit tests verifying the PostgreSQL migration schema."""

import re
from pathlib import Path

MIGRATION_PATH = Path(__file__).resolve().parents[2] / "migrations" / "001_initial_schema.sql"


def test_migration_file_exists_and_is_not_empty():
    assert MIGRATION_PATH.is_file()
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0


def test_migration_declares_all_core_tables():
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    expected_tables = [
        "public.sources",
        "public.articles",
        "public.events",
        "public.event_articles",
        "public.briefings",
        "public.briefing_events",
        "public.briefing_event_evidence",
        "public.pipeline_runs",
    ]
    for table in expected_tables:
        pattern = rf"create\s+table\s+if\s+not\s+exists\s+{re.escape(table)}"
        assert re.search(pattern, content, re.IGNORECASE), f"Missing table definition: {table}"


def test_migration_has_balanced_parentheses_and_valid_structure():
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    # Strip comments
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)

    open_parens = cleaned.count("(")
    close_parens = cleaned.count(")")
    assert open_parens == close_parens, (
        f"Unbalanced parentheses: {open_parens} open vs {close_parens} close"
    )


def test_migration_enforces_cascade_deletes_on_evidence_and_associations():
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    # Verify briefing_event_evidence cascades on article_id
    assert re.search(
        r"article_id\s+uuid\s+not\s+null\s+references\s+public\.articles\(id\)\s+on\s+delete\s+cascade",
        content,
        re.IGNORECASE,
    ), "briefing_event_evidence.article_id must specify on delete cascade"

    # Verify event_articles cascades on both foreign keys
    assert re.search(
        r"event_id\s+uuid\s+not\s+null\s+references\s+public\.events\(id\)\s+on\s+delete\s+cascade",
        content,
        re.IGNORECASE,
    )
    assert re.search(
        r"article_id\s+uuid\s+not\s+null\s+references\s+public\.articles\(id\)\s+on\s+delete\s+cascade",
        content,
        re.IGNORECASE,
    )


def test_migration_defines_foreign_key_and_lookup_indexes():
    content = MIGRATION_PATH.read_text(encoding="utf-8")
    expected_indexes = [
        "articles_published_idx",
        "articles_source_published_idx",
        "articles_relevance_idx",
        "events_last_seen_idx",
        "event_articles_article_id_idx",
        "briefing_events_event_id_idx",
        "briefing_event_evidence_article_id_idx",
        "pipeline_runs_started_idx",
    ]
    for idx in expected_indexes:
        pattern = rf"create\s+index\s+if\s+not\s+exists\s+{re.escape(idx)}"
        assert re.search(pattern, content, re.IGNORECASE), f"Missing index: {idx}"


def test_draft_migration_003_exists_and_is_valid():
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "003_telegram_delivery_state.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "telegram_chat_id" in content
    assert "telegram_message_ids" in content
    assert "digest_hash" in content
    assert "chunks_delivered" in content

    # Check balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")


def test_draft_migration_004_exists_and_is_valid():
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "004_event_metadata_and_briefing_status.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "published_at" in content
    assert "is_analysis" in content
    assert "ambiguous" in content
    assert "briefings_status_check" in content

    # Check balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")


def test_draft_migration_005_exists_and_is_valid():
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "005_structured_summaries.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "headline" in content
    assert "key_details" in content
    assert "confidence_qualification" in content
    assert "model_used" in content

    # Check balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")


def test_draft_migration_006_atomic_event_persistence_exists_and_is_valid():
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "006_atomic_event_persistence.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "create or replace function public.upsert_events_with_articles" in content
    assert "p_events jsonb" in content
    assert "p_event_articles jsonb" in content
    assert "returns jsonb" in content
    assert "insert into public.events" in content
    assert "insert into public.event_articles" in content
    assert "on conflict (id) do update set" in content
    assert "on conflict (event_id, article_id) do nothing" in content
    assert "Orphan event rejected" in content
    assert "distinct on" in content
    assert "Malformed event payload" in content
    assert "Malformed association payload" in content
    assert "Unmatched association" in content
    assert "Invalid timestamps" in content
    assert "Duplicate event ID in batch" in content
    assert "grant execute on function public.upsert_events_with_articles" in content

    # Check balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # Check balanced dollar quotes for function body
    assert content.count("$$") % 2 == 0


def test_migration_006_uuid_regex_behavior():
    """Verify canonical UUID validation regex in migration 006 against edge cases."""
    import uuid

    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "006_atomic_event_persistence.sql"
    )
    content = migration_path.read_text(encoding="utf-8")
    # Extract the regex pattern from SQL
    pattern_match = re.search(r"\^\[0-9a-f\]\{8\}-\[0-9a-f\]\{4\}[^\']+", content)
    assert pattern_match is not None, "UUID regex pattern missing in migration 006"
    uuid_pattern = re.compile(pattern_match.group(0), re.IGNORECASE)

    # 1. Valid UUIDs
    v4 = str(uuid.uuid4())
    v5 = str(uuid.uuid5(uuid.NAMESPACE_URL, "geopulse:event:test_url"))
    assert uuid_pattern.match(v4)
    assert uuid_pattern.match(v5)
    assert uuid_pattern.match(v4.upper())

    # 2. Misplaced hyphens
    assert not uuid_pattern.match("1234567-8-1234-1234-1234-123456789012")
    assert not uuid_pattern.match("123456781234-1234-1234-123456789012")
    assert not uuid_pattern.match("12345678-123-41234-1234-123456789012")
    assert not uuid_pattern.match("12345678-1234-1234-1234123456789012-")

    # 3. Invalid characters
    assert not uuid_pattern.match("12345678-1234-1234-1234-12345678901g")
    assert not uuid_pattern.match("12345678-1234-1234-1234-12345678901z")
    assert not uuid_pattern.match("12345678-1234-1234-1234-12345678901!")
    assert not uuid_pattern.match("12345678-1234-1234-1234-12345678901 ")

    # 4. Invalid lengths
    assert not uuid_pattern.match("12345678-1234-1234-1234-12345678901")  # 35 chars
    assert not uuid_pattern.match("12345678-1234-1234-1234-1234567890123")  # 37 chars
    assert not uuid_pattern.match("")


def test_migration_006_least_privilege_grants_and_security_settings():
    """Verify static SQL text enforces least privilege, search_path, and security invoker.

    Note: This is a static test of migration source code and does not prove
    effective live-database permissions, which require PostgreSQL execution.
    """
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "006_atomic_event_persistence.sql"
    )
    content = migration_path.read_text(encoding="utf-8").lower()

    # 1. SECURITY INVOKER
    assert "security invoker" in content

    # 2. Pinned search_path
    assert "set search_path = public, pg_temp" in content

    # 3. Explicit revocation from PUBLIC
    assert re.search(
        r"revoke\s+execute\s+on\s+function\s+public\.upsert_events_with_articles\s*\([^\)]*\)\s+from\s+public",
        content,
    )

    # 4. Explicit revocation from anon and authenticated
    assert re.search(
        r"revoke\s+execute\s+on\s+function\s+public\.upsert_events_with_articles\s*\([^\)]*\)\s+from\s+anon\s*,\s*authenticated",
        content,
    )

    # 5. Execution granted strictly to service_role
    assert re.search(
        r"grant\s+execute\s+on\s+function\s+public\.upsert_events_with_articles\s*\([^\)]*\)\s+to\s+service_role",
        content,
    )

    # 6. Ensure no subsequent statements grant execution to authenticated, anon, or public
    assert not re.search(
        r"grant\s+execute\s+on\s+function\s+public\.upsert_events_with_articles[^\;]+to\s+(authenticated|anon|public)",
        content,
    )
