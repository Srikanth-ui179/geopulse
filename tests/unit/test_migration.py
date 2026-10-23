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

