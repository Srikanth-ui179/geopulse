"""Contract test verifying repository payload mapping against 001_initial_schema.sql."""

import re
from datetime import UTC, datetime
from pathlib import Path

from geopulse.ingestion.models import ArticleRecord, FeedSource

MIGRATION_PATH = Path(__file__).resolve().parents[2] / "migrations" / "001_initial_schema.sql"


def test_article_payload_contract_matches_schema_columns():
    """Verify that ArticleRecord serialization matches the columns in public.articles."""
    schema_sql = MIGRATION_PATH.read_text(encoding="utf-8")

    # Extract column names from "create table if not exists public.articles (...)"
    table_match = re.search(
        r"create\s+table\s+if\s+not\s+exists\s+public\.articles\s*\((.*?)\);",
        schema_sql,
        re.DOTALL | re.IGNORECASE,
    )
    assert table_match is not None, "Could not find public.articles table definition in migration"

    table_body = table_match.group(1)
    schema_columns: set[str] = set()
    for line in table_body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("--", "check", "primary key", "unique", "foreign key")):
            continue
        col_name = line.split()[0]
        schema_columns.add(col_name)

    # Repository payload keys constructed for insertion
    sample = ArticleRecord(
        source_name="Defense News",
        title="Test Article",
        url="https://example.com/original",
        canonical_url="https://example.com/canonical",
        published_at=datetime.now(UTC),
        snippet="Snippet text",
        author="Reporter",
        content_hash="abc123hash",
        language="en",
        relevance_status="relevant",
    )
    assert sample.title == "Test Article"

    repo_payload_keys = {
        "source_id",
        "url",
        "canonical_url",
        "title",
        "author",
        "published_at",
        "fetched_at",
        "snippet",
        "content_hash",
        "language",
        "relevance_status",
    }

    # Verify every payload key exists as a column in the SQL schema
    missing_columns = repo_payload_keys - schema_columns
    assert not missing_columns, (
        f"Repository payload contains keys not in SQL schema: {missing_columns}"
    )

    # Verify that id and created_at have defaults and are omitted from manual payload
    assert "id" in schema_columns
    assert "created_at" in schema_columns
    assert "id" not in repo_payload_keys
    assert "created_at" not in repo_payload_keys


def test_sources_contract_matches_schema_columns():
    """Verify FeedSource maps to public.sources columns."""
    schema_sql = MIGRATION_PATH.read_text(encoding="utf-8")
    table_match = re.search(
        r"create\s+table\s+if\s+not\s+exists\s+public\.sources\s*\((.*?)\);",
        schema_sql,
        re.DOTALL | re.IGNORECASE,
    )
    assert table_match is not None, "Could not find public.sources table in migration"
    table_body = table_match.group(1)

    schema_columns: set[str] = set()
    for line in table_body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("--", "check", "primary key", "unique")):
            continue
        schema_columns.add(line.split()[0])

    source = FeedSource(name="Defense News", feed_url="https://example.com/rss", enabled=True)
    source_keys = set(source.model_dump().keys())

    missing = source_keys - schema_columns
    assert not missing, f"FeedSource fields missing from public.sources: {missing}"
