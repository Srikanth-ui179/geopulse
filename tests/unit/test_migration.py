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


def test_draft_migration_007_exists_and_is_valid():
    """Verify migration 007 file structure, table, columns, and indexes."""
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "007_article_revisions.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "create table if not exists public.article_revisions" in content
    assert "article_id uuid not null references public.articles(id) on delete restrict" in content
    assert "unique (article_id, revision_sequence)" in content
    assert (
        "previous_revision_id uuid references public.article_revisions(id) on delete set null"
        in content
    )
    assert (
        "current_revision_id uuid references public.article_revisions(id) on delete set null"
        in content
    )
    assert "append_article_revision" in content
    assert "enforce_article_revision_immutability" in content

    # Check balanced parentheses and dollar quotes
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")
    assert content.count("$$") % 2 == 0


def test_migration_007_security_and_immutability_grants():
    """Verify migration 007 enforces RLS, function revocation, and immutability trigger."""
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "007_article_revisions.sql"
    )
    content = migration_path.read_text(encoding="utf-8").lower()

    # 1. SECURITY DEFINER with pinned search_path
    assert "security definer" in content
    assert "set search_path = public, pg_temp" in content

    # 2. Execution revoked from PUBLIC, anon, authenticated
    assert re.search(
        r"revoke\s+execute\s+on\s+function\s+public\.append_article_revision\s*\([^\)]*\)\s+from\s+public",
        content,
    )
    assert re.search(
        r"revoke\s+execute\s+on\s+function\s+public\.append_article_revision\s*\([^\)]*\)\s+from\s+anon\s*,\s*authenticated",
        content,
    )

    # 3. Execution granted strictly to service_role
    assert re.search(
        r"grant\s+execute\s+on\s+function\s+public\.append_article_revision\s*\([^\)]*\)\s+to\s+service_role",
        content,
    )

    # 4. Row Level Security enabled
    assert "alter table public.article_revisions enable row level security;" in content

    # 5. Table privileges revoked from untrusted roles
    assert (
        "revoke all on table public.article_revisions from public, anon, authenticated;" in content
    )

    # 6. Read policy for authenticated users and table grant
    assert "create policy article_revisions_select_policy" in content
    assert "grant select on table public.article_revisions to authenticated;" in content

    # 7. Immutability trigger defined
    assert "create or replace trigger trg_enforce_article_revision_immutability" in content


def test_migration_007_rollback_exists_and_is_valid():
    """Verify rollback migration 007 drops all added objects cleanly."""
    rollback_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "007_article_revisions_rollback.sql"
    )
    assert rollback_path.is_file()
    content = rollback_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0

    assert "drop trigger if exists trg_enforce_article_revision_immutability" in content
    assert "drop function if exists public.enforce_article_revision_immutability" in content
    assert (
        "drop policy if exists article_revisions_select_policy on public.article_revisions"
        in content
    )
    assert "drop function if exists public.append_article_revision" in content
    assert "drop table if exists public.article_revisions cascade" in content

    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")


def test_draft_migration_008_exists_and_is_valid():
    """Verify migration 008 file structure, syntax, and permission hardening."""
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "008_article_revisions_permissions.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Table-level SELECT revoked from authenticated
    assert "revoke select on table public.article_revisions from authenticated;" in content

    # 3. Column-level SELECT granted strictly on safe metadata columns
    assert "grant select (" in content
    expected_safe_cols = [
        "id",
        "article_id",
        "canonical_url",
        "content_hash",
        "title",
        "publisher",
        "published_at",
        "first_seen_at",
        "retrieved_at",
        "last_modified",
        "etag",
        "extraction_status",
        "extraction_method",
        "revision_sequence",
        "previous_revision_id",
        "revision_classification",
        "created_at",
    ]
    for col in expected_safe_cols:
        assert col in content, f"Expected column {col} in migration 008 grant list"

    # 4. Sensitive columns strictly excluded from grants
    grant_match = re.search(
        r"grant\s+select\s*\(([^\)]+)\)\s*on\s+table\s+public\.article_revisions\s+to\s+authenticated",
        content,
    )
    assert grant_match is not None, "Expected grant select (...) block in migration 008"
    granted_cols = [c.strip() for c in grant_match.group(1).split(",")]
    assert "body_text" not in granted_cols, "body_text MUST NOT be in granted columns"
    assert "relevant_metadata" not in granted_cols, (
        "relevant_metadata MUST NOT be in granted columns"
    )
    assert len(granted_cols) == 17, f"Expected exactly 17 granted columns, got {len(granted_cols)}"

    # 5. Service role permissions preserved
    assert (
        "grant select, insert, update on table public.article_revisions to service_role;" in content
    )


def test_migration_008_rollback_exists_and_is_valid():
    """Verify rollback migration 008 cleanly restores migration 007 permissions state."""
    rollback_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "008_article_revisions_permissions_rollback.sql"
    )
    assert rollback_path.is_file()
    content = rollback_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Clears authenticated column/table permissions before restoring
    assert "revoke all on table public.article_revisions from authenticated;" in content

    # 3. Restores table-level SELECT for authenticated matching 007
    assert "grant select on table public.article_revisions to authenticated;" in content

    # 4. Service role permissions reaffirmed
    assert (
        "grant select, insert, update on table public.article_revisions to service_role;" in content
    )


def test_draft_migration_009_exists_and_is_valid():
    """Verify migration 009 file structure, table alterations, and indexes."""
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "009_source_corroboration.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Add corroboration fields to events and briefing_events
    assert "add column if not exists corroboration_status text" in content
    assert "add column if not exists independent_chains_count integer not null default 0" in content
    assert (
        "add column if not exists corroboration_detail jsonb not null default '{}'::jsonb"
        in content
    )

    # 3. Add lineage fields to briefing_event_evidence
    assert "add column if not exists lineage_type text" in content
    assert "add column if not exists original_source text" in content

    # 4. Lookup indexes
    assert "create index if not exists events_corroboration_status_idx" in content
    assert "create index if not exists briefing_events_corroboration_status_idx" in content
    assert "create index if not exists briefing_evidence_lineage_idx" in content


def test_migration_009_rollback_exists_and_is_valid():
    """Verify rollback migration 009 cleanly drops all added columns and indexes."""
    rollback_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "009_source_corroboration_rollback.sql"
    )
    assert rollback_path.is_file()
    content = rollback_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Drop indexes
    assert "drop index if exists public.briefing_evidence_lineage_idx;" in content
    assert "drop index if exists public.briefing_events_corroboration_status_idx;" in content
    assert "drop index if exists public.events_corroboration_status_idx;" in content

    # 3. Drop columns
    assert "drop column if exists lineage_type" in content
    assert "drop column if exists original_source" in content
    assert "drop column if exists corroboration_status" in content
    assert "drop column if exists independent_chains_count" in content
    assert "drop column if exists corroboration_detail" in content


def test_draft_migration_010_exists_and_is_valid():
    """Verify migration 010 file structure, columns, constraints, and indexes."""
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "010_classification_and_corroboration.sql"
    )
    assert migration_path.is_file()
    content = migration_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Category updates
    assert "intelligence_espionage" in content
    assert "military_technology" in content

    # 3. Evidence and persistent event columns
    assert "add column if not exists is_update boolean" in content
    assert "add column if not exists persistent_event_id text" in content
    assert "add column if not exists semantic_signature text" in content
    assert "add column if not exists evidence_label text" in content
    assert "add column if not exists origin_type text" in content

    # 4. Indexes
    assert "create index if not exists events_semantic_signature_idx" in content
    assert "create index if not exists events_persistent_id_idx" in content


def test_migration_010_rollback_exists_and_is_valid():
    """Verify rollback migration 010 cleanly drops all added columns and indexes."""
    rollback_path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "010_classification_and_corroboration_rollback.sql"
    )
    assert rollback_path.is_file()
    content = rollback_path.read_text(encoding="utf-8").lower()
    assert len(content.strip()) > 0

    # 1. Balanced parentheses
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    cleaned = "\n".join(lines)
    assert cleaned.count("(") == cleaned.count(")")

    # 2. Drop indexes
    assert "drop index if exists events_semantic_signature_idx;" in content
    assert "drop index if exists events_persistent_id_idx;" in content

    # 3. Drop columns
    assert "drop column if exists is_update" in content
    assert "drop column if exists persistent_event_id" in content
    assert "drop column if exists semantic_signature" in content
    assert "drop column if exists evidence_label" in content
