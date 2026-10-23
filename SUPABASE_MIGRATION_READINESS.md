# Geopulse Supabase Migration Readiness Review (Phase 6I)

**Audit Date:** 2026-10-24  
**Project Workspace:** `Geopulse` (Repository Root)  
**Target Environment:** PostgreSQL 15+ (Supabase)  
**Verification Scope:** Migrations `001` through `005`, Repository Layer, Pipeline Execution Semantics  
**Test Suite Baseline:** 209 passing unit tests (0 failures, 0 warnings, clean Ruff)  

---

## 1. Executive Summary & Readiness Verdict

### Final Verdict: **READY WITH CONDITIONS**

The database schema defined across migrations `001_initial_schema.sql` through `005_structured_summaries.sql` correctly aligns with the application code (`SupabaseArticleRepository`, `SupabaseEventRepository`, `SupabaseDeliveryRepository`, and `BriefingPipeline`). However, calling the migration suite unconditionally "idempotent" is technically inaccurate: individual statements have strict ordering dependencies, lack transactional wrapping, and can leave a partially migrated schema if executed without transactional safeguards.

Before executing migrations or running persistent digests, four mandatory conditions must be satisfied:

1. **Condition 1 (Strict Application Order & Inclusivity):**  
   All five migrations (`001` ➔ `002` ➔ `003` ➔ `004` ➔ `005`) **must be applied in full**. Migration `005_structured_summaries.sql` is **mandatory**; the application code already constructs payloads referencing `headline`, `key_details` (JSONB), `confidence_qualification`, and `model_used`. Omission will cause PostgREST schema mismatch errors (HTTP 400).
2. **Condition 2 (Transactional Wrapping per Script):**  
   Because the `.sql` migration files do not contain explicit `BEGIN; ... COMMIT;` blocks, each script should be wrapped in a transaction block in the Supabase SQL Editor to prevent partial statement commits on unexpected network or query errors.
3. **Condition 3 (Service Role Authentication):**  
   The `.env` configuration for `SUPABASE_KEY` must use the secret `service_role` key (which bypasses PostgreSQL Row Level Security for backend pipeline writes), never the public `anon` key.
4. **Condition 4 (Deep Diagnostic Verification Prior to Persistent Runs):**  
   The operator must execute the comprehensive diagnostic script in Section 7 to verify not just object names, but expected data types, primary keys, foreign key cascades, unique constraints, check constraints, and indexes.

---

## 2. Statement-by-Statement Audit: Idempotency, Rerun Safety & Failure Modes

A strict statement-by-statement review reveals that while many DDL statements use `IF NOT EXISTS`, the migrations as a whole are **not unconditionally idempotent**:

| Migration | Statement / Clause | Rerun Safety on Clean / Existing DB | Failure Mode / Partial Application Risk |
| :--- | :--- | :---: | :--- |
| **001** | `CREATE EXTENSION IF NOT EXISTS pgcrypto;` | **Safe** | Native in Supabase; succeeds if already loaded. |
| **001** | `CREATE TABLE IF NOT EXISTS public.<tables>` (8 tables) | **Conditionally Safe** | Safe if rerun on a database where tables were created by this script. **Risk:** If a table already existed with conflicting column types, `IF NOT EXISTS` silently ignores creation, masking schema drift. |
| **001** | `CREATE INDEX IF NOT EXISTS <indexes>` (8 indexes) | **Safe** | Idempotent on existing tables. |
| **002** | `ALTER TABLE IF EXISTS public.articles ADD COLUMN IF NOT EXISTS topic ...` | **Conditionally Safe** | If run out of order (before `001`), silently skips table modification with `NOTICE: relation "public.articles" does not exist`. |
| **002** | `CREATE INDEX IF NOT EXISTS articles_topic_idx ...` | **Ordering Dependency** | **FAILS with fatal error** (`relation "public.articles" does not exist`) if applied before `001`. |
| **002** | `CREATE INDEX IF NOT EXISTS events_category_first_seen_idx ...` | **Ordering Dependency** | **FAILS with fatal error** (`relation "public.events" does not exist`) if applied before `001`. |
| **002** | `CREATE INDEX IF NOT EXISTS events_confidence_label_idx ...` | **Ordering Dependency** | **FAILS with fatal error** (`relation "public.events" does not exist`) if applied before `001`. |
| **003** | `ALTER TABLE IF EXISTS public.briefings ADD COLUMN IF NOT EXISTS ...` | **Conditionally Safe** | Safe rerun on existing table. Silently skips if `briefings` does not exist. |
| **003** | `CREATE INDEX IF NOT EXISTS briefings_digest_hash_idx / chat_id_idx` | **Ordering Dependency** | **FAILS with fatal error** if applied before `001`. |
| **004** | `ALTER TABLE IF EXISTS public.events ADD COLUMN IF NOT EXISTS published_at, is_analysis` | **Safe on existing DB** | Non-breaking column additions with default values. |
| **004** | `DROP CONSTRAINT IF EXISTS briefings_status_check;` followed by `ADD CONSTRAINT briefings_status_check ...` | **PARTIAL FAILURE RISK** | **Not atomic without a transaction block.** If a network interruption occurs between `DROP` and `ADD`, or if existing rows violate the new check constraint, `ADD CONSTRAINT` fails and the table is left **without any status check constraint**. |
| **004** | `CREATE INDEX IF NOT EXISTS events_is_analysis_idx ...` | **Ordering Dependency** | **FAILS with fatal error** if applied before `001`. |
| **005** | `ALTER TABLE IF EXISTS public.events ADD COLUMN IF NOT EXISTS ...` | **Safe on existing DB** | All 4 added columns are nullable. Safe to rerun. |
| **005** | `ALTER TABLE IF EXISTS public.briefing_events ADD COLUMN IF NOT EXISTS ...` | **Safe on existing DB** | All 4 added columns are nullable. Safe to rerun. |

---

## 3. Credential Security & Secret Isolation Audit

A project-wide audit of credentials (`SUPABASE_KEY` / `SUPABASE_SERVICE_ROLE_KEY`) was performed across all source files, models, CLI modules, and tests:

1. **Backend Isolation:**  
   `supabase_key` is declared as `pydantic.SecretStr` in [`src/geopulse/config.py`](src/geopulse/config.py#L27). In Pydantic v2, `SecretStr` overrides `__str__` and `__repr__` to return `'**********'`, preventing accidental leakage in log formatters, traceback printouts, or debugger inspections.
2. **Read Access Control:**  
   The unmasked secret (`.get_secret_value()`) is accessed strictly inside [`Settings.require_supabase()`](src/geopulse/config.py#L108-L118) and passed directly to `create_client(url, key)` in repository constructors ([`SupabaseArticleRepository`](src/geopulse/ingestion/repository.py#L121), [`SupabaseEventRepository`](src/geopulse/clustering/repository.py#L132), [`SupabaseDeliveryRepository`](src/geopulse/delivery/repository.py#L458)).
3. **No Leakage in CLI, Logs, or Exceptions:**  
   - [`src/geopulse/cli.py`](src/geopulse/cli.py#L62) only prints `settings.is_supabase_configured` (a boolean `True`/`False`), never the token value.
   - Error messages in `cli.py` reference only the variable names (e.g. `"requires SUPABASE_URL and SUPABASE_KEY to be configured"`).
   - No frontend bundles, client-side templates, or public assets exist in the codebase that could expose backend secrets.
   - Sample configuration files ([`.env.example`](.env.example)) leave values empty.

---

## 4. Pipeline Execution Timing & Write-Order Vulnerability Analysis

A critical architectural analysis of the persistent pipeline (`geopulse brief --persist --send`) reveals an important distinction between credential checks and schema validation:

```
[CLI Startup]
  │
  ├── 1. Syntactic Flag Checks (Mutually exclusive flags: preview vs send, state flags) [PRE-FLIGHT]
  ├── 2. Env Var Presence Check (is_supabase_configured: checks non-empty strings) [PRE-FLIGHT]
  │
  ▼ [WRITES BEGIN]
  ├── 3. Ingestion Service (fetch_feed_data from RSS sources)
  ├── 4. Sources Table Write (get_or_create_source -> INSERT INTO public.sources) <--- 1st DB WRITE
  ├── 5. Articles Table Write (persist_articles -> UPSERT INTO public.articles)   <--- 2nd DB WRITE
  ├── 6. Canonical URL Lookup (get_article_ids_by_urls -> SELECT FROM articles)
  ├── 7. Clustering Service (temporal & semantic clustering in memory)
  ├── 8. Events Table Write (persist_events -> UPSERT INTO public.events)         <--- 3rd DB WRITE
  ├── 9. Event-Articles Join Write (persist_events -> UPSERT INTO event_articles) <--- 4th DB WRITE
  ├── 10. Ollama Summarization (EventSummarizer executes local inference)
  ├── 11. Event Summary Updates (update_event_summaries -> UPDATE public.events)  <--- 5th DB WRITE
  ├── 12. Telegram Delivery (deliver_digest -> UPSERT INTO public.briefings)       <--- 6th DB WRITE
  ├── 13. Telegram API Network Dispatch (TelegramClient.send_message per chunk)
  ├── 14. Delivery State Incremental Progress (UPDATE public.briefings)           <--- 7th DB WRITE
  └── 15. Briefing Events & Evidence (record_briefing_events -> UPSERT)           <--- 8th DB WRITE
```

### Key Findings & Timing Vulnerabilities:
1. **No Upfront Database Connectivity Check:**  
   The application validates that `SUPABASE_URL` and `SUPABASE_KEY` are populated strings, but **does not ping the database or check connectivity before initiating writes**. An invalid key or unapplied migration will only fail when the first database write occurs at Step 4 (`sources`).
2. **Late Schema Failures Result in Partial Pipeline Execution:**  
   - If Migration `001` was applied, but `004` was omitted (`published_at`, `is_analysis`), articles will be persisted to `sources` and `articles` before the pipeline crashes at Step 8 (`events`).
   - If Migration `005` was omitted, articles and events are persisted, and expensive Ollama LLM summarization completes, only to crash at Step 11 (`update_event_summaries`) due to missing columns (`headline`, `key_details`).
   - If Migration `005` is missing on `briefing_events`, the crash occurs at Step 15 **after Telegram messages have already been delivered to the recipient**.
3. **Mitigation Requirement:**  
   Operators **must never run `--persist` until the Section 7 diagnostic script confirms 100% schema readiness**.

---

## 5. Recovery Runbook: Partial Failures

### Scenario A: Partial Migration Failure in Supabase
If a migration fails midway through execution (e.g. timeout, connection drop, typo):
1. **Diagnose:** Execute the Diagnostic SQL in Section 7 to determine exactly which tables, columns, constraints, or indexes succeeded and which failed.
2. **Remediate:**
   - If `001` failed partway: Because all statements use `IF NOT EXISTS`, fix the underlying error and re-run the entire script. Alternatively, execute the clean rollback script in Section 8 and restart from `001`.
   - If `004` failed on constraint replacement: Verify whether existing rows violate the status check. Run:
     ```sql
     UPDATE public.briefings 
     SET status = 'failed' 
     WHERE status NOT IN ('pending', 'processing', 'sent', 'partial', 'failed', 'ambiguous');
     ```
     Then re-apply `migrations/004_event_metadata_and_briefing_status.sql`.
   - If `005` failed: Re-running is safe because all columns use `ADD COLUMN IF NOT EXISTS`.

### Scenario B: Partial Digest Delivery to Telegram
If a multi-chunk briefing fails midway through Telegram delivery (e.g. chunk 1 delivers, chunk 2 encounters a network timeout):
1. **Incremental State Guarantee:**  
   [`TelegramDeliveryService`](src/geopulse/delivery/service.py#L272-L289) updates `chunks_delivered` and appends the Telegram message ID to `telegram_message_ids` immediately after each individual chunk succeeds.
2. **Safe Resumption:**  
   Re-running the command:
   ```powershell
   .\.venv\Scripts\python.exe -m geopulse brief --persist --send
   ```
   The service detects `status = 'partial'`, verifies that the `digest_hash` matches (guaranteeing the digest content has not changed), skips Chunk 1, and delivers only Chunk 2.
3. **Handling Ambiguous Timeouts:**  
   If a read timeout occurred on Telegram API dispatch (uncertain whether Telegram posted the message), the briefing is marked `status = 'ambiguous'`. The service intentionally halts to prevent duplicate messages.  
   - **Step 1:** The operator checks the private Telegram chat to see if the chunk arrived.  
   - **Step 2:** If it did not arrive, pass `--force` to resume delivery:  
     `.\.venv\Scripts\python.exe -m geopulse brief --persist --send --force`  
   - **Step 3:** If it did arrive, manually resolve the state in Supabase:  
     `UPDATE public.briefings SET status = 'sent', completed_at = now() WHERE briefing_date = CURRENT_DATE;`

---

## 6. Exact Manual Application Sequence (Operator Runbook)

To apply migrations safely without exposing secrets:

### Step 1: Open Supabase Project
1. Log in to [supabase.com](https://supabase.com).
2. Select your project.
3. In the left navigation menu, open the **SQL Editor**.

### Step 2: Preflight Verification Query
Run the following SQL snippet to confirm the target database is clean:
```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('sources', 'articles', 'events', 'briefings');
```
*(Expected output: 0 rows on a fresh project).*

### Step 3: Apply Migrations in Transaction Blocks
Copy and run each migration script sequentially, wrapping each in a transaction block:

```sql
-- 1. Apply Migration 001
BEGIN;
\i migrations/001_initial_schema.sql  -- or paste file contents
COMMIT;

-- 2. Apply Migration 002
BEGIN;
\i migrations/002_clustering_and_topics.sql
COMMIT;

-- 3. Apply Migration 003
BEGIN;
\i migrations/003_telegram_delivery_state.sql
COMMIT;

-- 4. Apply Migration 004
BEGIN;
\i migrations/004_event_metadata_and_briefing_status.sql
COMMIT;

-- 5. Apply Migration 005
BEGIN;
\i migrations/005_structured_summaries.sql
COMMIT;
```

---

## 7. Deep Diagnostic Verification SQL

Execute the following comprehensive diagnostic script in the Supabase SQL Editor. It inspects table existence, exact column data types, JSONB structures, primary keys, foreign key cascade actions, unique constraints, check constraints, and performance indexes:

```sql
-- Comprehensive Geopulse Post-Migration Deep Diagnostic Health Check
WITH 
-- 1. Table Verification
expected_tables AS (
    SELECT unnest(ARRAY[
        'sources', 'articles', 'events', 'event_articles',
        'briefings', 'briefing_events', 'briefing_event_evidence', 'pipeline_runs'
    ]) AS table_name
),
actual_tables AS (
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
),
table_checks AS (
    SELECT 
        '1. Table Presence' AS check_category,
        et.table_name AS target_object,
        CASE WHEN at.table_name IS NOT NULL THEN 'PASS' ELSE 'FAIL (Missing Table)' END AS status
    FROM expected_tables et
    LEFT JOIN actual_tables at ON et.table_name = at.table_name
),

-- 2. Column Type & JSONB Verification
expected_columns AS (
    SELECT * FROM (VALUES
        ('articles', 'topic', 'text'),
        ('briefings', 'telegram_chat_id', 'text'),
        ('briefings', 'telegram_message_ids', 'jsonb'),
        ('briefings', 'digest_hash', 'text'),
        ('briefings', 'chunks_delivered', 'integer'),
        ('events', 'published_at', 'timestamp with time zone'),
        ('events', 'is_analysis', 'boolean'),
        ('events', 'headline', 'text'),
        ('events', 'key_details', 'jsonb'),
        ('events', 'confidence_qualification', 'text'),
        ('events', 'model_used', 'text'),
        ('briefing_events', 'headline', 'text'),
        ('briefing_events', 'key_details', 'jsonb'),
        ('briefing_events', 'confidence_qualification', 'text'),
        ('briefing_events', 'model_used', 'text')
    ) AS t(table_name, column_name, expected_type)
),
actual_columns AS (
    SELECT table_name, column_name, data_type, udt_name
    FROM information_schema.columns 
    WHERE table_schema = 'public'
),
column_checks AS (
    SELECT 
        '2. Column Data Type' AS check_category,
        ec.table_name || '.' || ec.column_name || ' (' || ec.expected_type || ')' AS target_object,
        CASE 
            WHEN ac.column_name IS NULL THEN 'FAIL (Missing Column)'
            WHEN ac.data_type = ec.expected_type OR ac.udt_name = ec.expected_type THEN 'PASS'
            ELSE 'FAIL (Type Mismatch: found ' || ac.data_type || ')'
        END AS status
    FROM expected_columns ec
    LEFT JOIN actual_columns ac ON ec.table_name = ac.table_name AND ec.column_name = ac.column_name
),

-- 3. Primary Key Verification
expected_pks AS (
    SELECT unnest(ARRAY[
        'sources', 'articles', 'events', 'event_articles',
        'briefings', 'briefing_events', 'briefing_event_evidence', 'pipeline_runs'
    ]) AS table_name
),
actual_pks AS (
    SELECT tc.table_name
    FROM information_schema.table_constraints tc
    WHERE tc.table_schema = 'public' AND tc.constraint_type = 'PRIMARY KEY'
),
pk_checks AS (
    SELECT 
        '3. Primary Key' AS check_category,
        ep.table_name || '_pkey' AS target_object,
        CASE WHEN ap.table_name IS NOT NULL THEN 'PASS' ELSE 'FAIL (Missing PK)' END AS status
    FROM expected_pks ep
    LEFT JOIN actual_pks ap ON ep.table_name = ap.table_name
),

-- 4. Foreign Key Cascade Verification
expected_fks AS (
    SELECT * FROM (VALUES
        ('articles', 'sources', 'CASCADE'),
        ('event_articles', 'events', 'CASCADE'),
        ('event_articles', 'articles', 'CASCADE'),
        ('briefing_events', 'briefings', 'CASCADE'),
        ('briefing_events', 'events', 'CASCADE'),
        ('briefing_event_evidence', 'briefing_events', 'CASCADE'),
        ('briefing_event_evidence', 'articles', 'CASCADE')
    ) AS t(child_table, parent_table, expected_action)
),
actual_fks AS (
    SELECT 
        tc.table_name AS child_table,
        ccu.table_name AS parent_table,
        rc.delete_rule AS actual_action
    FROM information_schema.table_constraints tc
    JOIN information_schema.referential_constraints rc ON tc.constraint_name = rc.constraint_name
    JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
    WHERE tc.table_schema = 'public' AND tc.constraint_type = 'FOREIGN KEY'
),
fk_checks AS (
    SELECT 
        '4. Foreign Key Cascade' AS check_category,
        ef.child_table || ' -> ' || ef.parent_table AS target_object,
        CASE 
            WHEN af.child_table IS NULL THEN 'FAIL (Missing FK)'
            WHEN af.actual_action = ef.expected_action THEN 'PASS'
            ELSE 'FAIL (Delete Rule: ' || af.actual_action || ')'
        END AS status
    FROM expected_fks ef
    LEFT JOIN actual_fks af ON ef.child_table = af.child_table AND ef.parent_table = af.parent_table
),

-- 5. Unique Constraint Verification
expected_uniques AS (
    SELECT * FROM (VALUES
        ('sources', 'sources_feed_url_key'),
        ('articles', 'articles_canonical_url_key'),
        ('briefings', 'briefings_briefing_date_key'),
        ('briefing_events', 'briefing_events_briefing_id_rank_key')
    ) AS t(table_name, constraint_name)
),
actual_uniques AS (
    SELECT table_name, constraint_name 
    FROM information_schema.table_constraints
    WHERE table_schema = 'public' AND constraint_type = 'UNIQUE'
),
unique_checks AS (
    SELECT 
        '5. Unique Constraint' AS check_category,
        eu.constraint_name AS target_object,
        CASE WHEN au.constraint_name IS NOT NULL THEN 'PASS' ELSE 'FAIL (Missing Unique Constraint)' END AS status
    FROM expected_uniques eu
    LEFT JOIN actual_uniques au ON eu.table_name = au.table_name AND eu.constraint_name = au.constraint_name
),

-- 6. Check Constraint Verification
expected_checks AS (
    SELECT * FROM (VALUES
        ('briefings', 'briefings_status_check'),
        ('articles', 'articles_relevance_status_check'),
        ('events', 'events_confidence_label_check')
    ) AS t(table_name, constraint_name)
),
actual_checks AS (
    SELECT table_name, constraint_name 
    FROM information_schema.table_constraints
    WHERE table_schema = 'public' AND constraint_type = 'CHECK'
),
check_checks AS (
    SELECT 
        '6. Check Constraint' AS check_category,
        ec.constraint_name AS target_object,
        CASE WHEN ac.constraint_name IS NOT NULL THEN 'PASS' ELSE 'FAIL (Missing Check Constraint)' END AS status
    FROM expected_checks ec
    LEFT JOIN actual_checks ac ON ec.table_name = ac.table_name AND ec.constraint_name = ac.constraint_name
),

-- 7. Index Verification
expected_indexes AS (
    SELECT unnest(ARRAY[
        'articles_published_idx', 'articles_source_published_idx', 'articles_relevance_idx', 'articles_topic_idx',
        'events_last_seen_idx', 'events_category_first_seen_idx', 'events_confidence_label_idx', 'events_is_analysis_idx',
        'event_articles_article_id_idx', 'briefings_digest_hash_idx', 'briefings_chat_id_idx',
        'briefing_events_event_id_idx', 'briefing_event_evidence_article_id_idx', 'pipeline_runs_started_idx'
    ]) AS index_name
),
actual_indexes AS (
    SELECT indexname 
    FROM pg_indexes 
    WHERE schemaname = 'public'
),
index_checks AS (
    SELECT 
        '7. Index Verification' AS check_category,
        ei.index_name AS target_object,
        CASE WHEN ai.indexname IS NOT NULL THEN 'PASS' ELSE 'FAIL (Missing Index)' END AS status
    FROM expected_indexes ei
    LEFT JOIN actual_indexes ai ON ei.index_name = ai.indexname
)

-- Combined Diagnostic Report
SELECT * FROM table_checks
UNION ALL
SELECT * FROM column_checks
UNION ALL
SELECT * FROM pk_checks
UNION ALL
SELECT * FROM fk_checks
UNION ALL
SELECT * FROM unique_checks
UNION ALL
SELECT * FROM check_checks
UNION ALL
SELECT * FROM index_checks
ORDER BY check_category ASC, target_object ASC;
```

*(Expected Output: Exactly 59 verified objects, all returning `PASS`).*

---

## 8. Rollback Procedures (Disaster Recovery)

If emergency rollback is ever required, execute in reverse order:

```sql
-- Emergency Full Rollback Script
DROP TABLE IF EXISTS public.briefing_event_evidence CASCADE;
DROP TABLE IF EXISTS public.briefing_events CASCADE;
DROP TABLE IF EXISTS public.pipeline_runs CASCADE;
DROP TABLE IF EXISTS public.briefings CASCADE;
DROP TABLE IF EXISTS public.event_articles CASCADE;
DROP TABLE IF EXISTS public.events CASCADE;
DROP TABLE IF EXISTS public.articles CASCADE;
DROP TABLE IF EXISTS public.sources CASCADE;
```
