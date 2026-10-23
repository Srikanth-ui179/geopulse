# Geopulse Phase 6J: Supabase Preflight Readiness Report

## Executive Summary

Phase 6I identified a critical operational risk: commands running in persistent mode (`--persist`, `--supabase-state`, or `--send`) could write sources, articles, or event clusters to Supabase before discovering unapplied migrations (such as `005_structured_summaries.sql`), resulting in orphaned rows, corrupted state, or runtime crashes during briefing generation.

Phase 6J implements and validates a non-destructive, read-only preflight mechanism (`geopulse.preflight`) that verifies database connectivity and schema compatibility *before* initiating any database mutations or dispatching any Telegram messages.

---

## 1. How Preflight Works

### 1.1 Non-Destructive Probing Mechanism
The preflight check queries Supabase PostgREST endpoints using read-only `limit(0)` statements with specific column projections:
```python
client.table(table_name).select(",".join(required_columns)).limit(0).execute()
```
- **Zero Rows Read/Written**: `limit(0)` executes against PostgREST's schema metadata cache without scanning table rows or generating table locks.
- **Fail-Fast Isolation**: If a table query fails, preflight performs an isolated wildcard probe (`select("*").limit(0)`) to distinguish between a **missing relation** (table not yet created) and **missing columns** (unapplied migration on an existing table).
- **Zero Leaks**: All exception details are sanitized; raw URLs and service-role keys are never included in exception logs or CLI terminal output.

### 1.2 Probed Schema Objects by Scope

| Scope | Command Context | Checked Tables | Verified Columns |
|---|---|---|---|
| `ingest` | `geopulse ingest --persist` | `sources`, `articles` | `sources`: id, name, feed_url, enabled<br>`articles`: id, source_id, url, canonical_url, title, relevance_status, topic |
| `cluster` | `geopulse cluster --persist` | `sources`, `articles`, `events`, `event_articles` | Ingest scope plus:<br>`events`: id, title, category, confidence_label, published_at, is_analysis<br>`event_articles`: event_id, article_id |
| `full` | `geopulse brief --persist`<br>`geopulse brief --send`<br>`BriefingPipeline.run(persistent=True)` | `sources`, `articles`, `events`, `event_articles`, `briefings`, `briefing_events`, `briefing_event_evidence` | Cluster scope plus:<br>`events`: headline, key_details, confidence_qualification, model_used (Migration 005)<br>`briefings`: id, briefing_date, status, telegram_chat_id, telegram_message_ids, digest_hash, chunks_delivered<br>`briefing_events`: briefing_id, event_id, rank, summary, headline, key_details, confidence_qualification, model_used<br>`briefing_event_evidence`: briefing_id, event_id, article_id |

---

## 2. Failure Modes, Error Messages, and Credential Safety

### 2.1 Failure Modes and Diagnoses

1. **Missing Configuration**:
   - *Condition*: `SUPABASE_URL` or `SUPABASE_KEY` missing from `.env` or settings.
   - *Terminal Output*: `Error: Supabase credentials are not configured. Set SUPABASE_URL and SUPABASE_KEY in .env.`
   - *Exit Code*: `1` (aborts before any network call or pipeline activity).

2. **Connection / Host Unreachable**:
   - *Condition*: DNS failure, bad credentials, or unreachable Supabase URL.
   - *Terminal Output*: `Error: Supabase schema preflight failed: Failed to connect to Supabase: [error]`
   - *Credential Safety*: Client initialization errors suppress sensitive query strings and tokens.

3. **Missing Migration (Table Level)**:
   - *Condition*: Migration `001` or `002` unapplied (`relation does not exist` / 404).
   - *Terminal Output*:
     ```
     Error: Supabase schema preflight failed: Missing required table(s): briefing_events, briefing_event_evidence
     Details: Pending database migrations (001-005) must be applied to Supabase before persistent execution can proceed.
     ```
   - *Guarantees*: Zero records inserted into `sources` or `articles`.

4. **Missing Migration (Column Level)**:
   - *Condition*: Migration `005` unapplied (`events.headline` or `briefing_events.headline` missing).
   - *Terminal Output*:
     ```
     Error: Supabase schema preflight failed: Missing required column(s): events (headline, key_details, confidence_qualification, model_used); briefing_events (headline, key_details, confidence_qualification, model_used)
     Details: Pending database migrations (001-005) must be applied to Supabase before persistent execution can proceed.
     ```
   - *Guarantees*: Pipeline terminates immediately without saving intermediate ingestion or event rows.

5. **Preview / In-Memory Isolation**:
   - *Condition*: `geopulse brief --preview` or `--in-memory-state`.
   - *Behavior*: Preflight is bypassed entirely; runs in-memory without contacting Supabase or Telegram.

6. **Persistent Non-Delivery Isolation**:
   - *Condition*: `geopulse brief --persist` (without `--send`).
   - *Behavior*: Preflight validates schema; writes database; strictly sends zero Telegram messages.

---

## 3. Verification Results and Test Coverage

### 3.1 Unit Test Coverage (`tests/unit/test_preflight.py`)
12 offline tests comprehensively validate the preflight module:
- `test_preflight_missing_credentials`: Verifies fail-closed behavior when credentials are unset.
- `test_preflight_connection_failure`: Verifies network failure handling without leaking `SUPABASE_KEY`.
- `test_preflight_missing_required_table`: Simulates 404/PGRST200 table-missing response.
- `test_preflight_missing_migration_005_columns`: Simulates missing `headline` / `key_details` on `events`.
- `test_preflight_success_full_scope`: Verifies passing outcome across all 7 tables and required columns.
- `test_preflight_scope_ingest_checks_only_sources_and_articles`: Enforces selective scope.
- `test_preflight_failure_blocks_pipeline_writes`: Verifies zero calls to PostgREST `insert()` when preflight fails.
- `test_preflight_success_allows_pipeline_writes`: Verifies pipeline writes proceed only when preflight succeeds.
- `test_preview_mode_never_sends_messages`: Confirms `--preview` never touches Telegram client.
- `test_persist_without_send_never_sends_messages`: Confirms `--persist` never calls Telegram.
- `test_send_blocked_when_preflight_fails`: Confirms `--persist --send` aborts prior to Telegram delivery on preflight error.
- `test_cli_ingest_preflight_blocking`: Confirms `geopulse ingest --persist` fails before ingestion on missing tables.

### 3.2 Full Test Suite Results
```
============================ 221 passed in 37.08s =============================
```
- Total tests: **221 passed, 0 failed**.
- Linter status: **Ruff check passed (0 errors)**.

---

## 4. Release Readiness Status

| Checkpoint | Status | Detail |
|---|---|---|
| **Editorial & Relevance Filters** | Verified | Administrative filtering active; analysis detection enabled. |
| **Telegram Delivery** | Verified | Tested live in Phase 6H (Message ID 4). Mock tests enforce dry-run safety. |
| **Idempotent Migrations** | Verified | Migrations 001–005 audited. `005_structured_summaries.sql` ready. |
| **Preflight Schema Enforcement** | **RESOLVED** | Read-only check fails closed before any DB mutation or Telegram delivery. |
| **Credential Safety** | Verified | Service keys never emitted in error logs, traces, or terminal output. |

### Next Step
Execute the staged runbook from Phase 6F:
1. Apply migrations `001_initial_schema.sql` through `005_structured_summaries.sql` in Supabase SQL Editor.
2. Run `geopulse brief --persist --limit 1` to verify preflight passes and populates initial persistent state without sending.
3. Inspect Supabase tables to confirm rows for briefing, events, and evidence.
4. Execute `geopulse brief --persist --send` for the first authorized live digest.
