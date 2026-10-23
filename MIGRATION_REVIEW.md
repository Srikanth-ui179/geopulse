# Geopulse Database Schema & Migration Review (Phase 6)

## Overview & Architecture

Geopulse uses a relational schema hosted on PostgreSQL (via Supabase) designed to support an automated, audit-grade OSINT intelligence pipeline.

### Data Flow Architecture

The data pipeline transitions data through distinct, verifiable stages:
```
1. RSS Sources (sources.py)
   │
   ▼
2. Ingestion & Normalization (SupabaseArticleRepository)
   - Canonical URL deduplication (unique constraint)
   - Content hashing (SHA-256)
   - Initial status: 'pending'
   │
   ▼
3. Relevance & Classification (HeuristicRelevanceClassifier)
   - Relevance status: 'relevant', 'irrelevant', 'review'
   - Topic classification: 11 categories (articles.topic)
   │
   ▼
4. Event Clustering (SupabaseEventRepository)
   - Temporal & semantic clustering into EventCluster
   - Metadata: first_seen_at, last_seen_at, published_at, is_analysis
   - Corroboration: confidence_label ('corroborated', 'single_source', 'disputed')
   - Relational link: public.event_articles (M:N mapping)
   │
   ▼
5. Structured Summarization (OllamaSummarizer)
   - LLM generation: headline, what_happened, key_details, significance
   - Strict source preservation
   │
   ▼
6. Briefing Compilation & Telegram Delivery (SupabaseDeliveryRepository)
   - Formatted HTML digest chunks (< 4096 chars)
   - Deduplication: unique briefing_date & digest_hash
   - Multi-chunk progress tracking: chunks_delivered, telegram_message_ids
   - Resumption & lock protection: inter-process file locking
   - Provenance records: public.briefing_events & public.briefing_event_evidence
```

---

## Migration Manifest

| Migration File | Stage | Purpose | Compatibility |
| :--- | :--- | :--- | :--- |
| `001_initial_schema.sql` | Base Schema | Core tables: sources, articles, events, event_articles, briefings, briefing_events, briefing_event_evidence, pipeline_runs. | Initial deployment |
| `002_clustering_and_topics.sql` | Phase 3 | Adds `articles.topic` and supporting composite indexes. | Fully backward-compatible |
| `003_telegram_delivery_state.sql` | Phase 5 | Adds multi-chunk delivery state (`chunks_delivered`, `telegram_message_ids`, `digest_hash`, `telegram_chat_id`). | Fully backward-compatible |
| `004_event_metadata_and_briefing_status.sql` | Phase 6 | Adds `events.published_at`, `events.is_analysis`, and updates check constraint to permit `'ambiguous'` status. | Fully backward-compatible |
| `005_structured_summaries.sql` | Phase 6B (Draft) | Adds `key_details` (JSONB), `headline`, `confidence_qualification`, and `model_used` to `events` and `briefing_events`. | Non-breaking extension (draft only) |

---

## Detailed Migration Review

### Migration `001_initial_schema.sql`

#### Purpose
Establishes the foundational schema for the intelligence pipeline.

#### Key Objects Created
- **`public.sources`**: Monitored feeds (`id` UUID PK, `name`, `feed_url` UNIQUE, `enabled`).
- **`public.articles`**: Ingested articles (`id` UUID PK, `source_id` FK, `canonical_url` UNIQUE, `content_hash`, `relevance_status`).
- **`public.events`**: Clustered geopolitical events (`id` UUID PK, `title`, `category`, `confidence_label`, `first_seen_at`, `last_seen_at`).
- **`public.event_articles`**: Junction table for event-to-article grounding (`event_id`, `article_id` composite PK).
- **`public.briefings`**: Daily digest delivery record (`id` UUID PK, `briefing_date` UNIQUE DATE, `status`, `telegram_message_count`).
- **`public.briefing_events`**: Ranked events included in a briefing (`briefing_id`, `event_id` composite PK, `rank` UNIQUE per briefing).
- **`public.briefing_event_evidence`**: Full provenance trail connecting briefing events to specific source articles.
- **`public.pipeline_runs`**: Operational metrics and telemetry logging.

#### Constraints & Indexes
- Foreign keys with `ON DELETE CASCADE` prevent orphaned records while preserving referential integrity.
- Check constraints ensure valid `relevance_status`, `confidence_label`, `pipeline_runs.status`, and chronology (`last_seen_at >= first_seen_at`, `completed_at >= started_at`).
- B-tree indexes on `published_at`, `source_id`, `relevance_status`, `last_seen_at`, and foreign key columns.

#### Rollback SQL
```sql
drop table if exists public.briefing_event_evidence cascade;
drop table if exists public.briefing_events cascade;
drop table if exists public.pipeline_runs cascade;
drop table if exists public.briefings cascade;
drop table if exists public.event_articles cascade;
drop table if exists public.events cascade;
drop table if exists public.articles cascade;
drop table if exists public.sources cascade;
```

---

### Migration `002_clustering_and_topics.sql`

#### Purpose
Enhances the schema to support fine-grained topic classification and accelerated clustering lookups.

#### Changes
- Adds `topic` column to `public.articles` with a CHECK constraint validating the 11 domain topics:
  - `armed_conflict`, `military_operations`, `defence_policy`, `weapons_procurement`, `diplomacy`, `sanctions`, `terrorism_insurgency`, `cyber_hybrid_warfare`, `security_incident`, `other_geopolitical`, `uncertain`.
- Adds index `articles_topic_idx` on `(topic, published_at desc)`.
- Adds composite index `events_category_first_seen_idx` on `(category, first_seen_at desc)`.
- Adds index `events_confidence_label_idx` on `(confidence_label)`.

#### Prerequisites
Requires `001_initial_schema.sql` applied.

#### Data Loss Risks
None. Adding nullable columns with checks and indexes is non-destructive.

#### Rollback SQL
```sql
drop index if exists public.events_confidence_label_idx;
drop index if exists public.events_category_first_seen_idx;
drop index if exists public.articles_topic_idx;
alter table if exists public.articles drop column if exists topic;
```

---

### Migration `003_telegram_delivery_state.sql`

#### Purpose
Enables multi-chunk Telegram delivery state tracking, persistent deduplication, and atomic resumption in `public.briefings`.

#### Changes
- Adds `telegram_chat_id` (TEXT) to identify target chat.
- Adds `telegram_message_ids` (JSONB default `'[]'::jsonb`) storing Telegram message IDs for delivered chunks.
- Adds `digest_hash` (TEXT) storing deterministic SHA-256 hash of the complete digest content.
- Adds `chunks_delivered` (INTEGER default 0) tracking confirmed delivered chunk index.
- Adds indexes on `digest_hash` and `telegram_chat_id`.

#### Prerequisites
Requires `001_initial_schema.sql` applied.

#### Data Loss Risks
- Forward application: None (adding nullable/defaulted columns and indexes is non-destructive).
- Rollback risk: Dropping `telegram_message_ids`, `digest_hash`, `chunks_delivered`, and `telegram_chat_id` irreversibly destroys the audit history of which Telegram messages were delivered and breaks partial resumption for any interrupted briefings.

#### Rollback SQL
```sql
drop index if exists public.briefings_chat_id_idx;
drop index if exists public.briefings_digest_hash_idx;
alter table if exists public.briefings
    drop column if exists telegram_chat_id,
    drop column if exists telegram_message_ids,
    drop column if exists digest_hash,
    drop column if exists chunks_delivered;
```

---

### Migration `004_event_metadata_and_briefing_status.sql`

#### Purpose
Aligns the database schema with Phase 4-5 domain model evolutions:
1. Editorial analysis separation (`is_analysis`, `published_at` on `EventCluster`).
2. Delivery safety resilience (`ambiguous` status for read/network timeouts during Telegram API delivery).

#### Changes
- Adds `published_at` (TIMESTAMPTZ) to `public.events`.
- Adds `is_analysis` (BOOLEAN NOT NULL DEFAULT FALSE) to `public.events`.
- Modifies `briefings_status_check` constraint on `public.briefings` to include `'ambiguous'`:
  `CHECK (status in ('pending', 'processing', 'sent', 'partial', 'failed', 'ambiguous'))`.
- Adds index `events_is_analysis_idx` on `(is_analysis, first_seen_at desc)`.

#### Prerequisites
Requires `001_initial_schema.sql`, `002_clustering_and_topics.sql`, and `003_telegram_delivery_state.sql`.

#### Data Loss & Constraint Risks
- **Rollback Pre-condition**: If any briefings rows have `status = 'ambiguous'`, attempting to restore the original check constraint `('pending', 'processing', 'sent', 'partial', 'failed')` will fail immediately with a PostgreSQL check constraint violation. Before rolling back, existing `ambiguous` records must be manually resolved or updated (e.g., to `'failed'`).
- Dropping `is_analysis` or `published_at` columns will permanently drop that event metadata from historical events.

#### Rollback SQL
```sql
drop index if exists public.events_is_analysis_idx;
-- Resolve any rows with status 'ambiguous' before restoring stricter constraint
update public.briefings set status = 'failed' where status = 'ambiguous';
alter table if exists public.briefings drop constraint if exists briefings_status_check;
alter table if exists public.briefings
    add constraint briefings_status_check
        check (status in ('pending', 'processing', 'sent', 'partial', 'failed'));
alter table if exists public.events drop column if exists published_at;
alter table if exists public.events drop column if exists is_analysis;
```

---

### Migration `005_structured_summaries.sql` (Draft Only - Not Applied)

#### Purpose & Schema Gap Analysis
Under migrations `001–004`, structured summaries from `EventSummary` are only partially persisted:
- `what_happened` is stored in `public.events.summary` and `public.briefing_events.summary`.
- `significance` is stored in `public.events.significance` and `public.briefing_events.why_it_matters`.

**Identified Schema Gap**: The rich factual components extracted by local Ollama — `key_details` (`actors`, `location`, `timing`, `action`), `headline`, `confidence_qualification`, and `model_used` — have no dedicated columns in `001–004`.

Draft migration `005` addresses this gap cleanly by adding nullable/JSONB columns without breaking existing data:
```sql
alter table if exists public.events
    add column if not exists headline text,
    add column if not exists key_details jsonb,
    add column if not exists confidence_qualification text,
    add column if not exists model_used text;

alter table if exists public.briefing_events
    add column if not exists headline text,
    add column if not exists key_details jsonb,
    add column if not exists confidence_qualification text,
    add column if not exists model_used text;
```

#### Rollback SQL
```sql
alter table if exists public.briefing_events
    drop column if exists headline,
    drop column if exists key_details,
    drop column if exists confidence_qualification,
    drop column if exists model_used;

alter table if exists public.events
    drop column if exists headline,
    drop column if exists key_details,
    drop column if exists confidence_qualification,
    drop column if exists model_used;
```

---

## Domain Model Mapping

| Domain Model | PostgreSQL Table | Key Field Mappings |
| :--- | :--- | :--- |
| `ArticleRecord` (`src/geopulse/ingestion/models.py`) | `public.articles` | `canonical_url` ➔ `canonical_url`<br>`relevance_status` ➔ `relevance_status`<br>`topic` ➔ `topic`<br>`content_hash` ➔ `content_hash` |
| `EventCluster` (`src/geopulse/clustering/models.py`) | `public.events` | `id` ➔ `id`<br>`title` ➔ `title`<br>`category.value` ➔ `category`<br>`confidence_label.value` ➔ `confidence_label`<br>`is_analysis` ➔ `is_analysis`<br>`published_at` ➔ `published_at` |
| `(event, article)` relationship | `public.event_articles` | `event_id` ➔ `event_id`<br>`article_id` ➔ `article_id`<br>`relationship_type` ➔ `reports_on` |
| `BriefingRecord` (`src/geopulse/delivery/models.py`) | `public.briefings` | `briefing_date` ➔ `briefing_date`<br>`status` ➔ `status`<br>`digest_hash` ➔ `digest_hash`<br>`chunks_delivered` ➔ `chunks_delivered`<br>`telegram_message_ids` ➔ `telegram_message_ids` (JSONB) |
| `SummarizationResult` (`src/geopulse/summarization/models.py`) | `public.briefing_events` | `event.id` ➔ `event_id`<br>`rank` ➔ `rank`<br>`summary.what_happened` ➔ `summary`<br>`summary.significance` ➔ `why_it_matters`<br>`summary.headline` ➔ `headline`<br>`summary.key_details` ➔ `key_details` (JSONB)<br>`summary.confidence_qualification` ➔ `confidence_qualification`<br>`summary.model_used` ➔ `model_used` |
| `EventSummary` (`src/geopulse/summarization/models.py`) | `public.events` | `summary.what_happened` ➔ `summary`<br>`summary.significance` ➔ `significance`<br>`summary.headline` ➔ `headline`<br>`summary.key_details` ➔ `key_details` (JSONB)<br>`summary.confidence_qualification` ➔ `confidence_qualification`<br>`summary.model_used` ➔ `model_used`<br>*(Reconstructable via `reconstruct_event_summary()`)* |
| Source attribution links | `public.briefing_event_evidence` | `briefing_id`, `event_id`, `article_id` |

---

## Transaction Boundaries & PostgREST HTTP Limitations

Geopulse interacts with PostgreSQL via the Supabase client utilizing PostgREST HTTP endpoints. Operators and developers must understand the transactional boundaries inherent to this architecture:

### 1. HTTP-Level Per-Table Atomicity
- Each Supabase Python client method invocation (`.upsert()`, `.update()`, `.insert()`) translates to a single HTTP request executing within an isolated PostgreSQL transaction on the database server.
- Individual table writes are atomic: an upsert of 50 article rows either commits entirely or aborts entirely.

### 2. Lack of Distributed Cross-Table ACID Transactions
- PostgREST does not support distributed ACID transactions spanning multiple HTTP requests (`articles` ➔ `events` ➔ `event_articles` ➔ `briefings` ➔ `briefing_events`).
- A network partition or client crash occurring between `persist_events` and `update_event_summaries` would leave event rows persisted without summary columns populated.

### 3. Geopulse Consistency & Safety Guarantees
To ensure strict audit-grade integrity without distributed transactions, Geopulse enforces:
1. **Prerequisite Gating (Fail-Closed Semantics)**:
   In `BriefingPipeline`, each persistence stage is strictly sequenced. If an exception occurs in `persist_articles`, `persist_events`, or `update_event_summaries`, the pipeline aborts immediately, logs the exact error, sets `PipelineResult.error_message`, and **guarantees that `delivery_service.deliver_digest()` is never called**. No Telegram messages are ever delivered based on partially persisted state.
2. **Deterministic Primary Keys & Idempotent Upserts**:
   - `events.id` is derived deterministically (or mapped by UUIDv5).
   - `articles.canonical_url` is unique; repeat ingestion uses `on_conflict="canonical_url"`.
   - `briefing_events` uses `on_conflict="briefing_id,event_id"`.
   - Repeat pipeline executions safely converge to the exact same database state without duplicate rows.
3. **OS-Level Kernel Delivery Locking**:
   - `JsonFileDeliveryRepository` enforces cross-process mutual exclusion via OS-level non-blocking kernel locks (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX).
   - Concurrent delivery workers are serialized, preventing duplicate briefings or race conditions.

---

## Pipeline Execution Modes & Flag Matrix (Phase 6C)

Geopulse enforces three distinct execution modes with strict fail-closed flag validation:

| Mode | Invocations / CLI Flags | Article Persistence | Event & Summary Persistence | Telegram Delivery | Briefing Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode 1: Default Preview** | `geopulse brief` (or with `--preview`) | **0 (In-Memory)** | **0 (In-Memory)** | **0 (Never sent)** | **0 (None)** |
| **Mode 2: Persistent Processing** | `geopulse brief --persist` (without `--send`) | **Written to `articles`** | **Written to `events` & `event_articles`** | **0 (Never sent)** | **0 (None)** |
| **Mode 3: Live Delivery** | `geopulse brief --persist --send` | **Written to `articles`** | **Written to `events` & `event_articles`** | **Delivered via Telegram Bot API** | **Written to `briefings`, `briefing_events`, `evidence`** |

### Mutually Exclusive Flag Validations
The CLI parser and `handle_brief` reject contradictory flags before performing any network, database, or LLM operations:
- `--preview` + `--send`: Rejected (`Error: Cannot specify both --preview and --send.`).
- `--in-memory-state` + `--state-file`: Rejected (`Error: Cannot specify both --in-memory-state and --state-file.`).
- `--persist` / `--supabase-state` + `--state-file`: Rejected (`Error: Cannot specify both persistent state and --state-file.`).
- `--in-memory-state` + `--persist` / `--supabase-state`: Rejected (`Error: Cannot specify both --in-memory-state and --persist.`).

---

## Pipeline Wiring vs. In-Memory Execution Status

It is critical to distinguish between repository capabilities implemented in code versus what the CLI currently invokes during pipeline commands:

| Pipeline Stage | Subcommand | Repository Implementation | Active CLI Wiring | Notes / Database Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | `geopulse ingest` | `SupabaseArticleRepository` | **Wired when `--persist` passed** | Defaults to in-memory dry-run unless `--persist` is explicitly provided. |
| **Clustering** | `geopulse cluster` | `SupabaseEventRepository` | **Wired when `--persist` passed** | Persists `EventCluster` to `public.events` and article links to `public.event_articles`. |
| **Summarization**| `geopulse summarize` | None (Direct LLM generation) | **In-memory terminal inspection** | Summaries generated via local Ollama. |
| **Briefing Pipeline**| `geopulse brief` | `BriefingPipeline` & all Supabase Repositories | **Fully wired across all 3 modes** | End-to-end orchestration of ingestion, clustering, summarization, formatting, and delivery with prerequisite fail-closed gating. |

---

## Migration Reapplication Safety

All DDL statements across migrations `001` through `004` are idempotent:
- Tables use `CREATE TABLE IF NOT EXISTS`
- Columns use `ADD COLUMN IF NOT EXISTS`
- Indexes use `CREATE INDEX IF NOT EXISTS`
- Constraints use `DROP CONSTRAINT IF EXISTS` followed by `ADD CONSTRAINT`

Reapplying migrations `001–004` against an existing database is safe and non-destructive.

---

## Operational Verification & Live Readiness Checklist

Before applying migrations to a live Supabase environment:
1. **Environment Variables**:
   - `SUPABASE_URL`: HTTPS URL of the Supabase project.
   - `SUPABASE_KEY` or `SUPABASE_SERVICE_ROLE_KEY`: Service role secret key (never expose in public logs or VCS).
2. **Execution Order**:
   - Apply `001_initial_schema.sql`
   - Apply `002_clustering_and_topics.sql`
   - Apply `003_telegram_delivery_state.sql`
   - Apply `004_event_metadata_and_briefing_status.sql`
3. **Rollback Drill**:
   - Verify all rollback scripts can be applied cleanly in staging before production execution.
4. **Offline Isolation**:
   - All automated test suites (`pytest`) mock `postgrest` / `supabase` and require zero active credentials or outbound network requests.
