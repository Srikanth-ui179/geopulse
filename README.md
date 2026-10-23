# Geopulse

Geopulse is a source-grounded daily intelligence and briefing bot for geopolitical affairs, defense strategy, and military developments. It aggregates multi-source reporting, clusters related articles into discrete events, extracts structured factual summaries using a local LLM, and delivers curated HTML briefings to Telegram with verifiable source attribution.

## Key Capabilities

- **Source Ingestion & Deduplication:** Continuous feed processing across reputable defense and geopolitical sources, canonical URL resolution, and SHA-256 content fingerprinting.
- **Noise Filtering & Relevance Classification:** Heuristic relevance classification filtering out routine administrative notices, court center listings, and promotional noise while tagging topics across 11 defense domains.
- **Event Clustering & Corroboration:** Temporal and semantic cross-source article clustering that evaluates multi-source corroboration and flags contested claims (`corroborated`, `single_source`, `disputed`).
- **Source-Grounded Local Summarization:** Extracts structured facts (headline, what happened, key actors/actions/locations, strategic significance, confidence qualification) using a local Ollama model without external API costs or cloud data leaks.
- **Telegram HTML Digest Delivery:** Clean, structured HTML briefings with category badges, clickable source links, and length-bounded chunk splitting (<4,000 chars) that never truncates mid-sentence or breaks HTML tags.
- **Fail-Safe Delivery & Idempotency:** Bounded retries with exponential backoff, rate-limit compliance (`retry_after`), ambiguous timeout handling to prevent duplicate messages, and daily delivery deduplication.
- **Non-Destructive Preflight Schema Checks:** Probes database tables and column availability before executing persistent writes, halting cleanly if migrations are missing.

---

## Requirements

- **Python:** 3.12 or newer
- **Local LLM Daemon (Ollama):** Optional for offline previews/tests; required for live summarization ([ollama.com](https://ollama.com), recommended model: `llama3.2:3b`)
- **Database (Supabase PostgreSQL):** Optional for local preview mode; required for persistent runs
- **Telegram Bot API:** Optional for dry-run/preview; required for live briefing delivery

---

## Quickstart

### 1. Clone & Set Up Virtual Environment

#### Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,database]"
```

#### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,database]"
```

### 2. Configure Environment

Copy the example configuration file:
```powershell
# Windows
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` with your credentials:
```ini
APP_ENV=development
LOG_LEVEL=INFO
TIMEZONE=Asia/Kolkata

# Supabase (Required for --persist)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key

# Ollama Local LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Telegram (Required for --send)
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-target-chat-id
```

---

## CLI Usage

Geopulse provides three primary modes of operation:

### Mode 1: In-Memory Preview (Default)
Runs entirely in memory with zero database writes and zero Telegram calls. Safe to run at any time.

```powershell
# Preview briefing using test fixtures and mock model (fully offline)
geopulse brief --preview --fixture --mock-model

# Preview briefing with live RSS feeds and local Ollama
geopulse brief --preview --limit 3
```

### Mode 2: Persistent Processing (Database Only)
Ingests, clusters, summarizes, and records events in Supabase, but makes zero calls to Telegram.

```powershell
# Ingest and cluster events into Supabase
geopulse brief --persist --limit 5
```

### Mode 3: Live or Simulated Telegram Delivery
Formats and delivers the daily briefing to Telegram.

```powershell
# Test delivery safely with simulated mock Telegram client
geopulse brief --send --fixture --mock-model --mock-telegram

# Live persistent delivery (persists to Supabase and delivers to Telegram)
geopulse brief --persist --send --limit 5

# Force resend for today if a briefing was already recorded
geopulse brief --persist --send --limit 5 --force
```

### Granular CLI Subcommands

```powershell
# Validate configuration and environment
geopulse check-config

# Ingestion: dry-run or persistent
geopulse ingest --dry-run --limit 5
geopulse ingest --persist --limit 10

# Clustering: group articles into events
geopulse cluster --limit 5

# Summarization: test LLM extraction on events
geopulse summarize --limit 3 --model llama3.2:3b
```

---

## Database Migrations

When connecting to Supabase PostgreSQL, apply migrations in exact sequential order via the Supabase SQL Editor:

1. `migrations/001_initial_schema.sql`  
   Initializes baseline tables: `sources`, `articles`, `events`, `event_articles`, `briefings`, `briefing_events`, `briefing_event_evidence`, and `pipeline_runs`.
2. `migrations/002_clustering_and_topics.sql`  
   Adds `articles.topic` and composite performance indexes on `events (category, first_seen_at desc)`.
3. `migrations/003_telegram_delivery_state.sql`  
   Adds delivery audit and progress columns to `briefings`: `telegram_chat_id`, `telegram_message_ids`, `digest_hash`, and `chunks_delivered`.
4. `migrations/004_event_metadata_and_briefing_status.sql`  
   Adds `events.published_at`, `events.is_analysis`, and extends `briefings.status` constraints.
5. `migrations/005_structured_summaries.sql`  
   Adds structured summary columns (`headline`, `key_details`, `confidence_qualification`, `model_used`) to `events` and `briefing_events`.

*Note: Geopulse includes an automated read-only preflight check (`geopulse.preflight`) that verifies all required tables and columns exist in PostgREST before allowing any database writes.*

---

## Default Configured Feeds

Geopulse tracks verified defense and geopolitical outlets out of the box:

| Source | Domain / Focus |
|---|---|
| **Defense News** | Defense industry, acquisitions, military technology |
| **Breaking Defense** | Air, naval, land, and space warfare strategy |
| **War on the Rocks** | Foreign policy analysis and military affairs |
| **UK Ministry of Defence** | Official UK government defense announcements |
| **BBC World News** | International conflict and breaking crisis reporting |
| **Al Jazeera English** | Middle East, Africa, and global regional reporting |

---

## Testing & Quality Assurance

All core capabilities are tested offline with zero external network or database dependencies:

```powershell
# Run the complete test suite
pytest -v

# Run code style and lint checks
ruff check src tests
```

---

## Project Principles

1. **Strict Source Grounding:** Summaries never fabricate facts, dates, or URLs. Every point is anchored to source reporting.
2. **Preserve Contested Uncertainty:** Maintain dispute labels (`disputed`, `uncertain`) for active conflict claims.
3. **Fail-Closed Safety:** Database preflight, credential validation, and rate-limit safeguards abort before mutating state on errors.
4. **Offline Testability:** All parsing, clustering, formatting, and delivery mechanics run with mock providers in unit tests.
