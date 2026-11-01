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

# LLM Provider Configuration
LLM_PROVIDER=ollama

# Ollama Local LLM (Default)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Google Gemini API (Optional)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

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

### Mode 4: Continuous Service (Scheduler & Interactive Bot)
Runs Geopulse as a continuous dual-engine process attached to the terminal (logging live events to stdout):
1. **Briefing Scheduler**: Automatically evaluates system time against `TIMEZONE` and triggers the Morning briefing at 08:00 AM and Evening briefing at 08:00 PM, with automatic catch-up logic for missed windows and duplicate send prevention.
2. **Interactive Telegram Bot**: Long-polls for incoming updates and answers user queries (`/help`, `/latest`, and admin-only `/status`) using source-grounded retrieval with strict prompt-injection delimiters.

```powershell
# Safe verification smoke test (single evaluation pass using mocks with zero side effects)
geopulse service run --mock-telegram --mock-model --once

# Inspect current service schedule, configured timezone, and status
geopulse service status

# Start live continuous service in the foreground (interactive bot + scheduled dispatches)
geopulse service run

# Start live continuous service with persistent Supabase writes
geopulse service run --persist

# Run dedicated component only
geopulse service run --no-bot        # Scheduler only (no interactive Q&A bot)
geopulse service run --no-scheduler  # Interactive Q&A bot only (no scheduled briefings)
```

#### Stopping and Background Execution
- **Stopping**: In the active terminal, press `Ctrl+C` or send a standard `SIGINT` / `SIGTERM` signal. The service catches the signal, finishes any in-flight task, flushes delivery state, cleanly releases the OS file lock (`.geopulse/service.lock`), and exits gracefully.
- **Running Unattended in Background**: To run Geopulse as a headless system daemon, use the provided PowerShell runner (`powershell -File scripts\run_geopulse_windows.ps1 -Live`), register it as a Windows Scheduled Task / Windows Service (via NSSM), or configure a systemd unit on Linux.
- **Instance Mutual Exclusion**: An exclusive OS-level file lock prevents multiple service instances from running concurrently against the same state directory.

### Granular CLI Subcommands

```powershell
# Validate configuration and environment
geopulse check-config

# Preflight check: verify Supabase schema and stored procedure availability
geopulse check-preflight --scope full

# Ingestion: dry-run or persistent
geopulse ingest --dry-run --limit 5
geopulse ingest --persist --limit 10

# Clustering: group articles into events
geopulse cluster --limit 5

# Summarization: test LLM extraction on events
geopulse summarize --limit 3 --model llama3.2:3b
geopulse summarize --limit 3 --provider gemini --model gemini-2.5-flash

# Service daemon: inspect schedule and health
geopulse service status
```

---

## LLM Providers & Fallback Architecture

Geopulse supports flexible model provider selection for grounded event summarization:

| Provider | Mode Description | Fallback Behavior |
|---|---|---|
| `ollama` (Default) | Uses local Ollama instance (`llama3.2:3b`) without external API calls or cloud data transfer. | None (fails if Ollama daemon is unreachable). |
| `gemini` | Uses Google's Gemini API directly for fast, cloud-based summarization. Requires `GEMINI_API_KEY`. | None (fails if Gemini errors; no silent downgrade). |
| `auto` | Prefers Gemini when configured; automatically falls back to local Ollama on eligible transient API errors. | Falls back to Ollama on HTTP 5xx, timeouts, or 429 quota exhaustion. |

### Configuration

Set provider settings in your `.env` file or environment variables:

```ini
# Provider: 'ollama', 'gemini', or 'auto' (default: 'ollama')
LLM_PROVIDER=auto

# Ollama local settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Google Gemini settings
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_TIMEOUT=30.0
```

> [!IMPORTANT]
> **API Key Setup & Credits:**
> Google Gemini API keys are provisioned through [Google AI Studio](https://aistudio.google.com/) or Google Cloud Vertex AI. Note that a personal Google One AI Premium or Gemini Pro web subscription does **not** include developer API credits. API usage is billed or quota-allocated separately via Google Cloud / Google AI Studio.

### Transient Failure & Fallback Semantics

In `auto` mode, Geopulse balances cloud quality with local offline reliability:
- **Eligible Fallback Errors:**
  - Transient server errors (HTTP 500, 502, 503, 504) after bounded exponential backoff retries.
  - Network timeouts and connection drops after bounded retries.
  - Rate limit or quota exhaustion (HTTP 429 / `RESOURCE_EXHAUSTED`).
- **Non-Eligible Errors (Fail-Fast):**
  - Authentication errors (HTTP 401 Unauthorized, HTTP 403 Forbidden, invalid API key). These are permanent credential issues; Geopulse fails immediately to notify the operator rather than silently masking the error.
  - Malformed model names (HTTP 404 Not Found) or configuration errors.
- **Explicit Mode Integrity:**
  - When `--provider gemini` or `LLM_PROVIDER=gemini` is explicitly selected, Geopulse will **not** silently downgrade to Ollama. If the cloud API encounters an error, it fails cleanly with an explanatory error message.

### Switching Back to Ollama

To switch back to local Ollama operation at any time:
1. In `.env`: set `LLM_PROVIDER=ollama` or simply remove the `LLM_PROVIDER` line (Ollama is the default).
2. On CLI: pass the `--provider ollama` flag, e.g. `geopulse brief --preview --provider ollama`.
3. If `GEMINI_API_KEY` is not present and `LLM_PROVIDER=auto` is set, Geopulse smoothly defaults to Ollama without error.

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
6. `migrations/006_atomic_event_persistence.sql`
   Creates the PostgreSQL stored procedure `public.upsert_events_with_articles(p_events jsonb, p_event_articles jsonb)` to transactionally upsert events and link their `event_articles` associations within a single atomic database transaction. Enforces canonical UUID validation, foreign-key relationship integrity, and least-privilege security (revokes execution from `public`, `anon`, and `authenticated`; grants strictly to `service_role`).

> [!IMPORTANT]
> **Operator Migration Steps (Fresh vs. Existing Databases):**
> - **Fresh Supabase Database**: Apply all six migrations sequentially (`001` through `006`) via the Supabase SQL Editor or Supabase CLI before running persistent workloads.
> - **Existing Database (Already on 001–005)**: Apply only `migrations/006_atomic_event_persistence.sql` to install the `upsert_events_with_articles` atomic stored procedure and update execute grants.
> - **Preflight Verification**: Always run `geopulse check-preflight --scope full` before starting persistent runs. The check inspects table schemas, column availability, and RPC procedure access to verify complete database readiness.

*Note: Geopulse includes an automated read-only preflight check (`geopulse.preflight`) that verifies all required tables, columns, and RPC procedures exist in PostgREST before allowing any database writes.*

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
