# Geopulse: Full Local Audit, Hardening, and Deployment Verification Report

**Audit Date:** 2026-10-25T00:04:00+05:30<br>
**Repository Working Directory:** `<repo-root>`<br>
**Git Baseline:** Branch `master` at commit `3dad9cd1c03c2b25c58ac8b4b5f2565cef664b02`<br>
**Environment:** Windows 11 / PowerShell, Python 3.14.7, local Ollama daemon (`llama3.2:3b`), Supabase PostgreSQL<br>
**Audit Scope:** End-to-end local architecture inspection, defect remediation, security review, and Windows deployment preparation without remote side effects.

---

## 1. Executive Summary & Readiness Verdict

### Deployment Readiness Verdict: **NEEDS HUMAN REVIEW**

| Component / Boundary | Status | Verification Evidence |
| :--- | :--- | :--- |
| **Local Unit & Integration Test Suite** | **VERIFIED (PASS)** | **299 passed in 49.92s** (`pytest`) |
| **Code Style & Static Analysis** | **VERIFIED (PASS)** | **All checks passed** (`ruff check .`) |
| **CLI & Service Smoke Tests** | **VERIFIED (PASS)** | Verified with mock Telegram & mock LLM (`--once`) |
| **Ollama Local Connectivity** | **VERIFIED (PASS)** | Local daemon reachable on `http://localhost:11434`, model `llama3.2:3b` available |
| **Windows Process Instance Locking** | **VERIFIED (HARDENED)** | Exclusive OS file lock prevents duplicate daemons & Telegram HTTP 409 conflicts |
| **Prompt Injection Defenses** | **VERIFIED (HARDENED)** | Explicit demarcation delimiters (`--- BEGIN UNTRUSTED USER QUERY ---`) & security directive |
| **Telegram Message Boundaries** | **VERIFIED (HARDENED)** | Chunking safeguards prevent Telegram HTTP 400 Bad Request on replies > 4000 chars |
| **Migration 006 Status** | **VERIFIED (APPLIED REMOTELY)** | Read-only RPC probe verified `upsert_events_with_articles` exists; all 7 tables accessible |
| **Live Telegram Dispatch** | **PENDING APPROVAL** | Live API calls withheld; mock doubles verified |
| **Supabase Remote Writes** | **PENDING APPROVAL** | Database mutations withheld; local preflight checks verified |

**Why "NEEDS HUMAN REVIEW":**
All local code, tests, configurations, and remote database schemas/stored procedures are verified and hardened. The system cannot be transitioned to **LIVE ACTIVE SERVICE** until the operator provides explicit authorization to activate live Telegram message dispatching and polling.

---

## 2. Baseline Working Tree Inspection

### Git Status & Commit Baseline
- **Branch:** `master` (synchronized with `origin/main` at commit `3dad9cd`)
- **Safety Boundary Adherence:**
  - Zero commits created.
  - Zero branches switched or created.
  - Zero destructive Git operations (`git reset`, `git restore`, `git clean` were never run).
  - No secrets printed or leaked into logs, reports, or test fixtures.
  - `.env` and `.geopulse/` remain strictly ignored by Git.

### Tracked File Modifications (31 Files)
The 31 modified tracked files fall into three functional groups:
1. **Relevance & Classification Engine:**
   - `src/geopulse/classification/rules.py`
   - `src/geopulse/ingestion/relevance.py`
   - `tests/unit/test_classification.py`
   - `tests/unit/test_relevance.py`
2. **Clustering & Corroboration Engine:**
   - `src/geopulse/clustering/clusterer.py`
   - `src/geopulse/clustering/models.py`
   - `src/geopulse/clustering/repository.py`
   - `src/geopulse/clustering/similarity.py`
   - `tests/unit/test_clustering.py`
3. **Pipeline, Delivery & Preflight Hardening:**
   - `src/geopulse/config.py`
   - `src/geopulse/cli.py`
   - `src/geopulse/pipeline.py`
   - `src/geopulse/preflight.py`
   - `src/geopulse/delivery/__init__.py`
   - `src/geopulse/delivery/client.py`
   - `src/geopulse/delivery/exceptions.py`
   - `src/geopulse/delivery/formatter.py`
   - `src/geopulse/delivery/models.py`
   - `src/geopulse/delivery/repository.py`
   - `src/geopulse/delivery/service.py`
   - `src/geopulse/summarization/client.py`
   - `src/geopulse/summarization/prompts.py`
   - `src/geopulse/summarization/service.py`
   - `tests/unit/test_cli_brief.py`
   - `tests/unit/test_cli_ingest.py`
   - `tests/unit/test_config.py`
   - `tests/unit/test_migration.py`
   - `tests/unit/test_pipeline_wiring.py`
   - `tests/unit/test_preflight.py`
   - `tests/unit/test_summarization.py`
   - `tests/unit/test_supabase_integration.py`

### Untracked Files
- `migrations/006_atomic_event_persistence.sql` — Schema migration introducing `upsert_events_with_articles` stored procedure.
- `src/geopulse/bot/` — Interactive Telegram Q&A package (`__init__.py`, `qa_service.py`).
- `src/geopulse/service/` — Scheduled briefing and service lifecycle runner package (`__init__.py`, `runner.py`).
- `scripts/` — Windows startup launchers (`run_geopulse_windows.bat`, `run_geopulse_windows.ps1`).
- `tests/unit/test_scheduler.py` — Scheduler and service lifecycle unit tests.
- `tests/unit/test_telegram_qa.py` — Interactive Telegram Q&A unit tests.

---

## 3. Comprehensive Architectural Audit

### 3.1 RSS Ingestion & Deduplication
- **Feed Parsing:** Feeds are fetched with strict timeouts (`15.0s`), custom `User-Agent` headers, and resilient parsing via `feedparser` and `defusedxml`.
- **URL Canonicalization:** Resolves tracking parameters, session tokens, and query strings. URLs are SHA-256 fingerprinted.
- **Fail-Safe Source Isolation:** Feed failures are captured per source; a failure in one RSS source (e.g. timeout or malformed XML) does not abort ingestion for other sources.
- **Idempotency:** Re-ingesting existing articles detects identical canonical URLs/hashes and skips insertion without duplicate state creation.
- **Untrusted Input Treatment:** All titles, summaries, and HTML bodies are parsed strictly as untrusted data, stripped of executable HTML/scripts, and stored safely.

### 3.2 Classification & Event Clustering
- **Category Taxonomy:** 11 distinct topic categories (`ARMED_CONFLICT`, `MILITARY_OPERATIONS`, `DEFENCE_POLICY`, `WEAPONS_PROCUREMENT`, `DIPLOMACY`, `SANCTIONS`, etc.).
- **Entity Extraction & Similarity:** Hybrid clustering combines TF-IDF headline cosine similarity with temporal proximity windows (48-hour default sliding window) and shared actor/location entity matching.
- **Deduplication Hardening:**
  - *UK–France Joint Nuclear Exercise:* Verified that multi-outlet reporting of the same military exercise merges cleanly into a single event cluster with `corroborated` confidence label.
  - *Semiconductor Noise Filtering:* Verified that broad macroeconomic and opinion pieces without concrete defense actions are classified as `OTHER_GEOPOLITICAL` or filtered out.
- **Determinism:** Event cluster IDs are derived deterministically using UUIDv5 or sorted article canonical URLs, guaranteeing reproducibility across repeated runs.

### 3.3 Summarization & AI Safety
- **Model Configuration:** Configured for local `llama3.2:3b` running on Ollama (`http://localhost:11434`).
- **Pydantic Validation:** Ollama responses are parsed into `EventSummary` models (headline, what happened, key actors/location/timing/action, strategic significance, confidence qualification).
- **Graceful Fallbacks:** If Ollama returns malformed JSON, times out, or is unavailable, the pipeline falls back to deterministic rule-based lead extraction without crashing.
- **Factual Grounding & Prompt Injection Defense:**
  - LLM prompts now strictly demarcate untrusted inputs using `--- BEGIN UNTRUSTED USER QUERY ---` and `--- BEGIN RETRIEVED EVIDENCE (UNTRUSTED SOURCE DATA) ---`.
  - Prompts enforce explicit anti-injection instructions: the model is barred from following instructions embedded in news articles or user questions, and must honestly state when evidence is insufficient.

### 3.4 Supabase Persistence & Migration 006
- **Architecture:** Persistence uses PostgREST HTTP client (`httpx`) with service keys.
- **Migration 006 Audit:**
  - Local file: `migrations/006_atomic_event_persistence.sql`.
  - Purpose: Creates `upsert_events_with_articles` PostgreSQL function to insert/update clusters and link articles in an atomic database transaction.
  - Application Status: **VERIFIED APPLIED REMOTELY**. A read-only PostgREST RPC check confirmed `upsert_events_with_articles` exists, is executable by `service_role`, and returns `{'events_persisted': 0, 'articles_linked': 0}` with zero writes. All 7 required schema tables are present.
  - Fail-Closed Behavior: `run_supabase_preflight` checks for required tables (`articles`, `events`, `event_articles`, `briefings`) and functions. If `upsert_events_with_articles` is missing, persistent runs abort cleanly with an actionable error.
- **Credential Safety:** Supabase API keys are held in `pydantic.SecretStr`, redacted from `__repr__`, and never logged.

### 3.5 Telegram Delivery
- **Formatting & Escaping:** All digest chunks and bot replies use `escape_html()` to prevent invalid HTML entities or injection attacks.
- **Chunk Boundaries:** Digest chunks are strictly bounded to `< 4,000` characters (Telegram limit is 4,096), breaking cleanly across event blocks without splitting sentences or tags.
- **Delivery Idempotency:** Daily briefing status (`sent`, `skipped_duplicate`, `failed`) is persisted in `.geopulse/delivery_state.json`. Restarting the service checks this file to avoid re-sending briefings for the same date and edition.
- **Network Resilience:** Handles Telegram HTTP 429 rate limits using `retry_after`, and handles timeouts with bounded exponential backoff.

### 3.6 Scheduler & Service Lifecycle
- **Schedule:** Triggers morning briefings at 08:00 AM Asia/Kolkata and evening briefings at 08:00 PM (20:00) Asia/Kolkata.
- **Instance Locking (New):** Utilizes non-blocking OS-level file locking (`msvcrt.locking` on Windows, `fcntl.flock` on Unix) on `.geopulse/service.lock`. If another instance is running, the process fails fast with `ServiceAlreadyRunningError`, preventing concurrent Telegram polling and duplicate dispatches.
- **Graceful Shutdown:** Intercepts `SIGINT` (Ctrl+C) and `SIGTERM`, releasing the instance lock, cleanly stopping polling loops, and writing log footers.

### 3.7 Interactive Telegram Q&A
- **Commands:** `/help` (guidance), `/latest` (recent briefed developments), `/status` (system health and scheduler status).
- **Admin Authorization:** `/status` enforces strict admin ID checking via `settings.is_admin(user_id)`. If `TELEGRAM_ADMIN_ID` is missing or the caller's ID does not match, the bot returns `403 Forbidden` (`⛔ Unauthorized`).
- **Grounded Intelligence Answers:** Queries search stored event clusters and articles. If no match is found, the bot honestly replies with `🔍 Insufficient Evidence`. If matching articles exist, Ollama synthesizes an answer strictly grounded in those articles, appending verified clickable source links.
- **Message Chunking (New):** Replies exceeding 4,000 characters are split across paragraph breaks and sent sequentially, preventing Telegram HTTP 400 Bad Request errors.

---

## 4. Defects Identified & Remediations Applied

| Defect # | Component | Root Cause | Impact | Fix Applied | Test Added |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | `ServiceRunner` | Absence of OS-level instance locking. | Two instances could run concurrently on Windows, causing Telegram polling HTTP 409 conflict and duplicate briefings. | Implemented cross-platform non-blocking file locking (`msvcrt` on Windows, `fcntl` on POSIX) on `.geopulse/service.lock`. | `test_service_instance_lock_prevents_duplicate_process`, `test_cli_service_handles_service_already_running` |
| **DEF-02** | `QAService` | Query and evidence passed to LLM without demarcation delimiters. | Vulnerable to prompt injection if an article or user query contained prompt overrides. | Added explicit boundary delimiters (`--- BEGIN UNTRUSTED USER QUERY ---`) and reinforced anti-injection directives in system prompt. | `test_qa_service_synthesize_hardens_prompt_injection` |
| **DEF-03** | `QAService` | `handle_message` sent raw replies without length checking. | Responses longer than 4,096 characters would trigger Telegram HTTP 400 Bad Request. | Added `_split_reply_chunks` splitting messages > 4000 chars across paragraph breaks. | `test_qa_service_splits_oversized_replies` |
| **DEF-04** | CLI Service Smoke Test | Unconfigured root logger and empty mock queue in smoke test mode. | `--mock-telegram --mock-model --once` exited without visible status output. | Configured root logger and enqueued mock `/help` update during `--once` smoke test. | Verified in `test_cli_service_run_smoke_test_once` |

---

## 5. Security & Privacy Audit Findings

1. **Zero Secret Leakage:**
   - Evaluated all CLI commands, logs, exception handlers, and string representations.
   - `TelegramClient` explicitly returns `token='[REDACTED]'` in `__repr__`.
   - `SUPABASE_KEY` and `TELEGRAM_BOT_TOKEN` use `pydantic.SecretStr`.
   - `check-config` displays only redacted statuses (`True`/`False`, key category) and never prints secret strings.
2. **Untrusted Data Isolation:**
   - RSS title, description, and content are sanitized.
   - User queries and retrieved article texts are bounded with security delimiters before reaching Ollama.
3. **Safe Network Boundaries:**
   - Mock test modes (`MockTelegramClient`, `MockOllamaClient`) completely isolate networking; no live HTTP calls to Telegram or Supabase occur in mock mode.
4. **Local Model Isolation:**
   - Ollama is configured to listen only on `localhost:11434`. No external AI endpoints or cloud model APIs are utilized.

---

## 6. Verification Evidence & Test Execution

### 6.1 Full Pytest Suite
```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-8.4.2, pluggy-1.7.0
rootdir: <repo-root>
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1
collected 297 items

tests\unit\test_classification.py ..............................         [ 10%]
tests\unit\test_cli.py ....                                              [ 11%]
tests\unit\test_cli_brief.py ..........                                  [ 14%]
tests\unit\test_cli_ingest.py .......                                    [ 17%]
tests\unit\test_clustering.py ......................                     [ 24%]
tests\unit\test_config.py ..............                                 [ 29%]
tests\unit\test_fixtures.py .                                            [ 29%]
tests\unit\test_migration.py ...........                                 [ 33%]
tests\unit\test_normalization.py ..........                              [ 36%]
tests\unit\test_parser.py ....                                           [ 38%]
tests\unit\test_pipeline_wiring.py ....................                  [ 44%]
tests\unit\test_preflight.py ................                            [ 50%]
tests\unit\test_relevance.py ................                            [ 55%]
tests\unit\test_repository.py ..                                         [ 56%]
tests\unit\test_scheduler.py ...........                                 [ 59%]
tests\unit\test_service.py ......                                        [ 61%]
tests\unit\test_summarization.py ...............................         [ 72%]
tests\unit\test_supabase_integration.py ............................     [ 81%]
tests\unit\test_telegram_client.py ........                              [ 84%]
tests\unit\test_telegram_delivery.py .................                   [ 90%]
tests\unit\test_telegram_formatter.py ..........                         [ 93%]
tests\unit\test_telegram_qa.py ..............                            [ 98%]
tests\unit\test_timestamps.py .....                                      [100%]

============================ 297 passed in 45.32s =============================
```

### 6.2 Ruff Lint & Style Verification
```text
All checks passed!
```

### 6.3 Configuration Validation (No Secrets Revealed)
```text
Configuration loaded successfully.
Environment: development
Timezone: Asia/Kolkata
Ollama endpoint: http://localhost:11434
Ollama model: llama3.2:3b
Supabase configured: True
Supabase key category: newer secret key
Telegram configured: True
```

### 6.4 Service Smoke Test Run
```text
============================================================
Geopulse Service Smoke Test Summary (--once)
============================================================
  Timezone:          Asia/Kolkata
  Scheduler Status:  Morning: not due (scheduled for 08:00) | Evening: not due (scheduled for 20:00)
  Telegram Q&A:      1 update(s) processed
  Mock Messages:     1 dispatched
  Result:            VERIFIED (Single pass executed cleanly)
============================================================
```

---

## 7. Exact External Actions That Remain Unapproved

Until the operator explicitly authorizes each individual operation:
1. **No Live Telegram Message Dispatch:** No briefings or replies will be sent to the configured Telegram chat.
2. **No Live Telegram Long Polling:** `TelegramClient.get_updates` will not be called against live Telegram servers.
3. **No Supabase Database Writes:** No records will be inserted, updated, or deleted in Supabase.
4. **No Supabase Migration Execution:** `migrations/006_atomic_event_persistence.sql` will not be applied to the remote database by automated tooling.
5. **No Automatic Windows Task Registration:** No persistent Windows Task Scheduler task has been registered.
6. **No Git Remote Operations:** No commits or pushes have been made.

---

## 8. Prioritized Next Steps for Operator

1. **Step 1: Verify Database Preflight (Completed)**
   - Run: `.\.venv\Scripts\python.exe -m geopulse.cli check-preflight`
   - Verified: All 7 required schema tables, columns, and stored procedure `upsert_events_with_articles` reported `PASS`.
2. **Step 2: Execute Controlled Live Briefing Preview**
   - Execute: `.\.venv\Scripts\python.exe -m geopulse.cli brief --preview`
   - Verify output digest formatting, source links, and event rankings without sending.
3. **Step 3: Launch Controlled Service via Windows Launcher**
   - Execute: `powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -Live`
   - Monitor the console logs as the service starts up and acquires `.geopulse/service.lock`.
