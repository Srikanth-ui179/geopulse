# Geopulse: Windows Deployment, Operations, and Recovery Guide

**Target Environment:** Windows 10 / Windows 11 (PowerShell & Command Prompt)<br>
**Timezone:** `Asia/Kolkata` (UTC+05:30)<br>
**Inference Engine:** Local Ollama daemon (`llama3.2:3b`) on `http://localhost:11434`<br>
**Durable Store:** Supabase PostgreSQL<br>
**Delivery Channel:** Telegram Bot API (channel/group dispatches & interactive Q&A)

---

## 1. Safety Notice & Operating Policy

> [!CAUTION]
> **NOT FOR LIVE USE UNTIL APPROVED BY OPERATOR**<br>
> All startup scripts and deployment commands default to **safe mock verification mode**. Live Telegram message dispatches and Supabase database mutations require explicit operator authorization and the `-Live` flag.

---

## 2. Windows Prerequisites & Environment Setup

### 2.1 Software Requirements
| Dependency | Minimum Version | Verified Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Windows OS** | Windows 10 21H2+ | Windows 11 Pro 64-bit | Host operating environment |
| **Python** | 3.12+ | 3.14.7 | Application runtime |
| **Ollama** | 0.3.0+ | 0.5.0+ | Local privacy-preserving LLM inference |
| **Ollama Model** | `llama3.2:3b` | `llama3.2:3b` | Grounded summarization & Q&A synthesis |
| **Shell** | PowerShell 5.1+ | PowerShell 7.4+ | Script execution & service management |

### 2.2 UTF-8 Console Configuration
Ensure your PowerShell session handles Unicode characters (used in category badges and emojis):
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

### 2.3 Windows Power Management (Sleep & Standby)
The Geopulse continuous service runs scheduled briefings at **08:00 AM** and **08:00 PM** Asia/Kolkata.
If Windows enters Sleep or Modern Standby, scheduled runs may be delayed until the system wakes up:
- **Recommended Configuration:** In Windows Settings -> System -> Power -> Screen and sleep, configure the PC to **Never** sleep when plugged in.
- **Missed Run Catch-up:** If the system is asleep at 08:00 or 20:00 and wakes up later (e.g. at 08:15), `ServiceRunner` evaluates missed runs on wake and triggers the briefing automatically once running.

---

## 3. Local Verification & Health Checks

Before starting any service, verify local components non-destructively:

### 3.1 Validate Configuration (No Secrets Printed)
```powershell
.\.venv\Scripts\python.exe -m geopulse.cli check-config
```
*Expected Output:*
```text
Configuration loaded successfully.
Environment: development
Timezone: Asia/Kolkata
Ollama endpoint: http://localhost:11434
Ollama model: llama3.2:3b
Supabase configured: True
Telegram configured: True
```

### 3.2 Verify Local Ollama Inference
Check Ollama without pulling or downloading models:
```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/tags" | Select-Object -ExpandProperty models | Format-Table name, size
```
*Ensure `llama3.2:3b` is listed.*

### 3.3 Run Safe Service Smoke Test
Run a single-pass verification test that evaluates the scheduler and processes a mock bot message:
```powershell
.\.venv\Scripts\python.exe -m geopulse.cli service run --mock-telegram --mock-model --once
```

### 3.4 Preview Briefing Offline
Generate a sample briefing preview using cached fixtures:
```powershell
# Morning edition preview
.\.venv\Scripts\python.exe -m geopulse.cli brief --edition morning --preview --fixture --mock-model

# Evening edition preview
.\.venv\Scripts\python.exe -m geopulse.cli brief --edition evening --preview --fixture --mock-model
```

---

## 4. Service Architecture & Windows Lifecycle

### 4.1 Dual-Engine Architecture
`geopulse service run` operates two coordinated asynchronous engines in a single process:
1. **Briefing Scheduler Engine:**
   - Evaluates system clock against `Asia/Kolkata` timezone.
   - Triggers Morning briefing at 08:00 AM (prioritizing overnight developments).
   - Triggers Evening briefing at 08:00 PM (prioritizing developments since morning).
   - Enforces idempotent delivery; identical briefings are never resent.
2. **Interactive Telegram Q&A Engine:**
   - Long-polls Telegram Bot API (`getUpdates`).
   - Responds to `/help`, `/latest`, and admin-only `/status`.
   - Handles natural-language intelligence questions grounded in stored OSINT articles with clickable citations.

### 4.2 Windows Process Instance Locking
To prevent duplicate service instances on Windows (which cause Telegram polling HTTP 409 conflict and race conditions on delivery state), Geopulse implements OS-level non-blocking locking:
- **Lock File:** `.geopulse/service.lock`
- **Mechanism:** `msvcrt.locking` on Windows (`fcntl.flock` on Unix).
- **Behavior:** If an instance is already running, a second launch exits immediately with:
  ```text
  Error: Another Geopulse service instance is already running (locked at .geopulse\service.lock).
  ```
- **Cleanup:** Released automatically upon graceful shutdown (`SIGINT` / Ctrl+C) or process termination.

---

## 5. Startup & Shutdown Operations

### 5.1 Using PowerShell Launcher (Recommended)
The launcher script incorporates working directory resolution, virtual environment checks, Ollama connectivity checks, and safety boundaries.

**Safe Verification Mode (Default):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1
```

**Pre-flight Checks Only:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -CheckOnly
```

**Live Continuous Service (Requires Operator Approval):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -Live
```

### 5.2 Using Windows Batch Launcher
```cmd
:: Safe verification mode
scripts\run_geopulse_windows.bat

:: Live mode (after approval)
scripts\run_geopulse_windows.bat --live
```

### 5.3 Graceful Shutdown
To stop the continuous service:
- Press `Ctrl+C` in the console window.
- The service runner intercepts `SIGINT`, completes any in-flight message processing, releases `.geopulse/service.lock`, and exits cleanly.

---

## 6. Verified CLI Command Reference

All commands below have been inspected and verified against `src/geopulse/cli.py`:

| Command | Key Flags | Description |
| :--- | :--- | :--- |
| `check-config` | *(None)* | Validates `.env` and environment variables without printing secret tokens. |
| `check-preflight` | `--format {text,json}` | Performs read-only inspection of Supabase tables, columns, and RPC functions. |
| `service status` | *(None)* | Displays scheduler status, timezone, Ollama health, and delivery state counts. |
| `service run` | `--once` | Runs one scheduler evaluation and update poll pass, then exits (smoke-test). |
| `service run` | `--mock-telegram` | Uses mock Telegram client (no real network messages sent or polled). |
| `service run` | `--mock-model` | Uses mock summarizer (no Ollama queries). |
| `service run` | `--no-scheduler` | Disables automated 8 AM / 8 PM briefing triggers (runs Q&A bot only). |
| `service run` | `--no-bot` | Disables Telegram Q&A polling (runs scheduled briefings only). |
| `service run` | `--in-memory-state` | Uses transient memory state (bypasses `.geopulse/delivery_state.json`). |
| `service run` | `--state-file <PATH>` | Specifies custom JSON state file path. |
| `service run` | `--persist` | Writes briefings and clusters to Supabase (requires migration 006). |
| `brief` | `--preview` | Formats and outputs the briefing digest to stdout without sending. |
| `brief` | `--edition {morning,evening}` | Explicitly selects morning or evening edition logic. |
| `brief` | `--date YYYY-MM-DD` | Generates briefing for a specific historical date. |
| `brief` | `--fixture` | Uses cached offline test articles instead of live RSS feeds. |
| `brief` | `--send` | Delivers the generated briefing chunks to the configured Telegram chat. |
| `brief` | `--persist` | Persists ingested articles, clusters, and summaries to Supabase. |
| `ingest` | `--fixture` | Ingests offline fixture articles. |
| `ingest` | `--persist` | Persists ingested articles to Supabase. |

---

## 7. State Management & Operational Recovery

### 7.1 State Partitioning
- **Durable Enterprise State (Supabase):**
  - Stored tables: `articles`, `events`, `event_articles`, `briefings`.
  - Intended as the authoritative multi-device record for intelligence archives and longitudinal analysis.
- **Local Operational State (`.geopulse/delivery_state.json`):**
  - Records briefing send status (`sent`, `skipped_duplicate`, `failed`), timestamps, chunk hashes, and Telegram message IDs.
  - **Critical Role:** Ensures that if the service restarts, briefings for the current date/edition are not resent.
  - **Backup:** Back up `.geopulse/delivery_state.json` periodically. If lost, the service will rely on Supabase records if `--persist` is active.
- **Transient Locks (`.geopulse/service.lock`):**
  - Process instance lock file containing the active PID and start timestamp. Removed upon graceful shutdown.

### 7.2 Failure Modes & Automated Recovery
1. **Telegram HTTP 429 (Rate Limit):**
   - The bot client inspects `parameters.retry_after` from Telegram's response.
   - Automatically pauses polling/sending for the requested duration before retrying.
2. **Telegram HTTP 409 (Polling Conflict):**
   - Occurs if another client or webhook is listening with the same bot token.
   - The runner logs a warning, backs off for 10 seconds, and retries. Instance locking prevents internal duplicate instances.
3. **Ollama Temporary Outage:**
   - If Ollama is unreachable or times out, the summarizer falls back to heuristic rule-based lead extraction. Briefings are still generated without crashing.
4. **Supabase Unavailable / Missing Migration:**
   - The pipeline fails closed if `--persist` is enabled and required RPCs are missing, preventing partial or corrupted data writes.

---

## 8. Live Activation Sign-Off Checklist

Before running Geopulse in live continuous mode, the operator must complete and verify:

- [x] **1. Migration 006 & Database Schema Verified:**
  - Ran `.\.venv\Scripts\python.exe -m geopulse.cli check-preflight`.
  - Confirmed all 7 schema tables and stored procedure `upsert_events_with_articles` report `PASS`.
- [ ] **2. Confirm Telegram Target:**
  - Verify that `TELEGRAM_CHAT_ID` in `.env` corresponds to the intended private channel or chat.
  - Verify that `TELEGRAM_ADMIN_ID` corresponds to your personal Telegram user ID (for `/status` access).
- [ ] **3. Confirm Local Ollama Service:**
  - Verify Ollama daemon is running (`ollama serve`).
  - Verify `llama3.2:3b` model is loaded.
- [ ] **4. Run Final Briefing Preview:**
  - Run `.\.venv\Scripts\python.exe -m geopulse.cli brief --edition morning --preview` to verify live RSS feeds and formatting.
- [ ] **5. Start Live Service:**
  - Execute: `powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -Live`
