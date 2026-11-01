# Geopulse: First Live Run Checklist & Operational Guide

This document defines the strict, staged procedure for taking Geopulse live on Windows with local Ollama inference and Telegram delivery.

---

## 1. Safety & Architecture Verification Summary

Before running live, the following behavioral guarantees have been verified in the codebase:

### 1.1 Telegram Destination Security
- **Scheduled Briefings & CLI `brief --send`**: The destination is strictly locked to `TELEGRAM_CHAT_ID` configured in `.env`. The delivery pipeline does not allow any destination override during automated execution.
- **Interactive Bot Q&A**: Replies are sent directly to the incoming update's `chat_id` (the user or group initiating the question).
- **Secret Redaction**: `TELEGRAM_BOT_TOKEN`, `SUPABASE_KEY`, and other credentials are never printed to console or logs; string representations strictly output `[REDACTED]`.

### 1.2 Briefing Pipeline & Grounding
- **Ingestion**: Monitored feeds (Defense News, UK MoD, BBC World, Reuters Defense) are filtered via defense keyword heuristics. Pure tech/broad opinions (e.g. semiconductor commodity commentary) are filtered out.
- **Clustering**: Entity extraction, temporal windowing (48h), and title similarity cluster multi-source reports into unified events.
- **Consecutive Deduplication**: Stories briefed in the previous cycle are automatically suppressed unless `--force` is specified.
- **Summarization**: Local Ollama (`llama3.2:3b`) synthesizes verified facts vs. analysis.
- **Formatting**: Chunks are bounded to <4000 characters (within Telegram's 4096 limit) with valid HTML escaping, category badges, confidence indicators (`[CONFIRMED]`, `[CORROBORATED]`), and clickable source hyperlinks.

### 1.3 Scheduler Startup & Catch-Up Behavior
- **Morning Briefing Hour**: `08:00 AM Asia/Kolkata`
- **Evening Briefing Hour**: `08:00 PM Asia/Kolkata`
- **Startup Logic**:
  - **Started before target hour** (e.g., 07:30 AM): The service waits idle, polling bot Q&A, and triggers the morning briefing at 08:00 AM.
  - **Started after target hour with no prior send** (e.g., 10:30 AM without today's morning briefing recorded as `sent` in `.geopulse/delivery_state.json`): **The service triggers an immediate catch-up morning briefing on its first loop pass.**
  - **Started after target hour with prior send recorded**: The service recognizes today's morning briefing is already `sent`, skips it, and waits for 08:00 PM.

### 1.4 Delivery Idempotency & Crash Recovery
- **Deduplication**: Content SHA-256 hash is computed. If today's edition is already marked `sent`, re-delivery is blocked with `status="skipped_duplicate"` unless `--force` is passed.
- **Chunk Resumption**: After every delivered chunk, state is flushed immediately (`chunks_delivered = N`, `status = "processing"`). If the process crashes or network fails on chunk 3 of 5, restarting the process automatically skips chunks 1 and 2 and resumes delivery at chunk 3.
- **Ambiguous Timeouts**: If a network timeout occurs after request transmission (`httpx.ReadTimeout`), status transitions to `"ambiguous"`. Automatic retries halt immediately to avoid duplicate messages in Telegram.
- **Instance Mutual Exclusion**: An OS-level file lock (`.geopulse/service.lock`) prevents multiple runner processes from executing concurrently.

### 1.5 Database Persistence Boundaries
- **Default Mode**: Persistence is **OFF by default**. The service runs in-memory and records delivery state locally to `.geopulse/delivery_state.json`. Zero writes are sent to Supabase.
- **Persistent Mode (`--persist`)**: Writes articles to `articles`, events to `events` (via RPC `upsert_events_with_articles`), briefings to `briefings`, and relationships to `briefing_events` and `briefing_evidence`.

### 1.6 Interactive Bot Security & Q&A
- `/help`: Returns public guidance and intelligence scope.
- `/latest`: Displays up to 5 briefed events with clickable source links.
- `/status`: **Admin-only**. Requests from non-admin user IDs (not matching `TELEGRAM_ADMIN_ID`) are rejected with an access-denied notice.
- **Prompt Injection Defense**: User queries and retrieved evidence are wrapped with explicit demarcation boundaries (`--- BEGIN UNTRUSTED USER QUERY ---`) and Ollama is constrained by strict security directives to prevent instruction overrides.

---

## 2. Preconditions Checklist

Before executing the live service, verify every item below:

- [ ] **Windows Console Encoding**: Console must support UTF-8 (for emojis and formatting):
  ```powershell
  chcp 65001
  ```
- [ ] **Ollama Service Active**: Local Ollama server must be running and model loaded:
  ```powershell
  ollama list
  # Confirm 'llama3.2:3b' (or configured model) is listed
  curl.exe http://127.0.0.1:11434/api/tags
  ```
- [ ] **Environment Configuration (.env)**:
  - `TELEGRAM_BOT_TOKEN` is populated with a valid bot token from @BotFather.
  - `TELEGRAM_CHAT_ID` matches your designated target chat or channel.
  - `TELEGRAM_ADMIN_ID` matches your personal numeric Telegram user ID (for `/status` access).
  - `TIMEZONE="Asia/Kolkata"`
  - `OLLAMA_MODEL="llama3.2:3b"`
- [ ] **Windows Power & Sleep Settings**:
  - Disable Windows automatic sleep / standby while the bot is running to avoid missing the 08:00 AM or 08:00 PM briefing windows.
- [ ] **Test Suite & Lint Cleanliness**:
  ```powershell
  .venv\Scripts\pytest -q
  .venv\Scripts\ruff check .
  ```

---

## 3. Staged Activation Procedure

Execute each stage sequentially. Do not advance to the next stage until the current stage succeeds.

### Stage 1: Configuration & Preflight Verification
Verify environment settings and Supabase schema readiness:
```powershell
# 1.1 Verify configuration loading and secrets detection
.venv\Scripts\python.exe -m geopulse.cli check-config

# 1.2 Verify Supabase schema and RPC readiness (read-only verification)
.venv\Scripts\python.exe -m geopulse.cli check-preflight --scope full
```
*Expected Result*: `Configuration loaded successfully.` and `Preflight check passed.`

---

### Stage 2: Dry-Run Single Briefing Preview
Simulate the full briefing pipeline with live RSS feeds and local Ollama summarization without sending messages:
```powershell
.venv\Scripts\python.exe -m geopulse.cli brief --edition morning --preview
```
*Expected Result*:
- Live articles fetched and defense relevance evaluated.
- Multi-source clustering executed.
- Ollama generates event summaries and significance ratings.
- Telegram HTML chunks are printed to the console.
- Output ends with: `PREVIEW COMPLETE: No messages were sent to Telegram.`

---

### Stage 3: Controlled Single Live Digest Send
Deliver exactly one real morning briefing to the configured Telegram chat:
```powershell
.venv\Scripts\python.exe -m geopulse.cli brief --edition morning --send
```
*Expected Result*:
- Digest chunks arrive in the target Telegram chat with correct formatting, badges, and clickable links.
- Console displays successful delivery with message IDs.
- Delivery record is saved to `.geopulse\delivery_state.json`.

*Verification in Telegram*:
1. Inspect the received Telegram message chunks.
2. Confirm source links open valid news articles.
3. Confirm formatting and emojis render cleanly without raw HTML tags.

---

### Stage 4: Continuous Service Activation (Live Mode)

#### Option A: Using the PowerShell Launcher (Recommended)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_geopulse_windows.ps1 -Live
```

#### Option B: Direct Python Execution
```powershell
.venv\Scripts\python.exe -m geopulse.cli service run
```

*(Note: Add `--persist` if Supabase database storage is desired)*:
```powershell
.venv\Scripts\python.exe -m geopulse.cli service run --persist
```

*Expected Behavior*:
- Acquires lock at `.geopulse\service.lock`.
- Displays startup banner showing timezone (`Asia/Kolkata`), morning/evening schedule hours, and Ollama model.
- If started after 08:00 AM and today's morning briefing was not yet sent, sends the catch-up morning briefing immediately.
- Enters polling loop: checks schedules every 1s and polls Telegram for incoming Q&A updates.

---

## 4. Operational Monitoring & Interactive Verification

Once the service is running in Stage 4:

1. **Verify `/help`**: Send `/help` to your bot in Telegram. It should return available commands and grounding principles.
2. **Verify `/latest`**: Send `/latest`. It should display the top briefed events from the latest cycle.
3. **Verify `/status` (Admin)**: Send `/status`.
   - From admin account: Displays health, cached articles/events count, and last briefing timestamp.
   - From non-admin account: Displays `Access Restricted` warning.
4. **Verify Grounded Q&A**: Ask a question related to a recent briefed conflict (e.g. *"What is the latest situation in Ukraine?"*).
   - Bot should respond with concise facts and a `Sources:` footer.
5. **Verify Out-of-Scope / Non-Reported Topic**: Ask about an unreported or fabricated topic.
   - Bot should respond with `Insufficient Evidence` rather than hallucinating.

---

## 5. Rollback & Stop Instructions

### Graceful Stop
In the active terminal window running Geopulse:
- Press `Ctrl + C`.
- The service catches `SIGINT`, logs `Shutdown signal received. Stopping Geopulse service runner...`, releases the lock at `.geopulse\service.lock`, and exits cleanly.

### Force Termination (if unresponsive)
Open a separate PowerShell window (execute commands from the repository root):
```powershell
$repoRoot = (Get-Location).Path
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$repoRoot*" } | Stop-Process -Force
```

### Clearing a Stale Lock
If the machine shut down abruptly while the service was running, remove the orphaned lock file before restarting (execute from the repository root):
```powershell
$repoRoot = (Get-Location).Path
$lockPath = Join-Path $repoRoot ".geopulse\service.lock"
Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
```

### Inspecting Delivery State
To inspect the current local delivery state (execute from the repository root):
```powershell
$repoRoot = (Get-Location).Path
$statePath = Join-Path $repoRoot ".geopulse\delivery_state.json"
Get-Content $statePath | ConvertFrom-Json
```

### Forcing a Re-Send after Failure or Ambiguous Timeout
If a briefing was interrupted or halted due to an ambiguous network timeout:
```powershell
.venv\Scripts\python.exe -m geopulse.cli brief --edition morning --send --force
```
