# Geopulse V1 — Operator User Guide

**System Version:** V1.0  
**Classification:** Operational / Technical Guidance  
**Target Audience:** Intelligence Analysts, System Operators, Geopolitical Monitors  

---

## 1. What Geopulse Does

Geopulse is an operator-supervised geopolitical intelligence briefing system designed to transform voluminous open-source reporting into concise, source-grounded intelligence digests.

### Core Processing Pipeline
1. **Multi-Source Ingestion:** Ingests articles from verified global wire services, government ministries, regional defence outlets, and analytical publications.
2. **Text Normalization & Deduplication:** Cleans HTML, extracts canonical URLs, sanitizes text, and computes SHA-256 cryptographic hashes to eliminate duplicates.
3. **Event Clustering:** Groups topically related articles into distinct geopolitical events using lexical, temporal, and entity heuristics.
4. **Factual Claim & Evidence Extraction:** Identifies discrete, verifiable factual assertions and maps them directly to exact supporting evidence spans in source text.
5. **Source Lineage & Dependency Tracking:** Distinguishes original investigative reporting from wire syndication, official press releases, and derivative reporting.
6. **Contradiction & Conflict Detection:** Identifies contested claims and military/diplomatic denials, applying conservative evidentiary standards.
7. **Human-in-the-Loop Supervision:** Intercepts every candidate event claiming independent corroboration. Unapproved events cannot bypass operator sign-off.
8. **Structured Briefing Generation:** Formats validated events into Telegram-ready HTML digests structured with clear editorial priorities and confidence badges.

---

## 2. System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 22.04+ recommended).
- **Python:** Python 3.11 or higher (Python 3.14 compatible).
- **Memory & Storage:** Minimum 8 GB RAM; at least 5 GB free disk space for local models and runtime caches.
- **Local LLM Engine (Default):** Ollama daemon running `llama3.2:3b` (or `mistral`/`llama3`).
- **Cloud LLM (Optional Alternative):** Google Gemini API (`gemini-3.8-flash` or `gemini-1.5-pro`).
- **Database (Optional Persistence):** Supabase PostgreSQL (operating in safe in-memory dry-run mode if unconfigured).

---

## 3. Installation & Setup

### Step 1: Clone or Open Workspace
```powershell
cd "c:\Users\KIIT\Downloads\WAR ROOM"
```

### Step 2: Create & Activate Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -e .
```

### Step 4: Configure Local LLM (Ollama)
Ensure Ollama is installed and running:
```powershell
ollama serve
ollama pull llama3.2:3b
```

---

## 4. Configuration (`.env`)

Create a local `.env` file in the project root:
```ini
# Application Environment
APP_ENV=development
TIMEZONE=Asia/Kolkata

# LLM Configuration (Default: local Ollama)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Optional Cloud LLM (Gemini)
# GEMINI_API_KEY=your-api-key-here
# GEMINI_MODEL=gemini-3.8-flash

# Optional Database Persistence (Supabase)
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-service-role-key

# Optional Live Telegram Delivery (Dry-Run by default)
# TELEGRAM_BOT_TOKEN=your-bot-token
# TELEGRAM_CHAT_ID=your-chat-id
```

> [!IMPORTANT]
> Never commit `.env` or paste live API keys into public logs. Geopulse operates in dry-run mode by default; credentials are only required if live database persistence or live Telegram broadcasting is explicitly authorized.

---

## 5. Operational Health & Configuration Probes

Before running briefing passes, verify your environment:

### A. Configuration Check
Inspect environment variables and model selection without revealing credentials:
```powershell
geopulse check-config
```

### B. Comprehensive Health Check
Execute live connectivity and readiness probes across LLM endpoints, RSS sources, writable directories, and supervisory storage:
```powershell
geopulse check-health
# Or convenience alias:
geopulse health
```
Expected output:
```
============================================================
Geopulse V1 Operational Health Check
============================================================
[PASS] Configuration loaded successfully.
Checking LLM Provider (ollama)...
[PASS] Ollama reachable. Model 'llama3.2:3b' available.
Checking configured RSS Feeds (6 enabled)...
[PASS] RSS feeds probe: reachability verified (3/3 sampled feeds responded).
[PASS] Local runtime directory (.geopulse/) writable.
[INFO] Supervisory Pilot Storage: 0 review decisions, 0 events in queue.
============================================================
Health check completed successfully.
============================================================
```

---

## 6. Daily Intelligence Briefing Operations

### A. Running Ingestion
Collect and store articles from all configured open-source RSS feeds:
```powershell
# Dry-run mode (in-memory evaluation)
geopulse ingest

# Or target a specific feed
geopulse ingest --feed "Reuters World"
```

### B. Supervised Daily Briefing (Dry-Run Preview)
Generate the morning briefing under operator supervision:
```powershell
# Generate and preview in terminal without sending to Telegram
geopulse brief --pilot-supervision --preview

# Run using local evaluation fixtures (isolated from live feeds)
geopulse brief --pilot-supervision --fixture --mock-model
```

If any candidate event proposes independent corroboration, the CLI triggers an immediate interception alert:
```
[PILOT SUPERVISION ALERT]: Intercepted 1 unapproved [CORROBORATED] event(s).
  - Held in review queue: 'Intercepted Naval Operation' (Claims: 1, Sources: 2)
These events are held from the digest until approved by a human analyst.
```

---

## 7. Operator Supervisory Review Workflow

The supervisory pilot gives analysts full visibility into the evidence graph before publication.

### Command Reference

| Action | Command | Purpose |
| :--- | :--- | :--- |
| **List Queue** | `geopulse pilot queue` | Inspect events awaiting review or approved |
| **Filter Queue** | `geopulse pilot queue --status pending` | View only items needing immediate sign-off |
| **Inspect Event** | `geopulse pilot inspect <event-id>` | View complete claims, spans, and reporting lineages |
| **Approve** | `geopulse pilot review <id> --action approve ...` | Verify primary corroboration and release event |
| **Downgrade** | `geopulse pilot review <id> --action downgrade_shared_origin ...` | Downgrade to shared origin or official statement |
| **Reject** | `geopulse pilot review <id> --action reject ...` | Exclude unverified or untrustworthy event |
| **Preview Final** | `geopulse pilot preview` | Preview final digest with decisions applied |

### Step-by-Step Operator Journey

#### Step 1: Check the Review Queue
```powershell
geopulse pilot queue
```
Output:
```
============================================================
Geopulse Operator Supervisory Review Queue
Total Events Enqueued: 1 (Filter: all)
============================================================
#01 [PENDING_APPROVAL]         [CORROBORATED]         (FP:dc97ed30)
    Event ID: 40daee18-cf7a-4a0a-9701-4103549019fb
    Title:    Allied Naval Forces Patrol Red Sea
    Sources:  2 publishers, 2 articles
------------------------------------------------------------
```

#### Step 2: Inspect Event Evidence & Lineage
```powershell
geopulse pilot inspect 40daee18-cf7a-4a0a-9701-4103549019fb
```
Examine:
- **Central Factual Claims:** Verbatim evidence spans and source URLs.
- **Reporting Lineage:** Whether sources are independent field correspondents, syndicated wire dispatches, or official announcements.
- **Candidate Fingerprint:** 16-character hash binding the decision to exact text and evidence.

#### Step 3: Record Review Decision
Analyst explanations must be at least 10 characters:

- **Approve genuinely corroborated events:**
  ```powershell
  geopulse pilot review 40daee18-cf7a-4a0a-9701-4103549019fb `
    --action approve `
    --analyst-id "ANALYST-NAV-01" `
    --reason "Verified primary reporting from distinct on-scene correspondents."
  ```

- **Downgrade shared origin or syndicate quotes:**
  ```powershell
  geopulse pilot review 40daee18-cf7a-4a0a-9701-4103549019fb `
    --action downgrade_shared_origin `
    --analyst-id "ANALYST-NAV-01" `
    --reason "Both outlets quote the identical primary wire dispatch."
  ```

- **Downgrade joint allied press releases:**
  ```powershell
  geopulse pilot review <event-id> `
    --action downgrade_official `
    --analyst-id "ANALYST-NAV-01" `
    --reason "Joint communique from coalition partner ministries."
  ```

- **Reject unverified or fabricated reports:**
  ```powershell
  geopulse pilot review <event-id> `
    --action reject `
    --analyst-id "ANALYST-NAV-01" `
    --reason "Disinformation claim unsupported by satellite monitoring."
  ```

#### Step 4: Preview Final Briefing
Preview the completed, post-review intelligence briefing:
```powershell
geopulse pilot preview
```

---

## 8. Confidence Badges & Evidentiary Meaning

Every briefed event carries an explicit evidentiary badge:

| Badge | Meaning | Evidentiary Standard Required |
| :--- | :--- | :--- |
| `[CORROBORATED]` | Independently Verified | $\ge 2$ demonstrably independent reporting chains with verified claim evidence **AND** signed human analyst approval. |
| `[SHARED ORIGIN]` | Multi-Outlet Syndication | Multiple outlets republishing or referencing the same underlying reporting or wire copy. |
| `[OFFICIAL ANNOUNCEMENT]` | Institutional / State Release | Government ministry, military command, or official institutional statement. |
| `[SINGLE SOURCE]` | Single-Outlet Reporting | Factual claims reported by exactly one outlet; unverified by external sources. |
| `[DISPUTED]` | Conflicting Evidence | Active informational contestation or formal denials between official spokespersons and independent actors. |
| `[PRELIMINARY]` | Early / Incomplete | Developing incident with insufficient corroboration or unverified casualty/damage claims. |
| `[NOT ASSESSED]` | Unassessed / Legacy | Historical event or pipeline record without claim-level corroboration analysis. |

---

## 9. Evidentiary Integrity & Cryptographic Fingerprints

To prevent stale approvals from silently applying to modified intelligence:

- Every candidate event is assigned a SHA-256 **Candidate Fingerprint** derived from its headline, narrative, claim IDs, claim classifications, verbatim evidence spans, supporting URLs, and article contents.
- When an analyst approves or downgrades an event, the decision is cryptographically bound to that exact fingerprint.
- **Fail-Closed Guarantee:** If newly ingested articles modify the event's claims, add evidence spans, or alter narrative facts, the fingerprint changes. The prior approval is automatically invalidated, the event status transitions to `STALE_RE_REVIEW_REQUIRED`, and the event is intercepted for re-review.

---

## 10. Troubleshooting & Common Operational Errors

### 1. "Ollama daemon not reachable"
- **Cause:** Local Ollama service is stopped.
- **Fix:** Open a terminal and run `ollama serve`. In a separate window, verify `ollama list`.

### 2. "Model 'llama3.2:3b' not found"
- **Cause:** Model has not been pulled locally.
- **Fix:** Execute `ollama pull llama3.2:3b`.

### 3. "Error: Pilot supervision mode is strictly dry-run. Live Telegram delivery is prohibited."
- **Cause:** Both `--pilot-supervision` and `--send` were passed.
- **Fix:** The pilot workflow strictly prohibits automated network delivery. Use `--preview` instead.

### 4. "Event ID not found in the pilot review queue"
- **Cause:** Event has not been processed in a supervised briefing pass or was purged.
- **Fix:** Run `geopulse brief --pilot-supervision --preview` first to populate the review queue.

### 5. "Validation Error: Analyst rationale must be a meaningful explanation"
- **Cause:** Rationale was blank or under 10 characters.
- **Fix:** Provide substantive context (e.g., `--reason "Verified against primary naval tracking logs"`).

---

## 11. Known Limitations & Operating Guardrails

1. **Unattended Autonomous Publication Disabled:** By system invariant, autonomous publication of corroborated intelligence is forbidden. All `[CORROBORATED]` events require explicit analyst sign-off.
2. **RSS Snippet Constraints:** When news feeds only provide short RSS summaries without full body text, claim extraction operates on available snippets.
3. **Local Compute Performance:** Running local Ollama inference on consumer CPUs typically requires 15–30 seconds per clustered event. For faster processing, utilize Gemini (`LLM_PROVIDER=gemini`).
4. **Task Scheduler:** Background automated scheduling remains disabled by default; manual or supervised CLI execution is standard for V1.
