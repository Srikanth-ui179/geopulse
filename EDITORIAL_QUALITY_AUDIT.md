# Geopulse Editorial Quality Audit & MVP Readiness Review

**Audit Date:** 2026-10-24  
**Phase:** Phase 6G — Editorial Precision & Noise Filtering  
**Environment:** Windows, Python 3.14.7, Ollama (`llama3.2:3b`), Telegram HTML Digest  
**Project Workspace:** `Geopulse` (Repository Root)  
**Test Suite Status:** 209 passed / 0 failed (49.02s runtime)  
**Code Quality & Linter:** Ruff check 0 errors (`All checks passed!`)

---

## 1. Executive Summary & Quality Gate Status

* **Editorial Quality Gate Status:** **PASSED**
* **Readiness Verdict:** **GO FOR STAGED OPERATOR-CONTROLLED PRIVATE TEST**
* **Baseline vs Final Test Count:** 202 passed ➔ **209 passed** (+7 new regression tests, 0 failures, 0 warnings)
* **Real-Feed Preview Outcome (Phase 6G):**
  - **Total Events:** **8 high-signal events** (reduced from 10; exactly 2 routine UK MoD administrative court notices filtered out).
  - **Analysis De-prioritization:** Operational breaking developments lead the digest (#1 through #7); strategic analysis ("The Endless Chip War") is cleanly de-prioritized to the final position (#8/8).
  - **Telegram Chunks:** 2 chunks (3,805 and 2,554 characters), with generous headroom below Telegram's 4,096-character limit.
  - **Safety Boundaries:** Zero database mutations, zero Telegram calls, zero delivery-state writes.

---

## 2. Before vs. After Gate D Comparison

| Metric | Gate D Baseline (Phase 6F) | Gate D Post-Filter (Phase 6G) | Impact & Editorial Significance |
|:---|:---:|:---:|:---|
| **Articles Ingested** | 12 (4 feeds, limit 3) | 12 (4 feeds, limit 3) | Same live RSS corpus evaluated. |
| **Events Formed** | 10 events | **8 events** | 2 routine court-martial administrative listings eliminated. |
| **High-Signal Intelligence Ratio** | 80% (8/10 events) | **100% (8/8 events)** | 100% of briefed stories are strategic geopolitical/defence news. |
| **Administrative Noise** | 2 notices (#7 sentencing, #10 listings) | **0 notices (Filtered out)** | Routine legal dockets excluded from briefing; archived in raw articles table. |
| **Lead Headline Position (#1)** | *The Endless Chip War* (Analysis) | *Natilus Air Force BWB Contract* (Breaking) | Operational news now leads the briefing. |
| **Analysis Representation** | Position #1, labelled `[ANALYSIS]` | **Position #8 (Final)**, labelled `[ANALYSIS]` | Commentary de-prioritized to background section; publication date preserved. |
| **Telegram Chunk 1 Length** | 3,775 characters | 3,805 characters | Safely below 4,096 ceiling (safe margin: 291 chars). |
| **Telegram Chunk 2 Length** | 3,985 characters (tight) | **2,554 characters (spacious)** | Safe margin expanded from 111 chars to **1,542 chars**. |
| **Automated Tests** | 202 passed | **209 passed** | +7 regression tests covering noise filters, analysis ranking, and corroboration. |

---

## 3. Implemented Improvements & Code-Level Details

### 3.1 Targeted Administrative-Noise Filtering
- **Problem:** Official UK Ministry of Defence feeds publish routine domestic court-martial transparency records and daily court center dockets (e.g. `Military Court Service sentencing remarks 2026`, `Military court centres court listings`). These passed default relevance because UK MoD is a dedicated defence source.
- **Solution:**
  - Added targeted regex patterns to `IRRELEVANT_KEYWORDS` in `src/geopulse/ingestion/relevance.py` and `IRRELEVANT_PATTERNS` in `src/geopulse/classification/rules.py` covering:
    - `transparency data:\s*military court`
    - `court listings? (?:for|at|of)?\s*military court(?: centres?)?`
    - `military court(?: centres?)? court listings?`
    - `(?:military )?court[- ]martial (?:court )?listings?`
    - `sentencing remarks in (?:the )?court[- ]martial`
    - `military court service sentencing remarks`
    - `transcripts of sentencing remarks`
  - **Strategic Override Protection:** Added `_STRATEGIC_LEGAL_OVERRIDE_PATTERN` (`treason`, `espionage`, `classified`, `secrets`, `war crimes`, `atrocities`, `homicide`, `murder`, `bribery`, `corruption`, `reform`, `mutiny`, `sabotage`, `trafficking`). If a court-martial story involves high-consequence misconduct, espionage, or major policy reform, it is **never** filtered out.
  - **Archival Integrity:** Filtered stories are assigned `relevance_status = "irrelevant"`, meaning they are preserved in persistent article storage for auditability, but excluded from clustering and daily briefings.

### 3.2 Strategic Analysis De-prioritization
- **Problem:** In Gate D, "The Endless Chip War" appeared as Event #1. While correctly labeled as `[ANALYSIS]` and `DEFENCE_POLICY`, having an essay lead the briefing distorted the primary purpose of a daily operational digest.
- **Solution:**
  - In `src/geopulse/clustering/clusterer.py`: Updated `EventClusterer.cluster_articles` to sort finalized clusters with operational news (`is_analysis=False`) first and analytical commentary (`is_analysis=True`) last.
  - In `src/geopulse/delivery/formatter.py`: Updated `TelegramDigestFormatter.format_digest` to sort successful events with `key=lambda r: 1 if getattr(r.event, "is_analysis", False) else 0`.
  - In `src/geopulse/pipeline.py` and `src/geopulse/cli.py`: Synchronized briefing event ranking so database persistence and chat formatting maintain the exact same sequence.
  - Result: All operational developments are briefed under ranks `#1` through `#7`, with "The Endless Chip War" briefed as `#8/8` with its explicit `[ANALYSIS]` badge and original publication date (`October 22, 2026`).

### 3.3 Rigorous Corroboration Semantics & Circular Wire Echo Guardrails
- **Problem:** If multiple outlets merely quoted the same state news agency (e.g. TASS, RIA Novosti) or repeated a single defence ministry press release, previous naive logic upgraded the event to `[CORROBORATED]` solely because `source_count >= 2`.
- **Solution:**
  - Added `has_single_origin_dependency()` in `src/geopulse/clustering/similarity.py`:
    1. **Derived Reporting Detection:** Checks if one outlet in the cluster explicitly attributes its report to another source in the cluster (e.g. *Defense News* writing "According to a UK Ministry of Defence announcement...").
    2. **Single-Wire Echo Detection:** Uses regex `_ATTRIBUTION_PATTERN` to detect if all outlets in the cluster attribute the story to the exact same originating wire or state agency (`TASS`, `RIA Novosti`, `Interfax`, `SANA`, `state news agency`, etc.).
  - In `src/geopulse/clustering/clusterer.py`: When `has_single_origin_dependency` returns `True`, the cluster confidence is locked to `single_source`, preventing unearned promotion to `corroborated`.
  - In `src/geopulse/summarization/service.py`: Assessment texts for corroborated events explicitly state:
    `"Corroborated across X reporting outlets; military claims unconfirmed independently."`
    And for single-origin derived multi-outlet coverage:
    `"Single-origin reporting across outlets; unverified independently."`
    Geopulse never claims to automatically verify factual ground truth.

---

## 4. Regression Verification & New Tests

| Test Module | Baseline Count | Final Count | New Tests Added |
|:---|:---:|:---:|:---|
| `tests/unit/test_relevance.py` | 9 | **11** | • `test_routine_administrative_court_listings_filtered_as_irrelevant`<br>• `test_consequential_military_justice_and_reforms_preserved_as_relevant` |
| `tests/unit/test_clustering.py` | 16 | **20** | • `test_multiple_outlets_citing_same_wire_agency_remain_single_source`<br>• `test_outlet_citing_another_outlets_press_release_remains_single_source`<br>• `test_multiple_articles_from_single_source_remain_single_source`<br>• `test_operational_events_precede_analytical_essays_in_cluster_ordering` |
| `tests/unit/test_telegram_formatter.py` | 9 | **10** | • `test_format_digest_deprioritizes_analysis_behind_operational_events` |
| **Total Test Suite** | **202** | **209** | **+7 passing tests (0 failures, 0 warnings)** |

---

## 5. Summary of Changed Files

1. `src/geopulse/ingestion/relevance.py`: Added administrative court listing patterns and `_STRATEGIC_LEGAL_OVERRIDE_PATTERN`.
2. `src/geopulse/classification/rules.py`: Aligned `IRRELEVANT_PATTERNS` and added `STRATEGIC_LEGAL_OVERRIDE_PATTERN` in rule-based classifier.
3. `src/geopulse/clustering/similarity.py`: Implemented `_ATTRIBUTION_PATTERN` and `has_single_origin_dependency()`.
4. `src/geopulse/clustering/clusterer.py`: Integrated single-origin dependency check in `to_event_cluster()` and operational-first sorting in `cluster_articles()`.
5. `src/geopulse/delivery/formatter.py`: Enforced operational-first event ordering in `format_digest()`.
6. `src/geopulse/summarization/service.py`: Added explicit single-origin qualification guardrail when multiple outlets cite a single origin.
7. `tests/unit/test_relevance.py`: Added positive and negative test cases for court listings vs consequential military justice.
8. `tests/unit/test_clustering.py`: Added tests for wire echoing, press release derivation, same-source multi-article clustering, and analysis de-prioritization.
9. `tests/unit/test_telegram_formatter.py`: Added test verifying operational news precedes analysis in formatted Telegram chunks.

---

## 6. Known Remaining Limitations

1. **Syndicated Wire Feeds:** If multiple outlets publish identical wire copy from Reuters or AP without including the wire service name in the RSS title or snippet, string-matching attribution heuristics cannot distinguish it from independent parallel coverage. Full body-text entity extraction in Phase 7 will address this.
2. **Analysis Detection Heuristics:** Detection currently relies on designated analysis sources (War on the Rocks) and explicit title/snippet patterns (`analysis`, `commentary`, `opinion`, `review`, `interview`, `essay`). Long-form reporting that contains subtle analysis without keyword cues is treated as operational reporting.
3. **RSS Feed Depth:** When testing with `--limit 3`, stories published more than ~12 hours ago can be pushed below the limit by high-volume outlets. Production operation should use `--limit 10`.

---

## 7. Recommendation & Next Steps

### **VERDICT: GO FOR PRIVATE INTERNAL TESTING**

The editorial quality gate is fully satisfied. The digest exhibits 100% strategic relevance, routine administrative noise has been purged, analysis is properly separated and de-prioritized, and corroboration semantics are strictly honest about what multi-outlet coverage represents.

**Recommended Next Step:**
Proceed to **Gate E** of the staged runbook:
```powershell
.\.venv\Scripts\python.exe -m geopulse test-telegram
```
*(Requires operator approval to send exactly one diagnostic ping message to verify Telegram credentials).*
