# Geopulse V1 — First Real Run Diagnostic

**Diagnostic Date**: November 05, 2026  
**Scope**: Read-Only Post-Operational Diagnostic of First Real-World Briefing Run  
**Evaluated Run**: `task-16165` (`geopulse brief --pilot-supervision --preview`) and `task-16179` (`geopulse pilot preview`)  
**Git HEAD**: `eb41b38bc351ffa0e1fac0de08f01065328680e9`  

---

## Executive Overview

The first operational run of Geopulse V1 processed 60 articles across 6 configured news feeds, successfully producing an operator-supervised intelligence briefing digest. This diagnostic analyzes the clustering dynamics, LLM extraction fidelity across 47 Ollama inference calls, the five selected briefing items, and why zero events entered the pending corroboration queue.

### High-Level Diagnostic Summary

| Investigation Area | Key Diagnostic Finding | System Health |
| :--- | :--- | :--- |
| **Cluster Structure** | 19 clusters formed, all singletons (1 article, 1 publisher each). | Functional, but under-clustered |
| **Cross-Publisher Ingestion** | BBC and Al Jazeera (20 articles) were 100% filtered out by relevance rules. War on the Rocks contributed 5 analysis pieces. Only Defense News (8) and Breaking Defense (6) provided breaking news. | Feed imbalance / strict classification |
| **Full-Text Extraction** | 19/19 clustered articles achieved cached full-text extraction via Trafilatura (88 to 1,422 words). | High Fidelity (100% full text) |
| **Ollama Inference (47 Calls)** | 47 calls executed. 1 request hit the 320-token `num_predict` limit; 1 prompt timed out on a 1,422-word longform article, producing a fallback summary. | Resilient failover, minor headline truncation |
| **Briefing Accuracy (Top 5)** | All 5 selected events accurately reflect source content, but represent official government/corporate announcements rather than corroborated investigative reporting. One headline was clipped mid-sentence. | Factually grounded, headline edge-cases |
| **Corroboration Queue** | Zero items in `pending` queue because 0 events claimed independent multi-chain corroboration. All 19 events safely received `[PASS_THROUGH]` with explicit `[SINGLE SOURCE]` disclaimers. | Zero false positives (Safe) |

---

## 1. Inventory & Lineage of All 19 Event Clusters

Every cluster formed during the operational run originated from a single article and publisher. All 19 articles successfully had their full body text extracted and cached.

| # | Event Title / Generated Headline | Articles | Publishers | Reporting Origin & Lineage | Pub Date (UTC) | Extraction Status |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **1** | **Australia preps Bluebottle USV for submarine-hunting role**<br>_HL: Australia preps Bluebottle USV for submarine-hunting role_ | 1 | 1 | Defense News (Gordon Arthur)<br>`original_reporting` (Origin: Defense News) | 2026-11-03 12:19 | Full Text<br>(481 words, success) |
| **2** | **French defense minister kicks off ‘pact’ with industry in rush to field naval drones**<br>_HL: French defense minister kicks off ‘pact’ with industry in rush to field naval drones_ | 1 | 1 | Defense News (Sebastian Sprenger)<br>`official_statement` (Origin: Ministry of Defense) | 2026-11-03 15:14 | Full Text<br>(419 words, success) |
| **3** | **EXCLUSIVE: SNC pulls out of Navy’s trainer jet competition, leaving Textron likely winner**<br>_HL: The service is now planning to issue an award for the UJTS program between July and the end of September 2027._ | 1 | 1 | Breaking Defense (Diana Stancy)<br>`original_reporting` (Origin: Breaking Defense) | 2026-11-03 23:19 | Full Text<br>(561 words, cached) |
| **4** | **MBDA’s high-speed strike missile to have range ‘far’ exceeding Exocet**<br>_HL: MBDA’s high-speed strike missile to have range ‘far’ exceeding Exocet_ | 1 | 1 | Defense News (Rudy Ruitenberg)<br>`derivative` (Origin: Gauffin / Euronaval Briefing) | 2026-11-04 06:30 | Full Text<br>(777 words, success) |
| **5** | **France tasks Naval Group to develop large oceangoing underwater drone**<br>_HL: France tasks Naval Group to develop large oceangoing underwater drone_ | 1 | 1 | Defense News (Rudy Ruitenberg)<br>`original_reporting` (Origin: Defense News) | 2026-11-04 09:51 | Full Text<br>(472 words, success) |
| **6** | **Babcock launches digital platform to link warship construction and fleet support**<br>_HL: Babcock launches digital platform to link warship construction and fleet support_ | 1 | 1 | Defense News (Linus Hoeller)<br>`original_reporting` (Origin: Defense News) | 2026-11-04 10:26 | Full Text<br>(225 words, success) |
| **7** | **Israel Weapons Industry expands into Southeast Asia with Thailand’s deployment of counter-drone system**<br>_HL: Israel Weapons Industry expands into Southeast Asia with Thailand’s deployment of counter-drone system_ | 1 | 1 | Breaking Defense (Seth J. Frantzman)<br>`original_reporting` (Origin: Breaking Defense / IWI) | 2026-11-04 11:09 | Full Text<br>(537 words, updated) |
| **8** | **Saildrone expands its European market reach with manufacturing partner in Norway**<br>_HL: Saildrone has partnered with Norwegian naval builder Umoe Mandal to produce the Surveyor multipurpose surveillance craft locally_ | 1 | 1 | Defense News (Sebastian Sprenger)<br>`original_reporting` (Origin: Defense News) | 2026-11-04 11:47 | Full Text<br>(455 words, success) |
| **9** | **Leonardo, Fincanteri revamp joint venture, aiming to trim frigate build time by 18 months**<br>_HL: Leonardo, Fincanteri revamp joint venture, aiming to trim frigate build time by 18 months_ | 1 | 1 | Breaking Defense (Elisabeth Gosselin-Malo)<br>`original_reporting` (Origin: Breaking Defense) | 2026-11-04 14:14 | Full Text<br>(423 words, cached) |
| **10** | **Shield AI cofounder teases ‘snowball’ of X-BAT fighter drone announcements in Europe**<br>_HL: Shield AI cofounder teases ‘snowball’ of X-BAT fighter drone announcements in Europe_ | 1 | 1 | Breaking Defense (Tim Martin)<br>`original_reporting` (Origin: Breaking Defense) | 2026-11-04 15:47 | Full Text<br>(390 words, cached) |
| **11** | **EXCLUSIVE: House China committee pushes Air Force on think tank cuts**<br>_HL: EXCLUSIVE: House China committee pushes Air Force on think tank cuts_ | 1 | 1 | Breaking Defense (Michael Marrow)<br>`original_reporting` (Origin: Breaking Defense) | 2026-11-04 16:28 | Full Text<br>(511 words, success) |
| **12** | **India taps two domestic shipbuilders for $5.7B corvette program**<br>_HL: India taps two domestic shipbuilders for $5.7B corvette program_ | 1 | 1 | Breaking Defense (Mike Yeo)<br>`original_reporting` (Origin: Breaking Defense) | 2026-11-04 17:37 | Full Text<br>(397 words, cached) |
| **13** | **Under threat of air strikes, US looks to bolster base infrastructure with 3D printing**<br>_HL: The US Air Force plans to develop mobile 3D concrete printers to erect buildings quickly at its bases in the Middle East damaged by Iranian_ | 1 | 1 | Defense News (Michael Peck)<br>`official_statement` (Origin: Department of Defense) | 2026-11-04 20:16 | Full Text<br>(434 words, success) |
| **14** | **What Israel’s Election Results Mean for the Middle East and Beyond**<br>_HL: Israel faces multiple conflicts_ | 1 | 1 | War on the Rocks (Michael Koplow)<br>`original_reporting` (Origin: War on the Rocks) | 2026-10-29 18:15 | Full Text<br>(421 words, success) |
| **15** | **The Pentagon-Silicon Complex and the Rise of the Neo-Primes**<br>_HL: The Pentagon-Silicon Complex and the Rise of the Neo-Primes_ | 1 | 1 | War on the Rocks (Jonathan Panter)<br>`original_reporting` (Origin: War on the Rocks) | 2026-11-02 09:00 | Full Text<br>(88 words, success) |
| **16** | **On ‘NATO’s lake,’ Nordic allies worry about escalation with Russia**<br>_HL: The Nord Stream pipeline attack in September 2022 was a significant development, with Russia consistently denying responsibility for acts_ | 1 | 1 | Defense News (Christopher Woody)<br>`original_reporting` (Origin: Defense News) | 2026-11-03 14:55 | Full Text<br>(1,075 words, success) |
| **17** | **The G.I. and the Paperback: A World War II Love Story**<br>_HL: 122,951,031 copies_ | 1 | 1 | War on the Rocks (Alana Davis)<br>`original_reporting` (Origin: War on the Rocks) | 2026-11-04 08:30 | Full Text<br>(1,422 words, success) |
| **18** | **The Army’s Command-and-Control Reboot**<br>_HL: The Army’s Command-and-Control Reboot_ | 1 | 1 | War on the Rocks (Ryan Evans)<br>`original_reporting` (Origin: War on the Rocks) | 2026-11-04 18:23 | Full Text<br>(107 words, success) |
| **19** | **Why Are Wars Increasingly Difficult to End?**<br>_HL: Why Are Wars Increasingly Difficult to End?_ | 1 | 1 | War on the Rocks (Monica Duffy Toft)<br>`original_reporting` (Origin: War on the Rocks) | 2026-11-04 18:30 | Full Text<br>(422 words, success) |

---

## 2. Cross-Publisher Clustering Investigation

A detailed audit of the 60 fetched articles was conducted to investigate why no cross-publisher clusters were formed:

### 2.1 Feed Breakdown & Classifier Filtering
- **BBC World News (10 articles)**: **100% filtered out**. Articles covered general international interest (e.g. US midterm politics, Ronaldo's sports career, French union strikes, Hong Kong judicial sentencing, Canadian bear attacks). None matched high-confidence defense keywords; 8 were categorized as `pending`/`uncertain` and 2 as `irrelevant`/`other_geopolitical`.
- **Al Jazeera English (10 articles)**: **100% filtered out**. Articles covering Gaza, US midterms, and Indonesian politics were classified as `pending`/`uncertain` by `RuleBasedClassifier` due to conversational headline formats (e.g., _"Can Iran’s new supreme leader make peace from behind the scenes?"_, _"‘Healthcare not bombs’: How Gaza shaped..."_).
- **UK Ministry of Defence (10 articles)**: Ingested routine administrative notices, low-flying training schedules, and medical doctrine factfiles rather than breaking geopolitical events.
- **War on the Rocks (10 articles)**: 5 were excluded as general essay commentary; 5 formed single-article analysis clusters.
- **Defense News & Breaking Defense (20 articles)**: Provided all 14 hard-news defense stories.

### 2.2 Coverage Overlap Analysis
Did different feeds cover the same underlying event?
1. **Euronaval 2026 Maritime Drones (Concrete Overlap)**:
   - **Defense News (Cluster #2)**: _"French defense minister kicks off ‘pact’ with industry in rush to field naval drones"_ (Nov 3).
   - **Defense News (Cluster #5)**: _"France tasks Naval Group to develop large oceangoing underwater drone"_ (Nov 4).
   - **Breaking Defense (Cluster #20 / Queue Item)**: _"Naval Group unveils new drone carrier concept"_ (Nov 4).
   - **Diagnosis**: All three articles covered announcements made at the Euronaval exhibition in Paris regarding the French Navy's drone modernization. However, because each outlet led with a different contract or concept ship, headline lexical similarity was below the `0.6` threshold, separating them into 3 distinct singleton clusters.
2. **Israeli Counter-Drone Deployment in Southeast Asia (Concrete Overlap)**:
   - **Breaking Defense (Cluster #7)**: _"Israel Weapons Industry expands into Southeast Asia with Thailand’s deployment of counter-drone system"_ (Auth: Seth J. Frantzman, Nov 4).
   - **Defense News (Subsequent feed cycle)**: _"Israeli defense company says this trigger is programmed to shoot drones"_ (Also reported by Seth J. Frantzman).
   - **Diagnosis**: The same defence correspondent covered this technological deployment for both publications under distinct angles. Because the 10-article feed window captured Breaking Defense's version first, cross-publisher merging did not trigger within the initial window.

### 2.3 Clustering Errors Assessment
- **Incorrectly Grouped Articles (False Positives)**: **0**. Precision was 100%; no unrelated articles were erroneously joined.
- **Incorrectly Separated Articles (False Negatives)**: **3 articles** covering the Euronaval 2026 French drone initiatives were split into separate clusters due to title lexical divergence.

---

## 3. Ollama Inference Call Telemetry & Evidence Grounding

Across the pipeline execution, **47 Ollama inference calls** were executed using `llama3.2:3b`.

### 3.1 Token Prediction Limit & Timeout Analysis
1. **The Truncated Request**:
   - In `task-16179.log`, Ollama logged:  
     `Ollama response truncated: hit num_predict limit (320 tokens); model=llama3.2:3b, elapsed=62.24s`  
     `Ollama request timed out after 6506.14s (prompt_chars=10423)`
   - **Root Cause**: The prompt for Cluster #17 (_"The G.I. and the Paperback: A World War II Love Story"_) contained 1,422 words (10,423 prompt characters) of historical prose. Ollama attempted to generate an exhaustive JSON extraction, hitting the 320-token ceiling (`STAGE_BUDGET_CLAIMS`) mid-generation.
   - **Pipeline Behavior**: The fail-safe summarization client caught the truncation error, rejected the invalid truncated JSON, and gracefully degraded to deterministic claim extraction without terminating the batch.

### 3.2 Evidence Grounding & Loss Assessment
- **Lost Grounding in Cluster #17**:
  - The model emitted: `Claim #1: "122,951,031 copies"` with `Evidence Span: "<short exact verbatim quote under 25 words>"`.
  - The model copied the instruction prompt verbatim as its evidence reference because token exhaustion prevented actual quote extraction.
- **Weak or Fragmentary Claims**:
  - **Cluster #3 (SNC Trainer Jet)**: The headline was extracted as `"The service is now planning to issue an award for the UJTS program between July and the end of September 2027."` rather than the primary event (SNC withdrawing from the competition).
  - **Cluster #15 (Pentagon-Silicon Complex)**: Sourced from an 88-word podcast summary. Extracted claim (`"She delves into the Pentagon’s disputes with SpaceX and Anthropic..."`) reflects promotional podcast copy rather than hard news facts.
  - **Cluster #18 (Army C2 Reboot)**: Extracted claim #1 was a 5-word title fragment: `"Next Gen Command and Control"`.

---

## 4. Evaluation of the Five Selected Briefing Events

The Telegram digest selected 5 top-ranked events (`max_events=5`):

```text
1. [🚨 SECURITY INCIDENT] Under threat of air strikes, US looks to bolster base infrastructure with 3D printing (Defense News)
2. [🏛️ DEFENCE POLICY] India taps two domestic shipbuilders for $5.7B corvette program (Breaking Defense)
3. [🏛️ DEFENCE POLICY] EXCLUSIVE: House China committee pushes Air Force on think tank cuts (Breaking Defense)
4. [🚀 WEAPONS PROCUREMENT] Israel Weapons Industry expands into Southeast Asia with Thailand’s deployment of counter-drone system (Breaking Defense)
5. [🚀 WEAPONS PROCUREMENT] French defense minister kicks off ‘pact’ with industry in rush to field naval drones (Defense News)
```

### 4.1 Headline Representation & Truncation
- **Event 1 (Air Force 3D Printing)**: **Truncated Headline**. The generated headline abruptly cuts off:  
  `"The US Air Force plans to develop mobile 3D concrete printers to erect buildings quickly at its bases in the Middle East damaged by Iranian"` (missing `"bombardment"` or `"attacks"`).
- **Events 2, 3, 4, 5**: Headlines accurately and completely represent their source reporting.

### 4.2 Narrative Fidelity & Unsupported Details
- **Zero Hallucinated Details**: In all 5 events, narrative sentences in `what_happened` correspond to exact verbatim quotes from the full-text cache.
- **Grounding Spans**:
  - Event 1: Air Force SBIR solicitation quote and DoD IG report on 8 Middle East countries.
  - Event 2: Exact shipyard names (Garden Reach Shipbuilders & Engineers, Goa Shipyard Limited) and contract scope (5 and 3 corvettes).
  - Event 3: Verbatim quotation of the congressional letter to Air Force Secretary Troy Meink.
  - Event 4: ARBEL Anti-Drone System Thai deployment and company statements.
  - Event 5: Catherine Vautrin's French drone ecosystem initiative.

### 4.3 Official Announcements vs. Independent Events
- **Event 1**: **Official Solicitation / Government Report** (DoD IG & SBIR solicitation).
- **Event 2**: **Government Contract Award** (Indian Ministry of Defence procurement).
- **Event 3**: **Congressional Investigation** (House Select Committee on China letter).
- **Event 4**: **Corporate / Vendor Promotion** (Israel Weapons Industry marketing & Thai military procurement).
- **Event 5**: **Government Ministerial Policy Announcement** (French Ministry of Defense).
- **Takeaway**: None of the top 5 stories represents an independently corroborated battlefield incident or covert operation; all 5 are institutional announcements, contract awards, or corporate disclosures reported by single specialized outlets.

---

## 5. Corroboration Review Queue Dynamics

The supervisory pilot queue reported:
- **Total Enqueued**: 19 events
- **Pending Review**: **0 events**
- **Pass-Through**: **19 events**

### 5.1 Why Zero Events Entered the Review Queue
1. **Safety Gate Architecture**: The Operator Supervisory Pilot enforces human review (`requires_approval = True`) strictly for events claiming `[CORROBORATED]`. Single-source, disputed, shared-origin, and preliminary events do not bypass review—they are transparently labeled with their exact qualification badge.
2. **Corroboration Engine Conservatism**: Geopulse V1 strictly forbids granting corroboration to single articles or single publishers. Because all 19 clusters were singletons, the corroboration engine correctly classified 100% of them as `[SINGLE SOURCE]`.
3. **Absence of False Positives**: In previous versions, single-source events with multiple internal quotes were occasionally mislabeled as corroborated. The Phase 3 conservatism fixes completely prevented this failure mode. Zero unverified stories claimed corroboration.

### 5.2 Expected Outcome vs. Clustering Limitation
- **Expected Outcome**: Zero pending items was the **expected and correct safety outcome** given that all clusters were singletons. The system acted with mathematical safety by not presenting any singleton as corroborated.
- **Clustering Limitation**: Having only singleton clusters was an **upstream ingestion limitation**. By sampling only 10 articles per feed and dropping 100% of BBC and Al Jazeera articles, cross-outlet overlap was severely restricted.

---

## 6. Recommendations (Non-Breaking, Prioritized)

The following three high-priority improvements are recommended for future operational releases:

### 1. Refine Ingestion Keyword Classification & Expand Per-Feed Ingestion Depth
- **Finding**: Dropping 100% of BBC and Al Jazeera articles eliminated the primary cross-publisher wires that naturally corroborate breaking geopolitical news. Furthermore, a 10-article limit is too narrow to catch overlapping coverage.
- **Recommendation**:
  1. Increase the default `feed_limit` from 10 to 25–30 articles.
  2. Add geopolitical conflict terms (e.g., _Gaza, Lebanon, Red Sea, airstrike, sanctions, ceasefire, Taiwan Strait_) to `RuleBasedClassifier` so major wire stories are retained rather than classified as `uncertain` or `other_geopolitical`.

### 2. Implement Entity and Venue-Aware Cluster Merging
- **Finding**: Related stories from the same conference or operation (e.g. Euronaval 2026 naval drone announcements in Paris) were split across 3 single-source clusters because article headlines focused on different sub-components.
- **Recommendation**: In `compute_article_similarity`, boost similarity when articles share multiple named entities (e.g. `Naval Group`, `French Navy`), identical locations (`Paris`), and overlapping 48-hour timestamps, allowing multi-angle conference coverage to cluster together while tracking separate reporting lineages.

### 3. Headline Truncation Safeguard & Context Window Budgeting
- **Finding**: Ollama `llama3.2:3b` occasionally produced clipped headlines (e.g. `"...damaged by Iranian"`) or grabbed minor sentences when processing long articles, and hit the 320-token prediction limit on 1,400-word articles.
- **Recommendation**:
  1. Increase `STAGE_BUDGET_CLAIMS` from 320 to 512 tokens.
  2. Cap input article text passed to Ollama at 800 words (the most substantive opening paragraphs).
  3. Enforce a validation check on generated headlines: if a headline ends with an incomplete word or conjunction, or is under 4 words, automatically fall back to the publisher's original article title.

