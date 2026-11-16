"""Prompt templates and builders for source-grounded event summarization."""

from geopulse.clustering.models import EventCluster
from geopulse.ingestion.extractor import select_bounded_article_passages
from geopulse.summarization.evidence_matcher import clean_article_text

SYSTEM_PROMPT = """You are a senior geopolitical and defence intelligence analyst for Geopulse.
Produce strictly factual, source-grounded event summaries for a concise intelligence briefing.

CRITICAL OPERATIONAL RULES:
1. OUTPUT FORMAT: Output MUST be valid JSON only matching the schema.
   No markdown code fences, comments, or extra text.
2. STRICT FACTUAL GROUNDING: Rely SOLELY on the provided source articles.
   NEVER invent casualties, dates, locations, military outcomes, actors,
   weapons, commodities, or strategic motives. Omit unsupported details cleanly
   (set to null). NEVER emit placeholder phrases like 'Unspecified', 'Unknown',
   or '[Unspecified in reporting]'.
3. ENTITY & ACTOR INTEGRITY: Every person, military unit, company, or state
   named in 'actors' or claims MUST appear explicitly in source reporting.
   Publishers and news outlets (e.g. BBC, Reuters, Defense News) are NEVER event
   actors. Use full equipment designations (e.g., 'F-47', 'Su-57', 'T-90').
4. EVIDENCE ATTRIBUTION: Every factual claim must specify:
   - 'src': The integer index of the cited article (e.g. 1 for [Source 1]).
   - 'evidence': A short, exact verbatim phrase under 5 words directly from that source.
     Do not paraphrase or alter words.
   - Pair claims strictly with the source containing the verbatim evidence span.
5. ATTRIBUTION & UNCERTAINTY:
   - Attribute statements, allegations, denials, or third-party data (e.g., Oryx, SIPRI)
     to the named speaker, institution, or monitor in 'attribution' (simple string or null,
     NEVER an object). Never invert accuser and accused.
   - Report a detail as unknown/unverified ONLY when source explicitly states it.
   - If reporting is insufficient to establish what happened, output:
     'what_happened': 'Insufficient evidence provided in source reporting.'
6. SEPARATION OF FACTS, ANALYSIS & STAGES:
   - If marked Strategic Analysis, summarize the analytical thesis, NOT breaking news.
   - Distinguish approvals and proposals from signed contracts or completed fielding.
   - Frame manufacturer claims as vendor assertions (e.g., 'company claims').
   - Set 'significance': null unless reporting directly explains a concrete strategic
     cause-and-effect mechanism. Avoid generic boilerplate slogans.
7. CLAIM REQUIREMENTS:
   - Output EXACTLY 2 claims:
     * 'c1' (central=true): Primary breaking event development and operational mechanism.
     * 'c2' (central=false): Distinct supporting operational context, figures, or scope.
   - Claims must be complete, dense, informative factual sentences grounded in text.
"""

CORRECTION_SYSTEM_PROMPT = (
    "You are a JSON correction assistant for Geopulse intelligence summaries.\n"
    "Fix the identified schema, claim, or grounding issue and return valid JSON only.\n"
    "Rely solely on the provided source articles. Pair claims with exact verbatim\n"
    "evidence spans under 5 words, cite sources by integer index, designate exactly one\n"
    "central claim, and output no markdown."
)


def build_event_prompt(event: EventCluster) -> str:
    """Build the user prompt for an EventCluster using all attached source articles."""
    articles_text_blocks = []
    for idx, art in enumerate(event.articles, start=1):
        pub_str = (
            art.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if art.published_at
            else "Date not specified in source (do NOT assume current date)"
        )
        title_str = clean_article_text(art.title) if art.title else "No title available."
        is_full = bool(getattr(art, "is_full_text", False) and getattr(art, "body_text", None))
        if is_full:
            full_text = clean_article_text(art.body_text or "")
            bounded_text, _passages = select_bounded_article_passages(
                full_text,
                max_chars=1800,
            )
            content_label = "Article Passages (Extracted Full Text Coverage)"
            content_str = bounded_text
        else:
            content_label = "Snippet (Limited RSS text)"
            content_str = (
                clean_article_text(art.snippet) if art.snippet else "No snippet available."
            )

        block = (
            f"[Source {idx}] (Outlet: {art.source_name} | Published: {pub_str})\n"
            f"Title: {title_str}\n"
            f"{content_label}:\n{content_str}\n"
        )
        articles_text_blocks.append(block)

    articles_section = "\n".join(articles_text_blocks)

    event_nature = (
        "Strategic Analysis / Commentary (NOT a breaking news event)"
        if getattr(event, "is_analysis", False)
        else "Breaking News Development"
    )

    prompt = f"""Identify the material claims for this geopolitical event based SOLELY on the
provided source articles. Return claim-level evidence; the application will render the final
narrative from accepted claims and will not use independently generated narrative prose.

EVENT METADATA:
Category: {event.category.value}
Event Nature: {event_nature}
Reported Region: {event.region or "Unspecified"}
Clustering Confidence Label (event-level hint, not claim evidence): {event.confidence_label}
Distinct Reporting Sources: {event.source_count} ({event.article_count} total reports)

GROUNDING INSTRUCTIONS:
- EXACTLY 2 SUBSTANTIVE CLAIMS REQUIRED (TARGET 2 INFORMATIVE, HIGH-VALUE SENTENCES):
  * Output EXACTLY 2 claims (c1 and c2). Do NOT output a third claim to avoid token cutoff.
  * Central factual claim ("central": true, "claim_id": "c1"): State the primary breaking
    development along with its core operational mechanism, key actors, locations, dates,
    and official actions:
    - For ministerial communiques & summits: state the meeting date, location, primary consensus,
      and any formal joint commitments (such as intelligence sharing, maritime domain awareness,
      or early warning mechanisms).
    - For defense industrial production & assembly: state the concrete milestone (units begun,
      facility location, contract volume) and facility production capacity or delivery timeline.
    - For industrial offset & regulatory policies: state the announcing authority, effective date,
      governing body, and contract-value threshold triggering mandatory domestic economic benefits.
    - For military mission networks, space data relays & operations centers: state the concrete
      award or fielding and explain its functional role (such as integrating satellite
      constellations into the data backbone, dynamic data routing, or missile defense tracking).
      Never quote empty contract buzzwords like 'design, field, and scale'.
    - For arms sales & procurement approvals: state the specific regulatory approval milestone
      (distinguishing regulatory authorization from signed contracts or physical deliveries),
      recipient, system quantities, and stated dollar ceiling.
  * Distinct supporting claim ("central": false, "claim_id": "c2"): Capture the wider operational
    context, technical integration, multi-domain framework scope, or third-party data from a
    different source passage:
    - For communiques with third-party casualty or damage assessments: state the casualty, damage,
      or incident figures, strictly preserving third-party reporting attribution (e.g. reporting
      based on specialized tracking agencies or maritime monitors).
    - For bilateral frameworks & industrial pacts: detail the wider cooperation scope under the
      partnership, enumerating other established joint programs and distinguishing active
      cooperation from prospective or proposed future initiatives.
    - For industrial offset policies: detail the nature of required domestic economic benefits
      (such as local supply chains, facility investment, and technology transfer) and the maximum
      allowable percentage of contract value that commitments may reach.
    - For military mission networks: detail partner force system integration during exercises
      or maneuvers, including specific software platforms or targeting systems utilized.
    - For procurement approvals: detail partner or multilateral funding mechanisms, financing
      instruments, or secondary delivery options.
  * Each claim must be a complete, dense, informative factual sentence grounded in the source text.
    Never output generic filler, repeated headlines, or ungrounded assertions.
- STRUCTURED KEY DETAILS & SIGNIFICANCE:
  * Extract 'key_details' with specific actors (countries, ministries, forces), location,
    timing, and action. For 'actors', provide ONLY discrete proper-noun entity names (e.g.
    ['Bulgaria', 'Romania', 'Turkey']). NEVER list weather phenomena (rain, floods),
    munitions/explosives ('Explosive', 'bomb'), attack types, inanimate structures
    (buildings, homes), or headline fragments with trailing articles ('Panama An'). If
    perpetrators are unknown, set 'actors': [].
  * For 'location', extract ONLY the actual geographic setting where the event, strike,
    or investigation occurred (or null if ungrounded). NEVER extract locations mentioned
    only as background context, historical comparisons, past programs (e.g. do not extract
    France when discussing Italy's GCAP budget because France is mentioned regarding FCAS),
    or illustrative threat examples (e.g. do not extract Germany when investigating an
    espionage case in the UK because Germany is mentioned in a background report).
  * For 'action', state a concise 3-8 word summary of the concrete action or agreement.
    Never quote empty buzzwords like 'design, field, and scale'.
  * Always set 'significance': null unless reporting explains a concrete strategic mechanism.
- STAGES & MILESTONES: Distinguish regulatory/diplomatic approval (such as Foreign Military Sale
  approval) and proposals from completed deliveries, signed contracts, or physical deployments.
- THRESHOLDS vs. OBLIGATIONS: Distinguish contract-value thresholds that trigger a policy from
  the required domestic investment or offset benefit amounts.
- OPERATIONAL SCOPE & CONTEXT: Explain the concrete mechanisms of agreements and networks (tools,
  data flows, multi-domain scope), distinguishing established programs from proposed initiatives.
- SOURCE INDEX & EXACT EVIDENCE: Cite the article by its integer index in 'src' (e.g. 1 for
  [Source 1], 2 for [Source 2]). Copy a short, contiguous verbatim phrase under 5
  words directly from that source into 'evidence'. Do not paraphrase or reorder words.
  Do NOT output full URLs. Ensure claims are paired with the exact article containing the span.
- ATTRIBUTION & PROVENANCE: Explicitly attribute claims, allegations, or assessments to the
  speaker or institution making them in 'attribution' as a simple string or null (NEVER an object
  or dictionary). Never invert accuser and accused. If a statistic or finding is attributed to a
  third-party publication or data source (e.g. Financial Times citing Ambrey/IMO, Oryx, SIPRI),
  retain that attribution.
- CONTRACTING ROLES: When an intermediary prime contractor selects a defense supplier for a
  military program, do not describe the military branch as directly awarding the supplier the
  contract; preserve the distinct contracting tiers.
- BUDGET COMPONENT PRESERVATION: When an intelligence or defence budget request reports distinct
  programs and totals (e.g. Military Intelligence Program / MIP at $30.9B and National
  Intelligence Program / NIP at $78.1B), preserve both programs in the claims and narrative.
- PRESERVE CONCRETE DETAILS: Preserve specific source-supported quantities, counts, and named
  entities (e.g. '12 Ukrainian manufacturers'). Do not replace specifics with generic prose.
- TECHNICAL ACRONYMS & EXPANSIONS: Technical acronyms and their expansions MUST be taken strictly
  from source-supported terminology. Never invent an expansion based on familiar military
  acronyms or historical vessel classifications (e.g. do NOT expand LSM as LSTH or Landing
  Ship, Tank, Helicopter). When the source does not provide an explicit expansion, preserve
  the acronym verbatim without expanding it.
- COMMODITY & TRANSACTION INTEGRITY: Rely strictly on the specific commodity reported (e.g. diesel).
  Never substitute LNG, natural gas, or other ungrounded commodities. Never invert buyer and seller.
- PRESERVE CAVEATS: Retain essential operational caveats, negative conditions, or stated limits.
- CLASSIFICATION: Classify each claim as reported_fact, attributed_statement, source_analysis,
  inference, or disputed_claim.
- TIMELINES & PRECISION: State a timeline or quantity only when explicitly stated in source text.
- EXPLICIT UNCERTAINTY ONLY: Report details as unknown, unclear, or unconfirmed ONLY when the
  source text explicitly states that they are unknown or unverified (e.g. 'cause remains unknown').
  Do not convert omitted details or absence of evidence into an assertion that something is unknown.
- INCIDENT & AGREEMENT BOUNDARIES: Maintain distinct circumstances for multiple strikes or casualty
  incidents; never combine separate events into a single attack. Keep separate diplomatic treaties
  and defense-industrial arrangements distinct without conflating their provisions.

SOURCE ARTICLES:
{articles_section}

REQUIRED JSON STRUCTURE (valid JSON only, no markdown):
{{
  "claims": [
    {{
      "claim_id": "c1",
      "text": "<concise central factual claim stating primary development>",
      "src": 1,
      "evidence": "<exact verbatim span under 5 words from source 1>",
      "classification": "reported_fact",
      "attribution": null,
      "central": true
    }},
    {{
      "claim_id": "c2",
      "text": "<distinct supporting factual context, figures, scope, or mechanisms>",
      "src": 1,
      "evidence": "<exact verbatim span under 5 words from source 1>",
      "classification": "reported_fact",
      "attribution": null,
      "central": false
    }}
  ],
  "key_details": {{
    "actors": ["State or Organization Name"],
    "location": null,
    "timing": null,
    "action": "Short action summary"
  }},
  "significance": null
}}
"""
    return prompt


def build_grounding_correction_prompt(
    event: EventCluster,
    unsupported_entities: list[str] | tuple[str, ...] = (),
    malformed_issues: list[str] | tuple[str, ...] = (),
) -> str:
    """Build a corrective prompt instructing the model to fix entities or malformed grammar."""
    base_prompt = build_event_prompt(event)
    corrections: list[str] = []
    if unsupported_entities:
        entities_str = ", ".join(f"'{e}'" for e in unsupported_entities)
        corrections.append(
            f"Your previous summary introduced entities or actors absent from the source: "
            f"[{entities_str}]. Use ONLY actors explicitly mentioned in the source articles. "
            f"Never list news outlets or publishers as actors."
        )
    if malformed_issues:
        issues_str = "; ".join(malformed_issues)
        corrections.append(
            f"Your previous summary contained grammatically broken sentences ({issues_str}). "
            f"Ensure EVERY sentence is grammatically complete with a clear subject and predicate. "
            f"Do NOT write incomplete clauses like ', as residents who survived on weeds...'."
        )

    correction_items = "\n".join(f"- {c}" for c in corrections)
    correction_header = f"""CORRECTION REQUIRED:
Please correct the following defect(s) in your previous output:
{correction_items}

CRITICAL INSTRUCTIONS FOR THIS ATTEMPT:
1. Ensure full grammatical completeness with no dangling or broken clauses.
2. Rely strictly on the facts in the SOURCE ARTICLES below.
3. If the source text is too sparse, output:
   'what_happened': 'Insufficient evidence provided in source reporting.'

"""
    return correction_header + base_prompt
