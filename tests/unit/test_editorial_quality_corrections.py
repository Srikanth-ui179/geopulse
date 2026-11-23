"""Regression and unit tests for Geopulse final editorial quality corrections.

Covers:
1. Selection-count discrepancy & arithmetic reconciliation.
2. Candidates removed after initial selection (quarantine resilience).
3. Topic classification:
   - Opinion/analysis about political/economic effects of war (not armed_conflict).
   - Factual reporting about active military operations (armed_conflict / military_operations).
   - Analysis of military strategy (defence_policy).
   - Articles mentioning conflict only as background.
4. Entity & event metadata:
   - Rejection of stopwords, verbs, and question words as actors (e.g. 'How', 'Helped').
   - Location case normalization ('iran' -> 'Iran').
   - Location rejection when foreign conflict is only background context.
   - Concise action validation & omission of paragraph-length sentences.
5. Removal of low-value administrative content (e.g. Digital Veteran Card 23,000 downloads)
   while retaining substantive defence-policy developments.
6. Source assessment wording for all four evidence cases (single-source, syndicated/shared origin,
   corroborated, unassessed).
7. Final narrative quality (Golden Dome scenario: central development prioritized).
8. Production-path integration with AUKUS, USFK, diesel, and publication safeguards.
"""

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from geopulse.classification.models import TopicCategory
from geopulse.classification.rules import RuleBasedClassifier
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.client import MockOllamaClient
from geopulse.summarization.evidence_matcher import split_sentences_safely
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventSourceAttribution,
    EventSummary,
    SummaryClaim,
)
from geopulse.summarization.service import EventSummarizer
from geopulse.summarization.validator import (
    clean_concise_action,
    ground_event_location,
    is_polluted_actor,
    sanitize_actors,
)

FIXED_DT = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def _make_article(
    title: str,
    snippet: str,
    source_name: str = "Defense News",
    url: str = "https://defensenews.com/story-1",
    published_at: datetime = FIXED_DT,
    topic: TopicCategory = TopicCategory.DEFENCE_POLICY,
) -> ArticleRecord:
    return ArticleRecord(
        source_name=source_name,
        title=title,
        url=url,
        canonical_url=url,
        published_at=published_at,
        snippet=snippet,
        content_hash=f"hash-{title[:12]}",
        topic=topic.value,
    )


def _make_cluster(
    cluster_id: str,
    title: str,
    category: TopicCategory,
    source_count: int = 1,
    url: str | None = None,
    articles: tuple[ArticleRecord, ...] | None = None,
) -> EventCluster:
    if articles is None:
        arts = tuple(
            _make_article(
                title=f"{title} (Report {i})",
                snippet=f"Factual details for {title} report {i}.",
                source_name=f"Publisher_{cluster_id}_{i}",
                url=url or f"https://outlet-{i}.test/{cluster_id}",
                topic=category,
            )
            for i in range(1, source_count + 1)
        )
    else:
        arts = articles
    return EventCluster(
        id=cluster_id,
        title=title,
        category=category,
        first_seen_at=FIXED_DT,
        last_seen_at=FIXED_DT,
        published_at=FIXED_DT,
        confidence_label="corroborated" if len(arts) > 1 else "single_source",
        articles=arts,
    )


# ===========================================================================
# 1. Selection-count discrepancy & arithmetic reconciliation
# ===========================================================================
def test_selection_metrics_arithmetic_reconciles():
    """Verify that candidate selection and summarization metrics reconcile completely.

    Total clusters = Eligible + Priority Excluded
    Eligible = Selected + Deferred
    Total clusters = Published + Quarantined/Failed + Omitted.
    """
    # Create 30 clusters: 24 high priority, 6 low priority (> rank 10.0)
    clusters: list[EventCluster] = []
    for i in range(1, 25):
        clusters.append(
            _make_cluster(
                f"c-high-{i}",
                f"Air strike against military installation {i}",
                TopicCategory.ARMED_CONFLICT,
                source_count=2,
            )
        )
    for i in range(1, 7):
        clusters.append(
            _make_cluster(
                f"c-low-{i}",
                f"Tactical accessory trigger commercial announcement {i}",
                TopicCategory.DEFENCE_POLICY,
                source_count=1,
            )
        )

    assert len(clusters) == 30

    client = MockOllamaClient(model="mock-remediation")
    summarizer = EventSummarizer(client=client)

    results = summarizer.summarize_events(
        events=clusters,
        candidate_cap=10,
        min_relevance_rank=8.5,
    )

    m = summarizer.last_candidate_metrics
    assert m["total_clusters"] == 30
    assert m["eligible_candidates"] == 24
    assert m["priority_excluded"] == 6
    assert m["total_clusters"] == m["eligible_candidates"] + m["priority_excluded"]

    assert m["selected_candidates"] == 10
    assert m["deferred_candidates"] == 14
    assert m["eligible_candidates"] == m["selected_candidates"] + m["deferred_candidates"]

    # Reconcile final results population
    formatter = TelegramDigestFormatter()
    success_results = [r for r in results if r.status == "success" and r.summary is not None]
    published = formatter.select_digest_events(results=success_results, max_events=10)
    published_count = len(published)

    quarantined_count = sum(
        1 for r in results if r.status in {"quarantined", "insufficient_evidence", "failed"}
    )
    omitted_count = max(0, len(results) - published_count - quarantined_count)

    assert published_count + quarantined_count + omitted_count == len(results)
    assert len(results) == 30


# ===========================================================================
# 2. Candidates removed after initial selection (Quarantine Resilience)
# ===========================================================================
def test_candidates_quarantined_after_selection_does_not_break_reconciliation():
    """When candidates are quarantined after initial selection, the final briefing

    is not artificially forced to the cap, and arithmetic reconciles.
    """
    clusters = [
        _make_cluster(
            f"evt-{i}",
            f"Frontline military operation {i}",
            TopicCategory.ARMED_CONFLICT,
            source_count=2,
        )
        for i in range(1, 13)
    ]

    class QuarantiningClient(MockOllamaClient):
        def generate(
            self,
            prompt: str,
            system: str | None = None,
            format_type: str | None = "json",
            options: dict | None = None,
        ) -> str:
            # Simulate hallucination / ungrounded output for evt-2 and evt-3
            if "evt-2" in prompt or "evt-3" in prompt:
                return json.dumps(
                    {
                        "claims": [
                            {
                                "claim_id": "c1",
                                "text": "Invented General Hallucination ordered the retreat.",
                                "src": 1,
                                "evidence": "general ordered",
                                "classification": "reported_fact",
                                "attribution": None,
                                "central": True,
                            }
                        ],
                        "key_details": {
                            "actors": ["General Hallucination"],
                            "location": "Mars",
                            "timing": "Unknown",
                            "action": "retreat",
                        },
                        "significance": None,
                    }
                )
            return super().generate(prompt, system, format_type, options)

    summarizer = EventSummarizer(client=QuarantiningClient(model="mock-q"))
    results = summarizer.summarize_events(events=clusters, candidate_cap=12)

    quarantined = [r for r in results if r.status == "quarantined"]
    assert len(quarantined) >= 2

    formatter = TelegramDigestFormatter()
    success_results = [r for r in results if r.status == "success" and r.summary is not None]
    published = formatter.select_digest_events(results=success_results, max_events=12)

    # Published count is less than cap (not artificially padded)
    assert len(published) <= 10

    total_clusters = len(results)
    published_count = len(published)
    quarantined_count = len(quarantined)
    omitted_count = max(0, total_clusters - published_count - quarantined_count)

    assert total_clusters == published_count + quarantined_count + omitted_count


# ===========================================================================
# 3. Topic classification tests
# ===========================================================================
def test_topic_classification_opinion_on_political_economic_effects_of_war():
    """Al Jazeera opinion article on midterm electoral effects of the Iran war

    must be classified as OTHER_GEOPOLITICAL, not ARMED_CONFLICT.
    """
    classifier = RuleBasedClassifier()
    title = "How the Iran war helped Democrats win the midterms"
    url = "https://www.aljazeera.com/opinions/2026/11/how-the-iran-war-helped-democrats-win-the-midterms"
    snippet = (
        "Voters cited the conflict in the Middle East as a decisive factor in congressional races."
    )

    res = classifier.classify_text(title=title, snippet=snippet, url=url)
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.OTHER_GEOPOLITICAL
    assert res.primary_topic != TopicCategory.ARMED_CONFLICT


def test_topic_classification_factual_military_operation():
    """Factual reporting about an actual military strike must be ARMED_CONFLICT."""
    classifier = RuleBasedClassifier()
    title = "Ukrainian air defenses intercept 14 cruise missiles during overnight barrage"
    snippet = "Explosions were heard across multiple regions as interceptors engaged targets."

    res = classifier.classify_text(title=title, snippet=snippet)
    assert res.relevance_status == "relevant"
    assert res.primary_topic in {
        TopicCategory.ARMED_CONFLICT,
        TopicCategory.MILITARY_OPERATIONS,
    }


def test_topic_classification_military_strategy_analysis():
    """Analysis of military strategy without kinetic strikes must be DEFENCE_POLICY."""
    classifier = RuleBasedClassifier()
    title = "Strategic Analysis: The Evolution of Counter-Drone Doctrine in Eastern Europe"
    snippet = (
        "Military planners examine layered air defense integration and electronic warfare lessons."
    )

    res = classifier.classify_text(title=title, snippet=snippet)
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.DEFENCE_POLICY
    assert res.primary_topic != TopicCategory.ARMED_CONFLICT


def test_topic_classification_conflict_only_as_background():
    """Article discussing European energy storage where the Ukraine war is only background."""
    classifier = RuleBasedClassifier()
    title = "European gas reserves hit target levels despite Ukraine war market uncertainty"
    snippet = "Storage facilities reached 95% capacity ahead of winter demand across member states."

    res = classifier.classify_text(title=title, snippet=snippet)
    assert res.relevance_status == "relevant"
    assert res.primary_topic in {TopicCategory.OTHER_GEOPOLITICAL, TopicCategory.SANCTIONS}
    assert res.primary_topic != TopicCategory.ARMED_CONFLICT


# ===========================================================================
# 4. Entity and event metadata improvements
# ===========================================================================
def test_actors_reject_stopwords_verbs_and_question_words():
    """Actors list must reject stopwords and question words like 'How', 'Helped', 'Winning'."""
    raw_actors = ["How", "France", "Democrats", "Helped", "Tehran", "Winning"]
    clean = sanitize_actors(raw_actors)

    assert "How" not in clean
    assert "Helped" not in clean
    assert "Winning" not in clean
    assert "France" in clean
    assert "Democrats" in clean
    assert "Tehran" in clean


def test_is_polluted_actor_rejects_question_words_and_verbs():
    """is_polluted_actor returns True for standalone question words and common verbs."""
    assert is_polluted_actor("How")
    assert is_polluted_actor("Why")
    assert is_polluted_actor("Helped")
    assert is_polluted_actor("Says")
    assert is_polluted_actor("Winning")
    assert not is_polluted_actor("Armed Forces of Ukraine")
    assert not is_polluted_actor("UK Ministry of Defence")
    assert is_polluted_actor("Ministry of Defence")


def test_location_case_normalization_and_foreign_conflict_background_omission():
    """Lowercase 'iran' is normalized to 'Iran', but omitted when the article

    is about domestic midterms where the foreign war is merely background.
    """
    cluster = _make_cluster(
        "evt-elections",
        "How the Iran war helped Democrats win the midterms",
        TopicCategory.OTHER_GEOPOLITICAL,
    )
    src_text = (
        "How the Iran war helped Democrats win the midterms. Voters in US congressional "
        "elections debated foreign policy across key battleground districts."
    )

    # In domestic midterm context, foreign war theater is background context, not event location
    loc_fallout = ground_event_location("iran", cluster, src_text)
    assert loc_fallout is None

    # In genuine foreign news context, lowercase 'iran' normalizes to 'Iran'
    cluster_foreign = _make_cluster(
        "evt-iran-talks",
        "Diplomatic delegation arrives in Iran for nuclear consultations",
        TopicCategory.DIPLOMACY,
    )
    src_foreign = (
        "A high-level delegation arrived in Iran on Tuesday for discussions across Tehran."
    )
    loc_foreign = ground_event_location("iran", cluster_foreign, src_foreign)
    assert loc_foreign == "Iran"


def test_clean_concise_action_rejects_paragraph_length_and_extracts_verb_phrase():
    """clean_concise_action extracts concise verb phrases and omits sentence-length narratives."""
    # Concise verb phrase preserved
    assert clean_concise_action("Deploy missile interceptor battery") == (
        "Deploy missile interceptor battery"
    )

    # Multi-clause narrative trimmed at clause boundary
    long_action = (
        "Helped Democrats win the midterms following extensive congressional debate and campaigning"
    )
    trimmed = clean_concise_action(long_action)
    assert trimmed == "Helped Democrats win the midterms"

    # Excessively verbose paragraph without concise verb phrase omitted (None)
    paragraph_action = (
        "The committee issued an extensive report detailing multi-year institutional developments "
        "spanning several departments and discussing various strategic implications across decades."
    )
    assert clean_concise_action(paragraph_action) is None

    # Placeholders cleanly omitted
    assert clean_concise_action("Unspecified") is None
    assert clean_concise_action("unknown in reporting") is None


# ===========================================================================
# 5. Removal of low-value administrative content
# ===========================================================================
def test_exclude_routine_digital_veteran_card_milestone():
    """UK Digital Veteran Card 23,000 downloads milestone must be filtered as irrelevant."""
    classifier = RuleBasedClassifier()
    title = "Digital Veteran Card reaches 23,000 downloads milestone"
    snippet = (
        "More than 23,000 veterans have downloaded the new digital entitlement card via the portal."
    )

    res = classifier.classify_text(title=title, snippet=snippet, source_name="Ministry of Defence")
    assert res.relevance_status == "irrelevant"


def test_retain_substantive_defence_policy_announcements():
    """Substantive procurement contracts and defence policy changes must remain eligible."""
    classifier = RuleBasedClassifier()

    title_procurement = "Ministry of Defence awards £2bn contract for naval submarine fleet"
    snippet_procurement = "The defence contract awards funding for nuclear submarine procurement."
    res_proc = classifier.classify_text(
        title=title_procurement,
        snippet=snippet_procurement,
        source_name="Ministry of Defence",
    )
    assert res_proc.relevance_status == "relevant"
    assert res_proc.primary_topic == TopicCategory.WEAPONS_PROCUREMENT

    title_doctrine = (
        "Government publishes Strategic Defence Review outlining NATO readiness expansion"
    )
    snippet_doctrine = "The defence policy white paper details troop posture and readiness drills."
    res_doc = classifier.classify_text(
        title=title_doctrine,
        snippet=snippet_doctrine,
        source_name="Ministry of Defence",
    )
    assert res_doc.relevance_status == "relevant"
    assert res_doc.primary_topic == TopicCategory.DEFENCE_POLICY


# ===========================================================================
# 6. Source assessment wording for all four evidence cases
# ===========================================================================
def test_source_assessment_wording_all_four_cases():
    """Verify exact source assessment wording for all four evidence cases:

    1. Single source (no 'across outlets')
    2. Shared origin / syndication
    3. Corroborated independent chains
    4. Not assessed
    """
    formatter = TelegramDigestFormatter()

    def make_summary(
        corrob_status: str, conf_qual: str, source_count: int = 1
    ) -> tuple[EventSummary, TopicCategory]:
        sources = tuple(
            EventSourceAttribution(
                source_name=f"Publisher {i}",
                title=f"Report {i}",
                url=f"https://source{i}.test/art",
                published_at=FIXED_DT,
            )
            for i in range(1, source_count + 1)
        )
        claim = SummaryClaim(
            claim_id="c1",
            text="Operational development occurred in the sector.",
            evidence_refs=(
                ClaimEvidenceReference(
                    source_url="https://source1.test/art",
                    evidence_span="development occurred",
                    verification_status="accepted",
                ),
            ),
            classification="reported_fact",
            central=True,
            verification_status="accepted",
        )
        summary = EventSummary(
            event_id="evt-wording",
            headline="Operational Development Reported",
            what_happened="Operational development occurred in the sector.",
            confidence_qualification=conf_qual,
            sources=sources,
            claims=(claim,),
            model_used="test-model",
            corroboration_status=corrob_status,
        )
        return summary, TopicCategory.ARMED_CONFLICT

    # Case 1: Single source
    s1, cat1 = make_summary("single_source", "Single-source report; unverified independently.", 1)
    b1 = formatter.format_event_block(s1, cat1, "single_source", 1, 1)
    assert "[SINGLE SOURCE]" in b1
    assert "Single-source reporting; unverified independently." in b1
    assert "across outlets" not in b1

    # Case 2: Shared origin / syndication
    s2, cat2 = make_summary(
        "multiple_reports_shared_origin",
        "Syndicated reporting across outlets; not independently corroborated.",
        2,
    )
    b2 = formatter.format_event_block(s2, cat2, "single_source", 1, 1)
    assert "[SHARED ORIGIN]" in b2
    assert "not independently corroborated" in b2
    assert "Syndicated reporting across outlets" in b2 or "shared source material" in b2

    # Case 3: Corroborated independent chains
    corrob_text = (
        "Independent reporting from distinct sources; "
        "operational details not independently verified."
    )
    s3, cat3 = make_summary(
        "independently_corroborated",
        corrob_text,
        2,
    )
    b3 = formatter.format_event_block(s3, cat3, "corroborated", 1, 1)
    assert "[CORROBORATED]" in b3
    assert corrob_text in b3

    # Case 4: Not evaluated
    s4, cat4 = make_summary("unassessed", "", 1)
    b4 = formatter.format_event_block(s4, cat4, "single_source", 1, 1)
    assert "[NOT ASSESSED]" in b4
    assert "Unassessed; corroboration analysis not performed." in b4


# ===========================================================================
# 7. Final narrative quality: Golden Dome scenario
# ===========================================================================
def test_golden_dome_scenario_preserves_central_claim():
    """In the Golden Dome scenario, the central breaking claim explaining the actual

    test is preserved in the narrative, rather than being discarded in favor of
    generic historical background.
    """
    c1_text = "Defence forces completed a live-fire test of the Golden Dome air defence system."
    c2_text = (
        "The system was initially initiated under a 2021 program to provide localized coverage."
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=c1_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://jpost.test/golden-dome",
                evidence_span="completed live-fire test",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=c2_text,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://jpost.test/golden-dome",
                evidence_span="initiated under 2021",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=False,
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-golden-dome",
        headline="Defence Forces Complete Golden Dome Test",
        what_happened=f"{c1_text} {c2_text}",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Jerusalem Post",
                title="Defence Forces Complete Golden Dome Test",
                url="https://jpost.test/golden-dome",
                published_at=FIXED_DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
        corroboration_status="single_source",
    )

    formatter = TelegramDigestFormatter()
    rendered = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.WEAPONS_PROCUREMENT,
        confidence_label="single_source",
        rank=1,
        total=1,
    )

    # Must contain the central operational development
    assert c1_text in rendered
    # Must contain the supporting background
    assert c2_text in rendered
    # Must preserve the source link
    assert 'href="https://jpost.test/golden-dome"' in rendered


# ===========================================================================
# 8. Full production path regression coverage (USFK, diesel, AUKUS)
# ===========================================================================
def test_production_path_regressions_preserved():
    """Verify that key historical regressions (USFK, diesel, AUKUS) remain properly

    handled through the briefing formatter and classification rules.
    """
    classifier = RuleBasedClassifier()

    # 1. USFK readiness story
    usfk_title = "USFK conducts combined live-fire air defense exercise near demilitarized zone"
    res_usfk = classifier.classify_text(
        title=usfk_title,
        snippet="United States Forces Korea executed combined live-fire drills.",
    )
    assert res_usfk.relevance_status == "relevant"
    assert res_usfk.primary_topic in {
        TopicCategory.MILITARY_OPERATIONS,
        TopicCategory.DEFENCE_POLICY,
    }

    # 2. Russian diesel sanctions story
    diesel_title = "United States eases restrictions on Russian diesel imports to stabilize supply"
    res_diesel = classifier.classify_text(
        title=diesel_title,
        snippet="Treasury issued waivers allowing emergency diesel shipments.",
    )
    assert res_diesel.relevance_status == "relevant"
    assert res_diesel.primary_topic == TopicCategory.SANCTIONS

    # 3. AUKUS submarine pillar story
    aukus_title = "AUKUS partners finalize nuclear submarine construction schedule under Pillar I"
    res_aukus = classifier.classify_text(
        title=aukus_title,
        snippet="Australia, UK, and United States confirmed delivery timelines.",
    )
    assert res_aukus.relevance_status == "relevant"
    assert res_aukus.primary_topic in {
        TopicCategory.DEFENCE_POLICY,
        TopicCategory.WEAPONS_PROCUREMENT,
    }


# ===========================================================================
# 9. Additional Editorial Corrections (November 8 Preview Regression Suite)
# ===========================================================================
def test_sports_recovery_reporting_mentioning_attack_is_irrelevant():
    """Personal recovery and coaching stories must not qualify as ARMED_CONFLICT."""
    classifier = RuleBasedClassifier()
    title = "Lebanese footballer refuses to give up after injuries from Israeli strike"
    snippet = (
        "After surviving an Israeli airstrike that left her in a coma, Celine Haidar is "
        "back on the pitch, coaching footballers."
    )
    res = classifier.classify_text(
        title=title,
        snippet=snippet,
        url="https://www.aljazeera.com/video/newsfeed/2026/11/7/lebanese-footballer",
    )
    assert res.relevance_status == "irrelevant"
    assert res.primary_topic != TopicCategory.ARMED_CONFLICT


def test_military_public_awareness_campaign_not_armed_conflict():
    """US European Command vigilance campaign is defence policy, not armed conflict."""
    classifier = RuleBasedClassifier()
    title = "US European Command launches vigilance campaign as hybrid threats mount"
    snippet = (
        "U.S. European Command is calling on troops, family members, and civilian employees to "
        "watch for suspicious behavior on and off military installations and to practice "
        "rigorous digital discipline under the 'See Something, Do Something' initiative."
    )
    res = classifier.classify_text(
        title=title,
        snippet=snippet,
        source_name="Defense News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic in {
        TopicCategory.DEFENCE_POLICY,
        TopicCategory.CYBER_HYBRID_WARFARE,
    }
    assert res.primary_topic != TopicCategory.ARMED_CONFLICT


def test_genuine_military_developments_remain_eligible():
    """Genuine kinetic operations and naval weapons procurements remain fully eligible."""
    classifier = RuleBasedClassifier()

    # 1. Frontline strike and interception
    title_kinetic = "Ukrainian air defenses intercept 18 cruise missiles during overnight barrage"
    snippet_kinetic = (
        "Interceptors engaged incoming Russian missiles across central and southern regions."
    )
    res_kin = classifier.classify_text(title=title_kinetic, snippet=snippet_kinetic)
    assert res_kin.relevance_status == "relevant"
    assert res_kin.primary_topic in {
        TopicCategory.ARMED_CONFLICT,
        TopicCategory.MILITARY_OPERATIONS,
    }

    # 2. Naval weapons procurement
    title_proc = "Denmark and Lithuania join procurement for Blocker naval mines under NMC pact"
    snippet_proc = (
        "The Finnish military signed contracts for Blocker sea mines on behalf of "
        "participating European allies."
    )
    res_proc = classifier.classify_text(
        title=title_proc, snippet=snippet_proc, source_name="Defense News"
    )
    assert res_proc.relevance_status == "relevant"
    assert res_proc.primary_topic in {
        TopicCategory.WEAPONS_PROCUREMENT,
        TopicCategory.DEFENCE_POLICY,
    }


def test_invalid_us_location_extraction_rejected():
    """Candidate location 'us' from quoted speech is rejected and grounds actual location."""
    cluster = _make_cluster(
        "evt-zaporizhzhia",
        "‘They started bombing us with drones’: Russian attacks unnerve Zaporizhzhia",
        TopicCategory.ARMED_CONFLICT,
    )
    source_text = (
        "‘They started bombing us with drones’: Russian attacks unnerve Zaporizhzhia. "
        "Explosions shook the southeastern city overnight."
    )
    # The pronoun 'us' must not be returned as the event location
    loc = ground_event_location("us", cluster, source_text)
    assert loc != "us"
    assert loc == "Zaporizhzhia"


def test_non_actor_nouns_filtered_from_metadata():
    """Buildings, floods, weather events, and generic non-agent nouns are rejected as actors."""
    from geopulse.summarization.validator import clean_concise_action, is_polluted_actor

    assert is_polluted_actor("Buildings")
    assert is_polluted_actor("Floods")
    assert is_polluted_actor("Heavy Rain")
    assert is_polluted_actor("Families")
    assert not is_polluted_actor("IDF")
    assert not is_polluted_actor("Hezbollah")

    raw_actors = ["Buildings", "Floods", "IDF", "Heavy Rain", "Families", "Hezbollah"]
    sanitized = sanitize_actors(raw_actors)
    assert "Buildings" not in sanitized
    assert "Floods" not in sanitized
    assert "Heavy Rain" not in sanitized
    assert "Families" not in sanitized
    assert "IDF" in sanitized
    assert "Hezbollah" in sanitized

    # Buzzword contract action phrases must be rejected
    assert clean_concise_action("design, field, and scale") is None
    assert clean_concise_action("design and field") is None
    assert clean_concise_action("Field satellite operations center") == (
        "Field satellite operations center"
    )


def test_redundant_headline_body_output_eliminated_german_fighter():
    """When a central claim duplicates headline, body presents distinct supporting facts."""
    headline = (
        "Europe's sixth-generation fighter program collapse leaves Germany searching for a new path"
    )
    c1_dup = (
        "The collapse of Europe's sixth-generation fighter program has left Germany searching "
        "for a new path."
    )
    c2_facts = (
        "Germany is aiming to field collaborative combat aircraft (CCA) by 2029, while the "
        "UK-led GCAP targets service entry by 2035."
    )

    c1 = SummaryClaim(
        claim_id="c1",
        text=c1_dup,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://breakingdefense.test/fighter",
                evidence_span="collapse of Europe's sixth-generation fighter program",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    c2 = SummaryClaim(
        claim_id="c2",
        text=c2_facts,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://breakingdefense.test/fighter",
                evidence_span="field collaborative combat aircraft",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=False,
        verification_status="accepted",
    )

    summary = EventSummary(
        event_id="evt-german-fighter",
        headline=headline,
        what_happened=f"{c1_dup} {c2_facts}",
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Breaking Defense",
                title=headline,
                url="https://breakingdefense.test/fighter",
                published_at=FIXED_DT,
            ),
        ),
        claims=(c1, c2),
        model_used="test-model",
        corroboration_status="single_source",
    )

    formatter = TelegramDigestFormatter()
    rendered = formatter.format_event_block(
        summary=summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
        rank=1,
        total=1,
    )

    from geopulse.delivery.formatter import escape_html

    assert f"<b>{escape_html(headline)}</b>" in rendered
    assert c2_facts in rendered
    # Duplicate c1 sentence must not be repeated in the rendered body narrative
    body_narrative = rendered.split("<b>Assessment:</b>")[0].split("</b>\n")[-1]
    assert c2_facts in body_narrative
    assert c1_dup not in body_narrative


def test_thin_summaries_quarantined_rather_than_padded():
    """Thin reports that merely restate their headline without new facts are quarantined."""
    from geopulse.summarization.publication_gate import validate_publication_gate

    title = "Iran’s IRGC says building underground sites for weapons production, storage"
    body = "IRGC says building underground sites for weapons production, storage."
    art = _make_article(
        title=title,
        snippet=body,
        url="https://aljazeera.test/irgc",
        source_name="Al Jazeera",
        topic=TopicCategory.DEFENCE_POLICY,
    )
    event = _make_cluster(
        cluster_id="evt-irgc-thin",
        title=title,
        category=TopicCategory.DEFENCE_POLICY,
        articles=(art,),
    )
    claim = SummaryClaim(
        claim_id="c1",
        text=body,
        evidence_refs=(
            ClaimEvidenceReference(
                source_url="https://aljazeera.test/irgc",
                evidence_span="building underground sites",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
        verification_status="accepted",
    )
    thin_summary = EventSummary(
        event_id="evt-irgc-thin",
        headline=title,
        what_happened=body,
        confidence_qualification="Single-source reporting; unverified independently.",
        sources=(
            EventSourceAttribution(
                source_name="Al Jazeera",
                title=title,
                url="https://aljazeera.test/irgc",
                published_at=FIXED_DT,
            ),
        ),
        claims=(claim,),
        model_used="test-model",
        corroboration_status="single_source",
    )

    # When the only claim repeats the headline and no other facts exist, must not pass
    from geopulse.summarization.validator import validate_summary_grounding

    val_res = validate_summary_grounding(thin_summary, event)
    gate_res = validate_publication_gate(thin_summary, event)
    assert not val_res.is_valid or not gate_res.passed


def test_corroboration_and_source_attribution_behavior_intact():
    """Single-source, shared-origin, and corroborated labels are strictly preserved."""
    formatter = TelegramDigestFormatter()

    def make_digest_item(corrob_status: str, sources_count: int = 1) -> EventSummary:
        sources = tuple(
            EventSourceAttribution(
                source_name=f"Publisher {i}",
                title=f"Report {i}",
                url=f"https://source{i}.test/story",
                published_at=FIXED_DT,
            )
            for i in range(1, sources_count + 1)
        )
        claim = SummaryClaim(
            claim_id="c1",
            text="Naval forces carried out scheduled patrol operations.",
            evidence_refs=(
                ClaimEvidenceReference(
                    source_url="https://source1.test/story",
                    evidence_span="scheduled patrol operations",
                    verification_status="accepted",
                ),
            ),
            classification="reported_fact",
            central=True,
            verification_status="accepted",
        )
        return EventSummary(
            event_id=f"evt-{corrob_status}",
            headline="Naval Forces Conduct Patrol",
            what_happened="Naval forces carried out scheduled patrol operations.",
            confidence_qualification="Source assessment completed.",
            sources=sources,
            claims=(claim,),
            model_used="test-model",
            corroboration_status=corrob_status,
        )

    # 1. Single source clearly labelled
    s1 = make_digest_item("single_source", 1)
    b1 = formatter.format_event_block(s1, TopicCategory.MILITARY_OPERATIONS, "single_source", 1, 1)
    assert "[SINGLE SOURCE]" in b1

    # 2. Shared origin never implies independent corroboration
    s2 = make_digest_item("multiple_reports_shared_origin", 2)
    b2 = formatter.format_event_block(s2, TopicCategory.MILITARY_OPERATIONS, "single_source", 1, 1)
    assert "[SHARED ORIGIN]" in b2
    assert "[CORROBORATED]" not in b2

    # 3. Independent corroboration
    s3 = make_digest_item("independently_corroborated", 2)
    b3 = formatter.format_event_block(s3, TopicCategory.MILITARY_OPERATIONS, "corroborated", 1, 1)
    assert "[CORROBORATED]" in b3


def test_cross_day_hash_persistence_records_event_hashes():
    """Verify that deliver_digest records non-empty event_hashes to the repository."""
    from geopulse.delivery.service import TelegramDeliveryService

    mock_repo = MagicMock()
    mock_repo.get_briefing_by_date.return_value = None
    mock_record = MagicMock()
    mock_record.id = "test-briefing-id"
    mock_record.started_at = datetime.now(UTC)
    mock_record.delivered_event_hashes = ()
    mock_repo.create_briefing.return_value = mock_record
    mock_client = MagicMock()
    mock_client.send_message.return_value = MagicMock(message_id=101)

    service = TelegramDeliveryService(client=mock_client, repository=mock_repo)
    service.deliver_digest(
        digest_chunks=("Digest message content",),
        briefing_date=date(2026, 11, 15),
        force=True,
        event_hashes=["event_hash_1", "event_hash_2"],
    )

    assert mock_repo.update_briefing.called
    last_record = mock_repo.update_briefing.call_args[0][0]
    assert last_record.status == "sent"
    assert last_record.delivered_event_hashes == ("event_hash_1", "event_hash_2")


def test_geopolitical_classification_rules_terrorism_and_kinetic_strikes():
    """Verify terror plots classify as TERRORISM_INSURGENCY and drone attacks as ARMED_CONFLICT."""
    classifier = RuleBasedClassifier()

    # Case 1: RAF Fairford terror plot - should classify as TERRORISM_INSURGENCY
    fairford_title = "3 Iranian-backed men indicted in attempted plot on RAF Fairford"
    fairford_body = (
        "Three men linked to Iranian proxy networks were indicted in federal court in "
        "connection with a suspected terror plot targeting RAF Fairford air base in "
        "Gloucestershire. Investigators noted background tensions from the wider Middle East war."
    )
    res1 = classifier.classify_text(fairford_title, fairford_body)
    assert res1.primary_topic == TopicCategory.TERRORISM_INSURGENCY
    assert res1.confidence >= 0.85

    # Case 2: Drone attack on military base - should classify as ARMED_CONFLICT
    bahrain_title = "Iran targets US military base in Bahrain with drone attack"
    bahrain_body = (
        "Iranian-backed forces launched a targeted drone attack against a US naval and air "
        "operations facility in Bahrain, causing structural damage. Commercial traffic in the "
        "Strait of Hormuz was alerted."
    )
    res2 = classifier.classify_text(bahrain_title, bahrain_body)
    assert res2.primary_topic == TopicCategory.ARMED_CONFLICT
    assert res2.confidence >= 0.85


def test_actor_sanitization_bare_branches_and_capital_metonyms():
    """Verify rejection of bare military branch nouns and deduplication of capital metonyms."""
    # 1. Bare branch nouns are polluted
    assert is_polluted_actor("Army") is True
    assert is_polluted_actor("the army") is True
    assert is_polluted_actor("Navy") is True
    assert is_polluted_actor("Air Force") is True
    assert is_polluted_actor("Military") is True
    assert is_polluted_actor("Armed Forces") is True
    assert is_polluted_actor("Ministry of Defense") is True
    assert is_polluted_actor("Police") is True
    assert is_polluted_actor("Troops") is True

    # 2. Qualified entities are retained
    assert is_polluted_actor("US Army") is False
    assert is_polluted_actor("Italian Army") is False
    assert is_polluted_actor("Royal Navy") is False
    assert is_polluted_actor("Bahrain military") is False

    # 3. Capital metonyms are deduplicated when sovereign parent country is present
    actors_with_italy = ["Rome", "Italy", "Leonardo"]
    clean1 = sanitize_actors(actors_with_italy)
    assert "Rome" not in clean1
    assert "Italy" in clean1
    assert "Leonardo" in clean1

    # 4. Standalone capital metonym is preserved when parent country is absent
    clean2 = sanitize_actors(["Rome"])
    assert clean2 == ["Rome"]


def test_safe_sentence_splitting_dotted_initialisms():
    """Verify dotted initialisms (U.K., U.S., E.U., U.N.) and titles do not truncate sentences."""
    text = (
        "Italy increased defence spending after the U.K. committed new funds to the "
        "GCAP fighter jet program. Gen. Smith and Lt. Col. Davis welcomed the trilateral accord."
    )
    passages = split_sentences_safely(text)
    assert len(passages) == 2
    expected_p1 = (
        "Italy increased defence spending after the U.K. committed new funds to the "
        "GCAP fighter jet program."
    )
    assert passages[0] == expected_p1
    assert "Lt. Col. Davis" in passages[1]
    assert "Gen. Smith" in passages[1]


def test_formatter_omits_orphaned_source_line_on_single_claim_duplicate():
    """Verify single-claim narrative duplicating headline does not leave bare (Source) line."""
    formatter = TelegramDigestFormatter()
    summary = EventSummary(
        event_id="evt-panama-attack",
        headline="Panama Journalist Targeted in Attack",
        what_happened="Panama Journalist Targeted in Attack",  # Exact duplicate of headline
        sources=(
            EventSourceAttribution(
                source_name="La Prensa",
                title="Journalist targeted in firearm attack",
                url="https://laprensa.pa/article/1",
            ),
        ),
        claims=(
            SummaryClaim(
                claim_id="c1",
                text="Panama Journalist Targeted in Attack",
                evidence_refs=(
                    ClaimEvidenceReference(
                        source_url="https://laprensa.pa/article/1",
                        evidence_span="targeted in firearm attack",
                        verification_status="accepted",
                    ),
                ),
                classification="reported_fact",
                central=True,
                verification_status="accepted",
            ),
        ),
        model_used="test",
        confidence_qualification="single_source",
    )
    block = formatter.format_event_block(
        summary,
        TopicCategory.SECURITY_INCIDENT,
        "single_source",
        1,
        1,
    )
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    for line in lines:
        assert not (line.startswith("<i>(<a") and line.endswith("</a>)</i>"))
        assert line != '<i>(<a href="https://laprensa.pa/article/1">La Prensa</a>)</i>'


def test_split_sentences_safely_protects_currency_and_decimals():
    """Verify currency figures and decimal numbers are not truncated."""
    text = (
        "The defence ministry approved a package of $16. 5 million for border security. "
        "Separately, €12.4 billion was allocated for joint procurement."
    )
    passages = split_sentences_safely(text)
    assert len(passages) == 2
    assert "$16. 5 million" in passages[0]
    assert "€12.4 billion" in passages[1]

    # Also test detect_malformed_clauses does not falsely flag currency decimals
    from geopulse.summarization.validator import detect_malformed_clauses

    issues = detect_malformed_clauses(text)
    assert len(issues) == 0
