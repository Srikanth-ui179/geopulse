"""Unit and regression tests for factual grounding, relevance, and hallucination quarantine.

Verifies the 7 specific regression scenarios:
1. The Battle of Bailén historical article (fails validation and filtered as irrelevant).
2. Genuine current military development (passes relevance and validation).
3. Current defence-policy opinion article (passes relevance and validation).
4. Historical story with a genuine current news hook (DNA identification on the Somme is preserved).
5. Source containing too little text to summarize (rejected as insufficient_evidence).
6. Source mentioning an entity in an unrelated context (fabrications fail validation).
7. Model output inventing persons, organizations, or dates (fails validation and is quarantined).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from geopulse.classification.models import TopicCategory
from geopulse.classification.rules import RuleBasedClassifier
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.ingestion.relevance import classify_relevance
from geopulse.summarization.client import MockOllamaClient
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
)
from geopulse.summarization.service import EventSummarizer
from geopulse.summarization.validator import (
    clean_headline_grammar,
    detect_malformed_clauses,
    is_evidence_sufficient,
    is_publisher_or_outlet,
    sanitize_actors,
    validate_summary_grounding,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "grounding_regression_fixtures.json"
)


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_cluster_from_data(data: dict) -> EventCluster:
    pub_dt = (
        datetime.fromisoformat(data["published_at"])
        if data.get("published_at")
        else datetime.now(UTC)
    )
    art = ArticleRecord(
        source_name=data["source_name"],
        title=data["title"],
        url=data["url"],
        canonical_url=data["url"],
        published_at=pub_dt,
        snippet=data.get("snippet"),
        content_hash=f"hash-{data['title'][:10]}",
    )
    return EventCluster(
        title=data["title"],
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=pub_dt,
        last_seen_at=pub_dt,
        published_at=pub_dt,
        confidence_label="single_source",
        articles=(art,),
    )


# ---------------------------------------------------------------------------
# Scenario 1: Battle of Bailén historical article
# ---------------------------------------------------------------------------


def test_battle_of_bailen_relevance_and_classification_is_irrelevant():
    """Historical military essay without a contemporary news hook must be filtered out."""
    fixtures = _load_fixtures()["battle_of_bailen"]
    status = classify_relevance(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert status == "irrelevant"

    classifier = RuleBasedClassifier()
    cls_res = classifier.classify_text(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert cls_res.relevance_status == "irrelevant"
    assert "contemporary news hook" in cls_res.explanation.lower()


def test_battle_of_bailen_fails_validation_when_milley_nato_hallucinated():
    """Original hallucinated Mark Milley / NATO claim fails deterministic validation."""
    fixtures = _load_fixtures()["battle_of_bailen"]
    cluster = _build_cluster_from_data(fixtures)
    raw_out = fixtures["hallucinated_output"]

    summary = EventSummary(
        event_id=cluster.id,
        headline=raw_out["headline"],
        what_happened=raw_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(raw_out["key_details"]["actors"]),
            location=raw_out["key_details"]["location"],
            timing=raw_out["key_details"]["timing"],
            action=raw_out["key_details"]["action"],
        ),
        confidence_qualification=raw_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert not val_res.is_valid
    assert any("milley" in e.lower() for e in val_res.unsupported_entities)
    assert any("nato" in e.lower() for e in val_res.unsupported_entities)


def test_battle_of_bailen_summarization_quarantines_hallucination():
    """EventSummarizer must quarantine the event when hallucination persists."""
    fixtures = _load_fixtures()["battle_of_bailen"]
    cluster = _build_cluster_from_data(fixtures)
    hallucinated_json = json.dumps(fixtures["hallucinated_output"])

    client = MockOllamaClient(canned_response=hallucinated_json)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "quarantined"
    assert res.quarantine_reason is not None
    assert "unsupported entities" in res.quarantine_reason
    assert "milley" in res.quarantine_reason.lower()


# ---------------------------------------------------------------------------
# Scenario 2: Genuine current military development
# ---------------------------------------------------------------------------


def test_genuine_current_military_development_passes():
    """Current military operational event passes relevance and factual validation."""
    fixtures = _load_fixtures()["current_military_development"]
    status = classify_relevance(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert status == "relevant"

    cluster = _build_cluster_from_data(fixtures)
    valid_out = fixtures["valid_output"]
    summary = EventSummary(
        event_id=cluster.id,
        headline=valid_out["headline"],
        what_happened=valid_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(valid_out["key_details"]["actors"]),
            location=valid_out["key_details"]["location"],
            timing=valid_out["key_details"]["timing"],
            action=valid_out["key_details"]["action"],
        ),
        confidence_qualification=valid_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert val_res.is_valid
    assert len(val_res.unsupported_entities) == 0


# ---------------------------------------------------------------------------
# Scenario 3: Current defence-policy opinion article
# ---------------------------------------------------------------------------


def test_current_defence_policy_opinion_passes():
    """Contemporary defence policy analysis and official warnings are preserved and valid."""
    fixtures = _load_fixtures()["current_defence_policy_opinion"]
    status = classify_relevance(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert status == "relevant"

    cluster = _build_cluster_from_data(fixtures)
    valid_out = fixtures["valid_output"]
    summary = EventSummary(
        event_id=cluster.id,
        headline=valid_out["headline"],
        what_happened=valid_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(valid_out["key_details"]["actors"]),
            location=valid_out["key_details"]["location"],
            timing=valid_out["key_details"]["timing"],
            action=valid_out["key_details"]["action"],
        ),
        confidence_qualification=valid_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert val_res.is_valid


# ---------------------------------------------------------------------------
# Scenario 4: Historical story with a genuine current news hook
# ---------------------------------------------------------------------------


def test_historical_story_with_news_hook_preserved_and_valid():
    """Historical context is preserved when an active discovery or burial hook is present."""
    fixtures = _load_fixtures()["historical_story_with_news_hook"]
    status = classify_relevance(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert status == "relevant"

    classifier = RuleBasedClassifier()
    cls_res = classifier.classify_text(
        title=fixtures["title"],
        snippet=fixtures["snippet"],
        source_name=fixtures["source_name"],
    )
    assert cls_res.relevance_status == "relevant"

    cluster = _build_cluster_from_data(fixtures)
    valid_out = fixtures["valid_output"]
    summary = EventSummary(
        event_id=cluster.id,
        headline=valid_out["headline"],
        what_happened=valid_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(valid_out["key_details"]["actors"]),
            location=valid_out["key_details"]["location"],
            timing=valid_out["key_details"]["timing"],
            action=valid_out["key_details"]["action"],
        ),
        confidence_qualification=valid_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert val_res.is_valid


# ---------------------------------------------------------------------------
# Scenario 5: Source containing too little text to summarize
# ---------------------------------------------------------------------------


def test_source_containing_too_little_text_quarantined():
    """Articles with sparse or boilerplate snippets must be caught as insufficient_evidence."""
    fixtures = _load_fixtures()["source_too_little_text"]
    cluster = _build_cluster_from_data(fixtures)

    is_suff, reason = is_evidence_sufficient(cluster)
    assert not is_suff
    assert "insufficient evidence" in reason.lower()

    client = MockOllamaClient()
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "insufficient_evidence"
    assert "insufficient evidence" in (res.error_message or "").lower()


# ---------------------------------------------------------------------------
# Scenario 6: Source mentioning entity in unrelated context
# ---------------------------------------------------------------------------


def test_source_mentioning_entity_in_unrelated_context_fails_validation():
    """Fabrication of combat actions from a social dinner context must fail."""
    fixtures = _load_fixtures()["entity_in_unrelated_context"]
    cluster = _build_cluster_from_data(fixtures)
    fab_out = fixtures["fabricated_military_output"]

    summary = EventSummary(
        event_id=cluster.id,
        headline=fab_out["headline"],
        what_happened=fab_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(fab_out["key_details"]["actors"]),
            location=fab_out["key_details"]["location"],
            timing=fab_out["key_details"]["timing"],
            action=fab_out["key_details"]["action"],
        ),
        confidence_qualification=fab_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert not val_res.is_valid
    # "Austrian military" was fabricated and not in snippet (which was about university rectors)
    assert any("austrian" in e.lower() for e in val_res.unsupported_entities)


# ---------------------------------------------------------------------------
# Scenario 7: Model output inventing person, organization, date, or action
# ---------------------------------------------------------------------------


def test_model_output_invents_person_or_organization_quarantined():
    """Hallucinating unmentioned secretaries or alliances is quarantined."""
    fixtures = _load_fixtures()["model_output_invents_entity"]
    cluster = _build_cluster_from_data(fixtures)
    fab_out = fixtures["fabricated_output"]

    summary = EventSummary(
        event_id=cluster.id,
        headline=fab_out["headline"],
        what_happened=fab_out["what_happened"],
        key_details=EventKeyDetails(
            actors=tuple(fab_out["key_details"]["actors"]),
            location=fab_out["key_details"]["location"],
            timing=fab_out["key_details"]["timing"],
            action=fab_out["key_details"]["action"],
        ),
        confidence_qualification=fab_out["confidence_qualification"],
        sources=(),
        model_used="test-mock",
    )

    val_res = validate_summary_grounding(summary, cluster)
    assert not val_res.is_valid
    assert any("austin" in e.lower() for e in val_res.unsupported_entities)
    assert any("nato" in e.lower() for e in val_res.unsupported_entities)
    assert any("lockheed" in e.lower() for e in val_res.unsupported_entities)


# ---------------------------------------------------------------------------
# Additional Guardrails: Bounded Retry and Digest Formatter Isolation
# ---------------------------------------------------------------------------


def test_bounded_retry_succeeds_when_corrected():
    """If the LLM corrects an unsupported entity on retry, the corrected summary is accepted."""
    fixtures = _load_fixtures()["battle_of_bailen"]
    cluster = _build_cluster_from_data(fixtures)

    attempt = 0

    def custom_gen(prompt: str) -> str:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            # First attempt introduces unsupported NATO
            return json.dumps(
                {
                    "headline": "Generals discuss NATO operations",
                    "what_happened": "Officers discussed NATO readiness.",
                    "key_details": {
                        "actors": ["NATO"],
                        "location": "Europe",
                        "timing": "October 2026",
                        "action": "Readiness",
                    },
                    "confidence_qualification": "Single-source report; unverified independently.",
                }
            )
        else:
            # Second attempt corrects using source terms
            return json.dumps(
                {
                    "headline": "Spanish Army of Andalusia Attacks Forces of Emperor Napoleon",
                    "what_happened": (
                        "Captain-General Francisco Javier Castaños launched a surprise attack "
                        "against the forces of Emperor Napoleon Bonaparte in July 1808."
                    ),
                    "key_details": {
                        "actors": [
                            "Captain-General Francisco Javier Castaños",
                            "Napoleon Bonaparte",
                        ],
                        "location": "Bailén",
                        "timing": "July 16, 1808",
                        "action": "Surprise attack and combat",
                    },
                    "confidence_qualification": (
                        "Single-source report from War on the Rocks; unverified independently."
                    ),
                }
            )

    client = MockOllamaClient(custom_generator=custom_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert attempt == 2
    assert res.status == "success"
    assert res.summary is not None
    assert "Castaños" in res.summary.headline or "Napoleon" in res.summary.headline


def test_telegram_digest_formatter_excludes_quarantined_events():
    """Quarantined and insufficient-evidence events must never be delivered in Telegram chunks."""
    fixtures = _load_fixtures()["battle_of_bailen"]
    cluster = _build_cluster_from_data(fixtures)

    # SummarizationResult with status quarantined
    from geopulse.summarization.models import SummarizationResult

    quarantined_res = SummarizationResult(
        event_id=cluster.id,
        status="quarantined",
        error_message="Quarantined: unsupported entities ['General Mark Milley', 'NATO']",
        quarantine_reason="unsupported entities ['General Mark Milley', 'NATO']",
        event=cluster,
    )

    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest([quarantined_res], edition="morning")
    # Empty briefing chunk is generated when no successful events exist
    assert len(chunks) == 1
    assert "No significant new military or geopolitical developments" in chunks[0]
    assert "General Mark Milley" not in chunks[0]
    assert "NATO" not in chunks[0]


def test_vague_disagreement_phrase_without_source_dispute_fails_validation():
    """Vague phrases like 'this is disputed by some' fail validation if source has no dispute."""
    dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Germany announces new ammunition factory",
        url="https://defensenews.com/ammo-factory",
        canonical_url="https://defensenews.com/ammo-factory",
        published_at=dt,
        snippet="Germany announced the construction of a new 155mm artillery shell facility.",
        content_hash="ammo123",
    )
    cluster = EventCluster(
        title="Germany announces new ammunition factory",
        category="weapons_procurement",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id=cluster.id,
        headline="Germany Announces New Ammunition Facility",
        what_happened=(
            "Germany announced a new 155mm artillery facility. This is disputed by some."
        ),
        key_details=EventKeyDetails(actors=("Germany",)),
        confidence_qualification=(
            "Single-source report from Defense News; unverified independently."
        ),
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(summary, cluster)
    assert not val.is_valid
    assert any("vague disagreement" in c.lower() for c in val.unsupported_claims)


def test_operational_deployment_inference_without_source_support_fails_validation():
    """Unveiled or prototyped systems cannot be claimed as operationally deployed."""
    dt = datetime(2026, 10, 26, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="Helsing unveils new AI interceptor drone concept",
        url="https://breakingdefense.com/helsing-interceptor",
        canonical_url="https://breakingdefense.com/helsing-interceptor",
        published_at=dt,
        snippet=(
            "Helsing unveiled its new AI-driven interceptor prototype "
            "designed for counter-drone defense."
        ),
        content_hash="helsing123",
    )
    cluster = EventCluster(
        title="Helsing unveils new AI interceptor drone concept",
        category="weapons_procurement",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Summary falsely infers operational deployment
    summary = EventSummary(
        event_id=cluster.id,
        headline="Helsing Interceptor Drone Unveiled",
        what_happened=(
            "Helsing unveiled an AI-powered interceptor that is now operationally deployed "
            "to the front for combat missions."
        ),
        key_details=EventKeyDetails(actors=("Helsing",)),
        confidence_qualification=(
            "Single-source report from Breaking Defense; unverified independently."
        ),
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(summary, cluster)
    assert not val.is_valid
    assert any("operational deployment" in c.lower() for c in val.unsupported_claims)


def test_single_source_independent_verification_claim_fails_validation():
    """Single-source summaries cannot claim they are independently verified."""
    dt = datetime(2026, 10, 26, 15, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Military forces report perimeter defense",
        url="https://reuters.com/perimeter",
        canonical_url="https://reuters.com/perimeter",
        published_at=dt,
        snippet="Regional military commanders reported reinforcing perimeter security.",
        content_hash="reuters123",
    )
    cluster = EventCluster(
        title="Military forces report perimeter defense",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id=cluster.id,
        headline="Military Forces Reinforce Perimeter",
        what_happened="Military forces reinforced perimeter security.",
        key_details=EventKeyDetails(actors=("Military Forces",)),
        confidence_qualification="This report has been independently verified by defence analysts.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(summary, cluster)
    assert not val.is_valid
    assert any("independent verification" in c.lower() for c in val.unsupported_claims)


def test_clean_headline_grammar_normalizes_demonym_subjects_and_html_entities():
    from geopulse.summarization.validator import clean_headline_grammar

    assert (
        clean_headline_grammar("Turkish Unveils Ateş Serbest Drone System")
        == "Turkey Unveils Ateş Serbest Drone System"
    )
    assert (
        clean_headline_grammar("British Announces New Naval Frigate Order")
        == "UK Announces New Naval Frigate Order"
    )
    assert (
        clean_headline_grammar("&#8216;Cyprus to acquire new offshore patrol vessel&#8217;")
        == "Cyprus to acquire new offshore patrol vessel"
    )
    # Valid demonym modifying a noun should not be altered
    assert (
        clean_headline_grammar("Turkish Armed Forces Complete Joint Drill")
        == "Turkish Armed Forces Complete Joint Drill"
    )


def test_sanitize_actors_deduplicates_and_cleans():
    from geopulse.summarization.validator import sanitize_actors

    raw = ["Turkey", "turkey", "Turkey's", "Turkish Armed Forces", "unspecified", ""]
    cleaned = sanitize_actors(raw)
    assert cleaned == ["Turkey", "Turkish Armed Forces"]


def test_e5_statement_tautological_snippet_fails_evidence_sufficiency():
    """Tautological snippet provides no facts and must fail evidence sufficiency."""
    dt = datetime(2026, 10, 26, 17, 44, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="E5 Defence Ministers' Statement: 26 October 2026",
        url="https://gov.uk/e5-statement",
        canonical_url="https://gov.uk/e5-statement",
        published_at=dt,
        snippet="E5 Defence Ministers' Statement.",
        content_hash="hash-e5-tautology",
    )
    cluster = EventCluster(
        title="E5 Defence Ministers' Statement: 26 October 2026",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    is_sufficient, reason = is_evidence_sufficient(cluster)
    assert not is_sufficient
    assert reason is not None
    assert "document label without substantive reporting body" in reason.lower()


def test_e5_statement_hallucinated_nuclear_denial_fails_grounding_validation():
    """Summary inventing a denial and nuclear policy claims without source text fails validation."""
    dt = datetime(2026, 10, 26, 17, 44, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="E5 Defence Ministers' Statement: 26 October 2026",
        url="https://gov.uk/e5-statement",
        canonical_url="https://gov.uk/e5-statement",
        published_at=dt,
        snippet=(
            "Joint statement from the Defence Ministers of France, Germany, Italy, Poland "
            "and the UK on European security."
        ),
        content_hash="hash-e5-valid-snippet",
    )
    cluster = EventCluster(
        title="E5 Defence Ministers' Statement: 26 October 2026",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )

    # Hallucinated output claiming a denial of nuclear capabilities expansion
    summary = EventSummary(
        event_id=cluster.id,
        headline="UK Ministry of Defence Denies Nuclear Capabilities Expansion Plans",
        what_happened=(
            "The UK Ministry of Defence denied reports of nuclear capabilities expansion "
            "following European defence talks."
        ),
        key_details=EventKeyDetails(
            actors=("UK Ministry of Defence",),
            location="Europe",
            timing="October 26, 2026",
            action="Denial of nuclear capabilities expansion",
        ),
        confidence_qualification=(
            "Single-source report from UK Ministry of Defence; unverified independently."
        ),
        sources=(),
        model_used="hallucinated-model",
    )

    val = validate_summary_grounding(summary, cluster)
    assert not val.is_valid
    assert any("denial or rebuttal" in c.lower() for c in val.unsupported_claims)
    assert any("nuclear capability" in c.lower() for c in val.unsupported_claims)
    assert val.quarantine_reason is not None
    assert "denial or rebuttal" in val.quarantine_reason.lower()
    assert "nuclear capability" in val.quarantine_reason.lower()


def test_unspecified_in_reporting_never_appears_in_rendered_output():
    """Placeholder phrases like 'Unspecified in reporting' are sanitized and never rendered."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="BBC World News",
        title="Truce begins in wartorn Ukraine city after announcement",
        url="https://bbc.com/truce",
        canonical_url="https://bbc.com/truce",
        published_at=dt,
        snippet=("The localised ceasefire secured by diplomatic envoys will allow aid deliveries."),
        content_hash="hash-truce-test",
    )
    cluster = EventCluster(
        title="Truce begins in wartorn Ukraine city after announcement",
        category="diplomacy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model output that leaked "Unspecified in reporting" in several fields
    raw_model_json = json.dumps(
        {
            "headline": "Truce begins in Ukrainian city of [Unspecified in reporting]",
            "what_happened": (
                "A localised ceasefire has taken effect in the city of [Unspecified in reporting] "
                "to permit humanitarian deliveries."
            ),
            "key_details": {
                "actors": ["Unspecified in reporting", "Diplomatic envoys", "Unspecified"],
                "location": "Unspecified in reporting",
                "timing": "Date not specified in source",
                "action": "Unspecified in reporting",
            },
            "significance": "Unspecified in reporting",
            "confidence_qualification": "Single-source report; unverified independently.",
        }
    )
    client = MockOllamaClient(canned_response=raw_model_json)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "success"
    assert res.summary is not None
    assert "Unspecified in reporting" not in res.summary.headline
    assert "Unspecified in reporting" not in res.summary.what_happened
    assert res.summary.key_details.location is None
    assert res.summary.key_details.timing is None
    assert res.summary.key_details.action is None
    assert "Unspecified" not in res.summary.key_details.actors
    assert res.summary.significance is None

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=cluster.category,
        confidence_label=cluster.confidence_label,
        rank=1,
        total=1,
        published_at=dt,
    )
    assert "Unspecified in reporting" not in block
    assert "[Unspecified]" not in block
    assert "Location:" not in block
    assert "Action:" not in block


def test_event_location_not_inferred_from_actor_nationality_or_capital():
    """Space, orbital, and maritime events must not assign actor capitals as locations."""
    dt = datetime(2026, 10, 23, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        url="https://defensenews.com/space-satellite",
        canonical_url="https://defensenews.com/space-satellite",
        published_at=dt,
        snippet=(
            "Washington blamed Moscow for deliberately moving a Russian military satellite "
            "with an undisclosed mission close last week to a U.S.-registered satellite in orbit."
        ),
        content_hash="hash-space-sat",
    )
    cluster = EventCluster(
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model erroneously infers location as "Moscow" simply because Russia was blamed
    raw_model_json = json.dumps(
        {
            "headline": "Russia Manoeuvres Military Satellite Near US Spacecraft in Orbit",
            "what_happened": (
                "Washington blamed Moscow for deliberately manoeuvring a military satellite "
                "close to a US-registered satellite in orbit."
            ),
            "key_details": {
                "actors": ["Washington", "Moscow", "Russia", "US"],
                "location": "Moscow",
                "timing": "Last week",
                "action": "Military satellite proximity manoeuvre",
            },
            "significance": "Highlights growing counter-space capabilities.",
            "confidence_qualification": "Single-source report; unverified independently.",
        }
    )
    client = MockOllamaClient(canned_response=raw_model_json)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "success"
    assert res.summary is not None
    # Location "Moscow" must be removed since the event occurred in space/orbit
    assert res.summary.key_details.location is None

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=cluster.category,
        confidence_label=cluster.confidence_label,
        rank=1,
        total=1,
        published_at=dt,
    )
    assert "• <b>Location:</b> Moscow" not in block
    assert "• <b>Location:</b>" not in block


def test_unknown_locations_omitted_cleanly_from_rendering():
    """When no event location is supported, the Location bullet is omitted entirely."""
    dt = datetime(2026, 10, 26, 10, 0, tzinfo=UTC)
    summary = EventSummary(
        event_id="test-evt",
        headline="Strategic Cyber Infrastructure Advisory Issued",
        what_happened=(
            "National security agencies issued an advisory regarding state cyber operations."
        ),
        key_details=EventKeyDetails(
            actors=("National security agencies",),
            location=None,
            timing="October 26, 2026",
            action="Cyber advisory",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=summary,
        category="cyber_hybrid_warfare",  # type: ignore[arg-type]
        confidence_label="single_source",
        rank=1,
        total=1,
        published_at=dt,
    )
    assert "• <b>Location:</b>" not in block
    assert "• <b>Action:</b> Cyber advisory" in block


def test_publication_dates_not_misrepresented_as_event_dates():
    """An article from 4 days ago must not claim the event occurred 'Within the last 24 hours'."""
    pub_dt = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="China carrier voyages form evolving pattern in disputed seas",
        url="https://defensenews.com/carrier",
        canonical_url="https://defensenews.com/carrier",
        published_at=pub_dt,
        snippet=("Passages near Taiwan and Japan in regional waters raised alerts across states."),
        content_hash="hash-carrier-older",
    )
    cluster = EventCluster(
        title="China carrier voyages form evolving pattern in disputed seas",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=pub_dt,
        last_seen_at=pub_dt,
        published_at=pub_dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model erroneously outputs "Within the last 24 hours" for a story published Oct 23
    raw_model_json = json.dumps(
        {
            "headline": "China Carrier Group Passages Raise Regional Security Alerts",
            "what_happened": (
                "Aircraft carrier group passages near Taiwan and Japan raised regional alerts."
            ),
            "key_details": {
                "actors": ["China", "Taiwan", "Japan"],
                "location": "disputed Asian seas",
                "timing": "Within the last 24 hours",
                "action": "Aircraft carrier operations",
            },
            "significance": None,
            "confidence_qualification": "Single-source report; unverified independently.",
        }
    )
    client = MockOllamaClient(canned_response=raw_model_json)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "success"
    assert res.summary is not None
    # Corrected timing reflects the verified publication date
    assert "within the last 24 hours" not in res.summary.key_details.timing.lower()
    assert "October 23, 2026" in res.summary.key_details.timing

    formatter = TelegramDigestFormatter()
    block = formatter.format_event_block(
        summary=res.summary,
        category=cluster.category,
        confidence_label=cluster.confidence_label,
        rank=1,
        total=1,
        published_at=pub_dt,
    )
    assert "Within the last 24 hours" not in block
    assert "October 23, 2026" in block


def test_unsupported_model_generated_details_trigger_quarantine():
    """Model inventing kinetic strikes and casualties not in source text fails validation."""
    dt = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="Shipbuilder signs contract for three new offshore patrol vessels",
        url="https://breakingdefense.com/patrol-vessels",
        canonical_url="https://breakingdefense.com/patrol-vessels",
        published_at=dt,
        snippet=(
            "European shipyard signs deal to construct three offshore patrol vessels "
            "for maritime surveillance."
        ),
        content_hash="hash-ship-contract",
    )
    cluster = EventCluster(
        title="Shipbuilder signs contract for three new offshore patrol vessels",
        category="weapons_procurement",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model invents airstrikes and 40 casualties absent from source text
    fab_summary = EventSummary(
        event_id=cluster.id,
        headline="Shipbuilder Targeted in Lethal Airstrike Attack",
        what_happened="Naval forces executed heavy airstrikes against shipyard, leaving 40 dead.",
        key_details=EventKeyDetails(
            actors=("Naval forces", "Shipbuilder"),
            location="Europe",
            timing="October 26, 2026",
            action="Airstrike assault",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(fab_summary, cluster)
    assert not val.is_valid
    assert any("kinetic military action" in c.lower() for c in val.unsupported_claims)
    assert any("casualty" in c.lower() for c in val.unsupported_claims)


def test_valid_source_supported_claims_remain_in_summary():
    """Carrier passages raising alerts supported by source text pass validation."""
    dt = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title=(
            "China’s first aircraft carrier forms ‘evolving pattern’ of voyages in "
            "disputed Asian seas"
        ),
        url="https://defensenews.com/china-carrier",
        canonical_url="https://defensenews.com/china-carrier",
        published_at=dt,
        snippet=(
            "The carrier group’s passages not far from Taiwan, Japan and the Philippines have "
            "raised alerts among regional militaries."
        ),
        content_hash="hash-china-carrier-valid",
    )
    cluster = EventCluster(
        title="China carrier voyages form evolving pattern in disputed Asian seas",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    valid_summary = EventSummary(
        event_id=cluster.id,
        headline="China's Carrier Passages in Disputed Asian Seas Raise Regional Alerts",
        what_happened=(
            "The carrier group's passages near Taiwan, Japan and the Philippines have "
            "raised alerts among regional militaries as naval voyages form an evolving pattern."
        ),
        key_details=EventKeyDetails(
            actors=("China", "Taiwan", "Japan", "Philippines"),
            location="disputed Asian seas",
            timing="October 23, 2026",
            action="Carrier group voyages",
        ),
        confidence_qualification=(
            "Single-source report from Defense News; unverified independently."
        ),
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(valid_summary, cluster)
    assert val.is_valid
    assert len(val.unsupported_claims) == 0
    assert len(val.unsupported_entities) == 0


def test_contradictory_attribution_triggers_quarantine():
    """Summary claiming 'Russia blamed Moscow' contradicts source and fails validation."""
    dt = datetime(2026, 10, 23, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        url="https://defensenews.com/space-satellite",
        canonical_url="https://defensenews.com/space-satellite",
        published_at=dt,
        snippet=(
            "Washington blamed Moscow for deliberately moving a Russian military satellite "
            "with an undisclosed mission close last week to a U.S.-registered satellite."
        ),
        content_hash="hash-space-sat-blame",
    )
    cluster = EventCluster(
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Contradictory allegation: Russia blaming its own capital
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Russia Blamed Moscow for Moving Military Satellite",
        what_happened=(
            "Russia blamed Moscow for deliberately moving a Russian military satellite "
            "close last week to a U.S.-registered satellite."
        ),
        key_details=EventKeyDetails(
            actors=("Russia", "Moscow", "US"),
            location="Space",
            timing="Last week",
            action="Satellite movement",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any("contradictory attribution" in c.lower() for c in val.unsupported_claims)


def test_correct_attribution_survives_validation():
    """Summary correctly attributing 'Washington blamed Moscow' passes validation."""
    dt = datetime(2026, 10, 23, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        url="https://defensenews.com/space-satellite",
        canonical_url="https://defensenews.com/space-satellite",
        published_at=dt,
        snippet=(
            "Washington blamed Moscow for deliberately moving a Russian military satellite "
            "with an undisclosed mission close last week to a U.S.-registered satellite."
        ),
        content_hash="hash-space-sat-good",
    )
    cluster = EventCluster(
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    good_summary = EventSummary(
        event_id=cluster.id,
        headline="Washington Blames Moscow for Moving Military Satellite Near US Spacecraft",
        what_happened=(
            "Washington blamed Moscow for deliberately moving a Russian military satellite "
            "close last week to a U.S.-registered satellite."
        ),
        key_details=EventKeyDetails(
            actors=("Washington", "Moscow", "US", "Russia"),
            location="Space",
            timing="Last week",
            action="Satellite movement",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(good_summary, cluster)
    assert val.is_valid
    assert len(val.unsupported_claims) == 0


def test_unsupported_numerical_claim_triggers_quarantine():
    """Model generating unsupported percentage or duration fails validation."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="The Machine Staff",
        url="https://warontherocks.com/the-machine-staff",
        canonical_url="https://warontherocks.com/the-machine-staff",
        published_at=dt,
        snippet=(
            "At Nellis Air Force Base last year, an AI system was tasked with building courses "
            "of action. The AI system took less than a minute to complete its task with solutions "
            "rated at 97 percent viable. By contrast, human planners took roughly 19 minutes "
            "and only reached 48 percent viability."
        ),
        content_hash="hash-machine-staff",
    )
    cluster = EventCluster(
        title="The Machine Staff",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model invents 25 percent and 45 minutes not present in source text
    unsupported_summary = EventSummary(
        event_id=cluster.id,
        headline="Nellis Air Force Base AI Operational Experiment Results",
        what_happened=(
            "An AI system at Nellis Air Force Base completed problem-solving in under a minute, "
            "while human planners took nearly 45 minutes to reach 25 percent viability."
        ),
        key_details=EventKeyDetails(
            actors=("Nellis Air Force Base", "US Military"),
            location="Nellis Air Force Base",
            timing="October 27, 2026",
            action="AI planning experiment",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(unsupported_summary, cluster)
    assert not val.is_valid
    assert any("25 percent" in c or "45 minutes" in c for c in val.unsupported_claims)


def test_valid_source_supported_numerical_claims_survive():
    """Source-supported percentages (48%, 97%) and durations (19 min) pass validation."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="The Machine Staff",
        url="https://warontherocks.com/the-machine-staff",
        canonical_url="https://warontherocks.com/the-machine-staff",
        published_at=dt,
        snippet=(
            "At Nellis Air Force Base last year, an AI system was tasked with building courses "
            "of action. The AI system took less than a minute to complete its task with solutions "
            "rated at 97 percent viable. By contrast, human planners took roughly 19 minutes "
            "and only reached 48 percent viability."
        ),
        content_hash="hash-machine-staff-valid",
    )
    cluster = EventCluster(
        title="The Machine Staff",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    valid_summary = EventSummary(
        event_id=cluster.id,
        headline="AI Outperforms Human Planners in Nellis Air Force Base Experiment",
        what_happened=(
            "An AI system solved complex operational problems in under a minute with solutions "
            "rated at 97 percent viable, while human counterparts took roughly 19 minutes "
            "and reached 48 percent viability."
        ),
        key_details=EventKeyDetails(
            actors=("Nellis Air Force Base", "US Military"),
            location="Nellis Air Force Base",
            timing="Last year",
            action="AI operational testing",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(valid_summary, cluster)
    assert val.is_valid
    assert len(val.unsupported_claims) == 0


def test_unsupported_demonstration_claims_trigger_quarantine():
    """Model claiming unverified system was 'demonstrated' in testing fails validation."""
    dt = datetime(2026, 10, 26, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="Helsing rolls out Contra air defense design, plus two new inteceptors",
        url="https://breakingdefense.com/helsing-contra",
        canonical_url="https://breakingdefense.com/helsing-contra",
        published_at=dt,
        snippet=(
            "The company claims the Contra system can track and engage 'hundreds or thousands "
            "of drone and missile threats at the same time.'"
        ),
        content_hash="hash-helsing-contra",
    )
    cluster = EventCluster(
        title="Helsing rolls out Contra air defense design, plus two new inteceptors",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model hallucinates that the system was demonstrated in field trials
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Helsing Demonstrates Contra Air Defense System",
        what_happened=(
            "Helsing demonstrated its new interceptors during live-fire testing against "
            "hundreds or thousands of drone targets."
        ),
        key_details=EventKeyDetails(
            actors=("Helsing",),
            location=None,
            timing="October 26, 2026",
            action="Air defense demonstration",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any("demonstration" in c.lower() for c in val.unsupported_claims)


def test_vendor_capability_claims_with_source_support_survive():
    """Vendor capability claim matching source text ('hundreds or thousands') survives."""
    dt = datetime(2026, 10, 26, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="Helsing rolls out Contra air defense design, plus two new inteceptors",
        url="https://breakingdefense.com/helsing-contra",
        canonical_url="https://breakingdefense.com/helsing-contra",
        published_at=dt,
        snippet=(
            "The company claims the Contra system can track and engage 'hundreds or thousands "
            "of drone and missile threats at the same time.'"
        ),
        content_hash="hash-helsing-valid",
    )
    cluster = EventCluster(
        title="Helsing rolls out Contra air defense design, plus two new inteceptors",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    good_summary = EventSummary(
        event_id=cluster.id,
        headline="Helsing Unveils Contra Air Defense Design and Interceptors",
        what_happened=(
            "Helsing unveiled its Contra air defense design, which the company claims "
            "can track and engage hundreds or thousands of drone and missile threats "
            "at the same time."
        ),
        key_details=EventKeyDetails(
            actors=("Helsing",),
            location=None,
            timing="October 26, 2026",
            action="Air defense design unveil",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(good_summary, cluster)
    assert val.is_valid
    assert len(val.unsupported_claims) == 0


def test_unsupported_ceasefire_negotiation_role_triggers_quarantine():
    """Asserting non-combatant negotiated or enforced ceasefire fails validation."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="BBC World News",
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        url="https://bbc.com/truce-ukraine",
        canonical_url="https://bbc.com/truce-ukraine",
        published_at=dt,
        snippet=(
            "The ceasefire in Oleshky, secured by US First Lady Melania Trump, is meant "
            "to allow aid deliveries and the evacuation of civilians."
        ),
        content_hash="hash-truce-melania",
    )
    cluster = EventCluster(
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        category="diplomacy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Model invents that she negotiated the ceasefire terms between the warring parties
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Ukraine City Enters Truce with Melania Trump's Ceasefire",
        what_happened=(
            "Melania Trump negotiated the ceasefire in Oleshky between Russian and Ukrainian "
            "forces to permit aid deliveries."
        ),
        key_details=EventKeyDetails(
            actors=("Melania Trump", "Oleshky residents"),
            location="Oleshky",
            timing="October 27, 2026",
            action="Ceasefire negotiation",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any("ceasefire role" in c.lower() for c in val.unsupported_claims)


def test_supported_ceasefire_aid_announcement_survives():
    """Summary accurately reporting secured ceasefire for aid deliveries survives."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="BBC World News",
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        url="https://bbc.com/truce-ukraine",
        canonical_url="https://bbc.com/truce-ukraine",
        published_at=dt,
        snippet=(
            "The ceasefire in Oleshky, secured by US First Lady Melania Trump, is meant "
            "to allow aid deliveries and the evacuation of civilians."
        ),
        content_hash="hash-truce-aid",
    )
    cluster = EventCluster(
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        category="diplomacy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    good_summary = EventSummary(
        event_id=cluster.id,
        headline="Ceasefire in Oleshky Secured for Humanitarian Aid and Evacuations",
        what_happened=(
            "A ceasefire in Oleshky secured with US First Lady Melania Trump is meant to "
            "allow humanitarian aid deliveries and the evacuation of civilians."
        ),
        key_details=EventKeyDetails(
            actors=("Melania Trump", "Oleshky residents"),
            location="Oleshky",
            timing="October 27, 2026",
            action="Ceasefire for aid deliveries",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(good_summary, cluster)
    assert val.is_valid
    assert len(val.unsupported_claims) == 0


def test_malformed_headline_words_and_phrasing_cleaned():
    """Malformed words like 'Intercceptors' and awkward 'forms patterns' are cleaned."""
    h1 = clean_headline_grammar("Helsing Unveils Contra Air Defense System, New Intercceptors")
    assert "Interceptors" in h1
    assert "Intercceptors" not in h1

    h2 = clean_headline_grammar("China’s Aircraft Carrier Forms Patterns in Disputed Asian Seas")
    assert "Conducts Pattern of Voyages" in h2 or "Conducts Voyages" in h2
    assert "Forms Patterns" not in h2

    h3 = clean_headline_grammar("Ukraine City Enters Truce with Melania Trump's Ceasefire")
    assert "Melania Trump's Ceasefire" not in h3
    assert "Humanitarian Aid" in h3


def test_quarantined_summaries_never_reach_telegram_output():
    """Summaries with status 'quarantined' are never rendered into Telegram chunks."""
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Satellite movement reported in orbit",
        url="https://defensenews.com/sat",
        canonical_url="https://defensenews.com/sat",
        published_at=dt,
        snippet="Washington blamed Moscow for moving military satellite.",
        content_hash="hash-sat-quarantine",
    )
    cluster = EventCluster(
        title="Satellite movement reported in orbit",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    quarantined_result = SummarizationResult(
        event_id=cluster.id,
        status="quarantined",
        error_message="Contradictory attribution detected",
        quarantine_reason="Contradictory attribution detected",
        summary=EventSummary(
            event_id=cluster.id,
            headline="Quarantined Story Headline",
            what_happened="This text must never appear in Telegram output.",
            key_details=EventKeyDetails(),
            confidence_qualification="Single-source report; unverified independently.",
            sources=(),
            model_used="test-model",
        ),
        event=cluster,
    )
    formatter = TelegramDigestFormatter()
    selected = formatter.select_digest_events(results=[quarantined_result])
    assert len(selected) == 0

    chunks = formatter.format_digest(results=[quarantined_result], briefing_date=dt.date())
    for chunk in chunks:
        assert "Quarantined Story Headline" not in chunk
        assert "This text must never appear in Telegram output." not in chunk


def test_publishers_never_extracted_as_event_actors():
    """Publishers/news outlets must never be extracted as event actors."""
    # 1. Direct validator test: publisher in actors is flagged as unsupported entity
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Al Jazeera English",
        title="Drone strike kills at least 21 and injures 40 in Nyala, Darfur",
        url="https://aljazeera.com/nyala",
        canonical_url="https://aljazeera.com/nyala",
        published_at=dt,
        snippet=(
            "A drone strike occurred on Monday evening at a market in Nyala city "
            "in the Darfur region, resulting in at least 21 fatalities and 40 injuries, "
            "according to a resident's report."
        ),
        content_hash="hash-nyala",
    )
    cluster = EventCluster(
        title="Drone strike kills at least 21 and injures 40 in Nyala, Darfur",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    summary_with_publisher = EventSummary(
        event_id=cluster.id,
        headline="Drone Strike Kills At Least 21 in Nyala Market",
        what_happened=(
            "A drone strike occurred on Monday evening at a market in Nyala city, "
            "resulting in at least 21 fatalities according to a resident's report."
        ),
        key_details=EventKeyDetails(
            actors=("Al Jazeera English",),
            location="Nyala city in the Darfur region",
            timing="Monday evening",
            action="drone strike",
        ),
        confidence_qualification="Single-source report from Al Jazeera English; unverified.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(summary_with_publisher, cluster)
    assert not val.is_valid
    assert "Al Jazeera English" in val.unsupported_entities

    # 2. sanitize_actors strips publisher names
    sanitized = sanitize_actors(
        ["Al Jazeera English", "Defense News", "Sudan Armed Forces"],
        source_names=["Al Jazeera English"],
    )
    assert "Al Jazeera English" not in sanitized
    assert "Defense News" not in sanitized
    assert "Sudan Armed Forces" in sanitized
    assert is_publisher_or_outlet("Al Jazeera English")
    assert not is_publisher_or_outlet("Sudan Armed Forces")

    # 3. Nyala report end-to-end: preserves attribution to reporting resident
    # without inventing actors
    mock_client = MockOllamaClient(
        canned_response=json.dumps(
            {
                "headline": "Drone strike kills at least 21 and injures 40 in Nyala, Darfur",
                "what_happened": (
                    "A drone strike occurred on Monday evening at a market in Nyala city "
                    "in the Darfur region, resulting in at least 21 fatalities and 40 injuries, "
                    "according to a resident's report."
                ),
                "key_details": {
                    "actors": ["Al Jazeera English"],
                    "location": "Nyala city in the Darfur region",
                    "timing": "Monday evening",
                    "action": "drone strike",
                },
                "confidence_qualification": "Single-source report; unverified independently.",
            }
        )
    )
    summarizer = EventSummarizer(client=mock_client)
    res = summarizer.summarize_event(cluster)
    assert res.status == "success"
    assert res.summary is not None
    # Publisher must be eliminated
    assert "Al Jazeera English" not in res.summary.key_details.actors
    # Attribution to reporting resident preserved
    assert res.summary.key_details.actors == ("Reporting resident",)
    # No additional actors invented
    assert "RSF" not in res.summary.key_details.actors
    assert "SAF" not in res.summary.key_details.actors


def test_dangling_subordinate_clause_detected_and_quarantined():
    """Detect broken dangling subordinate clauses and quarantine if not corrected."""
    # 1. detect_malformed_clauses catches broken subordinate clauses
    broken_text = (
        "The ceasefire in Oleshky is meant to allow aid deliveries and the evacuation "
        "of civilians, as residents who survived on weeds during the conflict."
    )
    issues = detect_malformed_clauses(broken_text)
    assert len(issues) > 0
    assert any("as residents who survived on weeds" in i for i in issues)

    # 2. Grammatically complete version is NOT flagged
    clean_text = (
        "The ceasefire in Oleshky is meant to allow aid deliveries and the evacuation "
        "of civilians, in a city where residents had resorted to eating weeds to survive."
    )
    clean_issues = detect_malformed_clauses(clean_text)
    assert len(clean_issues) == 0

    # 3. validate_summary_grounding marks broken text as invalid
    dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Kyiv Independent",
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        url="https://kyiv.ua/truce",
        canonical_url="https://kyiv.ua/truce",
        published_at=dt,
        snippet="Ceasefire in Oleshky secured for humanitarian aid deliveries.",
        content_hash="hash-truce",
    )
    cluster = EventCluster(
        title="Truce begins in Ukraine city where residents ate weeds to survive",
        category=TopicCategory.DIPLOMACY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    broken_summary = EventSummary(
        event_id=cluster.id,
        headline="Truce Begins in Ukraine City",
        what_happened=broken_text,
        key_details=EventKeyDetails(
            actors=("residents of Oleshky",),
            location="Oleshky",
            action="ceasefire",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        sources=(),
        model_used="test-model",
    )
    val = validate_summary_grounding(broken_summary, cluster)
    assert not val.is_valid
    assert any("Incomplete/dangling clause" in c for c in val.unsupported_claims)

    # 4. Summarizer retries and quarantines if model keeps emitting malformed text
    persistent_broken_client = MockOllamaClient(
        canned_response=json.dumps(
            {
                "headline": "Truce Begins in Ukraine City",
                "what_happened": (
                    "Ceasefire was declared in the area, as refugees who escaped from the zone."
                ),
                "key_details": {"actors": ["civilians"], "location": "Oleshky"},
                "confidence_qualification": "Single-source report; unverified independently.",
            }
        )
    )
    summarizer = EventSummarizer(client=persistent_broken_client)
    res = summarizer.summarize_event(cluster)
    assert res.status == "quarantined"
    assert "Incomplete/dangling clause detected" in str(res.quarantine_reason)


def test_corroboration_label_distinguishes_official_announcement_from_independent_sources():
    """Distinguish official announcements from independent sources in badges and assessments."""
    dt = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
    formatter = TelegramDigestFormatter()

    # Case 1: Official announcement repetition (France-UK exercise with MoD source)
    mod_src = EventSourceAttribution(
        source_name="UK Ministry of Defence",
        title="First joint airborne nuclear exercise conducted",
        url="https://mod.uk/announcement",
        published_at=dt,
    )
    news_src = EventSourceAttribution(
        source_name="Defense News",
        title="France and UK hold airborne nuclear exercise",
        url="https://defensenews.com/exercise",
        published_at=dt,
    )
    france_uk_summary = EventSummary(
        event_id="evt-france-uk",
        headline="France and UK Carry Out First Joint Airborne Nuclear-Strike Exercise",
        what_happened=(
            "France and the UK conducted their first joint airborne nuclear-strike exercise."
        ),
        key_details=EventKeyDetails(
            actors=("France", "UK"),
            location="France",
            timing="October 23, 2026",
            action="joint airborne nuclear exercise",
        ),
        confidence_qualification=(
            "Reported across multiple outlets based on official announcement; "
            "military details not independently verified."
        ),
        sources=(mod_src, news_src),
        model_used="test-model",
    )
    block_official = formatter.format_event_block(
        summary=france_uk_summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[OFFICIAL ANNOUNCEMENT]" in block_official
    assert "[CORROBORATED]" not in block_official
    assert (
        "Reported across multiple outlets based on official announcement; "
        "military details not independently verified." in block_official
    )

    # Case 2: Independent reporting from distinct sources without official announcement
    reuters_src = EventSourceAttribution(
        source_name="Reuters",
        title="Naval units track vessel in international waters",
        url="https://reuters.com/naval",
        published_at=dt,
    )
    ap_src = EventSourceAttribution(
        source_name="Associated Press",
        title="Vessel tracked across disputed sea lane",
        url="https://apnews.com/vessel",
        published_at=dt,
    )
    independent_summary = EventSummary(
        event_id="evt-independent",
        headline="Naval Forces Track Vessel Across Disputed Sea Lane",
        what_happened="Naval forces monitored vessel movements across international waters.",
        key_details=EventKeyDetails(
            actors=("Naval forces",),
            location="South China Sea",
            action="tracking vessel",
        ),
        confidence_qualification="Corroborated across multiple independent sources.",
        sources=(reuters_src, ap_src),
        model_used="test-model",
    )
    block_independent = formatter.format_event_block(
        summary=independent_summary,
        category=TopicCategory.MILITARY_OPERATIONS,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[CORROBORATED]" in block_independent
    assert "[OFFICIAL ANNOUNCEMENT]" not in block_independent
    assert (
        "Independent reporting from distinct sources; "
        "operational details not independently verified." in block_independent
    )

    # Case 3: Independently verified operational details
    verified_summary = EventSummary(
        event_id="evt-verified",
        headline="Verified Airfield Strike Damages Radar Installation",
        what_happened="Commercial satellite imagery confirms extensive damage to airfield radar.",
        key_details=EventKeyDetails(
            actors=("Air force",),
            location="Airfield base",
            action="radar strike",
        ),
        confidence_qualification="Independently verified operational details.",
        sources=(reuters_src, ap_src),
        model_used="test-model",
    )
    block_verified = formatter.format_event_block(
        summary=verified_summary,
        category=TopicCategory.ARMED_CONFLICT,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[CORROBORATED]" in block_verified
    assert "Independently verified operational details." in block_verified


# ---------------------------------------------------------------------------
# Regressions: 28 October Briefing Failure Patterns
# ---------------------------------------------------------------------------


def test_tensions_rise_actor_location_mismatch_quarantined():
    """Verify that an intro topic list cannot assign China as actor or location."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="Tensions Rise",
        url="https://warontherocks.com/tensions-rise/",
        canonical_url="https://warontherocks.com/tensions-rise/",
        published_at=dt,
        snippet=(
            "Welcome to The Adversarial. Every other week, we'll provide you with expert analysis "
            "on America's greatest challengers: China, Russia, Iran, North Korea, and jihadists. "
            "Read more below.***Iran Indications of a return to direct hostilities are increasing. "
            "The United States is evacuating more troops and diplomatic staff and their families "
            "from the region, moving a third aircraft carrier strike group to the Middle East, "
            "and reportedly preparing strike options. Ahead of Israeli elections on Oct. 27, "
            "Israeli Prime Minister Benjamin Netanyahu has been campaigning on toppling the "
            "Iranian regime. Diplomatic efforts have stalled."
        ),
        content_hash="hash-tensions-rise-synthetic-test",
    )
    cluster = EventCluster(
        title="Tensions Rise",
        category="military_operations",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    src_attr = EventSourceAttribution(
        source_name="War on the Rocks",
        title="Tensions Rise",
        url="https://warontherocks.com/tensions-rise/",
        published_at=dt,
    )

    # 1. Erroneous summary identifying China as actor and location
    mismatched_summary = EventSummary(
        event_id="evt-mismatch",
        headline="US Deploys Third Carrier Strike Group to Middle East Amid Tensions with Iran",
        what_happened=(
            "The United States is evacuating more troops and diplomatic staff from the region "
            "and preparing strike options amid increasing indications of a return to direct "
            "hostilities with Iran."
        ),
        key_details=EventKeyDetails(
            actors=("China",),
            location="China",
            action="Deploying strike group",
        ),
        confidence_qualification=(
            "Single-source report from War on the Rocks; unverified independently."
        ),
        sources=(src_attr,),
        model_used="test-model",
    )
    res_mismatch = validate_summary_grounding(mismatched_summary, cluster)
    assert not res_mismatch.is_valid
    assert res_mismatch.quarantine_reason is not None
    assert (
        "Actor mismatch" in res_mismatch.quarantine_reason
        or "Location" in res_mismatch.quarantine_reason
    )

    # 2. Correctly grounded summary with US/Iran actors and Middle East location
    correct_summary = EventSummary(
        event_id="evt-correct",
        headline="US Deploys Third Carrier Strike Group to Middle East Amid Tensions with Iran",
        what_happened=(
            "The United States is evacuating troops and moving a third carrier strike group "
            "to the Middle East amid increasing indications of direct hostilities with Iran."
        ),
        key_details=EventKeyDetails(
            actors=("United States", "Iran"),
            location="Middle East",
            action="Deploying strike group",
        ),
        confidence_qualification=(
            "Single-source report from War on the Rocks; unverified independently."
        ),
        sources=(src_attr,),
        model_used="test-model",
    )
    res_correct = validate_summary_grounding(correct_summary, cluster)
    assert res_correct.is_valid


def test_nuclear_weapons_upgrade_unsupported_announcement_quarantined():
    """Verify that an analytical roundtable cannot be turned into an official upgrade plan."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="What's haunting the Pentagon this Halloween?",
        url="https://breakingdefense.com/2026/10/whats-haunting-the-pentagon-this-halloween/",
        canonical_url=(
            "https://breakingdefense.com/2026/10/whats-haunting-the-pentagon-this-halloween/"
        ),
        published_at=dt,
        snippet=(
            "Three special guests, and Mark, share a few frightening mysteries from their beats — "
            "from the dangers of war in space to the intricacies of upgrading America's nuclear "
            "weapons."
        ),
        content_hash="hash-halloween-synthetic-test",
    )
    cluster = EventCluster(
        title="What's haunting the Pentagon this Halloween?",
        category="defence_policy",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    src_attr = EventSourceAttribution(
        source_name="Breaking Defense",
        title="What's haunting the Pentagon this Halloween?",
        url="https://breakingdefense.com/2026/10/whats-haunting-the-pentagon-this-halloween/",
        published_at=dt,
    )

    # Erroneous summary claiming Pentagon announces plans to upgrade nuclear weapons
    hallucinated_announcement = EventSummary(
        event_id="evt-hallucinated-nuclear",
        headline="Pentagon Announces Plans to Upgrade Nuclear Weapons",
        what_happened=(
            "The Pentagon announces plans to upgrade America's nuclear weapons and address "
            "the dangers of war in space."
        ),
        key_details=EventKeyDetails(
            actors=("Pentagon",),
            location=None,
            action="Nuclear weapons upgrade plans",
        ),
        confidence_qualification=(
            "Single-source report from Breaking Defense; unverified independently."
        ),
        sources=(src_attr,),
        model_used="test-model",
    )
    res = validate_summary_grounding(hallucinated_announcement, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert "Nuclear weapons upgrade plan or announcement asserted" in res.quarantine_reason


def test_clean_headline_grammar_restores_dropped_equipment_prefix():
    """Verify that clean_headline_grammar restores dropped military alphanumeric prefixes."""
    from geopulse.summarization.validator import clean_headline_grammar

    # 1. F-47 with possessive: 47's -> F-47's
    source_f47 = "US Air Force to Study F-47 Performance in Comparison with F-15, F-22"
    awkward_f47 = "US Air Force to Study 47’s Performance in F-15, F-22 Comparison"
    restored_f47 = clean_headline_grammar(awkward_f47, source_title=source_f47)
    assert restored_f47 == "US Air Force to Study F-47’s Performance in F-15, F-22 Comparison"
    assert " 47’s" not in restored_f47
    assert "F-47’s" in restored_f47

    # 2. Su-57 bare designation: 57 -> Su-57
    source_su57 = "Russia deploys Su-57 fighters along border sector"
    awkward_su57 = "Russia Deploys 57 in Border Patrols"
    restored_su57 = clean_headline_grammar(awkward_su57, source_title=source_su57)
    assert restored_su57 == "Russia Deploys Su-57 in Border Patrols"

    # 3. B-21 with possessive: 21's -> B-21's
    source_b21 = "Air Force tests B-21 Raider stealth bomber"
    awkward_b21 = "Air Force Tests 21's Avionics in High-Altitude Flights"
    restored_b21 = clean_headline_grammar(awkward_b21, source_title=source_b21)
    assert restored_b21 == "Air Force Tests B-21's Avionics in High-Altitude Flights"


def test_invented_official_silence_quarantined():
    """Summary claiming official silence without source evidence must be quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        url="https://dn.com/space-1",
        canonical_url="https://dn.com/space-1",
        published_at=now,
        snippet=(
            "Washington blamed Moscow for deliberately moving a military satellite "
            "close to a US satellite."
        ),
        content_hash="hash-space-1",
    )
    cluster = EventCluster(
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-silence",
        headline="Washington blames Moscow as Russian satellite tails US spacecraft",
        what_happened=(
            "US officials accused Russia of tailing a satellite, but Russian officials "
            "remained silent."
        ),
        key_details=EventKeyDetails(
            actors=("US officials", "Russian military"),
            location=None,
            action="Satellite proximity operation",
            timing="Last week",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert "Claim of official silence or absence of comment" in res.quarantine_reason


def test_invented_western_concern_quarantined():
    """Summary claiming Western concern or alarm without source evidence must be quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        url="https://dn.com/space-2",
        canonical_url="https://dn.com/space-2",
        published_at=now,
        snippet=(
            "Washington blamed Moscow for deliberately moving a military satellite "
            "close to a US satellite."
        ),
        content_hash="hash-space-2",
    )
    cluster = EventCluster(
        title="Space wars heat up as Russia tails a Ukraine-linked US satellite",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-concern",
        headline="Washington blames Moscow as Russian satellite tails US spacecraft",
        what_happened=(
            "Washington accused Moscow of space maneuvers close to a US satellite, "
            "sparking widespread Western concern and international alarm."
        ),
        key_details=EventKeyDetails(
            actors=("US officials", "Russian military"),
            location=None,
            action="Satellite proximity operation",
            timing="Last week",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert "Claim of Western or international concern" in res.quarantine_reason


def test_administrative_guidance_without_breaking_event_insufficient_evidence():
    """Administrative JSP guidance without breaking news is marked insufficient evidence."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="Guidance: Defence Safety Management System (JSP 815)",
        url="https://gov.uk/jsp815",
        canonical_url="https://gov.uk/jsp815",
        published_at=now,
        snippet=(
            "Joint Service Publication 815 provides the Defence Safety Management System "
            "framework, enabling Defence organisations to develop tailored safety management."
        ),
        content_hash="hash-jsp-1",
    )
    cluster = EventCluster(
        title="Guidance: Defence Safety Management System (JSP 815)",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    is_sufficient, reason = is_evidence_sufficient(cluster)
    assert not is_sufficient
    assert reason is not None
    assert "administrative guidance" in reason.lower()


def test_tautological_summary_repeating_headline_quarantined():
    """Summary whose narrative simply echoes the headline is quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="European Allies Finalize Missile Production Accord",
        url="https://dn.com/accord",
        canonical_url="https://dn.com/accord",
        published_at=now,
        snippet="European defense allies agreed on common missile production terms.",
        content_hash="hash-accord",
    )
    cluster = EventCluster(
        title="European Allies Finalize Missile Production Accord",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-tautology",
        headline="European Allies Finalize Missile Production Accord",
        what_happened="European Allies Finalize Missile Production Accord",
        key_details=EventKeyDetails(
            actors=("European Allies",),
            location=None,
            action="Missile production agreement",
            timing="October 2026",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert "repeats headline" in res.quarantine_reason


def test_unwarranted_system_unveil_hallucination_quarantined():
    """Summary claiming an aircraft/system was unveiled without source support is quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="How the F-15 and F-22 should inform the F-47’s procurement",
        url="https://dn.com/f47-lessons",
        canonical_url="https://dn.com/f47-lessons",
        published_at=now,
        snippet=(
            "Budgetary lessons from the two legacy aircraft can help the Air Force "
            "anticipate hard choices for the new F-47, one analyst argues."
        ),
        content_hash="hash-f47",
    )
    cluster = EventCluster(
        title="How the F-15 and F-22 should inform the F-47’s procurement",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    # Hallucinated summary asserting unveiling
    summary = EventSummary(
        event_id="evt-f47-hallucination",
        headline="US Air Force Unveils F-47 Fighter Jet, Emphasizing Upgrade Over New Technology",
        what_happened="The US Air Force has unveiled its new F-47 fighter jet.",
        key_details=EventKeyDetails(
            actors=("US Air Force",),
            location=None,
            action="Unveiling of F-47 fighter jet",
            timing="October 2026",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert any(
        "unveiling" in c.lower() or "analytical commentary" in c.lower()
        for c in res.unsupported_claims
    )


def test_analytical_commentary_misrepresented_as_official_decision_quarantined():
    """Commentary or analyst argument falsely summarized as official order is quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Lessons for sixth-generation fighter requirements",
        url="https://dn.com/sixth-gen",
        canonical_url="https://dn.com/sixth-gen",
        published_at=now,
        snippet="One analyst argues that cost overruns must be avoided.",
        content_hash="hash-analyst",
    )
    cluster = EventCluster(
        title="Lessons for sixth-generation fighter requirements",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-commentary-mismatch",
        headline="Pentagon Ordered Sixth-Generation Fighter Architecture",
        what_happened="The military officially announced and ordered new airframes.",
        key_details=EventKeyDetails(
            actors=("Pentagon",),
            location=None,
            action="Fighter order",
            timing="October 2026",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert any("analytical commentary" in c.lower() for c in res.unsupported_claims)


def test_unattributed_negative_breach_claim_quarantined():
    """Summary stating no breaches occurred without attribution to authority is quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Kherson region ceasefire continues into third day",
        url="https://reuters.com/kherson-ceasefire",
        canonical_url="https://reuters.com/kherson-ceasefire",
        published_at=now,
        snippet="Local authorities reported calm across the frontline.",
        content_hash="hash-kherson",
    )
    cluster = EventCluster(
        title="Kherson region ceasefire continues into third day",
        category=TopicCategory.DIPLOMACY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-unattributed-negative",
        headline="Kherson Region Ceasefire Continues",
        what_happened="The ceasefire held across the region. There have been no breaches.",
        key_details=EventKeyDetails(
            actors=("Kherson authorities",),
            location="Kherson",
            action="Ceasefire compliance",
            timing="October 2026",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert any("negative assertion" in c.lower() for c in res.unsupported_claims)


def test_single_source_event_claiming_corroboration_in_narrative_quarantined():
    """Single-source cluster claiming corroboration in what_happened is quarantined."""
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Local News",
        title="Patrol vessel visits island outpost",
        url="https://local.com/patrol",
        canonical_url="https://local.com/patrol",
        published_at=now,
        snippet="A patrol vessel arrived at the outpost.",
        content_hash="hash-patrol",
    )
    cluster = EventCluster(
        title="Patrol vessel visits island outpost",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art,),
    )
    summary = EventSummary(
        event_id="evt-false-corrob",
        headline="Patrol Vessel Arrives at Island Outpost",
        what_happened=(
            "A patrol vessel docked at the island, corroborated across multiple independent "
            "sources."
        ),
        key_details=EventKeyDetails(
            actors=("Patrol vessel",),
            location=None,
            action="Patrol visit",
            timing="October 2026",
        ),
        confidence_qualification="Single-source reporting; unverified independently.",
        model_used="test-model",
    )
    res = validate_summary_grounding(summary, cluster)
    assert not res.is_valid
    assert res.quarantine_reason is not None
    assert any("multi-source corroboration" in c.lower() for c in res.unsupported_claims)


def test_exact_cross_theatre_conflation_middle_east_and_south_china_sea():
    """Conflating US Middle East carrier deployment with China in South China Sea is quarantined."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="Tensions Rise",
        url="https://warontherocks.com/tensions-rise/",
        canonical_url="https://warontherocks.com/tensions-rise/",
        published_at=dt,
        snippet=(
            "Welcome to The Adversarial. Every other week, we’ll provide you with expert "
            "analysis on America’s greatest challengers: China, Russia, Iran, North Korea, and "
            "jihadists. Read more below.***Iran Indications of a return to direct hostilities "
            "are increasing. The United States is evacuating more troops and diplomatic staff "
            "and their families from the region, moving a third aircraft carrier strike group "
            "to the Middle East, and reportedly preparing strike options. Ahead of Israeli "
            "elections on Oct. 27, Israeli Prime Minister Benjamin Netanyahu has been "
            "campaigning on toppling the Iranian regime. Diplomatic efforts have stalled."
        ),
        content_hash="hash-tensions-rise-exact",
    )
    cluster = EventCluster(
        title="Tensions Rise",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # The exact hallucination observed in the Oct 28 preview:
    conflated_summary = EventSummary(
        event_id=cluster.id,
        headline="China Increases Military Presence in the South China Sea",
        what_happened=(
            "China has deployed a new aircraft carrier to the South China Sea, amid increased "
            "tensions with neighboring countries. The move comes as the US evacuates troops "
            "and diplomatic staff from the region."
        ),
        key_details=EventKeyDetails(
            actors=("China", "United States"),
            location="South China Sea",
            action="deployment of a new aircraft carrier",
        ),
        confidence_qualification=(
            "Single-source report from War on the Rocks; unverified independently."
        ),
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(conflated_summary, cluster)
    assert not val.is_valid
    assert val.quarantine_reason is not None
    assert (
        "South China Sea" in val.quarantine_reason
        or "Cross-theatre conflation" in val.quarantine_reason
    )


def test_valid_china_south_china_sea_only_passes():
    """A genuine report focused exclusively on China and the South China Sea is accepted."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Naval News",
        title="Chinese aircraft carrier conducts operations in South China Sea",
        url="https://navalnews.com/scs-carrier",
        canonical_url="https://navalnews.com/scs-carrier",
        published_at=dt,
        snippet=(
            "The Chinese aircraft carrier Shandong conducted joint maritime flight drills "
            "in the northern South China Sea alongside guided-missile destroyers this week."
        ),
        content_hash="hash-scs-valid",
    )
    cluster = EventCluster(
        title="Chinese aircraft carrier conducts operations in South China Sea",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    valid_scs_summary = EventSummary(
        event_id=cluster.id,
        headline="Chinese Aircraft Carrier Conducts Flight Drills in South China Sea",
        what_happened=(
            "The Chinese aircraft carrier Shandong conducted joint maritime flight operations "
            "in the South China Sea alongside guided-missile destroyers."
        ),
        key_details=EventKeyDetails(
            actors=("Chinese aircraft carrier Shandong",),
            location="South China Sea",
            action="Maritime flight drills",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report from Naval News; unverified independently.",
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(valid_scs_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_valid_middle_east_carrier_report_passes():
    """A genuine report on US carrier movements in the Middle East is accepted."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="USNI News",
        title="US moves third aircraft carrier strike group into Middle East",
        url="https://usni.org/carrier-me",
        canonical_url="https://usni.org/carrier-me",
        published_at=dt,
        snippet=(
            "The United States is moving a third aircraft carrier strike group into the Middle "
            "East and evacuating non-essential personnel amid rising tensions with Iran."
        ),
        content_hash="hash-me-valid",
    )
    cluster = EventCluster(
        title="US moves third aircraft carrier strike group into Middle East",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    valid_me_summary = EventSummary(
        event_id=cluster.id,
        headline="United States Deploys Third Carrier Strike Group to Middle East",
        what_happened=(
            "The United States is relocating a third aircraft carrier strike group into the "
            "Middle East and evacuating non-essential personnel amid heightened regional tensions."
        ),
        key_details=EventKeyDetails(
            actors=("United States",),
            location="Middle East",
            action="Carrier deployment and staff evacuation",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report from USNI News; unverified independently.",
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(valid_me_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_legitimate_multi_theatre_article_with_separate_attributions_passes():
    """An article covering multiple theatres passes when actions remain tied to their passages."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Global military movements: Red Sea patrols and South China Sea drills",
        url="https://reuters.com/global-movements",
        canonical_url="https://reuters.com/global-movements",
        published_at=dt,
        snippet=(
            "In the Middle East, US Navy warships conducted defensive patrols across the Red Sea. "
            "Separately, in the South China Sea, Philippine and allied vessels conducted "
            "maritime exercises."
        ),
        content_hash="hash-multi-theatre",
    )
    cluster = EventCluster(
        title="Global military movements: Red Sea patrols and South China Sea drills",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    valid_multi_summary = EventSummary(
        event_id=cluster.id,
        headline="US Warships Patrol Red Sea as Philippine Vessels Exercise in South China Sea",
        what_happened=(
            "US Navy warships conducted defensive security patrols in the Red Sea, "
            "while Philippine and allied vessels held separate maritime exercises in the "
            "South China Sea."
        ),
        key_details=EventKeyDetails(
            actors=("US Navy", "Philippine and allied vessels"),
            location=None,
            action="Naval patrols and exercises",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report from Reuters; unverified independently.",
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(valid_multi_summary, cluster)
    assert val.is_valid


def test_empty_and_headline_repeating_summaries_quarantined():
    """Summaries with empty/trivial narrative or repeating headline are quarantined."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Special operations lessons for defence acquisition",
        url="https://defensenews.com/specops",
        canonical_url="https://defensenews.com/specops",
        published_at=dt,
        snippet="Acquisition officials discussed procurement strategies.",
        content_hash="hash-specops",
    )
    cluster = EventCluster(
        title="Special operations lessons for defence acquisition",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # 1. Headline-repeating summary
    headline_repeat_summary = EventSummary(
        event_id=cluster.id,
        headline="Special Operations Lessons for Defence Acquisition",
        what_happened="Special Operations Lessons for Defence Acquisition",
        key_details=EventKeyDetails(actors=("Acquisition officials",)),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val_repeat = validate_summary_grounding(headline_repeat_summary, cluster)
    assert not val_repeat.is_valid
    assert "repeats headline" in (val_repeat.quarantine_reason or "").lower()

    # 2. Empty / trivial narrative
    empty_summary = EventSummary(
        event_id=cluster.id,
        headline="Special Operations Lessons for Defence Acquisition",
        what_happened="No details.",
        key_details=EventKeyDetails(actors=()),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val_empty = validate_summary_grounding(empty_summary, cluster)
    assert not val_empty.is_valid
    assert "insufficient narrative detail" in (val_empty.quarantine_reason or "").lower()

    # 3. Completely empty narrative (constructed via model_construct)
    empty_construct = EventSummary.model_construct(
        event_id=cluster.id,
        headline="Special Operations Lessons for Defence Acquisition",
        what_happened="",
        key_details=EventKeyDetails(actors=()),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val_empty_construct = validate_summary_grounding(empty_construct, cluster)
    assert not val_empty_construct.is_valid
    assert "insufficient narrative detail" in (val_empty_construct.quarantine_reason or "").lower()


def test_unsupported_actors_actions_dates_locations_quarantined():
    """Summaries asserting unsupported actors, locations, dates, or actions are quarantined."""
    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Border guards patrol northern checkpoint",
        url="https://reuters.com/checkpoint",
        canonical_url="https://reuters.com/checkpoint",
        published_at=dt,
        snippet=(
            "Local border guards conducted routine vehicle inspections at northern checkpoint."
        ),
        content_hash="hash-checkpoint",
    )
    cluster = EventCluster(
        title="Border guards patrol northern checkpoint",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Unsupported actor, location, and action
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Australian Special Forces Launch Submarine Strike in Antarctica",
        what_happened=(
            "Australian Special Forces launched a submarine missile strike against "
            "bases in Antarctica."
        ),
        key_details=EventKeyDetails(
            actors=("Australian Special Forces",),
            location="Antarctica",
            action="Submarine missile strike",
            timing="December 2038",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert val.quarantine_reason is not None


def test_retry_exhaustion_leads_to_quarantine_not_delivery():
    """When an LLM repeatedly returns unsupported entities across retries, it is quarantined."""
    from geopulse.summarization import EventSummarizer

    dt = datetime(2026, 10, 28, 6, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Logistics convoy delivers winter supplies",
        url="https://reuters.com/convoy",
        canonical_url="https://reuters.com/convoy",
        published_at=dt,
        snippet="A supply convoy delivered winter coats and heating units to base personnel.",
        content_hash="hash-convoy",
    )
    cluster = EventCluster(
        title="Logistics convoy delivers winter supplies",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )

    hallucinated_json = json.dumps(
        {
            "headline": "Martian Defense Forces Deploy Hypersonic Railguns",
            "what_happened": (
                "Martian Defense Forces deployed orbital railguns to defend the logistics convoy."
            ),
            "key_details": {
                "actors": ["Martian Defense Forces"],
                "location": "Mars Base Alpha",
                "timing": "October 2026",
                "action": "Railgun deployment",
            },
            "confidence_qualification": "Single-source report; unverified independently.",
        }
    )

    class StubbornHallucinatingClient(MockOllamaClient):
        def generate(
            self,
            prompt: str,
            system: str | None = None,
            format_type: str | None = None,
        ) -> str:
            return hallucinated_json

    client = StubbornHallucinatingClient()
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "quarantined"
    assert res.quarantine_reason is not None
    assert "Martian Defense Forces" in res.quarantine_reason

    # Formatter must exclude it from Telegram digest chunks
    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest([res], edition="morning")
    full_text = "\n".join(chunks)
    assert "Martian Defense Forces" not in full_text
    assert "Hypersonic Railguns" not in full_text


def test_wotr_dod_oversight_rejects_unsupported_executive_orders():
    """Event #6 failure pattern: invented executive orders are quarantined."""
    dt = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="The Next Congress Needs to Reclaim Its Oversight of the Department of Defense",
        url=(
            "https://warontherocks.com/the-next-congress-needs-to-reclaim-its-oversight"
            "-of-the-department-of-defense/"
        ),
        canonical_url=(
            "https://warontherocks.com/the-next-congress-needs-to-reclaim-its-oversight"
            "-of-the-department-of-defense/"
        ),
        published_at=dt,
        snippet=(
            "Since returning to office in January 2025, the Trump administration has aggressively "
            "exercised executive authority across the federal government. Meanwhile, Congress has "
            "not adequately applied its oversight role to rein in the executive branch, "
            "particularly over the Department of Defense. We have watched these dynamics "
            "with specific interest, in discussions with congressional staff over bills and "
            "appropriations, and in conversations with our clients. Congress adjourned at the "
            "end of September until after the midterm elections following an unproductive summer "
            "legislative session, particularly on foreign policy. While Congress passed a "
            "long-debated Russia sanctions bill in September, President Donald Trump has already "
            "issued a waiver to it in Moscow's favor."
        ),
        content_hash="hash-wotr-oversight",
    )
    cluster = EventCluster(
        title="The Next Congress Needs to Reclaim Its Oversight of the Department of Defense",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # The actual failure pattern: summary claiming Trump issued executive orders
    hallucinated_summary = EventSummary(
        event_id=cluster.id,
        headline="Congress Passes Russia Sanctions Bill as Trump Issues Executive Orders",
        what_happened=(
            "Congress passed a Russia sanctions bill in September while President Donald Trump "
            "issued executive orders overriding legislative provisions."
        ),
        key_details=EventKeyDetails(
            actors=("US Congress", "Donald Trump"),
            location=None,
            action="Issuing executive orders on Russia sanctions",
            timing="September 2026",
        ),
        confidence_qualification=(
            "Single-source report from War on the Rocks; unverified independently."
        ),
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(hallucinated_summary, cluster)
    assert not val.is_valid
    assert val.quarantine_reason is not None
    assert any("executive order" in c.lower() for c in val.unsupported_claims)


def test_wotr_dod_oversight_grounded_summary_passes():
    """Grounded analysis summary of DoD oversight essay passes validation."""
    dt = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="The Next Congress Needs to Reclaim Its Oversight of the Department of Defense",
        url=(
            "https://warontherocks.com/the-next-congress-needs-to-reclaim-its-oversight"
            "-of-the-department-of-defense/"
        ),
        canonical_url=(
            "https://warontherocks.com/the-next-congress-needs-to-reclaim-its-oversight"
            "-of-the-department-of-defense/"
        ),
        published_at=dt,
        snippet=(
            "Since returning to office in January 2025, the Trump administration has aggressively "
            "exercised executive authority across the federal government. Meanwhile, Congress has "
            "not adequately applied its oversight role to rein in the executive branch, "
            "particularly over the Department of Defense."
        ),
        content_hash="hash-wotr-oversight-valid",
    )
    cluster = EventCluster(
        title="The Next Congress Needs to Reclaim Its Oversight of the Department of Defense",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    grounded_summary = EventSummary(
        event_id=cluster.id,
        headline="Congress Urged to Reassert Oversight Role Over Department of Defense",
        what_happened=(
            "An analytical assessment argues that Congress has not adequately applied its "
            "constitutional oversight role to rein in executive authority over the Department "
            "of Defense."
        ),
        key_details=EventKeyDetails(
            actors=("US Congress", "Department of Defense"),
            location=None,
            action="Congressional defense oversight",
            timing="October 2026",
        ),
        confidence_qualification=(
            "Single-source report from War on the Rocks; unverified independently."
        ),
        model_used="llama3.2:3b",
    )
    val = validate_summary_grounding(grounded_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_unsupported_price_and_currency_claims_quarantined():
    """Summaries asserting ungrounded currency and contract values are quarantined."""
    dt = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Navy explores autonomous swarm surveillance",
        url="https://dn.com/swarm",
        canonical_url="https://dn.com/swarm",
        published_at=dt,
        snippet="Naval commanders met to evaluate maritime autonomous capabilities.",
        content_hash="hash-swarm",
    )
    cluster = EventCluster(
        title="Navy explores autonomous swarm surveillance",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Navy Awards $500 Million Contract for Autonomous Swarm Systems",
        what_happened=(
            "The Navy awarded a $500 million contract to develop autonomous swarm vessels."
        ),
        key_details=EventKeyDetails(
            actors=("US Navy",),
            action="Contract award",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any("price" in c.lower() or "$500 million" in c for c in val.unsupported_claims)


def test_unsupported_technical_specification_quarantined():
    """Summaries asserting ungrounded technical specifications are quarantined."""
    dt = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Interceptor test demonstrates high-altitude defense",
        url="https://dn.com/test",
        canonical_url="https://dn.com/test",
        published_at=dt,
        snippet="Engineers completed a test of the interceptor system at the proving grounds.",
        content_hash="hash-test-spec",
    )
    cluster = EventCluster(
        title="Interceptor test demonstrates high-altitude defense",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Interceptor Reaches Mach 5 in Proving Ground Trials",
        what_happened=(
            "The newly tested interceptor reached speeds exceeding Mach 5 during the flight test."
        ),
        key_details=EventKeyDetails(
            actors=("Engineers",),
            action="Interceptor test",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any(
        "technical specification" in c.lower() or "mach 5" in c.lower()
        for c in val.unsupported_claims
    )


def test_unsupported_strategic_intent_claims_quarantined():
    """Summaries claiming unverified strategic intent or motive are quarantined."""
    dt = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="Regional naval units deploy to western approaches",
        url="https://reuters.com/naval-units",
        canonical_url="https://reuters.com/naval-units",
        published_at=dt,
        snippet="Naval patrol vessels departed port to monitor coastal shipping lanes.",
        content_hash="hash-naval-intent",
    )
    cluster = EventCluster(
        title="Regional naval units deploy to western approaches",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="Naval Deployment Intended to Provoke Neighboring Garrison",
        what_happened=(
            "Naval patrol vessels deployed to shipping lanes, intended to provoke "
            "neighboring coastal garrisons."
        ),
        key_details=EventKeyDetails(
            actors=("Naval patrol vessels",),
            action="Naval deployment",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert any("strategic intent" in c.lower() for c in val.unsupported_claims)


def test_actor_acronym_deduplication_usfk():
    """Actor list deduplicates redundant acronyms when expanded name is present."""
    from geopulse.summarization.validator import sanitize_actors

    raw_actors = ["US Forces Korea", "USFK", "Department of Defense", "DoD", "US Congress"]
    cleaned = sanitize_actors(raw_actors)
    assert "US Forces Korea" in cleaned
    assert "USFK" not in cleaned
    assert "Department of Defense" in cleaned
    assert "DoD" not in cleaned
    assert "US Congress" in cleaned


def test_footer_accurate_event_count_when_below_cap():
    """Digest footer accurately distinguishes briefings below cap vs briefings reaching cap."""
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization.models import EventSourceAttribution, SummarizationResult

    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)
    dummy_results = [
        SummarizationResult(
            event_id=f"ev-{i}",
            status="success",
            summary=EventSummary(
                event_id=f"ev-{i}",
                headline=f"Operational update #{i}",
                what_happened=f"Operational units completed scheduled security patrol #{i}.",
                key_details=EventKeyDetails(actors=("Security patrol",)),
                sources=(
                    EventSourceAttribution(
                        source_name=f"Outlet {i}",
                        title=f"Title {i}",
                        url=f"https://example.com/{i}",
                        published_at=fixed_dt,
                    ),
                ),
                confidence_qualification="Single-source report.",
                model_used="test-model",
            ),
            event=EventCluster(
                id=f"ev-{i}",
                title=f"Operational update #{i}",
                category=TopicCategory.MILITARY_OPERATIONS,
                first_seen_at=fixed_dt,
                last_seen_at=fixed_dt,
                confidence_label="single_source",
                articles=(),
            ),
        )
        for i in range(1, 7)
    ]
    # Add 4 omitted candidate results
    omitted_results = [
        SummarizationResult(
            event_id=f"ev-omit-{i}",
            status="success",
            summary=EventSummary(
                event_id=f"ev-omit-{i}",
                headline=f"Deferred memo #{i}",
                what_happened=f"Deferred memo #{i}",
                confidence_qualification="single_source",
                model_used="omitted",
            ),
            event=EventCluster(
                id=f"ev-omit-{i}",
                title=f"Deferred memo #{i}",
                category=TopicCategory.DEFENCE_POLICY,
                first_seen_at=fixed_dt,
                last_seen_at=fixed_dt,
                confidence_label="single_source",
                articles=(),
            ),
        )
        for i in range(1, 5)
    ]
    all_results = dummy_results + omitted_results

    formatter = TelegramDigestFormatter()
    # 1. When fewer events qualify than max_events (6 < 10)
    chunks_below = formatter.format_digest(all_results, max_events=10, edition="morning")
    full_below = "\n".join(chunks_below)
    assert "6 event(s) briefed (configured limit: 10)" in full_below
    assert "capped at top 10 events" not in full_below
    assert "4 lower-priority event(s) omitted" in full_below

    # 2. When events reach max_events (6 >= 6)
    chunks_capped = formatter.format_digest(all_results, max_events=6, edition="morning")
    full_capped = "\n".join(chunks_capped)
    assert "Briefing capped at top 6 events" in full_capped
    assert "4 lower-priority event(s) omitted" in full_capped


def test_russian_diesel_summary_with_lng_conflation_and_unsupported_volume_quarantined():
    """Summary conflating Russian diesel with LNG and invented volume is quarantined."""
    dt = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Trump’s Russian diesel deal benefits Moscow more than US consumers, experts say",
        url="https://www.defensenews.com/global/europe/2026/10/28/trumps-russian-diesel-deal",
        canonical_url=(
            "https://www.defensenews.com/global/europe/2026/10/28/trumps-russian-diesel-deal"
        ),
        published_at=dt,
        snippet=(
            "American diesel prices rose as Russia's seaborne crude exports surged. "
            "President Trump moved to ease restrictions on Russian diesel imports to the US, "
            "drawing criticism that the deal puts oil money in Russian hands."
        ),
        content_hash="hash-diesel-test",
    )
    cluster = EventCluster(
        title="Trump’s Russian diesel deal benefits Moscow more than US consumers, experts say",
        category=TopicCategory.SANCTIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Hallucinated summary: LNG commodity mismatch, 1.7 bcm, two decades, US selling to Russia
    hallucinated_summary = EventSummary(
        event_id=cluster.id,
        headline="US Agrees to Sell Russia 1.7 Billion Cubic Metres of LNG Over Two Decades",
        what_happened=(
            "The US agreed to sell Russia 1.7 billion cubic metres of LNG over two decades "
            "as part of a major energy agreement."
        ),
        key_details=EventKeyDetails(
            actors=("US", "Russia"),
            location=None,
            action="agreed to sell LNG to Russia",
            timing="October 2026",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(hallucinated_summary, cluster)
    assert not val.is_valid
    assert val.quarantine_reason is not None
    # Verify commodity mismatch detected
    assert any("lng" in c.lower() for c in val.unsupported_claims)
    # Verify ungrounded volume / decimal quantity detected
    assert any("1.7" in c or "volume" in c.lower() for c in val.unsupported_claims)
    # Verify ungrounded timeframe / duration detected
    assert any("decade" in c.lower() for c in val.unsupported_claims)
    # Verify inverted trade transaction detected
    assert any(
        "inverted transaction" in c.lower() or "selling" in c.lower()
        for c in val.unsupported_claims
    )


def test_russian_diesel_grounded_summary_passes():
    """Factual summary of Russian diesel import deal passes validation."""
    dt = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Trump’s Russian diesel deal benefits Moscow more than US consumers, experts say",
        url="https://www.defensenews.com/global/europe/2026/10/28/trumps-russian-diesel-deal",
        canonical_url=(
            "https://www.defensenews.com/global/europe/2026/10/28/trumps-russian-diesel-deal"
        ),
        published_at=dt,
        snippet=(
            "American diesel prices rose as Russia's exports reached new levels. "
            "President Trump moved to ease restrictions on Russian diesel imports to the US, "
            "drawing criticism from Ukraine that the deal funds Moscow while Russian forces hit "
            "Ukrainian energy infrastructure."
        ),
        content_hash="hash-diesel-pass",
    )
    cluster = EventCluster(
        title="Trump’s Russian diesel deal benefits Moscow more than US consumers, experts say",
        category=TopicCategory.SANCTIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    grounded_summary = EventSummary(
        event_id=cluster.id,
        headline="Trump Eases Restrictions on Russian Diesel Imports",
        what_happened=(
            "The Trump administration moved to ease restrictions on Russian diesel fuel imports, "
            "prompting concerns from Ukrainian officials that the agreement provides "
            "revenue for Moscow."
        ),
        key_details=EventKeyDetails(
            actors=("Donald Trump", "Moscow", "Ukraine"),
            location=None,
            action="easing restrictions on diesel fuel",
            timing="October 2026",
        ),
        confidence_qualification=(
            "Single-source report from Defense News; unverified independently."
        ),
        model_used="test-model",
    )
    val = validate_summary_grounding(grounded_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_usfk_endurance_misrepresented_as_jamming_duration_quarantined():
    """Misrepresenting system survival threshold as duration of jamming is quarantined."""
    dt = datetime(2026, 10, 28, 10, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title=(
            "With daily jamming of US forces on Korean Peninsula, "
            "industry needs resilient solutions"
        ),
        url="https://breakingdefense.com/2026/10/with-daily-jamming-of-us-forces",
        canonical_url="https://breakingdefense.com/2026/10/with-daily-jamming-of-us-forces",
        published_at=dt,
        snippet=(
            "Electronic systems used by US forces are jammed daily on the Korean Peninsula. "
            "Gen. Xavier Brunson stated that any capability requiring pristine satellite "
            "communications will not survive 48 hours on the peninsula."
        ),
        content_hash="hash-usfk-test",
    )
    cluster = EventCluster(
        title=(
            "With daily jamming of US forces on Korean Peninsula, "
            "industry needs resilient solutions"
        ),
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    # Incorrect summary asserting 48 hours of jamming
    bad_summary = EventSummary(
        event_id=cluster.id,
        headline="US Forces Korea Subjected to 48 Hours of Jamming",
        what_happened=(
            "US forces on the Korean Peninsula endured 48 hours of jamming against satellites."
        ),
        key_details=EventKeyDetails(
            actors=("US Forces Korea", "Gen. Xavier Brunson"),
            location="Korean Peninsula",
            action="jamming",
            timing="48 hours",
        ),
        confidence_qualification="Single-source report; unverified independently.",
        model_used="test-model",
    )
    val = validate_summary_grounding(bad_summary, cluster)
    assert not val.is_valid
    assert val.quarantine_reason is not None
    assert any(
        "endurance" in c.lower() or "survival threshold" in c.lower()
        for c in val.unsupported_claims
    )


def test_usfk_grounded_summary_with_accurate_endurance_passes():
    """Summary accurately preserving 48 hours as survival threshold passes validation."""
    dt = datetime(2026, 10, 28, 10, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title=(
            "With daily jamming of US forces on Korean Peninsula, "
            "industry needs resilient solutions"
        ),
        url="https://breakingdefense.com/2026/10/with-daily-jamming-of-us-forces",
        canonical_url="https://breakingdefense.com/2026/10/with-daily-jamming-of-us-forces",
        published_at=dt,
        snippet=(
            "Electronic systems used by US forces are jammed daily on the Korean Peninsula. "
            "Gen. Xavier Brunson stated that any capability requiring pristine satellite "
            "communications will not survive 48 hours on the peninsula."
        ),
        content_hash="hash-usfk-pass",
    )
    cluster = EventCluster(
        title=(
            "With daily jamming of US forces on Korean Peninsula, "
            "industry needs resilient solutions"
        ),
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    good_summary = EventSummary(
        event_id=cluster.id,
        headline="USFK Commander Warns Non-Resilient Systems Will Not Survive Korean Jamming",
        what_happened=(
            "Gen. Xavier Brunson warned that systems requiring pristine satellite links "
            "would not survive 48 hours on the peninsula due to daily jamming."
        ),
        key_details=EventKeyDetails(
            actors=("US Forces Korea", "Gen. Xavier Brunson"),
            location="Korean Peninsula",
            action="warning on communications resilience",
            timing="October 2026",
        ),
        confidence_qualification=(
            "Single-source report from Breaking Defense; unverified independently."
        ),
        model_used="test-model",
    )
    val = validate_summary_grounding(good_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_aukus_maritime_big_play_grounded_summary_passes():
    """AUKUS Maritime Big Play autonomous systems exercise summary passes validation."""
    dt = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="AUKUS partners test autonomous systems as Europe weighs maritime surveillance needs",
        url="https://breakingdefense.com/2026/10/aukus-partners-test-autonomous-systems",
        canonical_url="https://breakingdefense.com/2026/10/aukus-partners-test-autonomous-systems",
        published_at=dt,
        snippet=(
            "During the Maritime Big Play exercise, AUKUS partners tested autonomous systems "
            "off the coast of San Diego, California, and integrated them into maritime operations. "
            "The UK and Australia participated with uncrewed systems off the coast."
        ),
        content_hash="hash-aukus-pass",
    )
    cluster = EventCluster(
        title="AUKUS partners test autonomous systems as Europe weighs maritime surveillance needs",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )
    good_summary = EventSummary(
        event_id=cluster.id,
        headline="AUKUS Partners Test Autonomous Systems Off San Diego Coast",
        what_happened=(
            "The UK, US, and Australia tested autonomous maritime surveillance systems during the "
            "Maritime Big Play exercise off the coast of San Diego."
        ),
        key_details=EventKeyDetails(
            actors=("UK", "US", "Australia", "AUKUS"),
            location="off the coast of San Diego",
            action="conducted Maritime Big Play exercise",
            timing="October 2026",
        ),
        confidence_qualification=(
            "Single-source report from Breaking Defense; unverified independently."
        ),
        model_used="test-model",
    )
    val = validate_summary_grounding(good_summary, cluster)
    assert val.is_valid
    assert val.quarantine_reason is None


def test_usfk_exact_force_survival_misstatement_quarantined_in_production_path():
    """Do not substitute the survival of US forces for the source's capability warning."""
    dt = datetime(2026, 10, 28, 13, 18, tzinfo=UTC)
    source = (
        "Gen. Xavier Brunson, commander of US Forces Korea, said: If your capability requires "
        "pristine satellite "
        "communications, infinite bandwidth, and an uninterrupted fiber optic backbone "
        "to function, it will not survive 48 hours on the peninsula. Electronic systems are "
        "jammed daily on the Korean Peninsula."
    )
    art = ArticleRecord(
        source_name="Breaking Defense",
        title=(
            "With daily jamming of US forces on Korean Peninsula, industry needs resilient "
            "solutions: USFK commander"
        ),
        url="https://example.com/usfk-editorial-regression",
        canonical_url="https://example.com/usfk-editorial-regression",
        published_at=dt,
        snippet=source,
        content_hash="hash-usfk-exact-preview-regression",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=dt,
        last_seen_at=dt,
        published_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )

    def run_summary(headline: str, what_happened: str, action: str) -> SummarizationResult:
        raw = {
            "headline": headline,
            "what_happened": what_happened,
            "key_details": {
                "actors": ["US Forces Korea", "Gen. Xavier Brunson"],
                "location": "Korean Peninsula",
                "timing": "October 28, 2026",
                "action": action,
            },
            "significance": None,
            "confidence_qualification": "Single-source report; unverified independently.",
        }
        return EventSummarizer(
            client=MockOllamaClient(canned_response=json.dumps(raw))
        ).summarize_event(event)

    misleading = run_summary(
        "Brunson Warns US Forces Could Not Survive 48 Hours",
        (
            "US Forces Korea (USFK) Commander Gen. Xavier Brunson stated that US forces on "
            "the Korean Peninsula would not survive 48 hours under daily jamming threats."
        ),
        "warning about daily jamming threats",
    )
    assert misleading.status == "quarantined"
    assert "Subject mismatch" in (misleading.quarantine_reason or "")
    # The existing phrase and key-field checks do not catch this specific subject swap.
    assert "Incorrect paraphrase of endurance" not in (misleading.quarantine_reason or "")
    assert "Timing mismatch" not in (misleading.quarantine_reason or "")

    faithful = run_summary(
        "Brunson Warns Communications-Dependent Capabilities Will Fail Under Peninsula Conditions",
        (
            "Gen. Xavier Brunson said capabilities requiring pristine satellite "
            "communications, "
            "infinite bandwidth, and an uninterrupted fiber optic backbone would not survive "
            "48 hours on the peninsula. He described electronic jamming as a daily threat."
        ),
        "warning on communications-dependent capability endurance",
    )
    assert faithful.status == "success", faithful.quarantine_reason


def test_relative_election_weekday_requires_article_support_in_production_path():
    """The article must state an election weekday; publication context cannot supply it."""
    dt = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)

    def run(source: str) -> SummarizationResult:
        art = ArticleRecord(
            source_name="Local News",
            title="Serbia's national election remains a focus of political debate",
            url="https://example.com/serbia-election",
            canonical_url="https://example.com/serbia-election",
            published_at=dt,
            snippet=source,
            content_hash=f"serbia-{source}",
        )
        event = EventCluster(
            title=art.title,
            category=TopicCategory.DIPLOMACY,
            first_seen_at=dt,
            last_seen_at=dt,
            published_at=dt,
            confidence_label="single_source",
            articles=(art,),
        )
        raw = {
            "headline": "Serbia's National Election Takes Place Sunday",
            "what_happened": "Serbia's national election is scheduled for Sunday.",
            "key_details": {
                "actors": ["Serbia"],
                "location": "Serbia",
                "timing": "Sunday",
                "action": "national election",
            },
            "significance": None,
            "confidence_qualification": "Single-source report; unverified independently.",
        }
        return EventSummarizer(
            client=MockOllamaClient(canned_response=json.dumps(raw))
        ).summarize_event(event)

    unsupported = run(
        "Serbian parties discussed the national election, but the article did not give its date."
    )
    assert unsupported.status == "quarantined"
    assert "Relative election date 'Sunday'" in (unsupported.quarantine_reason or "")

    grounded = run("The national election is scheduled for Sunday, according to the commission.")
    assert grounded.status == "success"


def test_black_sea_material_counts_must_match_local_article_text_in_production_path():
    """Large vessel and seafarer counts are checked against the available source text."""
    dt = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)

    def run(source: str) -> SummarizationResult:
        art = ArticleRecord(
            source_name="Local News",
            title="Black Sea attacks threaten merchant shipping",
            url="https://example.com/black-sea-shipping",
            canonical_url="https://example.com/black-sea-shipping",
            published_at=dt,
            snippet=source,
            content_hash=f"black-sea-{source}",
        )
        event = EventCluster(
            title=art.title,
            category=TopicCategory.SECURITY_INCIDENT,
            first_seen_at=dt,
            last_seen_at=dt,
            published_at=dt,
            confidence_label="single_source",
            articles=(art,),
        )
        raw = {
            "headline": "Black Sea Attacks Hit 570 Vessels",
            "what_happened": (
                "More than 570 vessels were hit and at least 40 seafarers were killed."
            ),
            "key_details": {
                "actors": [],
                "location": "Black Sea",
                "timing": "October 2026",
                "action": "attacks on commercial shipping",
            },
            "significance": None,
            "confidence_qualification": "Single-source report; unverified independently.",
        }
        return EventSummarizer(
            client=MockOllamaClient(canned_response=json.dumps(raw))
        ).summarize_event(event)

    unsupported = run(
        "Black Sea attacks damaged merchant vessels, and crew fatalities were reported; "
        "the report supplied no verified totals."
    )
    assert unsupported.status == "quarantined"
    assert "Asset quantity claim '570 vessels'" in (unsupported.quarantine_reason or "")
    assert "Asset quantity claim '40 seafarers'" in (unsupported.quarantine_reason or "")

    grounded = run("Reports say 570 vessels were hit and 40 seafarers were killed.")
    assert grounded.status == "success"


def test_editorial_regressions_use_production_summarization_validation_path():
    """Production summarizer quarantines the four observed classes of editorial error."""
    dt = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)

    def summarize(title: str, source: str, output: dict) -> SummarizationResult:
        article_url = (
            "https://example.com/2020/editorial-regression"
            if "Tigray" in title
            else "https://example.com/editorial-regression"
        )
        art = ArticleRecord(
            source_name="Breaking Defense",
            title=title,
            url=article_url,
            canonical_url=article_url,
            published_at=dt,
            snippet=source,
            content_hash=f"hash-{title}",
        )
        event = EventCluster(
            title=title,
            category=TopicCategory.DEFENCE_POLICY,
            first_seen_at=dt,
            last_seen_at=dt,
            published_at=dt,
            confidence_label="single_source",
            articles=(art,),
        )
        return EventSummarizer(
            client=MockOllamaClient(canned_response=json.dumps(output))
        ).summarize_event(event)

    def output(headline: str, what: str, actors: list[str], timing: str, action: str) -> dict:
        return {
            "headline": headline,
            "what_happened": what,
            "key_details": {
                "actors": actors,
                "location": None,
                "timing": timing,
                "action": action,
            },
            "significance": None,
            "confidence_qualification": "Single-source report; unverified independently.",
        }

    usfk_bad = summarize(
        "USFK commander warns about systems resilience",
        (
            "Gen. Xavier Brunson said satellite-dependent capabilities would not survive "
            "48 hours under degraded communications. Jamming occurs daily."
        ),
        output(
            "USFK Endures 48 Hours of Jamming",
            "USFK endured 48 hours of jamming.",
            ["USFK"],
            "48 hours",
            "jamming",
        ),
    )
    assert usfk_bad.status == "quarantined"
    assert "Incorrect paraphrase of endurance" in (usfk_bad.quarantine_reason or "")

    usfk_good = summarize(
        "USFK commander warns about systems resilience",
        (
            "Gen. Xavier Brunson said satellite-dependent capabilities would not survive "
            "48 hours under degraded communications. Jamming occurs daily."
        ),
        output(
            "USFK Commander Warns Systems May Not Survive 48 Hours Under Jamming",
            (
                "Gen. Xavier Brunson warned that systems requiring pristine satellite links "
                "would not survive 48 hours on the peninsula, where jamming occurs daily."
            ),
            ["USFK", "Gen. Xavier Brunson"],
            "October 2026",
            "warning about capability endurance under degraded communications",
        ),
    )
    assert usfk_good.status == "success"

    aukus_omission = summarize(
        "AUKUS tests uncrewed technologies in Maritime Big Play",
        (
            "The exercise tested more than 40 uncrewed technologies and included an "
            "anti-submarine warfare component."
        ),
        output(
            "AUKUS Tests Maritime Systems",
            "AUKUS partners tested uncrewed maritime systems in an exercise.",
            ["AUKUS"],
            "October 2026",
            "tested maritime systems",
        ),
    )
    assert aukus_omission.status == "quarantined"
    assert "Material omission: source reports more than 40 exercise technologies" in (
        aukus_omission.quarantine_reason or ""
    )
    assert "Material omission: source identifies an anti-submarine" in (
        aukus_omission.quarantine_reason or ""
    )

    aukus_absent = summarize(
        "AUKUS tests uncrewed technologies in Maritime Big Play",
        "AUKUS partners tested uncrewed maritime systems during the Maritime Big Play exercise.",
        output(
            "AUKUS Tests 40+ Uncrewed Systems in Anti-Submarine Warfare",
            (
                "AUKUS tested 40+ uncrewed technologies and included an anti-submarine "
                "warfare component."
            ),
            ["AUKUS"],
            "October 2026",
            "tested uncrewed technologies",
        ),
    )
    assert aukus_absent.status == "quarantined"
    assert "Unsupported detail: 40+" in (aukus_absent.quarantine_reason or "")
    assert "Unsupported detail: anti-submarine warfare" in (aukus_absent.quarantine_reason or "")

    aukus_short = summarize(
        "AUKUS tests uncrewed technologies in Maritime Big Play",
        "AUKUS partners tested uncrewed maritime systems during the Maritime Big Play exercise.",
        output(
            "AUKUS Tests Uncrewed Maritime Systems",
            (
                "AUKUS partners tested uncrewed maritime systems during the Maritime Big Play "
                "exercise."
            ),
            ["AUKUS"],
            "October 2026",
            "tested uncrewed maritime systems",
        ),
    )
    assert aukus_short.status == "success"

    diesel_bad = summarize(
        "Trump's Russian diesel deal benefits Moscow",
        (
            "The US moved to ease restrictions on Russian diesel imports. The article "
            "discusses the deal's impact on Moscow."
        ),
        output(
            "US to Sell Russia LNG for Two Decades",
            "The US agreed to sell Russia 1.7 billion cubic metres of LNG for two "
            "decades, a move linked to Ukraine.",
            ["US", "Russia", "Ukraine"],
            "two decades",
            "selling LNG to Russia",
        ),
    )
    assert diesel_bad.status == "quarantined"
    for issue in ("Commodity mismatch", "Volume or capacity", "Inverted transaction"):
        assert issue in (diesel_bad.quarantine_reason or "")

    diesel_ukraine_causal = summarize(
        "Trump's Russian diesel deal benefits Moscow",
        (
            "The US moved to ease restrictions on Russian diesel imports. Separately, "
            "Ukraine's energy ministry reported outages in several regions."
        ),
        output(
            "US Diesel Deal Is Linked to Ukraine",
            "The Russian diesel deal was linked to Ukraine's military aid policy.",
            ["US", "Russia", "Ukraine"],
            "October 2026",
            "ease diesel import restrictions",
        ),
    )
    assert diesel_ukraine_causal.status == "quarantined"
    assert "Unsupported causal claim" in (diesel_ukraine_causal.quarantine_reason or "")

    diesel_grounded = summarize(
        "Trump's Russian diesel deal benefits Moscow",
        (
            "The US moved to ease restrictions on Russian diesel imports. Ukrainian officials "
            "criticized the measure, saying the revenue benefits Moscow as Russian forces "
            "target Ukraine's energy infrastructure."
        ),
        output(
            "US Moves to Ease Russian Diesel Import Restrictions",
            (
                "The US moved to ease restrictions on Russian diesel imports. Ukrainian "
                "officials said the revenue could benefit Moscow while Russian forces target "
                "Ukraine's energy infrastructure."
            ),
            ["US", "Russia", "Ukraine", "Russian forces"],
            "October 2026",
            "ease restrictions on Russian diesel imports",
        ),
    )
    assert diesel_grounded.status == "success"

    tigray_timeline = summarize(
        "Ethiopia and Tigray tensions remain unresolved",
        (
            "The report describes renewed tensions between Ethiopian federal authorities "
            "and Tigrayan leaders, without dating the origins of the conflict."
        ),
        output(
            "Ethiopia Faces Renewed Tigray Tensions",
            "The report describes renewed tensions between Ethiopian federal authorities "
            "and Tigrayan leaders.",
            ["Ethiopian federal authorities", "Tigrayan leaders"],
            "Conflict began in 2020",
            "renewed tensions",
        ),
    )
    assert tigray_timeline.status == "quarantined"
    assert "Event timeline year '2020'" in (tigray_timeline.quarantine_reason or "")

    # Source-supported precision is retained by the same real validation path.
    valid_aukus = summarize(
        "AUKUS tests uncrewed technologies in Maritime Big Play",
        (
            "The exercise was designed to showcase more than 40 technologies, including "
            "manned, unmanned and multi-domain autonomous systems. The UK led the "
            "anti-submarine warfare component."
        ),
        output(
            "AUKUS Shows More Than 40 Technologies in Maritime Exercise",
            (
                "The Maritime Big Play exercise showcased more than 40 technologies, "
                "including manned, unmanned and multi-domain autonomous systems. The UK "
                "led the anti-submarine warfare component."
            ),
            ["AUKUS"],
            "October 2026",
            "tested uncrewed technologies and anti-submarine warfare",
        ),
    )
    assert valid_aukus.status == "success"

    aukus_overstates_count_scope = summarize(
        "AUKUS tests autonomous systems in Maritime Big Play",
        (
            "The exercise showcased more than 40 technologies, including manned, unmanned "
            "and autonomous systems, and the UK led its anti-submarine warfare component."
        ),
        output(
            "AUKUS Tests 40+ Uncrewed Technologies",
            (
                "AUKUS tested more than 40 uncrewed technologies in its anti-submarine "
                "warfare component."
            ),
            ["AUKUS"],
            "October 2026",
            "tested uncrewed technologies",
        ),
    )
    assert aukus_overstates_count_scope.status == "quarantined"
    assert "does not state that 40+ technologies were all uncrewed" in (
        aukus_overstates_count_scope.quarantine_reason or ""
    )


def test_targeted_diesel_metadata_and_role_regressions_use_production_path():
    """Exercise diesel grounding, role attribution and rendered provenance end to end."""
    dt = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)

    def run(
        title: str,
        source: str,
        body: str,
        *,
        headline: str = "Russian Diesel Report",
        location: object = None,
        confidence: str = "Single-source report; unverified independently.",
        source_names: tuple[str, ...] = ("Defense News",),
        action: str = "diesel imports",
        significance: str | None = None,
        duplicate_url: bool = False,
        actors: list[str] | None = None,
    ) -> SummarizationResult:
        articles = tuple(
            ArticleRecord(
                source_name=name,
                title=title,
                url=(
                    "https://example.com/shared/editorial-regression"
                    if duplicate_url
                    else f"https://example.com/{index}/editorial-regression"
                ),
                canonical_url=(
                    "https://example.com/shared/editorial-regression"
                    if duplicate_url
                    else f"https://example.com/{index}/editorial-regression"
                ),
                published_at=dt,
                snippet=source,
                content_hash=f"targeted-regression-{index}-{name}",
            )
            for index, name in enumerate(source_names)
        )
        event = EventCluster(
            title=title,
            category=TopicCategory.SANCTIONS,
            first_seen_at=dt,
            last_seen_at=dt,
            published_at=dt,
            confidence_label=(
                "corroborated"
                if len(set(source_names)) >= 2 and not duplicate_url
                else "single_source"
            ),
            articles=articles,
        )
        raw = {
            "headline": headline,
            "what_happened": body,
            "key_details": {
                "actors": actors if actors is not None else ["US", "Russia"],
                "location": location,
                "timing": None,
                "action": action,
            },
            "significance": significance,
            "confidence_qualification": confidence,
        }
        return EventSummarizer(
            client=MockOllamaClient(canned_response=json.dumps(raw))
        ).summarize_event(event)

    source = (
        "President Trump moved to ease restrictions on Russian diesel imports to the US. "
        "The excerpt provides no delivery volumes, prices, or timeline."
    )
    unsupported = run(
        "Trump moves to ease Russian diesel import restrictions",
        source,
        "The US and Russia agreed to ease sanctions, allowing the US to export diesel to Russia. "
        "The deal will cost Ukraine one of its few levers against the Kremlin and includes LNG.",
        headline="US-Russia Diesel Agreement Sends LNG to Moscow",
    )
    assert unsupported.status == "quarantined"
    reasons = unsupported.quarantine_reason or ""
    assert "Unsupported transaction" in reasons
    assert "Reversed trade direction" in reasons
    assert "Unsupported causal claim" in reasons
    assert "Commodity" in reasons

    invented_terms = run(
        "Trump moves to ease Russian diesel import restrictions",
        source,
        "Russian diesel imports to the US will total 570,000 tons over two years at $3.90 "
        "per gallon.",
        headline="Russia to Send 570,000 Tons of Diesel to US",
    )
    assert invented_terms.status == "quarantined"
    assert any(
        marker in (invented_terms.quarantine_reason or "").lower()
        for marker in ("volume", "quantity", "timeframe", "unsupported number", "price")
    )

    grounded = run(
        "Trump moves to ease Russian diesel import restrictions",
        source,
        "President Trump moved to ease restrictions on Russian diesel imports to the US.",
        headline="Trump Moves to Ease Restrictions on Russian Diesel Imports",
    )
    assert grounded.status == "success", grounded.quarantine_reason

    grounded_ukraine_assessment = run(
        "Trump's Russian diesel deal benefits Moscow more than US consumers",
        (
            "The US eased restrictions on Russian diesel imports. Analysts said the deal "
            "would benefit Russia disproportionately and cost Kyiv one of its few levers "
            "against the Kremlin."
        ),
        (
            "Analysts said the diesel deal would benefit Russia disproportionately and cost "
            "Kyiv one of its few levers against the Kremlin."
        ),
        headline="Analysts Say Diesel Deal Benefits Moscow",
        action="ease restrictions on Russian diesel imports",
    )
    assert grounded_ukraine_assessment.status == "success", (
        grounded_ukraine_assessment.quarantine_reason
    )

    no_date = run(
        "Ethiopia and Tigray tensions remain unresolved",
        (
            "The report describes renewed tensions between Ethiopian federal authorities "
            "and Tigrayan leaders."
        ),
        (
            "The report describes renewed tensions between Ethiopian federal authorities "
            "and Tigrayan leaders."
        ),
        headline="Ethiopia Faces Renewed Tigray Tensions",
        location=None,
        action="renewed tensions",
    )
    assert no_date.status == "success", no_date.quarantine_reason
    invented_date = run(
        "Ethiopia and Tigray tensions remain unresolved",
        (
            "The report describes renewed tensions between Ethiopian federal authorities "
            "and Tigrayan leaders."
        ),
        "The conflict began in 2020 and tensions remain unresolved.",
        headline="Ethiopia Faces Renewed Tigray Tensions",
        action="renewed tensions",
    )
    assert invented_date.status == "quarantined"
    assert "Event timeline year '2020'" in (invented_date.quarantine_reason or "")

    role_mismatch = run(
        "UK's Healey says government will fill defense plan gap",
        "Finance minister John Healey said reprioritization would fill a defense plan gap.",
        "Chancellor John Healey said the government would fill a defense plan gap.",
        headline="Chancellor John Healey Outlines UK Defence Funding",
        action="fill a defense plan gap",
    )
    assert role_mismatch.status == "quarantined"
    assert "Unsupported officeholder attribution" in (role_mismatch.quarantine_reason or "")

    one_source_claim = run(
        "Diesel price restrictions discussed",
        "President Trump moved to ease restrictions on Russian diesel imports to the US.",
        "Multiple sources corroborate the terms of the diesel arrangement.",
        confidence="Multiple sources corroborate the terms.",
    )
    assert one_source_claim.status == "quarantined"
    assert "Single-source event improperly claims multi-source corroboration" in (
        one_source_claim.quarantine_reason or ""
    )

    exact_assessment = run(
        "Russian diesel restrictions discussed",
        "President Trump moved to ease restrictions on Russian diesel imports to the US.",
        "President Trump moved to ease restrictions on Russian diesel imports to the US.",
        headline="Trump Moves on Russian Diesel Restrictions",
        confidence=(
            "Based on a single source article, with some corroborating reports from "
            "other institutions."
        ),
        action="ease import restrictions",
    )
    assert exact_assessment.status == "success", exact_assessment.quarantine_reason
    assert exact_assessment.summary is not None
    assert "corroborat" not in exact_assessment.summary.confidence_qualification.lower()

    duplicate_link = run(
        "US imports Russian diesel",
        "The US imported Russian diesel under eased restrictions.",
        "Multiple reports corroborate that the US imported Russian diesel.",
        headline="Reports Corroborate US Diesel Imports",
        source_names=("Defense News", "Reuters"),
        duplicate_url=True,
        action="import Russian diesel",
    )
    assert duplicate_link.status == "quarantined"
    assert duplicate_link.event.source_count == 1
    assert "two distinct, attached source links" in (duplicate_link.quarantine_reason or "")

    unsupported_black_sea_significance = run(
        "Black Sea attacks damage merchant shipping",
        "Attacks damaged merchant vessels in the Black Sea and killed crew members.",
        "Attacks damaged merchant vessels in the Black Sea and killed crew members.",
        headline="Attacks Damage Black Sea Merchant Ships",
        action="damage merchant vessels",
        significance=(
            "The attacks are critical for global trade and the supply of grain fertilizers."
        ),
        actors=["Russia"],
    )
    assert unsupported_black_sea_significance.status == "quarantined"
    assert "Unsupported material significance" in (
        unsupported_black_sea_significance.quarantine_reason or ""
    )

    supported_black_sea_significance = run(
        "Black Sea attacks disrupt grain and fertilizer cargoes",
        (
            "Attacks damaged Black Sea merchant vessels carrying grain and fertilizer cargoes. "
            "The route supports international trade in those goods."
        ),
        "Attacks disrupted merchant shipments carrying grain and fertilizer cargoes.",
        headline="Attacks Disrupt Black Sea Grain Shipments",
        action="disrupt merchant shipments",
        significance=(
            "The attacks threaten grain and fertilizer cargoes moving through an "
            "international trade route."
        ),
        actors=["Russia"],
    )
    assert supported_black_sea_significance.status == "success", (
        supported_black_sea_significance.quarantine_reason
    )

    unsupported_diesel_significance = run(
        "Trump moves to ease Russian diesel import restrictions",
        source,
        "President Trump moved to ease restrictions on Russian diesel imports to the US.",
        headline="Trump Moves to Ease Russian Diesel Restrictions",
        action="ease diesel import restrictions",
        significance=(
            "The deal may disproportionately benefit Russia and negatively impact Ukraine."
        ),
    )
    assert unsupported_diesel_significance.status == "quarantined"
    assert "Unsupported causal claim" in (unsupported_diesel_significance.quarantine_reason or "")

    supported_diesel_significance = run(
        "Trump's Russian diesel deal benefits Moscow more than US consumers",
        (
            "Trump moved to ease restrictions on Russian diesel imports to the US. The deal would "
            "benefit Russia disproportionately and cost Kyiv one of its few levers, "
            "according to ISW and Carnegie analysts."
        ),
        "Trump moved to ease restrictions on Russian diesel imports to the US.",
        headline="Trump Moves to Ease Diesel Restrictions",
        action="ease Russian diesel import restrictions",
        significance=(
            "According to ISW and Carnegie analysts, the deal may disproportionately "
            "benefit Russia and negatively impact Ukraine."
        ),
    )
    assert supported_diesel_significance.status == "success", (
        supported_diesel_significance.quarantine_reason
    )

    supported_role = run(
        "Foreign Minister Maya Alvarez announces talks",
        "Foreign Minister Maya Alvarez announced regional talks.",
        "Foreign Minister Maya Alvarez announced regional talks.",
        headline="Foreign Minister Maya Alvarez Announces Regional Talks",
        action="announce regional talks",
        actors=["Maya Alvarez"],
    )
    assert supported_role.status == "success", supported_role.quarantine_reason

    listed_location = run(
        "Diesel imports discussed in the Black Sea region",
        "The tanker route crossed the Black Sea.",
        "A diesel shipment was discussed.",
        headline="Diesel Shipment Crosses Black Sea",
        location=["Black Sea"],
    )
    assert listed_location.status == "success", listed_location.quarantine_reason
    assert listed_location.summary is not None
    assert listed_location.summary.key_details.location == "Black Sea"
    rendered = TelegramDigestFormatter().format_event_block(
        summary=listed_location.summary,
        category=TopicCategory.SANCTIONS,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "Location:</b> Black Sea" in rendered
    assert "['Black Sea']" not in rendered
    assert "[SINGLE SOURCE]" in rendered

    multi_source = run(
        "Russian diesel imports to the US",
        "The US imported Russian diesel. The import eased US supply pressure.",
        "The US imported Russian diesel, easing supply pressure.",
        headline="US Imports Russian Diesel",
        confidence="Single-source report; unverified independently.",
        source_names=("Defense News", "Reuters"),
        action="imported Russian diesel",
    )
    assert multi_source.status == "success", multi_source.quarantine_reason
    assert multi_source.summary is not None
    assert "Corroborated across 2 reporting outlets" in (
        multi_source.summary.confidence_qualification
    )
    multi_rendered = TelegramDigestFormatter().format_event_block(
        summary=multi_source.summary,
        category=TopicCategory.SANCTIONS,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[CORROBORATED]" in multi_rendered
    assert "[SINGLE SOURCE]" not in multi_rendered

    same_link_sources = (
        multi_source.summary.sources[0],
        multi_source.summary.sources[1].model_copy(
            update={"url": multi_source.summary.sources[0].url}
        ),
    )
    same_link_summary = multi_source.summary.model_copy(update={"sources": same_link_sources})
    same_link_rendered = TelegramDigestFormatter().format_event_block(
        summary=same_link_summary,
        category=TopicCategory.SANCTIONS,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[SINGLE SOURCE]" in same_link_rendered
    assert "[CORROBORATED]" not in same_link_rendered

    missing_links_summary = multi_source.summary.model_copy(update={"sources": ()})
    missing_links_validation = validate_summary_grounding(missing_links_summary, multi_source.event)
    assert not missing_links_validation.is_valid
    assert "two distinct, attached source links" in (
        missing_links_validation.quarantine_reason or ""
    )
