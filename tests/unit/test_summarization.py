"""Unit tests for Phase 4: Local Ollama-powered event summarization."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from geopulse.classification.models import TopicCategory
from geopulse.cli import main
from geopulse.clustering import cluster_articles
from geopulse.clustering.models import EventCluster
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization import (
    EventSummarizer,
    MockOllamaClient,
    summarize_event,
)

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_clustering_articles.json"


def load_sample_events() -> list[EventCluster]:
    """Helper to load and cluster fixture articles for summarization tests."""
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    articles: list[ArticleRecord] = []
    for item in raw:
        pub_dt = datetime.fromisoformat(item["published_at"]) if item.get("published_at") else None
        articles.append(
            ArticleRecord(
                source_name=item["source_name"],
                title=item["title"],
                url=item["url"],
                canonical_url=item["canonical_url"],
                published_at=pub_dt,
                snippet=item.get("snippet"),
                content_hash=f"hash-{item['id']}",
            )
        )
    return list(cluster_articles(articles).clusters)


def test_normal_event_summary_success():
    events = load_sample_events()
    # Find the corroborated Houthi airstrikes event
    houthi_event = next(e for e in events if e.source_count >= 2)

    canned = json.dumps(
        {
            "headline": "US and UK carry out joint strikes on Houthi targets across Yemen",
            "what_happened": (
                "American and British forces targeted Houthi radar and missile sites "
                "following attacks on commercial shipping in the Red Sea."
            ),
            "key_details": {
                "actors": ["US military", "British forces", "Houthi movement"],
                "location": "Western Yemen and Sanaa",
                "timing": "Overnight on October 22",
                "action": "Joint airstrikes targeting radar and missile facilities",
            },
            "significance": (
                "Marks an escalation in international efforts to secure Red Sea shipping lanes."
            ),
            "confidence_qualification": (
                "Corroborated by independent reporting from both BBC World News and Al Jazeera."
            ),
        }
    )

    client = MockOllamaClient(canned_response=canned)
    res = summarize_event(houthi_event, client=client)

    assert res.status == "success"
    assert res.summary.headline == (
        "US and UK carry out joint strikes on Houthi targets across Yemen"
    )
    assert "American and British forces" in res.summary.what_happened
    assert "US military" in res.summary.key_details.actors
    assert res.summary.key_details.location == "Western Yemen and Sanaa"
    assert res.summary.significance is not None
    assert len(res.summary.sources) == 2


def test_single_source_event_with_appropriate_qualification():
    events = load_sample_events()
    # Find single-source event (UK MoD missile contract)
    single_source_event = next(
        e for e in events if e.confidence_label == "single_source" and "missile" in e.title.lower()
    )

    canned = json.dumps(
        {
            "headline": "UK Ministry of Defence signs £400m air defence missile contract",
            "what_happened": (
                "The British Ministry of Defence announced a £400m procurement contract "
                "for short-range air defence systems to modernise army readiness."
            ),
            "key_details": {
                "actors": ["UK Ministry of Defence"],
                "location": "United Kingdom",
                "timing": "October 22, 2026",
                "action": "Procurement contract signing",
            },
            "significance": "Enhances domestic air defence posture.",
            "confidence_qualification": (
                "Single-source report from Defense News; unverified independently."
            ),
        }
    )

    client = MockOllamaClient(canned_response=canned)
    res = summarize_event(single_source_event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    assert "Single-source" in res.summary.confidence_qualification
    assert res.summary.key_details.actors == ("UK Ministry of Defence",)


def test_conflicting_disputed_event_preserves_disagreement():
    events = load_sample_events()
    # Find disputed event (rebel claims vs army denial)
    disputed_event = next(e for e in events if e.confidence_label == "disputed")

    canned = json.dumps(
        {
            "headline": "Rebels claim downing of transport aircraft as army denies loss",
            "what_happened": (
                "Insurgent commanders claimed to have shot down a military Antonov transport "
                "aircraft, but the army denies the claim, calling it unconfirmed propaganda."
            ),
            "key_details": {
                "actors": ["Insurgent commanders", "Military authorities"],
                "location": "Unspecified in reporting",
                "timing": "October 22, 2026",
                "action": "Contested downing of transport aircraft",
            },
            "significance": "Could signal insurgent air-defence capability if verified.",
            "confidence_qualification": (
                "Disputed claim: rebel claim is contradicted by official military denial."
            ),
        }
    )

    client = MockOllamaClient(canned_response=canned)
    res = summarize_event(disputed_event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    assert "Disputed" in res.summary.confidence_qualification
    assert "denies" in res.summary.what_happened


def test_missing_snippets_handled_without_hallucination():
    base_time = datetime(2026, 10, 22, 10, 0, tzinfo=UTC)
    art_no_snippet = ArticleRecord(
        source_name="Breaking Defense",
        title="Air Force tests hypersonic missile off coastal proving grounds",
        url="https://breakingdefense.com/hypersonic-test",
        canonical_url="https://breakingdefense.com/hypersonic-test",
        published_at=base_time,
        snippet=None,  # Missing snippet
        content_hash="hash-no-snippet",
    )

    cluster = EventCluster(
        title="Air Force tests hypersonic missile off coastal proving grounds",
        category="weapons_procurement",  # type: ignore[arg-type]
        first_seen_at=base_time,
        last_seen_at=base_time,
        confidence_label="single_source",
        articles=(art_no_snippet,),
    )

    client = MockOllamaClient()
    res = summarize_event(cluster, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Source attribution is preserved even with missing snippet
    assert len(res.summary.sources) == 1
    assert res.summary.sources[0].url == "https://breakingdefense.com/hypersonic-test"


def test_malformed_ollama_output_fails_gracefully():
    events = load_sample_events()
    event = events[0]

    client = MockOllamaClient(should_return_malformed=True)
    res = summarize_event(event, client=client)

    assert res.status == "failed"
    assert res.summary is None
    assert "could not be parsed as valid JSON" in (res.error_message or "")
    # Original event and source articles are preserved intact
    assert res.event.id == event.id
    assert res.event.articles == event.articles


def test_ollama_unavailable_fails_gracefully():
    events = load_sample_events()
    event = events[0]

    client = MockOllamaClient(should_be_unavailable=True)
    res = summarize_event(event, client=client)

    assert res.status == "unavailable"
    assert res.summary is None
    assert "unavailable" in (res.error_message or "").lower()
    assert res.event.id == event.id


def test_ollama_timeout_fails_gracefully():
    events = load_sample_events()
    event = events[0]

    client = MockOllamaClient(should_timeout=True)
    res = summarize_event(event, client=client)

    assert res.status == "failed"
    assert res.summary is None
    assert "timed out" in (res.error_message or "").lower()
    assert res.event.id == event.id


def test_ollama_model_not_found_fails_gracefully():
    events = load_sample_events()
    event = events[0]

    client = MockOllamaClient(should_model_not_found=True)
    res = summarize_event(event, client=client)

    assert res.status == "failed"
    assert res.summary is None
    assert "not found" in (res.error_message or "").lower()


def test_source_attribution_and_urls_strictly_preserved():
    events = load_sample_events()
    houthi_event = next(e for e in events if e.source_count >= 2)

    client = MockOllamaClient()
    res = summarize_event(houthi_event, client=client)

    assert res.status == "success"
    assert res.summary is not None

    expected_urls = {a.url for a in houthi_event.articles}
    actual_urls = {s.url for s in res.summary.sources}
    assert actual_urls == expected_urls

    expected_titles = {a.title for a in houthi_event.articles}
    actual_titles = {s.title for s in res.summary.sources}
    assert actual_titles == expected_titles


def test_null_or_unspecified_significance_normalizes_to_none():
    events = load_sample_events()
    event = events[0]

    # Test string "null", "none", and "Unspecified in reporting"
    for raw_sig in ["null", "none", "n/a", "Unspecified in reporting", ""]:
        canned = json.dumps(
            {
                "headline": "Airstrikes reported in conflict zone",
                "what_happened": "Strikes targeted radar facilities in the sector overnight.",
                "key_details": {
                    "actors": ["Military forces"],
                    "location": "Sector A",
                    "timing": "Overnight",
                    "action": "Airstrikes",
                },
                "significance": raw_sig,
                "confidence_qualification": "Corroborated across 2 reporting outlets.",
            }
        )
        client = MockOllamaClient(canned_response=canned)
        res = summarize_event(event, client=client)
        assert res.status == "success"
        assert res.summary is not None
        assert res.summary.significance is None


def test_slug_confidence_qualifications_expand_to_rigorous_statements():
    events = load_sample_events()
    single_event = next(e for e in events if e.confidence_label == "single_source")
    corrob_event = next(e for e in events if e.confidence_label == "corroborated")
    dispute_event = next(e for e in events if e.confidence_label == "disputed")

    # 1. Single source slug "single-source"
    canned_single = json.dumps(
        {
            "headline": "Procurement contract announced",
            "what_happened": "A defence contract was awarded for air defence systems.",
            "key_details": {"actors": ["MoD"]},
            "significance": None,
            "confidence_qualification": "single-source",
        }
    )
    res_single = summarize_event(
        single_event, client=MockOllamaClient(canned_response=canned_single)
    )
    assert res_single.summary is not None
    assert "Single-source report from" in res_single.summary.confidence_qualification
    assert "unverified independently" in res_single.summary.confidence_qualification

    # 2. Corroborated slug "corroborated"
    canned_corrob = json.dumps(
        {
            "headline": "Airstrikes hit radar sites",
            "what_happened": "Forces struck multiple radar positions across the region.",
            "key_details": {"actors": ["Coalition"]},
            "significance": None,
            "confidence_qualification": "multi-source corroboration",
        }
    )
    res_corrob = summarize_event(
        corrob_event, client=MockOllamaClient(canned_response=canned_corrob)
    )
    assert res_corrob.summary is not None
    assert "Corroborated across 2 reporting outlets" in res_corrob.summary.confidence_qualification
    assert "unconfirmed independently" in res_corrob.summary.confidence_qualification

    # 3. Disputed slug "disputed"
    canned_dispute = json.dumps(
        {
            "headline": "Transport plane shot down, army denies",
            "what_happened": "Rebels claim aircraft downing, while the military denies any loss.",
            "key_details": {"actors": ["Rebels", "Army"]},
            "significance": None,
            "confidence_qualification": "disputed",
        }
    )
    res_dispute = summarize_event(
        dispute_event, client=MockOllamaClient(canned_response=canned_dispute)
    )
    assert res_dispute.summary is not None
    assert "Disputed reporting" in res_dispute.summary.confidence_qualification


def test_cli_summarize_fixture_mock_model_succeeds(capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    exit_code = main(["summarize", "--fixture", "--mock-model"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Geopulse Event Summarization [Local Ollama]" in captured
    assert "Model: mock-llama3.2:3b" in captured
    assert "HEADLINE:" in captured
    assert "WHAT HAPPENED:" in captured
    assert "KEY DETAILS:" in captured
    assert "SOURCES:" in captured
    assert "Summarization Summary:" in captured
    assert "Successful:     " in captured


def test_cli_summarize_unreachable_ollama_reports_clean_diagnostic(capsys, monkeypatch):
    # Running without --mock-model when Ollama is offline reports UNAVAILABLE cleanly
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    exit_code = main(["summarize", "--fixture", "--limit", "1"])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "UNAVAILABLE" in captured
    assert "Ollama is unreachable" in captured or "Ensure 'ollama serve' is running" in captured
    assert "HINT: Start Ollama with 'ollama serve' or pass '--mock-model'" in captured


def test_single_source_event_with_hallucinated_corroboration_is_strictly_corrected():
    events = load_sample_events()
    single_event = next(e for e in events if e.source_count == 1)
    hallucinated_response = json.dumps(
        {
            "headline": "Procurement contract announced",
            "what_happened": "A defence contract was awarded for air defence systems.",
            "key_details": {"actors": ["MoD"]},
            "significance": None,
            "confidence_qualification": (
                "Corroborated by multiple news outlets and defense analysts."
            ),
        }
    )
    res = summarize_event(
        single_event, client=MockOllamaClient(canned_response=hallucinated_response)
    )
    assert res.summary is not None
    assert "corroborat" not in res.summary.confidence_qualification.lower()
    assert "multiple" not in res.summary.confidence_qualification.lower()
    assert "Single-source report from" in res.summary.confidence_qualification
    assert "unverified independently" in res.summary.confidence_qualification


def test_analytical_event_summarization_prompt_and_mock_response():
    old_pub = datetime(2024, 5, 10, 12, 0, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="War on the Rocks",
        title="The Endless Chip War",
        url="https://warontherocks.com/the-endless-chip-war",
        canonical_url="https://warontherocks.com/the-endless-chip-war",
        published_at=old_pub,
        snippet="Interview with author of Chip Age on semiconductor supply chain competition.",
        content_hash="hash-chip-analysis",
    )
    event = cluster_articles([art]).clusters[0]
    assert event.is_analysis is True

    client = MockOllamaClient()
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    assert "military strikes" not in res.summary.headline.lower()
    assert "semiconductor" in res.summary.headline.lower()
    assert "strategic analysis" in res.summary.what_happened.lower()


def test_summarize_petraeus_preserves_speaker_attribution():
    """Verify Defense News Petraeus article preserves speaker attribution."""
    petraeus_title = (
        "US military needs 'urgent revolutionary institutional change' to prepare "
        "for modern war, Petraeus says"
    )
    art = ArticleRecord(
        source_name="Defense News",
        title=petraeus_title,
        url="https://defensenews.com/petraeus-autonomous-war",
        canonical_url="https://defensenews.com/petraeus-autonomous-war",
        published_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        snippet=(
            "Ret. Army Gen. David Petraeus praised the Pentagon's new Autonomous Warfare Command "
            "but said the U.S. must rapidly prepare for modern autonomous war, warning that "
            "the military requires urgent revolutionary institutional change."
        ),
        content_hash="hash-petraeus",
    )
    event = cluster_articles([art]).clusters[0]

    client = MockOllamaClient()
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # 1. Headline attributes Petraeus
    assert "petraeus" in res.summary.headline.lower()
    # 2. What happened attributes statement to Petraeus rather than stating overhaul as fact
    assert "petraeus" in res.summary.what_happened.lower()
    # 3. Key actors includes Petraeus
    assert any("petraeus" in a.lower() for a in res.summary.key_details.actors)
    # 4. Confidence qualification reflects single source and speaker statement
    assert "single-source" in res.summary.confidence_qualification.lower()
    assert "unverified independently" in res.summary.confidence_qualification.lower()


def test_summarize_uk_france_nuclear_exercise_preserves_non_nuclear_qualification():
    """Verify UK MoD article preserves qualification that no nuclear weapons were involved."""
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/uk-france-exercise",
        canonical_url="https://gov.uk/mod/uk-france-exercise",
        published_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        snippet=(
            "UK deepens nuclear defence ties with France in historic first joint air exercise "
            "as RAF Typhoon fighters join French Rafale jets in a French nuclear exercise. "
            "The Ministry of Defence confirmed that no nuclear weapons were involved in the drill."
        ),
        content_hash="hash-uk-france",
    )
    event = cluster_articles([art]).clusters[0]

    client = MockOllamaClient()
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # 1. Critical negative qualification is preserved in what_happened
    assert "no nuclear weapons" in res.summary.what_happened.lower()
    # 2. Scope precision: does not generalize to "first-ever joint exercise"
    assert "first-ever joint exercise" not in res.summary.what_happened.lower()
    assert "air" in res.summary.what_happened.lower()
    # 3. Actors include RAF and French forces
    assert any(
        "air force" in a.lower() or "raf" in a.lower() for a in res.summary.key_details.actors
    )
    # 4. Confidence qualification notes single-source government announcement
    assert "single-source" in res.summary.confidence_qualification.lower()


def test_summarizer_guardrail_injects_omitted_nuclear_caveat():
    """Verify guardrail injects 'no nuclear weapons' caveat if LLM generated output omits it."""
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/uk-france-exercise",
        canonical_url="https://gov.uk/mod/uk-france-exercise",
        published_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        snippet=(
            "RAF Typhoon fighters joined French Rafale jets in a French nuclear exercise. "
            "The Ministry of Defence confirmed that no nuclear weapons were involved in the drill."
        ),
        content_hash="hash-uk-france-caveat",
    )
    event = cluster_articles([art]).clusters[0]

    # LLM output that carelessly omitted the non-nuclear caveat
    omitted_caveat_json = json.dumps(
        {
            "headline": "UK-France hold first joint air exercise",
            "what_happened": (
                "RAF Typhoon fighters joined French Rafale aircraft in a French nuclear exercise "
                "to enhance bilateral interoperability and strategic deterrence cooperation."
            ),
            "key_details": {
                "actors": ["Royal Air Force", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Joint air exercise",
            },
            "significance": "Strengthens European deterrence.",
            "confidence_qualification": "Single-source report from UK Ministry of Defence.",
        }
    )

    client = MockOllamaClient(canned_response=omitted_caveat_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Guardrail must detect omission from source snippet and inject qualification
    assert "no nuclear weapons were involved" in res.summary.what_happened.lower()


def test_summarizer_guardrail_enforces_missing_speaker_attribution():
    """Verify guardrail restores speaker attribution when LLM states opinion as fact."""
    petraeus_title = (
        "US military needs 'urgent revolutionary institutional change' to prepare "
        "for modern war, Petraeus says"
    )
    art = ArticleRecord(
        source_name="Defense News",
        title=petraeus_title,
        url="https://defensenews.com/petraeus-autonomous-war",
        canonical_url="https://defensenews.com/petraeus-autonomous-war",
        published_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        snippet=(
            "David Petraeus said the U.S. military requires urgent revolutionary "
            "institutional change."
        ),
        content_hash="hash-petraeus-attr",
    )
    event = cluster_articles([art]).clusters[0]

    # LLM output that stripped "Petraeus" and stated his opinion as objective fact
    unattributed_json = json.dumps(
        {
            "headline": "US military requires urgent revolutionary institutional change",
            "what_happened": (
                "The United States military must undergo rapid institutional change "
                "to prepare for modern autonomous warfare."
            ),
            "key_details": {
                "actors": ["US military", "Pentagon"],
                "location": "United States",
                "timing": "October 2026",
                "action": "Institutional reform",
            },
            "significance": "Crucial for future autonomous warfare readiness.",
            "confidence_qualification": "Single-source report from Defense News.",
        }
    )

    client = MockOllamaClient(canned_response=unattributed_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Guardrail prepends speaker attribution to prevent presenting claims as facts
    assert "according to petraeus:" in res.summary.what_happened.lower()
    # Guardrail ensures Petraeus is present in actors
    assert any("petraeus" in a.lower() for a in res.summary.key_details.actors)


def test_summarizer_guardrail_replaces_scope_overstatement():
    """Verify guardrail corrects sweeping overstatements to the source-attested scope.

    When the source says 'first joint air exercise' (not 'first joint air
    nuclear defence exercise'), the guardrail must replace over-broad claims
    like 'first-ever joint exercise' with the exact scope attested in the
    source text — never a narrower synthesised claim.
    """
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/uk-france-exercise",
        canonical_url="https://gov.uk/mod/uk-france-exercise",
        published_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        snippet="Historic first joint air exercise supporting French nuclear operations.",
        content_hash="hash-uk-france-scope",
    )
    event = cluster_articles([art]).clusters[0]

    overstated_json = json.dumps(
        {
            "headline": "UK and France conduct first joint air exercise",
            "what_happened": (
                "British and French air forces conducted their first-ever joint exercise "
                "supporting bilateral nuclear defence cooperation in Western Europe."
            ),
            "key_details": {
                "actors": ["RAF", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Joint air drill",
            },
            "significance": "Deepens bilateral alliance.",
            "confidence_qualification": "Single-source report from UK Ministry of Defence.",
        }
    )

    client = MockOllamaClient(canned_response=overstated_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Over-broad claim must be removed
    assert "first-ever joint exercise" not in res.summary.what_happened.lower()
    # Must use the source-attested scope, not a synthesised narrower one
    assert "first joint air exercise" in res.summary.what_happened.lower()
    # Must NOT upgrade to a phrase absent from the source
    assert "first joint air nuclear defence exercise" not in res.summary.what_happened.lower()


def test_telegram_formatter_renders_petraeus_and_uk_france_summaries_with_grounding():
    """Verify Telegram digest renders Petraeus and UK-France events with grounding."""
    from geopulse.delivery.formatter import TelegramDigestFormatter

    petraeus_title = (
        "US military needs 'urgent revolutionary institutional change' to prepare "
        "for modern war, Petraeus says"
    )
    art_petraeus = ArticleRecord(
        source_name="Defense News",
        title=petraeus_title,
        url="https://defensenews.com/petraeus-autonomous-war",
        canonical_url="https://defensenews.com/petraeus-autonomous-war",
        published_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        snippet="David Petraeus said the military requires urgent revolutionary change.",
        content_hash="hash-petraeus-fmt",
    )
    art_uk_fr = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/uk-france-exercise",
        canonical_url="https://gov.uk/mod/uk-france-exercise",
        published_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        snippet=(
            "UK deepens nuclear defence ties with France in historic "
            "first joint air exercise as RAF Typhoon fighters join "
            "French Rafale jets in a French nuclear exercise."
        ),
        content_hash="hash-uk-fr-fmt",
    )

    ev_petraeus = cluster_articles([art_petraeus]).clusters[0]
    ev_uk_fr = cluster_articles([art_uk_fr]).clusters[0]

    client = MockOllamaClient()
    res_petraeus = summarize_event(ev_petraeus, client=client)
    res_uk_fr = summarize_event(ev_uk_fr, client=client)

    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest(
        [res_petraeus, res_uk_fr],
        briefing_date=date(2026, 10, 23),
    )

    assert len(chunks) == 1
    output = chunks[0]

    # Check Petraeus formatting
    assert "🏛️ DEFENCE POLICY" in output
    assert "Petraeus" in output
    assert "Single-source" in output

    # Check UK-France formatting — source text does NOT contain
    # "no nuclear weapons were involved", so the guardrail must not inject it
    assert "first joint air" in output
    assert '<a href="https://gov.uk/mod/uk-france-exercise">UK Ministry of Defence</a>' in output


def test_summarizer_guardrail_does_not_inject_caveat_when_unstated_in_source():
    """Verify guardrail never falsely injects nuclear caveat when absent from source text."""
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Le Figaro",
        title="French Air Force conducts routine deterrence drill in Brittany",
        url="https://lefigaro.fr/deterrence-drill-brittany",
        canonical_url="https://lefigaro.fr/deterrence-drill-brittany",
        published_at=pub,
        snippet="French Rafale aircraft conducted tactical exercises over northwestern France.",
        content_hash="hash-lefigaro-no-caveat",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    raw_json = json.dumps(
        {
            "headline": "French Air Force holds deterrence drill in northwestern France",
            "what_happened": (
                "French Rafale aircraft completed tactical maneuvers over Brittany "
                "as part of routine strategic training."
            ),
            "key_details": {
                "actors": ["French Air Force"],
                "location": "Brittany, France",
                "timing": "October 2026",
                "action": "Routine deterrence drill",
            },
            "significance": "Maintains readiness.",
            "confidence_qualification": "Single-source report from Le Figaro.",
        }
    )

    client = MockOllamaClient(canned_response=raw_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Must NOT contain the unstated caveat
    assert "no nuclear weapons" not in res.summary.what_happened.lower()
    assert "without nuclear weapons" not in res.summary.what_happened.lower()


def test_summarizer_guardrail_does_not_replace_scope_when_generic_in_source():
    """Verify generic exercises without specific air/nuclear scope are not rewritten."""
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Stars and Stripes",
        title="US and Poland conduct joint logistics exercise",
        url="https://stripes.com/us-poland-logistics",
        canonical_url="https://stripes.com/us-poland-logistics",
        published_at=pub,
        snippet="Troops completed a logistics drill near the eastern border.",
        content_hash="hash-stripes-logistics",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    raw_json = json.dumps(
        {
            "headline": "US and Poland complete joint logistics exercise",
            "what_happened": (
                "Allied units concluded a first-ever joint exercise focusing on "
                "rapid corridor resupply."
            ),
            "key_details": {
                "actors": ["US Army", "Polish Land Forces"],
                "location": "Poland",
                "timing": "October 2026",
                "action": "Logistics exercise",
            },
            "significance": "Improves interoperability on NATO eastern flank.",
            "confidence_qualification": "Single-source report from Stars and Stripes.",
        }
    )

    client = MockOllamaClient(canned_response=raw_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Must NOT replace with nuclear air exercise scope
    assert "nuclear" not in res.summary.what_happened.lower()
    assert "air nuclear defence" not in res.summary.what_happened.lower()


def test_summarizer_guardrail_general_speaker_attribution():
    """Verify general speaker extraction works for diverse officials without hardcoded names."""
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 11, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Reuters",
        title="European allies must accelerate ammunition production, Stoltenberg urges",
        url="https://reuters.com/stoltenberg-ammunition",
        canonical_url="https://reuters.com/stoltenberg-ammunition",
        published_at=pub,
        snippet="NATO Secretary General emphasized production scaling across the alliance.",
        content_hash="hash-reuters-stoltenberg",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    unattributed_json = json.dumps(
        {
            "headline": "European allies must accelerate ammunition production",
            "what_happened": (
                "European nations need to rapidly scale up industrial munitions production "
                "to sustain collective defence commitments."
            ),
            "key_details": {
                "actors": ["European allies"],
                "location": "Europe",
                "timing": "October 2026",
                "action": "Ammunition industrial expansion",
            },
            "significance": "Crucial for European stockpile sustainability.",
            "confidence_qualification": "Single-source report from Reuters.",
        }
    )

    client = MockOllamaClient(canned_response=unattributed_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Prepend speaker attribution
    assert "according to stoltenberg:" in res.summary.what_happened.lower()
    assert any("stoltenberg" in a.lower() for a in res.summary.key_details.actors)


def test_summarizer_guardrail_multi_source_different_speakers_does_not_overattribute():
    """Verify multi-source cluster with conflicting speakers does not overattribute to one."""
    art1 = ArticleRecord(
        source_name="Defense One",
        title="Pentagon must accelerate autonomous systems, Austin says",
        url="https://defenseone.com/austin-autonomous",
        canonical_url="https://defenseone.com/austin-autonomous",
        published_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        snippet="Secretary Austin addressed autonomous drone deployment priorities.",
        content_hash="hash-multi-austin",
    )
    art2 = ArticleRecord(
        source_name="Janes",
        title="NATO partners face doctrine modernisation gap, Bauer warns",
        url="https://janes.com/bauer-nato-gap",
        canonical_url="https://janes.com/bauer-nato-gap",
        published_at=datetime(2026, 10, 23, 15, 0, tzinfo=UTC),
        snippet="Admiral Bauer highlighted operational doctrinal challenges.",
        content_hash="hash-multi-bauer",
    )
    from geopulse.classification.models import TopicCategory

    multi_event = EventCluster(
        title="Allied defence transformation debate",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        last_seen_at=datetime(2026, 10, 23, 15, 0, tzinfo=UTC),
        articles=(art1, art2),
        confidence_label="corroborated",
    )

    raw_json = json.dumps(
        {
            "headline": "Allies debate defence doctrine modernisation and autonomous systems",
            "what_happened": (
                "Defence leaders discussed divergent approaches to autonomous integration "
                "and doctrine modernisation across alliance members."
            ),
            "key_details": {
                "actors": ["Pentagon", "NATO Military Committee"],
                "location": "United States and Europe",
                "timing": "October 2026",
                "action": "Policy debate on defence modernisation",
            },
            "significance": "Affects joint doctrine interoperability.",
            "confidence_qualification": "Corroborated across 2 defence reporting outlets.",
        }
    )

    client = MockOllamaClient(canned_response=raw_json)
    res = summarize_event(multi_event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Must NOT attribute the combined summary to just Austin or just Bauer
    assert not res.summary.what_happened.lower().startswith("according to austin:")
    assert not res.summary.what_happened.lower().startswith("according to bauer:")


def test_summarizer_guardrail_already_attributed_does_not_duplicate():
    """Verify that when summary already contains speaker attribution, it is not duplicated."""
    art = ArticleRecord(
        source_name="Defense News",
        title="US military needs urgent change, Petraeus says",
        url="https://defensenews.com/petraeus-change",
        canonical_url="https://defensenews.com/petraeus-change",
        published_at=datetime(2026, 10, 23, 14, 0, tzinfo=UTC),
        snippet="David Petraeus said urgent institutional reform is required.",
        content_hash="hash-petraeus-dup",
    )
    event = cluster_articles([art]).clusters[0]

    already_attributed_json = json.dumps(
        {
            "headline": "US military requires urgent change",
            "what_happened": (
                "According to David Petraeus: The United States military requires "
                "urgent institutional change to prepare for autonomous warfare."
            ),
            "key_details": {
                "actors": ["David Petraeus", "US military"],
                "location": "United States",
                "timing": "October 2026",
                "action": "Institutional change",
            },
            "significance": "Crucial for autonomous readiness.",
            "confidence_qualification": "Single-source report from Defense News.",
        }
    )

    client = MockOllamaClient(canned_response=already_attributed_json)
    res = summarize_event(event, client=client)

    assert res.status == "success"
    assert res.summary is not None
    # Must NOT have duplicated "According to"
    assert not res.summary.what_happened.startswith("According to Petraeus: According to")
    assert res.summary.what_happened.count("According to") == 1


# ---------------------------------------------------------------------------
# Source-grounding regression tests: scope fidelity & caveat evidence
# ---------------------------------------------------------------------------


def test_scope_guardrail_explicit_first_joint_air_exercise():
    """Source explicitly attests 'first joint air exercise' — guardrail must
    preserve that exact scope when correcting an LLM overstatement.
    """
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/exercise",
        canonical_url="https://gov.uk/mod/exercise",
        published_at=pub,
        snippet=(
            "UK deepens nuclear defence ties with France in historic "
            "first joint air exercise as RAF Typhoon fighters join "
            "French Rafale jets in a French nuclear exercise."
        ),
        content_hash="hash-scope-air-explicit",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    # LLM overstates to "first-ever joint exercise"
    overstated = json.dumps(
        {
            "headline": "UK-France hold first joint exercise",
            "what_happened": (
                "British and French forces conducted their "
                "first-ever joint exercise in Western Europe."
            ),
            "key_details": {
                "actors": ["RAF", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Joint exercise",
            },
            "significance": "Deepens bilateral alliance.",
            "confidence_qualification": "Single-source.",
        }
    )

    client = MockOllamaClient(canned_response=overstated)
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    wh = res.summary.what_happened.lower()
    # Must use the exact source-attested phrase
    assert "first joint air exercise" in wh
    # Must NOT synthesise a narrower claim absent from the source
    assert "first joint air nuclear defence exercise" not in wh


def test_scope_guardrail_first_joint_exercise_without_air_qualifier():
    """Source attests 'first joint exercise' (no domain qualifier) — guardrail
    must NOT upgrade to 'first joint air exercise' or narrower.
    """
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Le Monde",
        title="Allies hold first joint exercise since 2019",
        url="https://lemonde.fr/allies-exercise",
        canonical_url="https://lemonde.fr/allies-exercise",
        published_at=pub,
        snippet=(
            "Allied nations completed their first joint exercise "
            "since 2019 with both air and maritime components."
        ),
        content_hash="hash-scope-generic-first",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    # LLM output uses source-faithful wording
    faithful = json.dumps(
        {
            "headline": "Allies complete first joint exercise since 2019",
            "what_happened": (
                "Allied forces conducted their first-ever joint exercise "
                "incorporating air and maritime units since 2019."
            ),
            "key_details": {
                "actors": ["Allied forces"],
                "location": "Unspecified in reporting",
                "timing": "October 2026",
                "action": "Combined joint exercise",
            },
            "significance": "Restores interoperability.",
            "confidence_qualification": "Single-source.",
        }
    )

    client = MockOllamaClient(canned_response=faithful)
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    wh = res.summary.what_happened.lower()
    # Source says "first joint exercise" — guardrail replaces
    # "first-ever joint exercise" with "first joint exercise"
    assert "first joint exercise" in wh
    # Must NOT inject "air" or "nuclear defence" into the scope
    assert "first joint air" not in wh
    assert "nuclear defence exercise" not in wh


def test_caveat_guardrail_nuclear_context_without_explicit_exclusion():
    """Source discusses nuclear defence context but does NOT explicitly state
    'no nuclear weapons were involved' — guardrail must NOT fabricate the
    caveat.
    """
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/exercise",
        canonical_url="https://gov.uk/mod/exercise",
        published_at=pub,
        snippet=(
            "UK deepens nuclear defence ties with France in historic "
            "first joint air exercise as RAF Typhoon fighters join "
            "French Rafale jets in a French nuclear exercise."
        ),
        content_hash="hash-caveat-no-exclusion",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    # LLM output that omits nuclear qualification (correctly)
    raw = json.dumps(
        {
            "headline": "UK-France deepen nuclear defence ties",
            "what_happened": (
                "RAF Typhoon fighters joined French Rafale jets in a "
                "French nuclear exercise, marking the first joint air "
                "exercise supporting bilateral nuclear defence ties."
            ),
            "key_details": {
                "actors": ["RAF", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Joint air exercise",
            },
            "significance": "Deepens nuclear deterrence cooperation.",
            "confidence_qualification": "Single-source report.",
        }
    )

    client = MockOllamaClient(canned_response=raw)
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    wh = res.summary.what_happened.lower()
    # Source text does NOT contain "no nuclear weapons" — must NOT inject it
    assert "no nuclear weapons were involved" not in wh
    assert "source explicitly confirmed" not in wh


def test_caveat_guardrail_explicit_non_nuclear_statement_in_source():
    """Source explicitly states 'no nuclear weapons were involved' — guardrail
    must inject the caveat when the LLM omits it.
    """
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/exercise",
        canonical_url="https://gov.uk/mod/exercise",
        published_at=pub,
        snippet=(
            "RAF Typhoon fighters joined French Rafale jets in a "
            "French nuclear exercise. The Ministry of Defence "
            "confirmed that no nuclear weapons were involved "
            "in the drill."
        ),
        content_hash="hash-caveat-explicit",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    # LLM omits the caveat
    omitted = json.dumps(
        {
            "headline": "UK-France deepen nuclear defence ties",
            "what_happened": (
                "RAF Typhoon fighters joined French Rafale jets "
                "in a French nuclear exercise to enhance bilateral "
                "deterrence cooperation."
            ),
            "key_details": {
                "actors": ["RAF", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Joint air exercise",
            },
            "significance": "Deepens deterrence cooperation.",
            "confidence_qualification": "Single-source report.",
        }
    )

    client = MockOllamaClient(canned_response=omitted)
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    wh = res.summary.what_happened.lower()
    # Source DOES contain "no nuclear weapons" — guardrail must inject
    assert "no nuclear weapons were involved" in wh


def test_scope_guardrail_conservative_correction_of_exaggerated_summary():
    """LLM exaggerates source claim — guardrail must correct it conservatively
    to the source-attested scope, validating the factual scope of the output
    rather than just the presence of a replacement phrase.
    """
    from geopulse.classification.models import TopicCategory

    pub = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="UK Ministry of Defence",
        title="UK-France deepen nuclear defence ties in first joint exercise",
        url="https://gov.uk/mod/exercise",
        canonical_url="https://gov.uk/mod/exercise",
        published_at=pub,
        snippet=(
            "UK deepens nuclear defence ties with France in historic "
            "first joint air exercise as RAF Typhoon fighters join "
            "French Rafale jets in a French nuclear exercise."
        ),
        content_hash="hash-scope-conservative",
    )
    event = EventCluster(
        title=art.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=pub,
        last_seen_at=pub,
        articles=(art,),
        confidence_label="single_source",
    )

    # LLM grossly exaggerates
    exaggerated = json.dumps(
        {
            "headline": "UK-France conduct first-ever joint exercise",
            "what_happened": (
                "In a historic development, British and French air "
                "forces conducted their first-ever joint exercise, "
                "establishing an unprecedented bilateral military "
                "partnership."
            ),
            "key_details": {
                "actors": ["RAF", "French Air Force"],
                "location": "France",
                "timing": "October 2026",
                "action": "Historic first joint exercise",
            },
            "significance": "Unprecedented cooperation.",
            "confidence_qualification": "Single-source report.",
        }
    )

    client = MockOllamaClient(canned_response=exaggerated)
    res = summarize_event(event, client=client)
    assert res.status == "success"
    assert res.summary is not None
    wh = res.summary.what_happened

    # The corrected text must reflect exactly what the source attests
    wh_lower = wh.lower()
    # 1. Over-broad "first-ever joint exercise" removed
    assert "first-ever joint exercise" not in wh_lower
    # 2. Replaced with source-attested "first joint air exercise"
    assert "first joint air exercise" in wh_lower
    # 3. Not upgraded to the synthesised claim
    assert "first joint air nuclear defence exercise" not in wh_lower
    # 4. No fabricated caveat (source lacks explicit exclusion)
    assert "no nuclear weapons were involved" not in wh_lower
    # 5. The factual scope is narrower than "first-ever joint" but
    #    broader than "first joint air nuclear defence" — the output
    #    must sit at exactly the source-attested level
    assert "first joint air exercise" in wh_lower


def test_summarize_events_pre_ranking_and_cap():
    """Verify highest-priority events are selected before summarization
    without calling LLM for lower items.
    """
    from geopulse.classification.models import TopicCategory
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.summarization import EventSummarizer

    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    # 1. Consequential diplomacy (Tier 1, rank 1.5)
    dip_art1 = ArticleRecord(
        source_name="Reuters",
        title="Diplomatic envoys convene for high-stakes ceasefire and peace negotiations",
        url="https://example.com/peace1",
        canonical_url="https://example.com/peace1",
        published_at=fixed_dt,
        snippet="Belligerents meet to negotiate a ceasefire.",
        content_hash="h-dip1",
    )
    dip_art2 = ArticleRecord(
        source_name="AP",
        title="Envoys meet for ceasefire talks",
        url="https://example.com/peace2",
        canonical_url="https://example.com/peace2",
        published_at=fixed_dt,
        snippet="Negotiations for peace deal begin.",
        content_hash="h-dip2",
    )
    dip_event = EventCluster(
        id="ev-dip",
        title="Diplomatic envoys convene for high-stakes ceasefire and peace negotiations",
        category=TopicCategory.DIPLOMACY,
        first_seen_at=fixed_dt,
        last_seen_at=fixed_dt,
        confidence_label="corroborated",
        articles=(dip_art1, dip_art2),
    )

    # 2. Armed conflict (Tier 1, rank 1.0)
    conflict_art1 = ArticleRecord(
        source_name="Al Jazeera",
        title="Heavy fighting reported along southern front lines",
        url="https://example.com/conflict1",
        canonical_url="https://example.com/conflict1",
        published_at=fixed_dt,
        snippet="Clashes erupt between armed forces.",
        content_hash="h-conflict1",
    )
    conflict_art2 = ArticleRecord(
        source_name="BBC",
        title="Frontline clashes intensify",
        url="https://example.com/conflict2",
        canonical_url="https://example.com/conflict2",
        published_at=fixed_dt,
        snippet="Heavy fighting erupts in southern sector.",
        content_hash="h-conflict2",
    )
    conflict_event = EventCluster(
        id="ev-conflict",
        title="Heavy fighting reported along southern front lines",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=fixed_dt,
        last_seen_at=fixed_dt,
        confidence_label="corroborated",
        articles=(conflict_art1, conflict_art2),
    )

    # 3-7. Military operations (Rank 4.0)
    mil_events = [
        EventCluster(
            id=f"ev-mil-{i}",
            title=f"Military exercise operational deployment phase {i}",
            category=TopicCategory.MILITARY_OPERATIONS,
            first_seen_at=fixed_dt,
            last_seen_at=fixed_dt,
            confidence_label="single_source",
            articles=(
                ArticleRecord(
                    source_name=f"Outlet {i}",
                    title=f"Military exercise operational deployment phase {i}",
                    url=f"https://example.com/mil-{i}",
                    canonical_url=f"https://example.com/mil-{i}",
                    published_at=fixed_dt,
                    snippet="Troops deploy for training maneuvers.",
                    content_hash=f"h-mil-{i}",
                ),
            ),
        )
        for i in range(1, 6)
    ]

    # 8-10. Low-priority historical commemorations (Rank 10.5)
    commem_events = [
        EventCluster(
            id=f"ev-commem-{i}",
            title=f"Somme battlefield burial of First World War soldier {i}",
            category=TopicCategory.OTHER_GEOPOLITICAL,
            first_seen_at=fixed_dt,
            last_seen_at=fixed_dt,
            confidence_label="single_source",
            articles=(
                ArticleRecord(
                    source_name=f"Historic Outlet {i}",
                    title=f"Somme battlefield burial of First World War soldier {i}",
                    url=f"https://example.com/commem-{i}",
                    canonical_url=f"https://example.com/commem-{i}",
                    published_at=fixed_dt,
                    snippet="Ceremonial reinterment on the Western Front.",
                    content_hash=f"h-commem-{i}",
                ),
            ),
        )
        for i in range(1, 4)
    ]

    all_events = [dip_event, conflict_event] + mil_events + commem_events

    valid_json = json.dumps(
        {
            "headline": "Verified Geopolitical Event Summary",
            "what_happened": "Operational developments and strategic security actions reported.",
            "key_details": {
                "actors": ["Armed forces"],
                "location": None,
                "timing": "October 2026",
                "action": "Deployment",
            },
            "significance": "Strategic impact.",
            "confidence_qualification": "Corroborated across reporting.",
        }
    )

    class CountingMockClient(MockOllamaClient):
        def __init__(self):
            super().__init__(model="test-mock")
            self.calls = 0

        def generate(
            self, prompt: str, system: str | None = None, format_type: str | None = None
        ) -> str:
            self.calls += 1
            return valid_json

    client = CountingMockClient()
    summarizer = EventSummarizer(client=client)

    # Request max_events=3
    results = summarizer.summarize_events(all_events, max_events=3, min_relevance_rank=9.0)

    # LLM must be called EXACTLY 3 times for top 3 candidates (not 10 times!)
    assert client.calls == 3
    assert len(results) == 10

    # The 3 primary events must have been summarized with model
    summarized_results = [r for r in results if r.summary and r.summary.model_used != "omitted"]
    assert len(summarized_results) == 3

    # Check that tier 1 events (conflict and consequential diplomacy) were included in top 3
    summarized_categories = [r.event.category for r in summarized_results]
    assert TopicCategory.ARMED_CONFLICT in summarized_categories
    assert TopicCategory.DIPLOMACY in summarized_categories

    # Formatter omission note preserves the 7 omitted items
    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest(results, max_events=3, edition="morning")
    digest = "\n\n".join(chunks)
    assert "capped at top 3 events" in digest
    assert "7 lower-priority event(s) omitted" in digest


def test_summarize_events_bounded_backfill_on_quarantine():
    """Verify that if a primary candidate is quarantined, backfill draws from deferred candidate."""
    from geopulse.summarization import EventSummarizer

    fixed_dt = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)

    events = [
        EventCluster(
            id=f"ev-backfill-{i}",
            title=f"Border security surveillance patrols conducted {i}",
            category=TopicCategory.SECURITY_INCIDENT,
            first_seen_at=fixed_dt,
            last_seen_at=fixed_dt,
            confidence_label="single_source",
            articles=(
                ArticleRecord(
                    source_name=f"Source {i}",
                    title=f"Border security surveillance patrols conducted {i}",
                    url=f"https://example.com/ev-{i}",
                    canonical_url=f"https://example.com/ev-{i}",
                    published_at=fixed_dt,
                    snippet="Security teams patrol border perimeter.",
                    content_hash=f"h-backfill-{i}",
                ),
            ),
        )
        for i in range(1, 6)
    ]

    valid_json = json.dumps(
        {
            "headline": "Border security patrols deployed",
            "what_happened": "Security units completed routine perimeter patrols.",
            "key_details": {
                "actors": ["Security units"],
                "location": None,
                "timing": "October 2026",
                "action": "Patrol",
            },
            "significance": "Security posture.",
            "confidence_qualification": "Single-source report.",
        }
    )

    # First event produces fatal invented kinetic strike claim (triggers quarantine)
    quarantine_json = json.dumps(
        {
            "headline": "Deadly missile strike hits capital",
            "what_happened": "Kinetic missile strike killed 50 soldiers in unprovoked attack.",
            "key_details": {
                "actors": ["Missile strike"],
                "location": "Capital",
                "timing": "October 2026",
                "action": "Missile strike",
            },
            "significance": "Escalation.",
            "confidence_qualification": "Disputed.",
        }
    )

    class BackfillClient(MockOllamaClient):
        def __init__(self):
            super().__init__(model="test-mock")
            self.calls = 0

        def generate(
            self, prompt: str, system: str | None = None, format_type: str | None = None
        ) -> str:
            self.calls += 1
            if self.calls == 1:
                return quarantine_json
            return valid_json

    client = BackfillClient()
    summarizer = EventSummarizer(client=client)

    # Request max_events=2. Primary = events[0], events[1].
    # events[0] gets quarantined -> Backfill draws events[2]
    results = summarizer.summarize_events(events, max_events=2, max_backfill=2)

    # 2 primary + 1 backfill = 3 calls
    assert client.calls == 3
    # Exactly 2 successful summaries generated
    successful_llm = [
        r for r in results if r.status == "success" and r.summary.model_used != "omitted"
    ]
    assert len(successful_llm) == 2
    # Quarantined event is recorded
    quarantined = [r for r in results if r.status in {"quarantined", "insufficient_evidence"}]
    assert len(quarantined) == 1


def test_negative_absence_of_breach_claims_are_explicitly_attributed():
    """Unattributed negative claims like 'no breaches have been reported' must be attributed."""
    dt = datetime(2026, 10, 26, 10, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Network outage investigated at defense logistics facility",
        url="https://defensenews.com/network-outage",
        canonical_url="https://defensenews.com/network-outage",
        published_at=dt,
        snippet="Officials initiated an investigation into an unexpected server outage.",
        content_hash="hash-outage-1",
        relevance_status="relevant",
    )
    cluster = EventCluster(
        title="Network outage investigated at defense logistics facility",
        category="cyber_hybrid_warfare",  # type: ignore[arg-type]
        first_seen_at=dt,
        last_seen_at=dt,
        confidence_label="single_source",
        articles=(art,),
    )

    unattributed_canned = json.dumps(
        {
            "headline": "Network Outage Investigated at Defense Facility",
            "what_happened": (
                "A network outage disrupted server systems at a defense logistics depot. "
                "No breaches have been reported following the incident."
            ),
            "key_details": {
                "actors": ["defense officials"],
                "location": None,
                "timing": "October 26, 2026",
                "action": "Investigation into server outage",
            },
            "significance": (
                "Highlights dependency of logistics facilities on reliable network infrastructure."
            ),
            "confidence_qualification": (
                "Single-source report from Defense News; unverified independently."
            ),
        }
    )

    client = MockOllamaClient(canned_response=unattributed_canned)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(cluster)

    assert res.status == "success"
    assert res.summary is not None
    # Must be attributed, not stated as an absolute unverified fact
    assert (
        "officials stated that no breaches have been reported" in res.summary.what_happened.lower()
    )


def test_mock_client_fixture_summaries_are_demonstrably_event_specific():
    """Verify MockOllamaClient produces distinct, event-specific summaries for fixtures."""
    from geopulse.clustering import cluster_articles

    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "sample_clustering_articles.json"
    )
    raw_items = json.loads(fixture_path.read_text(encoding="utf-8"))
    articles = [
        ArticleRecord(
            source_name=item["source_name"],
            title=item["title"],
            url=item["url"],
            canonical_url=item["canonical_url"],
            published_at=datetime.fromisoformat(item["published_at"]),
            snippet=item.get("snippet"),
            content_hash=f"hash-{item['id']}",
            relevance_status="relevant",
        )
        for item in raw_items
    ]

    summary = cluster_articles(articles)
    client = MockOllamaClient()
    summarizer = EventSummarizer(client=client)
    results = summarizer.summarize_events(summary.clusters)

    successful = [r for r in results if r.status == "success" and r.summary is not None]
    assert len(successful) >= 5

    headlines = [r.summary.headline for r in successful]  # type: ignore[union-attr]
    what_happeneds = [r.summary.what_happened for r in successful]  # type: ignore[union-attr]

    # Every event headline must be distinct
    assert len(set(headlines)) == len(headlines)
    # Every event summary narrative must be distinct
    assert len(set(what_happeneds)) == len(what_happeneds)

    # Verify key specific event topics are present and distinct
    combined_headlines = " ".join(headlines).lower()
    assert "houthi" in combined_headlines
    assert "rebels claim" in combined_headlines
    assert "oil depot" in combined_headlines or "rostov" in combined_headlines
    assert "belgorod" in combined_headlines
    assert "air defence" in combined_headlines or "400m" in combined_headlines

    # Generic identical military strike text must NOT be present
    assert "Targeted military strikes reported across conflict sector" not in combined_headlines
