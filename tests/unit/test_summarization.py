"""Unit tests for Phase 4: Local Ollama-powered event summarization."""

import json
from datetime import UTC, datetime
from pathlib import Path

from geopulse.cli import main
from geopulse.clustering import cluster_articles
from geopulse.clustering.models import EventCluster
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization import (
    MockOllamaClient,
    summarize_event,
)

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_clustering_articles.json"


def load_sample_events() -> list[EventCluster]:
    """Helper to load and cluster fixture articles for summarization tests."""
    raw = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    articles: list[ArticleRecord] = []
    for item in raw:
        pub_dt = (
            datetime.fromisoformat(item["published_at"])
            if item.get("published_at")
            else None
        )
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
    single_source_event = next(e for e in events if e.confidence_label == "single_source")

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


def test_cli_summarize_fixture_mock_model_succeeds(capsys):
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


