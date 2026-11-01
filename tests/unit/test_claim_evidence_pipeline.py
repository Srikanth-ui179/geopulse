"""Offline production-path tests for claim-first evidence verification and rendering."""

import json
from datetime import UTC, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.client import MockOllamaClient
from geopulse.summarization.service import EventSummarizer


def _event(snippet: str, url: str = "https://example.test/article") -> EventCluster:
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    article = ArticleRecord(
        source_name="Fixture News",
        title="Regional forces test a communications network",
        url=url,
        canonical_url=url,
        published_at=now,
        snippet=snippet,
        content_hash=f"hash-{url}",
    )
    return EventCluster(
        title=article.title,
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(article,),
    )


def _claim(
    event: EventCluster,
    text: str,
    evidence: str,
    *,
    claim_id: str = "c1",
    central: bool = True,
    classification: str = "reported_fact",
    attribution: str | None = None,
) -> dict:
    return {
        "claim_id": claim_id,
        "text": text,
        "evidence_refs": [{"source_url": event.articles[0].url, "evidence_span": evidence}],
        "classification": classification,
        "attribution": attribution,
        "central": central,
    }


def _run(event: EventCluster, claims: list[dict], verdicts: list[dict]):
    output = json.dumps({"claims": claims})
    verdict_by_claim = {verdict["claim_id"]: verdict for verdict in verdicts}
    verifier = json.dumps(
        {
            "verdicts": [
                {
                    **verdict_by_claim[claim["claim_id"]],
                    "source_url": ref["source_url"],
                }
                for claim in claims
                for ref in claim["evidence_refs"]
                if claim["claim_id"] in verdict_by_claim
            ]
        }
    )

    def generate(prompt: str) -> str:
        return verifier if prompt.startswith("Check whether each claim") else output

    client = MockOllamaClient(custom_generator=generate)
    result = EventSummarizer(client=client).summarize_event(event)
    return result, client


def test_claim_first_path_accepts_exact_supported_span_and_renders_only_verified_claims():
    evidence = "The ministry said the network passed field trials in the northern region."
    event = _event(evidence)
    claim = _claim(
        event,
        "The ministry said its network passed field trials in the northern region.",
        evidence,
        classification="attributed_statement",
        attribution="the ministry",
    )
    result, client = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "The span states this."}],
    )

    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert len(client.calls) == 2
    assert result.summary.claims[0].verification_status == "accepted"
    assert result.summary.what_happened == result.summary.claims[0].text
    assert result.summary.sources[0].url == event.articles[0].url
    assert "independently generated narrative" not in result.summary.what_happened
    rendered = TelegramDigestFormatter().format_event_block(
        summary=result.summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="single_source",
        rank=1,
        total=1,
    )
    assert f'href="{event.articles[0].url}"' in rendered


def test_exact_quote_that_does_not_entail_claim_is_quarantined_by_verifier():
    evidence = (
        "Communications-dependent capabilities may fail after 48 hours without reliable links."
    )
    event = _event(evidence)
    misleading = _claim(
        event,
        "Forces cannot survive 48 hours of jamming.",
        evidence,
    )
    result, _ = _run(
        event,
        [misleading],
        [
            {
                "claim_id": "c1",
                "verdict": "unsupported",
                "reason": (
                    "The span describes capability endurance, not force survival or "
                    "jamming duration."
                ),
            }
        ],
    )
    assert result.status == "quarantined"
    assert "Central claim not verified" in (result.quarantine_reason or "")


def test_central_claim_gets_one_bounded_correction_and_verification_retry():
    evidence = (
        "Communications-dependent capabilities may fail after 48 hours without reliable links."
    )
    event = _event(evidence)
    bad_claims = [_claim(event, "Forces cannot survive 48 hours of jamming.", evidence)]
    corrected_claims = [
        _claim(
            event,
            "Communications-dependent capabilities may fail after 48 hours without reliable links.",
            evidence,
        )
    ]
    verifier_count = 0

    def generate(prompt: str) -> str:
        nonlocal verifier_count
        if prompt.startswith("Check whether each claim"):
            verifier_count += 1
            verdict = "unsupported" if verifier_count == 1 else "supported"
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "claim_id": "c1",
                            "source_url": event.articles[0].url,
                            "verdict": verdict,
                            "reason": "The first wording changes the subject and duration."
                            if verdict == "unsupported"
                            else "Faithful to the evidence span.",
                        }
                    ]
                }
            )
        if "CORRECTION REQUIRED" in prompt:
            return json.dumps({"claims": corrected_claims})
        return json.dumps({"claims": bad_claims})

    client = MockOllamaClient(custom_generator=generate)
    result = EventSummarizer(client=client).summarize_event(event)
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert result.summary.claims[0].text == corrected_claims[0]["text"]
    assert len(client.calls) == 4


def test_claim_evidence_must_exist_in_the_referenced_event_article():
    event = _event("Officials announced a new regional communications network.")
    claim = _claim(
        event,
        "Officials announced a communications network.",
        "This fabricated quotation is not in the article.",
    )
    result, client = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "irrelevant"}],
    )
    assert result.status == "quarantined"
    assert "evidence span is absent" in (result.quarantine_reason or "")
    assert len(client.calls) == 2


def test_claim_reference_cannot_point_to_an_unrelated_article_or_event():
    event = _event("Officials announced a regional communications network.")
    claim = _claim(
        event,
        "Officials announced a regional communications network.",
        event.articles[0].snippet or "",
    )
    claim["evidence_refs"][0]["source_url"] = "https://example.test/other-event"
    result, client = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "irrelevant"}],
    )
    assert result.status == "quarantined"
    assert "source outside this event" in (result.quarantine_reason or "")
    assert len(client.calls) == 2


def test_unsupported_secondary_claim_is_removed_while_grounded_central_claim_remains():
    evidence = "The ministry announced a new communications network for field units."
    event = _event(evidence)
    claims = [
        _claim(event, "The ministry announced a communications network.", evidence),
        _claim(
            event,
            "The network will transform global trade.",
            evidence,
            claim_id="c2",
            central=False,
            classification="source_analysis",
            attribution="the ministry",
        ),
    ]
    result, _ = _run(
        event,
        claims,
        [
            {"claim_id": "c1", "verdict": "supported", "reason": "Directly stated."},
            {"claim_id": "c2", "verdict": "unsupported", "reason": "No such implication."},
        ],
    )
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert [claim.claim_id for claim in result.summary.claims] == ["c1"]
    assert "global trade" not in result.summary.what_happened.lower()
    assert result.summary.significance is None


def test_supported_attributed_analysis_is_kept_as_analysis_not_recast_as_fact():
    evidence = "The analysts said the exercise could improve regional maritime surveillance."
    event = _event(evidence)
    claim = _claim(
        event,
        "The exercise could improve regional maritime surveillance.",
        evidence,
        classification="source_analysis",
        attribution="the analysts",
    )
    result, _ = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "Attributed assessment."}],
    )
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert result.summary.significance.startswith("According to the analysts,")
    assert result.summary.claims[0].classification == "source_analysis"


def test_one_claim_can_be_corroborated_by_two_distinct_article_evidence_spans():
    event = _event("Officials announced a new regional communications network.")
    second_article = event.articles[0].model_copy(
        update={
            "source_name": "Reuters",
            "url": "https://example.test/second",
            "canonical_url": "https://example.test/second",
            "content_hash": "second-source-hash",
            "snippet": "The ministry announced the regional communications network.",
        }
    )
    event = event.model_copy(
        update={"articles": (event.articles[0], second_article), "confidence_label": "corroborated"}
    )
    claim = _claim(
        event,
        "Officials announced a regional communications network.",
        event.articles[0].snippet or "",
    )
    claim["evidence_refs"].append(
        {"source_url": second_article.url, "evidence_span": second_article.snippet}
    )
    result, _ = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "Both state the announcement."}],
    )
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert len(result.summary.claims[0].evidence_refs) == 2
    assert len(result.summary.sources) == 2
    assert "Corroborated across 2" in result.summary.confidence_qualification


def test_event_level_corroboration_does_not_upgrade_single_source_central_claim():
    event = _event("The ministry announced a regional communications network.")
    second_article = event.articles[0].model_copy(
        update={
            "source_name": "Reuters",
            "url": "https://example.test/analysis",
            "canonical_url": "https://example.test/analysis",
            "content_hash": "analysis-source-hash",
            "snippet": "Analysts said the network could improve maritime surveillance.",
        }
    )
    event = event.model_copy(
        update={"articles": (event.articles[0], second_article), "confidence_label": "corroborated"}
    )
    claims = [
        _claim(
            event,
            "The ministry announced a communications network.",
            event.articles[0].snippet or "",
        ),
        {
            **_claim(
                event,
                "The network could improve maritime surveillance.",
                second_article.snippet or "",
                claim_id="c2",
                central=False,
                classification="source_analysis",
                attribution="analysts",
            ),
            "evidence_refs": [
                {"source_url": second_article.url, "evidence_span": second_article.snippet}
            ],
        },
    ]
    result, _ = _run(
        event,
        claims,
        [
            {"claim_id": "c1", "verdict": "supported", "reason": "Announcement stated."},
            {"claim_id": "c2", "verdict": "supported", "reason": "Attributed analysis."},
        ],
    )
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert "Single-source reporting" in result.summary.confidence_qualification

    from geopulse.delivery.formatter import TelegramDigestFormatter

    rendered = TelegramDigestFormatter().format_event_block(
        summary=result.summary,
        category=TopicCategory.DEFENCE_POLICY,
        confidence_label="corroborated",
        rank=1,
        total=1,
    )
    assert "[SINGLE SOURCE]" in rendered
    assert "[CORROBORATED]" not in rendered


def test_incomplete_or_invalid_verifier_output_quarantines_with_reason():
    evidence = "Officials reported a new exercise in the region."
    event = _event(evidence)
    claim = _claim(event, "Officials reported an exercise in the region.", evidence)
    result, _ = _run(event, [claim], [])
    assert result.status == "quarantined"
    assert "missing, duplicate, or unknown claim/source references" in (
        result.quarantine_reason or ""
    )


def test_non_mock_production_client_cannot_fall_back_to_unstructured_narrative():
    class LegacyClient:
        model = "offline-fixture"
        provider = "fixture"

        def generate(self, prompt: str, system: str | None = None, format_type: str | None = None):
            return json.dumps(
                {
                    "headline": "Regional Network Announced",
                    "what_happened": "Officials announced a regional communications network.",
                    "confidence_qualification": (
                        "Single-source reporting; unverified independently."
                    ),
                }
            )

        def is_available(self) -> bool:
            return True

    event = _event("Officials announced a regional communications network.")
    result = EventSummarizer(client=LegacyClient()).summarize_event(event)
    assert result.status == "quarantined"
    assert "omitted required claim-level source evidence" in (result.quarantine_reason or "")


def test_single_factual_claim_promoted_when_central_omitted():
    """If exactly one factual claim is provided without central=True, it is promoted to central."""
    evidence = "Regional forces completed trials of a tactical network."
    event = _event(evidence)
    claim = _claim(
        event,
        "Regional forces completed tactical network trials.",
        evidence,
        central=False,
    )
    result, _ = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "Explicitly stated"}],
    )
    assert result.status == "success", result.quarantine_reason
    assert result.summary is not None
    assert result.summary.claims[0].central is True
    assert result.summary.headline is not None


def test_analysis_only_claims_not_promoted_to_central():
    """If only analysis claims exist without central=True, they cannot be promoted to central."""
    evidence = "Analysts suggest the exercise indicates regional preparation."
    event = _event(evidence)
    claim = _claim(
        event,
        "The exercise indicates regional preparation.",
        evidence,
        central=False,
        classification="source_analysis",
        attribution="defence analysts",
    )
    result, _ = _run(
        event,
        [claim],
        [{"claim_id": "c1", "verdict": "supported", "reason": "Explicitly stated"}],
    )
    assert result.status in {"failed", "quarantined"}
    reason = (result.error_message or "") + " " + (result.quarantine_reason or "")
    assert "does not identify a central event claim" in reason


def test_multiple_factual_claims_not_arbitrarily_promoted():
    """If multiple factual claims exist without a central flag, none is arbitrarily picked."""
    evidence = "Forces deployed radar units. Air defence battalions held exercises."
    event = _event(evidence)
    c1 = _claim(
        event,
        "Forces deployed radar units.",
        "Forces deployed radar units.",
        claim_id="c1",
        central=False,
    )
    c2 = _claim(
        event,
        "Air defence battalions held exercises.",
        "Air defence battalions held exercises.",
        claim_id="c2",
        central=False,
    )
    result, _ = _run(
        event,
        [c1, c2],
        [
            {"claim_id": "c1", "verdict": "supported", "reason": "radar stated"},
            {"claim_id": "c2", "verdict": "supported", "reason": "battalions stated"},
        ],
    )
    assert result.status in {"failed", "quarantined"}
    reason = (result.error_message or "") + " " + (result.quarantine_reason or "")
    assert "does not identify a central event claim" in reason
