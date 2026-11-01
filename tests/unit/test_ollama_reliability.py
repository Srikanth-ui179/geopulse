"""Deterministic unit tests for Ollama reliability, retry policy, and timeout safeguards.

Covers:
- Connection reset handling and recovery.
- Transient HTTP error (502, 503, 504) handling and recovery.
- Timeout handling and fail-fast without indefinite retries.
- Token repeat limit error (HTTP 500) handling and exhaustion.
- Malformed structured output handling and safe quarantine.
- Retry exhaustion on persistent failures.
- No duplicate publication after ambiguous failures.
"""

import json
from datetime import UTC, datetime

import httpx
import pytest

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.client import MockOllamaClient, OllamaClient
from geopulse.summarization.exceptions import (
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from geopulse.summarization.service import EventSummarizer


def _make_dummy_event() -> EventCluster:
    now = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    art = ArticleRecord(
        source_name="Defense News",
        title="Allied air forces complete Baltic interception exercise",
        url="https://defensenews.com/baltic-drill",
        canonical_url="https://defensenews.com/baltic-drill",
        published_at=now,
        snippet="NATO allied fighters carried out live air-policing drills over the Baltic Sea.",
        content_hash="hash-baltic-01",
    )
    return EventCluster(
        id="evt-baltic-01",
        title=art.title,
        category=TopicCategory.MILITARY_OPERATIONS,
        region="baltic",
        first_seen_at=now,
        last_seen_at=now,
        published_at=now,
        is_analysis=False,
        confidence_label="single_source",
        articles=(art,),
    )


def test_ollama_connection_reset_recovery():
    """Client automatically recovers from a transient TCP connection reset on the first attempt."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return httpx.Response(
            200,
            json={"response": '{"claims": []}'},
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=2, transport=transport)

    result = client.generate(prompt="Test prompt", format_type="json")
    assert result == '{"claims": []}'
    assert call_count == 2


def test_ollama_transient_http_error_recovery():
    """Client automatically retries and recovers from transient 502/503/504 errors."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(
            200,
            json={"response": '{"claims": []}'},
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=2, transport=transport)

    result = client.generate(prompt="Test prompt", format_type="json")
    assert result == '{"claims": []}'
    assert call_count == 2


def test_ollama_timeout_handling_raises_timeout_error():
    """Client raises OllamaTimeoutError when request times out without indefinite retries."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("Read operation timed out after 60s")

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=2, transport=transport)

    with pytest.raises(OllamaTimeoutError) as exc_info:
        client.generate(prompt="Test prompt", format_type="json")

    assert "timed out" in str(exc_info.value).lower()
    assert call_count == 1


def test_ollama_token_repeat_limit_error_recovery():
    """Client handles 500 repeat error by tightening repeat options and recovering on retry."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.read())
        if call_count == 1:
            return httpx.Response(
                500, text='{"error": "prediction aborted, token repeat limit reached"}'
            )
        assert body["options"]["repeat_penalty"] >= 1.20
        assert body["options"]["num_predict"] <= 512
        return httpx.Response(200, json={"response": '{"claims": []}'})

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=2, transport=transport)

    result = client.generate(prompt="Test prompt", format_type="json")
    assert result == '{"claims": []}'
    assert call_count == 2


def test_ollama_token_repeat_limit_exhaustion():
    """When token repeat limit error persists, client cleanly raises OllamaResponseError."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            500, text='{"error": "prediction aborted, token repeat limit reached"}'
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=1, transport=transport)

    with pytest.raises(OllamaResponseError) as exc_info:
        client.generate(prompt="Test prompt", format_type="json")

    assert "token repeat limit reached" in str(exc_info.value).lower()
    assert call_count == 2


def test_ollama_malformed_structured_output_safely_quarantined():
    """Malformed non-JSON output from the model is safely rejected, never published."""
    client = MockOllamaClient(should_return_malformed=True)
    summarizer = EventSummarizer(client=client)
    event = _make_dummy_event()

    result = summarizer.summarize_event(event)

    assert result.status in {"failed", "quarantined", "error"}
    assert result.summary is None or result.quarantine_reason is not None
    assert "json" in (result.error_message or "").lower() or result.quarantine_reason is not None


def test_ollama_retry_exhaustion_on_persistent_connection_reset():
    """Persistent connection drops exhaust retries and cleanly raise OllamaUnavailableError."""
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.RemoteProtocolError("Server disconnected without sending a response")

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", max_retries=2, transport=transport)

    with pytest.raises(OllamaUnavailableError) as exc_info:
        client.generate(prompt="Test prompt", format_type="json")

    assert "connection reset" in str(exc_info.value).lower()
    assert call_count == 3


def test_ollama_ambiguous_failure_no_duplicate_publication():
    """When summarization encounters an ambiguous error, the event is not published."""
    client = MockOllamaClient(should_timeout=True)
    summarizer = EventSummarizer(client=client)
    event = _make_dummy_event()

    result = summarizer.summarize_event(event)
    assert result.status != "success"
    # A failed summarization result without a valid summary cannot be published
    assert result.summary is None


def test_ollama_stage_specific_output_budgets():
    """Verify that each inference stage specifies its stage-tuned token budget."""
    from geopulse.summarization.client import (
        STAGE_BUDGET_CLAIMS,
        STAGE_BUDGET_CORRECTION,
        STAGE_BUDGET_QA,
        STAGE_BUDGET_VERIFIER,
    )

    # 1. Check budget constants are aligned with CPU hardware profile (8.28 tok/s)
    assert STAGE_BUDGET_CLAIMS == 320
    assert STAGE_BUDGET_VERIFIER == 160
    assert STAGE_BUDGET_CORRECTION == 320
    assert STAGE_BUDGET_QA == 256

    # 2. Check EventSummarizer passes budgets to model calls
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "evidence_refs": [
                        {
                            "source_url": "https://defensenews.com/baltic-drill",
                            "evidence_span": "live air-policing drills over the Baltic Sea",
                        }
                    ],
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "source_url": "https://defensenews.com/baltic-drill",
                    "verdict": "supported",
                    "reason": "exact match",
                }
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert len(client.calls) == 2

    # Call 0: Initial claim extraction
    assert client.calls[0]["options"]["num_predict"] == STAGE_BUDGET_CLAIMS
    assert client.calls[0]["options"]["temperature"] == 0.1

    # Call 1: Entailment verifier
    assert client.calls[1]["options"]["num_predict"] == STAGE_BUDGET_VERIFIER
    assert client.calls[1]["options"]["temperature"] == 0.0


def test_ollama_telemetry_and_token_diagnostics():
    """Client captures and reports token counts, rates, and done_reason in last_metrics."""
    captured_payload = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "response": '{"claims": []}',
                "prompt_eval_count": 128,
                "eval_count": 64,
                "eval_duration": 7_730_000_000,  # 7.73s (~8.28 tok/s)
                "done_reason": "stop",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", transport=transport)

    resp = client.generate(prompt="Test prompt", format_type="json")
    assert resp == '{"claims": []}'

    metrics = client.last_metrics
    assert metrics["prompt_tokens"] == 128
    assert metrics["eval_tokens"] == 64
    assert 8.2 <= metrics["eval_tokens_per_second"] <= 8.4
    assert metrics["done_reason"] == "stop"


def test_ollama_truncated_response_detection_warning(caplog):
    """Client logs a warning when done_reason indicates truncation (hit num_predict)."""
    import logging

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": '{"claims": [{"text": "truncated claim text',
                "prompt_eval_count": 250,
                "eval_count": 384,
                "eval_duration": 46_000_000_000,
                "done_reason": "length",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", transport=transport)

    with caplog.at_level(logging.WARNING):
        client.generate(prompt="Test prompt", format_type="json")

    assert client.last_metrics["done_reason"] == "length"
    assert any("truncated" in record.message.lower() for record in caplog.records)


def test_verifier_compact_response_parsing_without_source_url():
    """Verifier parses compact responses where source_url was omitted by the model."""
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "evidence_refs": [
                        {
                            "source_url": "https://defensenews.com/baltic-drill",
                            "evidence_span": "live air-policing drills over the Baltic Sea",
                        }
                    ],
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )
    # Compact verifier verdict without source_url:
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reason": "direct match",
                }
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert len(res.summary.claims) == 1
    assert res.summary.claims[0].verification_status == "accepted"


def test_sequential_model_calls_are_bounded():
    """Verify that an event summarization executes at most 2 model calls on success."""
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "evidence_refs": [
                        {
                            "source_url": "https://defensenews.com/baltic-drill",
                            "evidence_span": "live air-policing drills over the Baltic Sea",
                        }
                    ],
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reason": "direct match",
                }
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert summarizer.model_calls_count == 2
    assert len(client.calls) == 2


def test_compact_response_parsed_correctly_with_source_index():
    """Verify that compact schema using integer 'src' resolves to exact article URL."""
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "src": 1,
                    "evidence": "live air-policing drills over the Baltic Sea",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "reason": "direct match",
                }
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert len(res.summary.claims) == 1
    c1 = res.summary.claims[0]
    assert c1.central is True
    assert c1.evidence_refs[0].source_url == event.articles[0].url
    assert c1.evidence_refs[0].evidence_span == "live air-policing drills over the Baltic Sea"


def test_compact_response_multi_source_index_resolution():
    """Verify that multi-source articles resolve accurately via integer indices 'src': 1, 2."""
    now = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Defense News",
        title="Allied air forces complete Baltic interception exercise",
        url="https://defensenews.com/baltic-drill",
        canonical_url="https://defensenews.com/baltic-drill",
        published_at=now,
        snippet="NATO allied fighters carried out live air-policing drills over the Baltic Sea.",
        content_hash="hash-baltic-01",
    )
    art2 = ArticleRecord(
        source_name="Reuters",
        title="Baltic drill confirms readiness under jamming",
        url="https://reuters.com/baltic-drill-readiness",
        canonical_url="https://reuters.com/baltic-drill-readiness",
        published_at=now,
        snippet="Officials confirmed satellite navigation remained resilient during the drill.",
        content_hash="hash-baltic-02",
    )
    event = EventCluster(
        id="evt-baltic-multi",
        title=art1.title,
        category=TopicCategory.MILITARY_OPERATIONS,
        region="baltic",
        first_seen_at=now,
        last_seen_at=now,
        published_at=now,
        is_analysis=False,
        confidence_label="corroborated",
        articles=(art1, art2),
    )

    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "src": 1,
                    "evidence": "live air-policing drills over the Baltic Sea",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                },
                {
                    "claim_id": "c2",
                    "text": "Officials confirmed satellite navigation remained resilient.",
                    "src": 2,
                    "evidence": "satellite navigation remained resilient during the drill",
                    "classification": "attributed_statement",
                    "attribution": "Officials",
                    "central": False,
                },
            ]
        }
    )
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {"claim_id": "c1", "verdict": "supported", "reason": "matches source 1"},
                {"claim_id": "c2", "verdict": "supported", "reason": "matches source 2"},
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert len(res.summary.claims) == 2
    assert res.summary.claims[0].evidence_refs[0].source_url == art1.url
    assert res.summary.claims[1].evidence_refs[0].source_url == art2.url


def test_compact_response_invalid_source_index_rejected():
    """Verify that an out-of-bounds source index (e.g. src=99) is rejected and quarantined."""
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "src": 99,
                    "evidence": "live air-policing drills over the Baltic Sea",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )

    client = MockOllamaClient(canned_response=canned_claims)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "quarantined"
    assert res.summary is None
    assert "invalid source index 99" in (res.error_message or "").lower()


def test_secondary_claim_failure_does_not_destroy_valid_central_claim():
    """When a secondary claim is unsupported, it is dropped and the valid central claim survives."""
    event = _make_dummy_event()
    canned_claims = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "NATO allied fighters completed live air-policing drills.",
                    "src": 1,
                    "evidence": "live air-policing drills over the Baltic Sea",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                },
                {
                    "claim_id": "c2",
                    "text": "The exercise included ungrounded hypothetical carrier strikes.",
                    "src": 1,
                    "evidence": "drills over the Baltic Sea",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": False,
                },
            ]
        }
    )
    # Verifier rejects secondary claim c2:
    canned_verdict = json.dumps(
        {
            "verdicts": [
                {"claim_id": "c1", "verdict": "supported", "reason": "exact match"},
                {"claim_id": "c2", "verdict": "unsupported", "reason": "carrier not mentioned"},
            ]
        }
    )

    def mock_gen(prompt: str) -> str:
        if "conservative source-entailment verifier" in prompt or "entailed" in prompt:
            return canned_verdict
        return canned_claims

    client = MockOllamaClient(custom_generator=mock_gen)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert len(res.summary.claims) == 1
    assert res.summary.claims[0].claim_id == "c1"
    assert "carrier" not in res.summary.what_happened.lower()


def test_output_budget_exhaustion_quarantined_and_not_published():
    """When output generation hits token budget and is truncated, event is quarantined."""
    event = _make_dummy_event()

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "response": '{"claims": [{"claim_id": "c1", "text": "NATO fighters completed',
                "prompt_eval_count": 200,
                "eval_count": 320,
                "eval_duration": 38_600_000_000,
                "done_reason": "length",
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = OllamaClient(base_url="http://localhost:11434", transport=transport)
    summarizer = EventSummarizer(client=client)
    res = summarizer.summarize_event(event)

    assert res.status == "quarantined"
    assert res.summary is None
    assert "truncated" in (res.quarantine_reason or "").lower()


def test_compact_schema_token_budgets_and_size_bounds():
    """Verify that compact schema payloads fit comfortably within STAGE_BUDGET_CLAIMS (320)."""
    from geopulse.summarization.client import STAGE_BUDGET_CLAIMS

    # 1. Single central claim
    single_claim_payload = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": (
                        "Space Force general urged allied collaboration to "
                        "counter daily GPS jamming."
                    ),
                    "src": 1,
                    "evidence": "urged allied collaboration to counter daily jamming",
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                }
            ]
        }
    )
    # LLaMA token approximation: ~3.8 chars per token
    approx_tokens_single = len(single_claim_payload) / 3.8
    assert approx_tokens_single < 100, f"Expected <100 tokens, got {approx_tokens_single:.1f}"

    # 2. Central + supporting claim with attribution
    two_claims_payload = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": (
                        "Space Force general warned satellite capabilities face daily "
                        "electronic jamming."
                    ),
                    "src": 1,
                    "evidence": (
                        "warned that daily electronic jamming threatens "
                        "satellite-dependent capabilities"
                    ),
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                },
                {
                    "claim_id": "c2",
                    "text": (
                        "General Brunson urged allied collaboration to sustain "
                        "communications under degraded conditions."
                    ),
                    "src": 1,
                    "evidence": "urged allied collaboration to maintain communications",
                    "classification": "attributed_statement",
                    "attribution": "General Xavier Brunson",
                    "central": False,
                },
            ]
        }
    )
    approx_tokens_two = len(two_claims_payload) / 3.8
    assert approx_tokens_two < 200, f"Expected <200 tokens, got {approx_tokens_two:.1f}"

    # 3. Two claims with long 25-word spans
    long_claims_payload = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": (
                        "The defense ministry officially approved the procurement and "
                        "delivery of advanced radar systems."
                    ),
                    "src": 1,
                    "evidence": (
                        "defense ministry officially approved the multi-year procurement and "
                        "delivery of sovereign advanced radar surveillance systems across all "
                        "northern territorial commands"
                    ),
                    "classification": "reported_fact",
                    "attribution": None,
                    "central": True,
                },
                {
                    "claim_id": "c2",
                    "text": (
                        "Authorities stated that sovereign production capabilities would remain "
                        "fully resilient under sanctions."
                    ),
                    "src": 2,
                    "evidence": (
                        "authorities stated sovereign production capabilities and critical "
                        "microelectronic components would remain fully resilient against "
                        "potential international export restrictions or sanctions"
                    ),
                    "classification": "attributed_statement",
                    "attribution": "Authorities",
                    "central": False,
                },
            ]
        }
    )
    approx_tokens_long = len(long_claims_payload) / 3.8
    assert approx_tokens_long < 280, f"Expected <280 tokens, got {approx_tokens_long:.1f}"
    assert approx_tokens_long < STAGE_BUDGET_CLAIMS
