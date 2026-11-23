"""Unit tests for Geopulse Gemini API provider, fallback mechanisms, and test doubles."""

import json

import httpx
import pytest
from pydantic import SecretStr

from geopulse.clustering.models import EventCluster
from geopulse.config import Settings
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization import (
    EventSummarizer,
    GeminiAuthError,
    GeminiClient,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    MockGeminiClient,
    MockOllamaClient,
    OllamaClient,
    resolve_summarization_clients,
)


def _create_sample_cluster() -> EventCluster:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    art1 = ArticleRecord(
        source_name="Defense News",
        title="Air Defense Procurement Finalized",
        url="https://defensenews.com/ad-procure",
        canonical_url="https://defensenews.com/ad-procure",
        published_at=now,
        snippet="Air defense procurement agreement signed for modern batteries.",
        content_hash="hash-cluster-1",
    )
    return EventCluster(
        title="Air Defense Procurement Finalized",
        category="weapons_procurement",  # type: ignore[arg-type]
        first_seen_at=now,
        last_seen_at=now,
        confidence_label="single_source",
        articles=(art1,),
    )


def test_gemini_client_initialization_and_secret_redaction():
    client = GeminiClient(api_key="secret-api-key-12345", model="gemini-2.5-flash")
    assert client.model == "gemini-2.5-flash"
    assert client.provider == "gemini"
    assert client.is_available() is True

    # Critical security check: API key must NEVER be printed in __repr__ or str
    repr_str = repr(client)
    assert "[REDACTED]" in repr_str
    assert "secret-api-key-12345" not in repr_str

    # SecretStr input also accepted
    secret_client = GeminiClient(
        api_key=SecretStr("secret-api-key-12345"),
        model="gemini-3.8-flash",
    )
    assert secret_client.model == "gemini-3.8-flash"
    assert "[REDACTED]" in repr(secret_client)

    # Empty key rejected
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY must not be empty"):
        GeminiClient(api_key="")


def test_gemini_client_successful_generation():
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gemini-3.8-flash"
        assert body["input"] == "Summarize the event"
        assert body["system_instruction"] == "You are an intelligence analyst."
        assert "generation_config" in body
        assert body["response_format"] == [{"type": "text", "mime_type": "application/json"}]

        mock_interaction_payload = {
            "id": "int_12345",
            "object": "interaction",
            "model": "gemini-3.8-flash",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "headline": "Air Defense Procurement Signed",
                                    "what_happened": "Officials signed air defense contracts.",
                                    "key_details": {
                                        "actors": ["Defense Ministry"],
                                        "location": None,
                                        "timing": "October 24, 2026",
                                        "action": "Procurement contract signing",
                                    },
                                    "significance": "Enhances defensive airspace protection.",
                                    "confidence_qualification": "Single-source report.",
                                }
                            ),
                        }
                    ],
                }
            ],
        }
        return httpx.Response(200, json=mock_interaction_payload)

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="valid-test-key",
        model="gemini-3.8-flash",
        transport=transport,
    )

    output = client.generate(
        prompt="Summarize the event",
        system="You are an intelligence analyst.",
        format_type="json",
    )

    parsed = json.loads(output)
    assert parsed["headline"] == "Air Defense Procurement Signed"
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.headers["x-goog-api-key"] == "valid-test-key"
    assert "/v1beta/interactions" in str(req.url)


def test_gemini_client_extraction_variants_and_fallbacks():
    # 1. output_text top-level field
    t1 = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"output_text": '{"result": "from_output_text"}'})
    )
    c1 = GeminiClient(api_key="key", transport=t1)
    assert json.loads(c1.generate("p"))["result"] == "from_output_text"

    # 2. outputs legacy array
    t2 = httpx.MockTransport(
        lambda r: httpx.Response(
            200, json={"outputs": [{"type": "text", "text": '{"result": "from_outputs"}'}]}
        )
    )
    c2 = GeminiClient(api_key="key", transport=t2)
    assert json.loads(c2.generate("p"))["result"] == "from_outputs"

    # 3. candidates backward-compatibility array
    t3 = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"result": "from_candidates"}'}]}}]
            },
        )
    )
    c3 = GeminiClient(api_key="key", transport=t3)
    assert json.loads(c3.generate("p"))["result"] == "from_candidates"

    # 4. Missing text output raises GeminiResponseError
    t4 = httpx.MockTransport(lambda r: httpx.Response(200, json={"steps": []}))
    c4 = GeminiClient(api_key="key", transport=t4)
    with pytest.raises(GeminiResponseError, match="missing text output"):
        c4.generate("p")

    # 5. Failed interaction status
    t5 = httpx.MockTransport(
        lambda r: httpx.Response(
            200, json={"status": "failed", "error": {"message": "Content filter blocked"}}
        )
    )
    c5 = GeminiClient(api_key="key", transport=t5)
    with pytest.raises(
        GeminiResponseError, match="Gemini interaction failed: Content filter blocked"
    ):
        c5.generate("p")


def test_gemini_client_authentication_failure():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="invalid-key",
        transport=transport,
        max_retries=2,
    )

    with pytest.raises(GeminiAuthError, match="authentication failed \\(403\\)"):
        client.generate("Test prompt")


def test_gemini_client_rate_limit_quota_exhaustion_no_repeated_retry():
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (check quota).",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="quota-exhausted-key",
        transport=transport,
        max_retries=2,
    )

    with pytest.raises(GeminiQuotaError, match="quota or rate limit exhausted \\(429\\)"):
        client.generate("Test prompt")

    # Critical requirement: Rate limits and quota exhaustion must NOT be repeatedly retried
    assert call_count == 1


def test_gemini_client_model_not_found():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": 404,
                    "message": "models/non-existent-model is not found.",
                    "status": "NOT_FOUND",
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="test-key",
        model="non-existent-model",
        transport=transport,
    )

    with pytest.raises(GeminiModelNotFoundError, match="not found \\(404\\)"):
        client.generate("Test prompt")


def test_gemini_client_transient_error_bounded_retries():
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"headline": "Recovered After Transient Failure"}'}]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="test-key",
        transport=transport,
        max_retries=2,
    )

    output = client.generate("Test prompt")
    assert "Recovered After Transient Failure" in output
    assert attempts == 3


def test_gemini_client_transient_error_exhausts_retries():
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(
        api_key="test-key",
        transport=transport,
        max_retries=2,
    )

    with pytest.raises(GeminiUnavailableError, match="server error \\(500\\) after 2 retries"):
        client.generate("Test prompt")

    assert attempts == 3  # initial + 2 retries


def test_gemini_client_safety_block_raises_response_error():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "SAFETY",
                        "content": {"parts": []},
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test-key", transport=transport)

    with pytest.raises(GeminiResponseError, match="blocked by safety filters"):
        client.generate("Test prompt")


def test_mock_gemini_client_simulation_modes():
    mock_client = MockGeminiClient(model="mock-gemini-2.5-flash")
    assert mock_client.provider == "gemini"
    assert mock_client.model == "mock-gemini-2.5-flash"
    assert mock_client.is_available() is True

    # Standard deterministic fixture generation
    out = mock_client.generate("Clustering Confidence Label: single_source\nSome prompt")
    parsed = json.loads(out)
    assert "headline" in parsed

    # Simulation flags
    auth_fail = MockGeminiClient(should_auth_fail=True)
    with pytest.raises(GeminiAuthError):
        auth_fail.generate("Test")

    quota_exhaust = MockGeminiClient(should_quota_exhaust=True)
    with pytest.raises(GeminiQuotaError):
        quota_exhaust.generate("Test")

    unavail = MockGeminiClient(should_be_unavailable=True)
    with pytest.raises(GeminiUnavailableError):
        unavail.generate("Test")

    timeout = MockGeminiClient(should_timeout=True)
    with pytest.raises(GeminiTimeoutError):
        timeout.generate("Test")

    not_found = MockGeminiClient(should_model_not_found=True)
    with pytest.raises(GeminiModelNotFoundError):
        not_found.generate("Test")


def test_resolve_summarization_clients_matrix():
    # 1. Default: Ollama
    s_default = Settings(_env_file=None, llm_provider="ollama")
    c1, f1 = resolve_summarization_clients(settings=s_default)
    assert isinstance(c1, OllamaClient)
    assert f1 is None

    # Mock Ollama
    c1_m, f1_m = resolve_summarization_clients(settings=s_default, mock=True)
    assert isinstance(c1_m, MockOllamaClient)
    assert f1_m is None

    # 2. Gemini explicitly configured
    s_gemini = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="real-test-key",
        gemini_model="gemini-3.8-flash",
    )
    c2, f2 = resolve_summarization_clients(settings=s_gemini)
    assert isinstance(c2, GeminiClient)
    assert c2.model == "gemini-3.8-flash"
    assert f2 is None

    # Gemini without key raises GeminiConfigError (no silent downgrade)
    s_gemini_nokey = Settings(_env_file=None, llm_provider="gemini", gemini_api_key=None)
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is not configured"):
        resolve_summarization_clients(settings=s_gemini_nokey)

    # 3. Auto mode with key configured: Gemini primary, Ollama fallback
    s_auto_with_key = Settings(
        _env_file=None,
        llm_provider="auto",
        gemini_api_key="auto-test-key",
    )
    c3, f3 = resolve_summarization_clients(settings=s_auto_with_key)
    assert isinstance(c3, GeminiClient)
    assert isinstance(f3, OllamaClient)

    # 4. Auto mode without key: gracefully defaults to Ollama
    s_auto_nokey = Settings(
        _env_file=None,
        llm_provider="auto",
        gemini_api_key=None,
    )
    c4, f4 = resolve_summarization_clients(settings=s_auto_nokey)
    assert isinstance(c4, OllamaClient)
    assert f4 is None


def test_event_summarizer_gemini_success_and_downstream_grounding():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(model="mock-gemini-2.5-flash")

    summarizer = EventSummarizer(client=mock_gemini)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-gemini-2.5-flash"
    assert len(res.summary.headline) >= 5
    assert len(res.summary.what_happened) >= 10
    assert res.summary.key_details is not None
    assert "Single-source" in res.summary.confidence_qualification


def test_event_summarizer_gemini_mode_does_not_fall_back_on_error():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_quota_exhaust=True,
    )

    # In gemini-only mode with extractive fallback disabled (no fallback_client provided)
    summarizer = EventSummarizer(
        client=mock_gemini, fallback_client=None, enable_extractive_fallback=False
    )
    res = summarizer.summarize_event(event)

    # Must fail without falling back to Ollama
    assert res.status == "failed"
    assert res.summary is None
    assert "quota exceeded" in (res.error_message or "").lower()


def test_gemini_quota_error_triggers_extractive_fallback():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_quota_exhaust=True,
    )

    # By default, enable_extractive_fallback is True
    summarizer = EventSummarizer(client=mock_gemini, fallback_client=None)
    res = summarizer.summarize_event(event)

    # Gracefully degrades to deterministic extractive fallback
    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-gemini-2.5-flash:extractive-fallback"
    assert len(res.summary.headline) >= 5
    assert len(res.summary.what_happened) >= 10
    assert len(res.summary.claims) >= 1
    assert res.summary.claims[0].central is True
    assert res.summary.claims[0].verification_status == "accepted"
    assert len(mock_gemini.calls) == 1


def test_gemini_timeout_triggers_extractive_fallback():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_timeout=True,
    )

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=None)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-gemini-2.5-flash:extractive-fallback"
    assert len(res.summary.headline) >= 5
    assert len(res.summary.what_happened) >= 10
    assert len(mock_gemini.calls) == 1


def test_gemini_model_not_found_triggers_extractive_fallback():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_model_not_found=True,
    )

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=None)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-gemini-2.5-flash:extractive-fallback"
    assert len(res.summary.headline) >= 5
    assert len(res.summary.what_happened) >= 10
    assert len(mock_gemini.calls) == 1


def test_gemini_response_error_triggers_extractive_fallback():
    event = _create_sample_cluster()
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_return_malformed=True,
    )

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=None)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-gemini-2.5-flash:extractive-fallback"
    assert len(res.summary.headline) >= 5
    assert len(res.summary.what_happened) >= 10


def test_event_summarizer_auto_mode_falls_back_on_transient_error():
    event = _create_sample_cluster()
    # Gemini client experiences 503 unavailable
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_be_unavailable=True,
    )
    # Ollama fallback client is healthy
    mock_ollama = MockOllamaClient(model="mock-llama3.2:3b")

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=mock_ollama)
    res = summarizer.summarize_event(event)

    # Successfully fell back to Ollama!
    assert res.status == "success"
    assert res.summary is not None
    # Crucial audit test: summary records the fallback model that actually generated it!
    assert res.summary.model_used == "mock-llama3.2:3b"
    assert len(mock_gemini.calls) == 1
    assert len(mock_ollama.calls) == 1


def test_event_summarizer_auto_mode_falls_back_on_quota_exhaustion():
    event = _create_sample_cluster()
    # Gemini client experiences 429 quota exhaustion
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_quota_exhaust=True,
    )
    mock_ollama = MockOllamaClient(model="mock-llama3.2:3b")

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=mock_ollama)
    res = summarizer.summarize_event(event)

    assert res.status == "success"
    assert res.summary is not None
    assert res.summary.model_used == "mock-llama3.2:3b"


def test_event_summarizer_auto_mode_does_not_fall_back_on_auth_failure():
    event = _create_sample_cluster()
    # Authentication failure is a non-transient configuration error
    mock_gemini = MockGeminiClient(
        model="mock-gemini-2.5-flash",
        should_auth_fail=True,
    )
    mock_ollama = MockOllamaClient(model="mock-llama3.2:3b")

    summarizer = EventSummarizer(client=mock_gemini, fallback_client=mock_ollama)
    res = summarizer.summarize_event(event)

    # Must fail immediately so user is notified of invalid credentials; must NOT call Ollama
    assert res.status == "failed"
    assert "key invalid" in (res.error_message or "").lower()
    assert len(mock_gemini.calls) == 1
    assert len(mock_ollama.calls) == 0


def test_event_summarizer_timeout_fallback_and_explicit_gemini_behavior():
    event = _create_sample_cluster()

    # Part 1: auto mode (with fallback_client) falls back to Ollama on GeminiTimeoutError
    mock_gemini_auto = MockGeminiClient(
        model="mock-gemini-3.8-flash",
        should_timeout=True,
    )
    mock_ollama_fallback = MockOllamaClient(model="mock-llama3.2:3b")

    auto_summarizer = EventSummarizer(
        client=mock_gemini_auto,
        fallback_client=mock_ollama_fallback,
    )
    res_auto = auto_summarizer.summarize_event(event)

    assert res_auto.status == "success"
    assert res_auto.summary is not None
    assert res_auto.summary.model_used == "mock-llama3.2:3b"
    assert len(mock_gemini_auto.calls) == 1
    assert len(mock_ollama_fallback.calls) == 1

    # Part 2: explicit Gemini mode with extractive fallback disabled
    # fails closed on timeout without Ollama
    mock_gemini_explicit = MockGeminiClient(
        model="mock-gemini-3.8-flash",
        should_timeout=True,
    )
    explicit_summarizer = EventSummarizer(
        client=mock_gemini_explicit,
        fallback_client=None,
        enable_extractive_fallback=False,
    )
    res_explicit = explicit_summarizer.summarize_event(event)

    assert res_explicit.status == "failed"
    assert res_explicit.summary is None
    assert "timed out" in (res_explicit.error_message or "").lower()
    assert len(mock_gemini_explicit.calls) == 1


def test_briefing_pipeline_auto_provider_gemini_timeout_fallback_produces_valid_digest():
    from datetime import date

    from geopulse.clustering.repository import InMemoryEventRepository
    from geopulse.delivery.formatter import TelegramDigestFormatter
    from geopulse.ingestion.repository import InMemoryArticleRepository
    from geopulse.pipeline import BriefingPipeline

    event = _create_sample_cluster()
    article = event.articles[0]

    mock_gemini = MockGeminiClient(
        model="mock-gemini-3.8-flash",
        should_timeout=True,
    )
    mock_ollama = MockOllamaClient(model="mock-llama3.2:3b")

    summarizer = EventSummarizer(
        client=mock_gemini,
        fallback_client=mock_ollama,
    )

    pipeline = BriefingPipeline(
        article_repo=InMemoryArticleRepository(),
        event_repo=InMemoryEventRepository(),
        summarizer=summarizer,
        formatter=TelegramDigestFormatter(),
    )

    result = pipeline.run(
        briefing_date=date(2026, 10, 24),
        articles=[article],
        persistent=False,
        send=False,
    )

    assert result.is_preview is True
    assert result.events_count == 1
    assert len(result.summaries) == 1
    summary_res = result.summaries[0]
    assert summary_res.status == "success"
    assert summary_res.summary is not None
    assert summary_res.summary.model_used == "mock-llama3.2:3b"
    assert len(result.digest_chunks) >= 1
    assert "Air Defense Procurement Finalized" in result.digest_chunks[0]
    assert len(mock_gemini.calls) == 1
    assert len(mock_ollama.calls) == 1
