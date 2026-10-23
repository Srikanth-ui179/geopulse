"""Unit tests for Phase 5 Telegram Bot API client and test doubles."""

import httpx
import pytest

from geopulse.delivery.client import MockTelegramClient, TelegramClient
from geopulse.delivery.exceptions import (
    TelegramConfigError,
    TelegramPermanentError,
    TelegramRateLimitError,
    TelegramTimeoutError,
)


def test_telegram_client_requires_non_empty_credentials():
    with pytest.raises(TelegramConfigError, match="token must not be empty"):
        TelegramClient(token="", chat_id="12345")

    with pytest.raises(TelegramConfigError, match="chat ID must not be empty"):
        TelegramClient(token="secret-token", chat_id="   ")


def test_telegram_client_redacts_token_in_repr():
    client = TelegramClient(token="my-super-secret-token", chat_id="12345678")
    repr_str = repr(client)
    assert "my-super-secret-token" not in repr_str
    assert "[REDACTED]" in repr_str
    assert "12345678" in repr_str


def test_telegram_client_send_message_success(monkeypatch):
    recorded_request = {}

    def mock_post(url, json=None, timeout=None):
        recorded_request["url"] = url
        recorded_request["json"] = json
        mock_response = httpx.Response(
            status_code=200,
            json={
                "ok": True,
                "result": {
                    "message_id": 9991,
                    "date": 1729680000,
                    "chat": {"id": 12345678, "type": "private"},
                    "text": "Hello, Geopulse",
                },
            },
            request=httpx.Request("POST", url),
        )
        return mock_response

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    client = TelegramClient(
        token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        chat_id="12345678",
    )
    resp = client.send_message("<b>Briefing</b>", parse_mode="HTML")

    assert resp.message_id == 9991
    assert resp.chat_id == 12345678
    expected_endpoint = "/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/sendMessage"
    assert recorded_request["url"].endswith(expected_endpoint)
    assert recorded_request["json"]["chat_id"] == "12345678"
    assert recorded_request["json"]["text"] == "<b>Briefing</b>"
    assert recorded_request["json"]["parse_mode"] == "HTML"
    assert recorded_request["json"]["disable_web_page_preview"] is True


def test_telegram_client_handles_429_rate_limit(monkeypatch):
    def mock_post(url, **kwargs):
        return httpx.Response(
            status_code=429,
            json={
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests: retry after 5",
                "parameters": {"retry_after": 5},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    client = TelegramClient(token="test-token", chat_id="12345")
    with pytest.raises(TelegramRateLimitError) as exc_info:
        client.send_message("test")

    assert exc_info.value.retry_after == 5.0
    assert "Too Many Requests" in str(exc_info.value)


def test_telegram_client_handles_permanent_errors(monkeypatch):
    # 400 Bad Request
    def mock_bad_request(url, **kwargs):
        return httpx.Response(
            status_code=400,
            json={"ok": False, "error_code": 400, "description": "Bad Request: chat not found"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: mock_bad_request(url, **kwargs)
    )
    client = TelegramClient(token="test-token", chat_id="invalid-chat")
    with pytest.raises(TelegramPermanentError) as exc_info:
        client.send_message("test")
    assert exc_info.value.status_code == 400
    assert "chat not found" in exc_info.value.description


def test_telegram_client_handles_network_timeout(monkeypatch):
    def mock_read_timeout(url, **kwargs):
        raise httpx.ReadTimeout("Read timed out")

    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: mock_read_timeout(url, **kwargs)
    )
    client = TelegramClient(token="test-token", chat_id="12345")
    with pytest.raises(TelegramTimeoutError) as exc_info:
        client.send_message("test")
    # Read timeout is ambiguous because the message may have been delivered
    assert exc_info.value.is_ambiguous is True


def test_mock_telegram_client_records_messages():
    mock = MockTelegramClient(chat_id="test-chat")
    resp = mock.send_message("Message 1")
    assert resp.message_id > 0
    assert resp.chat_id == "test-chat"
    assert len(mock.sent_messages) == 1
    assert mock.sent_messages[0]["text"] == "Message 1"


def test_mock_telegram_client_simulates_transient_failures():
    # Fails first 2 calls with rate limit, succeeds on call 3
    mock = MockTelegramClient(
        fail_first_n_calls=2,
        should_fail_rate_limit=1.5,
    )

    with pytest.raises(TelegramRateLimitError) as exc1:
        mock.send_message("Attempt 1")
    assert exc1.value.retry_after == 1.5

    with pytest.raises(TelegramRateLimitError) as exc2:
        mock.send_message("Attempt 2")
    assert exc2.value.retry_after == 1.5

    # 3rd attempt succeeds
    resp = mock.send_message("Attempt 3")
    assert resp.message_id > 0
    assert len(mock.sent_messages) == 1
