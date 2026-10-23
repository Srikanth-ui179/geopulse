"""Unit tests for Phase 5 CLI brief command."""

from geopulse.cli import main


def test_cli_brief_preview_mode_succeeds(capsys):
    exit_code = main(["brief", "--fixture", "--mock-model"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Geopulse Daily Briefing Preview [Telegram HTML Format]" in captured
    assert "PREVIEW COMPLETE: No messages were sent to Telegram." in captured
    assert "Telegram Message Chunk 1/" in captured
    assert "⚔️ ARMED CONFLICT" in captured
    assert "HINT: Pass '--send' with valid credentials to deliver." in captured


def test_cli_brief_send_without_credentials_fails_gracefully(capsys, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    exit_code = main(["brief", "--fixture", "--mock-model", "--send"])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Error: Telegram credentials are not configured" in captured
    assert "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env" in captured


def test_cli_brief_send_with_mock_telegram_succeeds(capsys, tmp_path):
    test_state = str(tmp_path / "state.json")
    exit_code = main([
        "brief",
        "--fixture",
        "--mock-model",
        "--send",
        "--mock-telegram",
        "--force",
        "--state-file",
        test_state,
    ])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Delivering briefing to Telegram" in captured
    assert "Status:           SENT" in captured
    assert "Chunks Delivered:" in captured


def test_cli_brief_unknown_feed_filter_fails(capsys):
    exit_code = main(["brief", "--feed", "NonexistentFeed", "--mock-model"])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "No feed matched 'NonexistentFeed'" in captured


def test_cli_brief_preview_never_instantiates_real_telegram_client(monkeypatch):
    """Confirm preview mode never creates a TelegramClient or attempts network calls."""
    def boom(*args, **kwargs):
        raise AssertionError("TelegramClient was instantiated during preview mode!")

    monkeypatch.setattr("geopulse.cli.TelegramClient", boom)
    exit_code = main(["brief", "--fixture", "--mock-model", "--preview"])
    assert exit_code == 0


def test_cli_brief_mock_telegram_never_instantiates_real_telegram_client(monkeypatch):
    """Confirm --mock-telegram never creates a real TelegramClient or makes network calls."""
    def boom(*args, **kwargs):
        raise AssertionError("TelegramClient was instantiated with --mock-telegram!")

    monkeypatch.setattr("geopulse.cli.TelegramClient", boom)
    exit_code = main([
        "brief",
        "--fixture",
        "--mock-model",
        "--send",
        "--mock-telegram",
        "--force",
    ])
    assert exit_code == 0


def test_cli_brief_deduplication_across_separate_invocations(tmp_path, capsys):
    """Verify that separate CLI invocations detect duplicates via local state file."""
    state_file = str(tmp_path / "cli_state.json")
    cmd_base = [
        "brief",
        "--fixture",
        "--mock-model",
        "--send",
        "--mock-telegram",
        "--state-file",
        state_file,
    ]

    # First run delivers successfully
    exit_code_1 = main(cmd_base)
    captured_1 = capsys.readouterr().out
    assert exit_code_1 == 0
    assert "Status:           SENT" in captured_1
    assert "Chunks Delivered: 2/2" in captured_1

    # Second run in a separate invocation detects duplicate via state_file and skips
    exit_code_2 = main(cmd_base)
    captured_2 = capsys.readouterr().out
    assert exit_code_2 == 0
    assert "Status:           SKIPPED_DUPLICATE" in captured_2
    assert "already been delivered" in captured_2

    # Third run with --force forces re-delivery
    exit_code_3 = main([*cmd_base, "--force"])
    captured_3 = capsys.readouterr().out
    assert exit_code_3 == 0
    assert "Status:           SENT" in captured_3


def test_cli_test_telegram_mock_succeeds(capsys):
    """'geopulse test-telegram --mock' sends 1 mock test message safely without network calls."""
    exit_code = main(["test-telegram", "--mock"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Sending Geopulse connection test message [MOCK]" in captured
    assert "[OK] Test message delivered successfully" in captured
    assert "Telegram Message ID:" in captured


def test_cli_test_telegram_without_credentials_fails(capsys, monkeypatch):
    """'geopulse test-telegram' fails closed when credentials are not configured."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    exit_code = main(["test-telegram"])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Error: Telegram credentials are not configured." in captured
    assert "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env" in captured


def test_cli_test_telegram_offline_simulation_verifies_message_format(capsys, monkeypatch):
    """Confirm test-telegram constructs the exact labelled test message with redaction."""
    sent_payloads = []

    class DummyClient:
        def __init__(self, token, chat_id, **kwargs):
            self.chat_id = chat_id

        def send_message(self, text, parse_mode="HTML", **kwargs):
            sent_payloads.append(text)
            from geopulse.delivery.models import TelegramMessageResponse
            return TelegramMessageResponse(message_id=99999, chat_id=self.chat_id, date=1700000000)

    monkeypatch.setattr("geopulse.cli.TelegramClient", DummyClient)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:FAKE_TOKEN_XYZ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100987654321")

    exit_code = main(["test-telegram"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "[OK] Test message delivered successfully (Telegram Message ID: 99999)" in captured
    assert len(sent_payloads) == 1
    msg = sent_payloads[0]
    assert "<b>🌍 Geopulse Connection Test</b>" in msg
    assert "Telegram Bot API integration verified successfully." in msg
    assert "Target Chat: <code>-100987654321</code>" in msg
    assert "FAKE_TOKEN_XYZ" not in captured
    assert "FAKE_TOKEN_XYZ" not in msg



