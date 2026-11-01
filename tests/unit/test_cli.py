import pytest

from geopulse.cli import main


def test_check_config_command_succeeds(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["geopulse", "check-config"])
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    result = main()

    output = capsys.readouterr().out
    assert result == 0
    assert "Configuration loaded successfully." in output
    assert "Supabase configured: False" in output
    assert "Telegram configured: False" in output


def test_check_config_reports_configured_services(monkeypatch, capsys):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb-secret-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987654321")

    result = main(["check-config"])

    output = capsys.readouterr().out
    assert result == 0
    assert "Supabase configured: True" in output
    assert "Telegram configured: True" in output
    # Verify secrets are NOT printed in output
    assert "sb-secret-key" not in output
    assert "123456:ABC" not in output


def test_check_config_fails_gracefully_on_invalid_config(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_BASE_URL", "not-a-valid-url")

    result = main(["check-config"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Configuration is invalid:" in captured.out


def test_unrecognized_argument_exits():
    with pytest.raises(SystemExit):
        main(["unknown-command"])


def test_check_preflight_command_success(monkeypatch, capsys):
    from geopulse.preflight import PreflightResult

    def mock_preflight(*args, **kwargs):
        return PreflightResult(success=True, details="All schema objects verified.")

    monkeypatch.setattr("geopulse.cli.run_supabase_preflight", mock_preflight)
    result = main(["check-preflight"])
    captured = capsys.readouterr().out
    assert result == 0
    assert "Preflight check passed." in captured
    assert "All schema objects verified." in captured


def test_check_preflight_command_failure(monkeypatch, capsys):
    from geopulse.preflight import PreflightResult

    def mock_preflight(*args, **kwargs):
        return PreflightResult(
            success=False,
            error_message="Missing function: upsert_events_with_articles",
            details="Apply migration 006.",
        )

    monkeypatch.setattr("geopulse.cli.run_supabase_preflight", mock_preflight)
    result = main(["check-preflight"])
    captured = capsys.readouterr().out
    assert result == 1
    assert "Preflight check failed:" in captured
    assert "Missing function: upsert_events_with_articles" in captured
    assert "Apply migration 006." in captured
