import pytest
from pydantic import ValidationError

from geopulse.config import Settings, get_settings


def test_defaults_load_without_external_secrets(monkeypatch):
    for key in (
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.timezone == "Asia/Kolkata"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.supabase_url is None
    assert settings.supabase_key is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_chat_id is None
    assert settings.is_supabase_configured is False
    assert settings.is_telegram_configured is False


def test_invalid_supabase_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, supabase_url="not-a-url")


def test_invalid_ollama_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url="localhost:11434")


def test_empty_ollama_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url="")


def test_supabase_require_gives_helpful_error_when_missing():
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY"):
        settings.require_supabase()


def test_supabase_credentials_are_returned_only_on_explicit_request():
    settings = Settings(
        _env_file=None,
        supabase_url="https://example.supabase.co/",
        supabase_key="test-secret",
    )

    url, key = settings.require_supabase()

    assert url == "https://example.supabase.co"
    assert key == "test-secret"
    assert settings.is_supabase_configured is True


def test_empty_secrets_and_chat_id_normalize_to_none():
    settings = Settings(
        _env_file=None,
        supabase_url="",
        supabase_key="   ",
        telegram_bot_token="   ",
        telegram_chat_id="   ",
    )

    assert settings.supabase_url is None
    assert settings.supabase_key is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_chat_id is None
    assert settings.is_supabase_configured is False
    assert settings.is_telegram_configured is False


def test_telegram_require_when_configured():
    settings = Settings(
        _env_file=None,
        telegram_bot_token="bot123456:ABC-DEF",
        telegram_chat_id="123456789",
    )

    token, chat_id = settings.require_telegram()
    assert token == "bot123456:ABC-DEF"
    assert chat_id == "123456789"
    assert settings.is_telegram_configured is True


def test_telegram_require_when_missing():
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="Telegram is not configured"):
        settings.require_telegram()


def test_invalid_app_env_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="staging")  # type: ignore[arg-type]


def test_empty_timezone_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, timezone="   ")


def test_get_settings_cached():
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

