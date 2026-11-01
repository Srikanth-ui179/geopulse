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


def test_classify_supabase_key_categories():
    import base64
    import json

    from geopulse.config import classify_supabase_key

    def _make_jwt(role: str) -> str:
        h = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode().rstrip("=")
        p = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
        return f"{h}.{p}.sig"

    assert classify_supabase_key("sb_publishable_abc123") == "publishable/anon"
    assert classify_supabase_key("sb_secret_xyz789") == "newer secret key"
    assert classify_supabase_key(_make_jwt("service_role")) == "legacy service_role"
    assert classify_supabase_key(_make_jwt("anon")) == "publishable/anon"
    assert classify_supabase_key("generic-opaque-token") == "unknown"
    assert classify_supabase_key(None) == "unknown"


def test_settings_supabase_key_category():
    s_anon = Settings(_env_file=None, supabase_key="sb_publishable_123456")
    assert s_anon.supabase_key_category == "publishable/anon"

    s_sec = Settings(_env_file=None, supabase_key="sb_secret_123456")
    assert s_sec.supabase_key_category == "newer secret key"

    s_none = Settings(_env_file=None, supabase_key=None)
    assert s_none.supabase_key_category == "unknown"


def test_feed_limit_default_and_validation(monkeypatch):
    monkeypatch.delenv("FEED_LIMIT", raising=False)
    s = Settings(_env_file=None)
    assert s.feed_limit == 10

    s_custom = Settings(_env_file=None, feed_limit=25)
    assert s_custom.feed_limit == 25

    with pytest.raises(ValidationError):
        Settings(_env_file=None, feed_limit=0)


def test_gemini_defaults_and_provider_settings(monkeypatch):
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)

    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.gemini_api_key is None
    assert s.gemini_model == "gemini-3.8-flash"
    assert s.gemini_base_url == "https://generativelanguage.googleapis.com"
    assert s.is_gemini_configured is False


def test_gemini_require_when_configured():
    s = Settings(
        _env_file=None,
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-3.8-flash",
    )
    assert s.is_gemini_configured is True
    key, model = s.require_gemini()
    assert key == "test-gemini-key"
    assert model == "gemini-3.8-flash"
    assert "test-gemini-key" not in repr(s)


def test_gemini_require_when_missing():
    s = Settings(_env_file=None)
    with pytest.raises(ValueError, match="Gemini is not configured"):
        s.require_gemini()


def test_empty_gemini_api_key_normalizes_to_none():
    s = Settings(_env_file=None, gemini_api_key="   ")
    assert s.gemini_api_key is None
    assert s.is_gemini_configured is False


def test_invalid_gemini_base_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gemini_base_url="not-a-url")


def test_invalid_llm_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="openai")


def test_valid_llm_providers_accepted():
    for p in ("ollama", "gemini", "auto", "OLLAMA", "Gemini", "AUTO"):
        s = Settings(_env_file=None, llm_provider=p)
        assert s.llm_provider == p.lower()
