"""Typed application configuration."""

import base64
import json
from datetime import UTC, timedelta, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    timezone: str = "Asia/Kolkata"

    supabase_url: str | None = None
    supabase_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 120.0
    ollama_request_timeout: float = 180.0
    ollama_max_retries: int = 1

    llm_provider: Literal["ollama", "gemini", "auto"] = "ollama"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout: float = 30.0
    gemini_max_retries: int = 2

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    telegram_admin_id: str | None = None
    telegram_api_base_url: str = "https://api.telegram.org"
    telegram_timeout: float = 15.0
    delivery_state_file: Path = Path(".geopulse/delivery_state.json")
    feed_limit: int = 25
    lookback_hours: int | None = None
    min_relevance_rank: float | None = 9.0
    include_low_priority: bool = False
    candidate_cap: int = 12

    # Phase 1: Local NLI claim verification settings
    nli_verification_enabled: bool = False
    nli_model_dir: Path = Path(".geopulse/models/nli-deberta-v3-small")
    nli_quantized: bool = True
    nli_entailment_threshold: float = 0.60
    nli_contradiction_threshold: float = 0.50
    nli_batch_size: int = 16
    nli_intra_threads: int = 4

    # Phase 2: Article Revision History & Retention Settings
    retention_policy: Literal["full_text", "evidence_snippets_only", "metadata_only"] = "full_text"
    retention_ttl_days: int = 90
    article_revisions_enabled: bool = True

    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("SUPABASE_URL must be a valid HTTP(S) URL")
        return value.strip().rstrip("/")

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("OLLAMA_BASE_URL must not be empty")
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OLLAMA_BASE_URL must be a valid HTTP(S) URL")
        return value.strip().rstrip("/")

    @field_validator("gemini_base_url")
    @classmethod
    def validate_gemini_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("GEMINI_BASE_URL must not be empty")
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GEMINI_BASE_URL must be a valid HTTP(S) URL")
        return value.strip().rstrip("/")

    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, value: str) -> str:
        clean = (value or "").strip()
        if not clean:
            raise ValueError("GEMINI_MODEL must not be empty")
        return clean

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_llm_provider(cls, value: object) -> str:
        clean = str(value or "").strip().lower()
        if clean not in {"ollama", "gemini", "auto"}:
            raise ValueError(
                f"LLM_PROVIDER must be one of 'ollama', 'gemini', or 'auto', got '{value}'"
            )
        return clean

    @field_validator("supabase_key", "telegram_bot_token", "gemini_api_key", mode="before")
    @classmethod
    def validate_secret_empty(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value

    @field_validator("telegram_chat_id", "telegram_admin_id", mode="before")
    @classmethod
    def validate_telegram_id_field(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return str(value) if value else None

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("TIMEZONE must not be empty")
        resolve_timezone(stripped)
        return stripped

    @field_validator("feed_limit")
    @classmethod
    def validate_feed_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("FEED_LIMIT must be at least 1")
        return value

    @field_validator("candidate_cap")
    @classmethod
    def validate_candidate_cap(cls, value: int) -> int:
        if value < 1:
            raise ValueError("CANDIDATE_CAP must be at least 1")
        return value

    @field_validator("ollama_request_timeout")
    @classmethod
    def validate_ollama_request_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("OLLAMA_REQUEST_TIMEOUT must be greater than 0")
        return float(value)

    @field_validator("ollama_max_retries")
    @classmethod
    def validate_ollama_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("OLLAMA_MAX_RETRIES must be non-negative")
        return value

    def is_admin(self, user_id: str | int | None) -> bool:
        """Check whether the given Telegram user ID matches the configured administrator ID."""
        if not self.telegram_admin_id or user_id is None:
            return False
        return str(user_id).strip() == self.telegram_admin_id.strip()

    @property
    def tz(self) -> tzinfo:
        """Return the resolved timezone object for the configured application timezone."""
        return resolve_timezone(self.timezone)

    @property
    def is_supabase_configured(self) -> bool:
        """Check whether valid Supabase configuration is present."""
        return bool(
            self.supabase_url and self.supabase_key and self.supabase_key.get_secret_value().strip()
        )

    @property
    def supabase_key_category(self) -> str:
        """Return the safe category of the configured Supabase key."""
        if not self.supabase_key:
            return "unknown"
        return classify_supabase_key(self.supabase_key.get_secret_value())

    @property
    def is_telegram_configured(self) -> bool:
        """Check whether valid Telegram delivery configuration is present."""
        return bool(
            self.telegram_bot_token
            and self.telegram_bot_token.get_secret_value().strip()
            and self.telegram_chat_id
            and self.telegram_chat_id.strip()
        )

    @property
    def is_gemini_configured(self) -> bool:
        """Check whether valid Gemini API configuration is present."""
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value().strip())

    def require_gemini(self) -> tuple[str, str]:
        """Return Gemini API key and model or raise a helpful error when not configured."""
        if not self.gemini_api_key:
            raise ValueError("Gemini is not configured. Set GEMINI_API_KEY in .env.")
        key = self.gemini_api_key.get_secret_value().strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is empty.")
        return key, self.gemini_model

    def require_supabase(self) -> tuple[str, str]:
        """Return database settings or raise a helpful error when not configured."""
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in .env."
            )
        key = self.supabase_key.get_secret_value().strip()
        if not key:
            raise ValueError("SUPABASE_KEY is empty.")
        return self.supabase_url, key

    def require_telegram(self) -> tuple[str, str]:
        """Return Telegram settings or raise a helpful error when not configured."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            raise ValueError(
                "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env."
            )
        token = self.telegram_bot_token.get_secret_value().strip()
        chat_id = self.telegram_chat_id.strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is empty.")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is empty.")
        return token, chat_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def classify_supabase_key(key: str | None) -> str:
    """Safely categorize Supabase API key type without exposing any sensitive material."""
    if not key or not key.strip():
        return "unknown"
    stripped = key.strip()
    if stripped.startswith("sb_publishable_"):
        return "publishable/anon"
    if stripped.startswith("sb_secret_"):
        return "newer secret key"
    if stripped.count(".") == 2:
        try:
            payload_b64 = stripped.split(".")[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes.decode("utf-8", errors="ignore"))
            role = payload.get("role")
            if role == "service_role":
                return "legacy service_role"
            if role == "anon":
                return "publishable/anon"
        except Exception:
            return "unknown"
    return "unknown"


def resolve_timezone(tz_name: str | None = None) -> tzinfo:
    """Resolve a timezone identifier to a tzinfo instance, resilient on all platforms."""
    clean = (tz_name or "").strip()
    if not clean:
        clean = "Asia/Kolkata"

    try:
        import zoneinfo

        return zoneinfo.ZoneInfo(clean)
    except Exception:
        pass

    lower = clean.lower()
    if lower in {"asia/kolkata", "asia/calcutta", "ist"}:
        return timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
    if lower in {"utc", "gmt", "z"}:
        return UTC

    msg = f"TIMEZONE '{clean}' cannot be resolved to a known timezone."
    raise ValueError(msg)
