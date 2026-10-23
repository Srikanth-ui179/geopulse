"""Typed application configuration."""

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

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    telegram_api_base_url: str = "https://api.telegram.org"
    telegram_timeout: float = 15.0
    delivery_state_file: Path = Path(".geopulse/delivery_state.json")


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

    @field_validator("supabase_key", "telegram_bot_token", mode="before")
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

    @field_validator("telegram_chat_id", mode="before")
    @classmethod
    def validate_telegram_chat_id(cls, value: object) -> str | None:
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
        return stripped

    @property
    def is_supabase_configured(self) -> bool:
        """Check whether valid Supabase configuration is present."""
        return bool(
            self.supabase_url
            and self.supabase_key
            and self.supabase_key.get_secret_value().strip()
        )

    @property
    def is_telegram_configured(self) -> bool:
        """Check whether valid Telegram delivery configuration is present."""
        return bool(
            self.telegram_bot_token
            and self.telegram_bot_token.get_secret_value().strip()
            and self.telegram_chat_id
            and self.telegram_chat_id.strip()
        )

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
