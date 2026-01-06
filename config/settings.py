"""
Configuration management using Pydantic Settings.
Loads and validates environment variables from .env file.
"""

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIKeysSettings(BaseSettings):
    """API keys for LLM providers."""

    deepseek_api_key: str = Field(..., description="DeepSeek API key")
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    openai_api_key: str = Field(..., description="OpenAI API key")
    anthropic_api_key: str = Field(..., description="Anthropic API key")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    # Model names
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek model name")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", description="DeepSeek API base URL")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name")
    openai_model: str = Field(default="gpt-4-turbo", description="OpenAI model name")
    anthropic_model: str = Field(default="claude-sonnet-4.5-20250929", description="Anthropic model name")

    # Cascade configuration
    cascade_order: str = Field(default="deepseek,gemini,gpt5,claude", description="Order of LLM cascade")
    max_retries: int = Field(default=3, description="Max retries per LLM", ge=1, le=10)
    request_timeout: int = Field(default=60, description="Request timeout in seconds", ge=10, le=300)
    temperature: float = Field(default=0.7, description="LLM temperature", ge=0.0, le=2.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cascade_order")
    @classmethod
    def validate_cascade_order(cls, v: str) -> str:
        """Validate cascade order contains valid LLM names."""
        valid_llms = {"deepseek", "gemini", "gpt5", "claude"}
        llms = [llm.strip() for llm in v.split(",")]
        invalid = set(llms) - valid_llms
        if invalid:
            raise ValueError(f"Invalid LLM names in cascade_order: {invalid}")
        return v

    def get_cascade_list(self) -> List[str]:
        """Get cascade order as a list."""
        return [llm.strip() for llm in self.cascade_order.split(",")]


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    redis_use_fakeredis: bool = Field(default=True, description="Use FakeRedis instead of real Redis")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port", ge=1, le=65535)
    redis_db: int = Field(default=0, description="Redis database number", ge=0, le=15)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class AppSettings(BaseSettings):
    """Application configuration settings."""

    log_level: str = Field(default="INFO", description="Logging level")
    max_conversation_length: int = Field(default=50, description="Max messages in conversation history", ge=10, le=1000)
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    plots_dir: Path = Field(default=Path("./plots"), description="Plots directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class Settings:
    """
    Main settings class that combines all configuration.

    Usage:
        settings = get_settings()
        api_key = settings.api_keys.deepseek_api_key
        model = settings.llm.deepseek_model
    """

    def __init__(self):
        self.api_keys = APIKeysSettings()
        self.llm = LLMSettings()
        self.redis = RedisSettings()
        self.app = AppSettings()

        # Ensure output directories exist
        self.app.ensure_directories()

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  LLM Cascade: {self.llm.get_cascade_list()}\n"
            f"  Redis: {'FakeRedis' if self.redis.redis_use_fakeredis else 'Real Redis'}\n"
            f"  Log Level: {self.app.log_level}\n"
            f")"
        )


# Singleton instance
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """
    Get settings singleton instance.

    Returns:
        Settings instance with all configuration loaded
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """
    Reload settings from .env file.

    Returns:
        New Settings instance
    """
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
