from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the LLM Universal Adapter.
    Loads values securely from environment variables or a .env file.
    """

    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_temperature: float = 0.7

    # Configuration for pydantic-settings
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instantiate global settings
settings = Settings()
