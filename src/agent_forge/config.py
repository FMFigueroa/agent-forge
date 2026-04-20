from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    orchestrator_model: str = "claude-opus-4-7"
    specialist_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"

    db_url: str = "sqlite:///./agent_forge.db"
    log_level: str = "INFO"


settings = Settings()  # type: ignore[call-arg]
