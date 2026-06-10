from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    app_name: str = "Actus"
    app_version: str = "0.1.0"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    database_url: str = "sqlite:///./actus.db"
    default_model: str = "ollama/mistral"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = False
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 10_080  # 7 days
    algorithm: str = "HS256"
    scheduler_enabled: bool = True
    # Retry settings
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    # Restrict which models callers may request via /llm/*. Empty = no restriction.
    allowed_models: list[str] = []
    # How long audit log entries (incl. IP addresses) are retained before purge
    audit_log_retention_days: int = 90
    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    # Redis pub/sub for cross-process SSE fan-out; falls back to DB polling if empty
    redis_url: str = ""

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if not self.debug:
            if self.secret_key == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set in production")
            if len(self.secret_key) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters")
            if "*" in self.cors_origins:
                raise ValueError(
                    "CORS_ORIGINS is '*' — set it to your actual domain(s) in production, "
                    "e.g. CORS_ORIGINS=https://app.example.com"
                )
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()