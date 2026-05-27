from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    app_name: str = "Actus"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    database_url: str = "sqlite:///./actus.db"
    default_model: str = "ollama/mistral"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"
    # Retry settings
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if not self.debug:
            if self.secret_key == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set in production")
            if len(self.secret_key) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters")
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()