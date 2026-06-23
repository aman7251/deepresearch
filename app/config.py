"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection
    llm_provider: str = "groq"  # "groq" | "ollama"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"

    # Infra
    redis_url: str = "redis://localhost:6379"
    api_base_url: str = "http://localhost:8000"

    # Behaviour
    demo_mode: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_results_per_query: int = 4
    max_subquestions: int = 4

    @property
    def active_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.groq_model

    @property
    def active_base_url(self) -> str:
        return self.ollama_base_url if self.llm_provider == "ollama" else self.groq_base_url

    @property
    def active_api_key(self) -> str:
        # Ollama ignores the key but the OpenAI client requires a non-empty string.
        return "ollama" if self.llm_provider == "ollama" else self.groq_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
