"""LLM client with persistent disk caching, retry logic, and MockLLM fallback."""

from __future__ import annotations

import concurrent.futures
import hashlib
import sqlite3
import time
from pathlib import Path

from openai import APIConnectionError, OpenAI, RateLimitError
from pydantic_settings import BaseSettings, SettingsConfigDict

from leo.llm.mock import MockLLM


class LLMSettings(BaseSettings):
    """Configuration for LLM client."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    leo_model_name: str = "gpt-3.5-turbo-0613"
    leo_mock_llm: bool = True
    leo_cache_dir: str = ".leo_cache"
    max_retries: int = 5
    initial_retry_delay: float = 1.0

    # Default optimization hyperparameters
    leo_default_pop_size: int = 10
    leo_default_max_iters: int = 30
    leo_default_jitter_scale: float = 1e-4
    leo_default_sol_to_port: int = 1


class CacheDB:
    """SQLite-backed key-value disk cache for LLM queries."""

    def __init__(self, cache_dir: str | Path = ".leo_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.sqlite3"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    model TEXT,
                    prompt TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_hash(model: str, prompt: str, temperature: float = 0.7) -> str:
        content = f"{model}::{temperature:.2f}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, key_hash: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response FROM prompt_cache WHERE prompt_hash = ?", (key_hash,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set(self, key_hash: str, model: str, prompt: str, response: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_cache (prompt_hash, model, prompt, response) VALUES (?, ?, ?, ?)",
                (key_hash, model, prompt, response),
            )
            conn.commit()


class LLMClient:
    """Robust LLM client handling API calls, rate limits, caching, and offline mock."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()
        # If API key is empty, force mock mode
        if not self.settings.openai_api_key or self.settings.openai_api_key == "sk-...":
            self.settings.leo_mock_llm = True

        self.cache = CacheDB(self.settings.leo_cache_dir)
        self.mock_llm = MockLLM()
        self._sync_client: OpenAI | None = None

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._sync_client

    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """Query LLM with persistent disk caching and exponential backoff retry."""
        if self.settings.leo_mock_llm:
            return self.mock_llm.generate(prompt)

        key_hash = self.cache.compute_hash(
            self.settings.leo_model_name, prompt, temperature
        )
        cached_res = self.cache.get(key_hash)
        if cached_res is not None:
            return cached_res

        # Execute API call with exponential backoff on HTTP 429
        delay = self.settings.initial_retry_delay
        for attempt in range(self.settings.max_retries):
            try:
                response = self.sync_client.chat.completions.create(
                    model=self.settings.leo_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    extra_body={"enable_thinking": False},
                )
                text = response.choices[0].message.content or ""
                self.cache.set(key_hash, self.settings.leo_model_name, prompt, text)
                return text
            except (RateLimitError, APIConnectionError):
                if attempt == self.settings.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0

        raise RuntimeError("Failed to complete LLM query after maximum retries.")

    def query_parallel(self, prompts: list[str], temperature: float = 0.7) -> list[str]:
        """Execute parallel LLM queries using ThreadPoolExecutor."""
        if len(prompts) == 1:
            return [self.query(prompts[0], temperature)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(self.query, p, temperature) for p in prompts]
            return [f.result() for f in futures]

    async def aquery(self, prompt: str, temperature: float = 0.7) -> str:
        """Async compatibility wrapper."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.query, prompt, temperature)

    async def aquery_parallel(
        self, prompts: list[str], temperature: float = 0.7
    ) -> list[str]:
        """Async parallel compatibility wrapper."""
        import asyncio
        tasks = [self.aquery(p, temperature) for p in prompts]
        return await asyncio.gather(*tasks)

