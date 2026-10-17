"""LLM client with persistent disk caching, retry logic, and MockLLM fallback."""

from __future__ import annotations

import concurrent.futures
import hashlib
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

from openai import (
    APIConnectionError,
    DefaultHttpxClient,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
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
                    latency REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            try:
                conn.execute(
                    "ALTER TABLE prompt_cache ADD COLUMN latency REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()

    @staticmethod
    def compute_hash(
        model: str, prompt: str, temperature: float = 0.7, extra: str = ""
    ) -> str:
        content = f"{model}::{temperature:.2f}::{extra}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, key_hash: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response FROM prompt_cache WHERE prompt_hash = ?", (key_hash,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_with_latency(self, key_hash: str) -> tuple[str, float] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, COALESCE(latency, 0.0) FROM prompt_cache WHERE prompt_hash = ?",
                (key_hash,),
            )
            row = cursor.fetchone()
            return (str(row[0]), float(row[1])) if row else None

    def set(
        self,
        key_hash: str,
        model: str,
        prompt: str,
        response: str,
        latency: float = 0.0,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_cache (prompt_hash, model, prompt, response, latency) VALUES (?, ?, ?, ?, ?)",
                (key_hash, model, prompt, response, latency),
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
            parsed = urlparse(self.settings.openai_base_url)
            # If target is localhost / 127.0.0.1, bypass any system proxies (e.g. Clash/Surge)
            # that might intercept localhost requests and return HTTP 503 Service Unavailable.
            is_local = parsed.hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
            http_client = DefaultHttpxClient(trust_env=False) if is_local else None
            self._sync_client = OpenAI(
                api_key=self.settings.openai_api_key or "sk-dummy-local",
                base_url=self.settings.openai_base_url,
                http_client=http_client,
            )
        return self._sync_client

    def query(
        self,
        prompt: str,
        temperature: float = 0.7,
        cache_key_extra: str = "",
        bypass_cache: bool = False,
    ) -> str:
        """Query LLM with persistent disk caching and exponential backoff retry."""
        if self.settings.leo_mock_llm:
            return self.mock_llm.generate(prompt)

        key_hash = self.cache.compute_hash(
            self.settings.leo_model_name, prompt, temperature, extra=cache_key_extra
        )
        if not bypass_cache:
            cached_res = self.cache.get(key_hash)
            if cached_res is not None:
                return cached_res

        # Execute API call with exponential backoff on HTTP 429/500/502/503
        delay = self.settings.initial_retry_delay
        for attempt in range(self.settings.max_retries):
            try:
                t_req = time.perf_counter()
                response = self.sync_client.chat.completions.create(
                    model=self.settings.leo_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=temperature,
                    # extra_body={"enable_thinking": False},
                    extra_body={"reasoning_effort": "none"},
                )
                req_latency = time.perf_counter() - t_req
                text = response.choices[0].message.content or ""
                self.cache.set(
                    key_hash,
                    self.settings.leo_model_name,
                    prompt,
                    text,
                    latency=req_latency,
                )
                return text
            except (RateLimitError, APIConnectionError, InternalServerError):
                if attempt == self.settings.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0

        raise RuntimeError("Failed to complete LLM query after maximum retries.")

    def query_parallel(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        cache_key_extras: list[str] | None = None,
        bypass_cache: bool = False,
    ) -> list[str]:
        """Execute parallel LLM queries using ThreadPoolExecutor."""
        if cache_key_extras is None:
            cache_key_extras = ["" for _ in prompts]

        if len(prompts) == 1:
            return [
                self.query(
                    prompts[0],
                    temperature,
                    cache_key_extra=cache_key_extras[0],
                    bypass_cache=bypass_cache,
                )
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            futures = [
                executor.submit(self.query, p, temperature, extra, bypass_cache)
                for p, extra in zip(prompts, cache_key_extras)
            ]
            return [f.result() for f in futures]

    async def aquery(
        self,
        prompt: str,
        temperature: float = 0.7,
        cache_key_extra: str = "",
        bypass_cache: bool = False,
    ) -> str:
        """Async compatibility wrapper."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.query, prompt, temperature, cache_key_extra, bypass_cache
        )

    async def aquery_parallel(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        cache_key_extras: list[str] | None = None,
        bypass_cache: bool = False,
    ) -> list[str]:
        """Async parallel compatibility wrapper."""
        import asyncio

        if cache_key_extras is None:
            cache_key_extras = ["" for _ in prompts]

        tasks = [
            self.aquery(p, temperature, extra, bypass_cache)
            for p, extra in zip(prompts, cache_key_extras)
        ]
        return await asyncio.gather(*tasks)
