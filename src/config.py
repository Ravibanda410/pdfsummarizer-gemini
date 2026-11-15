"""Settings loaded from the environment (with a .env fallback).

The `.env` file is optional at import time so that pure-offline tools (PDF
inspection, tests) keep working without an API key. Call `require_api_key()`
when you actually need to talk to Gemini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_CHUNK_SIZE = 6000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SINGLE_PASS_LIMIT = 300_000


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader. Real environment variables always win."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    single_pass_token_limit: int
    temperature: float
    request_timeout_seconds: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    chunk_size = _int_env("CHUNK_SIZE_TOKENS", DEFAULT_CHUNK_SIZE)
    overlap = _int_env("CHUNK_OVERLAP_TOKENS", DEFAULT_CHUNK_OVERLAP)
    if overlap >= chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP_TOKENS ({overlap}) must be smaller than "
            f"CHUNK_SIZE_TOKENS ({chunk_size})"
        )

    return Settings(
        api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=overlap,
        single_pass_token_limit=_int_env("SINGLE_PASS_TOKEN_LIMIT", DEFAULT_SINGLE_PASS_LIMIT),
        temperature=_float_env("GEMINI_TEMPERATURE", 0.2),
        request_timeout_seconds=_int_env("REQUEST_TIMEOUT_SECONDS", 180),
    )


def require_api_key() -> Settings:
    settings = get_settings()
    if not settings.has_api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export GEMINI_API_KEY in your shell."
        )
    return settings
