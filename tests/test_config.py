"""Configuration tests. load_dotenv is patched so a local .env cannot leak in."""

from __future__ import annotations

import pytest

import src.config as config
from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    ConfigError,
    get_settings,
    require_api_key,
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    for name in (
        "GEMINI_MODEL",
        "CHUNK_SIZE_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
        "SINGLE_PASS_TOKEN_LIMIT",
        "GEMINI_TEMPERATURE",
        "REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()
    assert settings.api_key == "abc"
    assert settings.has_api_key is True
    assert settings.model == DEFAULT_MODEL
    assert settings.chunk_size_tokens == DEFAULT_CHUNK_SIZE
    assert settings.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP
    assert settings.temperature == 0.2


def test_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "34")
    monkeypatch.setenv("SINGLE_PASS_TOKEN_LIMIT", "999")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "42")

    settings = get_settings()
    assert settings.model == "gemini-custom"
    assert settings.chunk_size_tokens == 1234
    assert settings.chunk_overlap_tokens == 34
    assert settings.single_pass_token_limit == 999
    assert settings.temperature == 0.7
    assert settings.request_timeout_seconds == 42


def test_missing_key_is_not_fatal_until_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = get_settings()
    assert settings.has_api_key is False
    with pytest.raises(ConfigError, match="GEMINI_API_KEY is not set"):
        require_api_key()


def test_overlap_must_be_smaller_than_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "100")
    with pytest.raises(ConfigError, match="must be smaller than"):
        get_settings()


def test_non_numeric_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "abc")
    with pytest.raises(ConfigError, match="must be an integer"):
        get_settings()


def test_negative_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "-5")
    with pytest.raises(ConfigError, match="must be positive"):
        get_settings()
