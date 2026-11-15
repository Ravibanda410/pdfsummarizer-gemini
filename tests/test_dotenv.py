"""`.env` resolution rules - the part that decides whether the app has a key.

These deliberately call the real `load_dotenv` (tests/test_config.py patches it
out), and restore the environment afterwards so tests stay order-independent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.config as config

KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL")


@pytest.fixture(autouse=True)
def _isolate_env():
    saved = {name: os.environ.get(name) for name in KEYS}
    for name in KEYS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        '# a comment\nGEMINI_API_KEY="from-file"\nGEMINI_MODEL=gemini-from-file\n',
        encoding="utf-8",
    )
    return path


def test_reads_values_from_file(env_file: Path) -> None:
    config.load_dotenv(env_file)
    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["GEMINI_MODEL"] == "gemini-from-file"


def test_blank_env_var_does_not_hide_the_file(
    monkeypatch: pytest.MonkeyPatch, env_file: Path
) -> None:
    # Regression: os.environ.setdefault() left the blank value in place, so the
    # app reported "GEMINI_API_KEY is not set" with a perfectly good .env on disk.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    config.load_dotenv(env_file)
    assert os.environ["GEMINI_API_KEY"] == "from-file"


def test_real_env_var_still_wins(monkeypatch: pytest.MonkeyPatch, env_file: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    config.load_dotenv(env_file)
    assert os.environ["GEMINI_API_KEY"] == "from-env"


def test_blank_value_in_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    config.load_dotenv(path)
    assert "GEMINI_API_KEY" not in os.environ


def test_missing_file_is_a_noop(tmp_path: Path) -> None:
    config.load_dotenv(tmp_path / "does-not-exist.env")
    assert "GEMINI_API_KEY" not in os.environ


def test_reload_settings_picks_up_a_new_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # get_settings() is lru_cached, so a new .env only applies after cache_clear.
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert config.reload_settings().has_api_key is False

    (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\n", encoding="utf-8")
    assert config.reload_settings().has_api_key is True
