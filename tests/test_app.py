"""UI smoke tests using Streamlit's AppTest harness (offline, no API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.config as config

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_renders_and_warns_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = _run()

    assert not at.exception
    assert any("GEMINI_API_KEY" in error.value for error in at.error)

    labels = {metric.label for metric in at.metric}
    assert {"Pages", "Tokens", "Strategy", "API calls"} <= labels


def test_renders_with_key_and_offers_summarize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-render-only")
    at = _run()

    assert not at.exception
    assert not at.error
    assert any(button.label == "Summarize" for button in at.button)


def test_options_are_wired_to_the_sidebar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-render-only")
    at = _run()

    assert len(at.selectbox) == 1
    assert len(at.selectbox[0].options) == 3
    assert "Executive" in at.selectbox[0].options[0]
    assert len(at.slider) == 1
    assert at.slider[0].value == 400
    assert any(toggle.label == "Stream output" for toggle in at.toggle)


def test_configuration_panel_offers_a_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = _run()

    assert not at.exception
    buttons = {button.label: button for button in at.button}
    assert "Reload settings" in buttons

    # Reload must clear the lru_cache and re-render cleanly.
    buttons["Reload settings"].click().run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in error.value for error in at.error)
